"""Urdu-MiniGPT — Streamlit demo.

Run from this folder:   streamlit run app.py
Assets (tokenizer, checkpoint, result JSONs) live in ./assets — refresh them with ./sync_assets.sh
"""
import html
import json
import time
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import streamlit as st
import torch
import torch.nn.functional as F

from minigpt import Config, MiniGPT

ASSETS = Path(__file__).parent / "assets"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Urdu-MiniGPT", page_icon=str(ASSETS / "logo.svg"), layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&family=Source+Sans+3:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
:root { --indigo:#4338ca; --indigo-soft:#eef2ff; --line:#e2e8f0; --muted:#64748b; --ink:#0f172a; }
html, body, [class*="css"], .stMarkdown, button, textarea { font-family:'Source Sans 3', sans-serif; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none; }
.block-container { max-width:1080px; padding-top:3.2rem; padding-bottom:3rem; }
/* hero */
.hero { text-align:center; padding:1.9rem 1.5rem 1.6rem; border-radius:22px; border:1px solid #dbe3f4;
        background:linear-gradient(180deg,#e3e9fb 0%,#f1f3fd 55%,#fafbff 100%); margin-bottom:1.4rem; }
.hero .logo { font-size:2.6rem; line-height:1; }
.hero .logo svg { width:52px; height:58px; display:inline-block; }
.hero h1 { margin:.35rem 0 .15rem; font-size:2.35rem; font-weight:800; color:var(--indigo); letter-spacing:-.01em; padding:0; }
.hero .sub { color:#475569; font-size:1.08rem; margin-bottom:1rem; }
.hero .sub .ur { font-family:'Noto Nastaliq Urdu',serif; }
.pill { display:inline-block; background:#fff; border:1px solid #dbe3f4; border-radius:999px;
        padding:.3rem .8rem; margin:.15rem .2rem; font-size:.86rem; font-weight:700; color:#1e293b; }
/* tabs */
.stTabs [data-baseweb="tab-list"] { gap:1.4rem; border-bottom:2px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding:.4rem 0; font-weight:600; }
/* inputs */
textarea { direction:rtl; font-family:'Noto Nastaliq Urdu',serif !important; font-size:1.2rem !important;
           line-height:2.2 !important; background:#f8fafc !important; border-radius:12px !important; }
.stButton > button, [data-testid="stPopover"] button { border-radius:10px; height:2.75rem; font-weight:600; }
.stButton > button[kind="primary"] { background:#2563eb; border-color:#2563eb; }
.hint { color:var(--muted); font-size:.92rem; margin:.4rem 0 0; }
/* result banner + cards */
.banner { display:flex; justify-content:space-between; align-items:center; margin:1.3rem 0 1rem;
          padding:.85rem 1.1rem; border-radius:12px; background:#ecfdf5; border:1px solid #a7f3d0; }
.banner b { color:#047857; font-size:1.05rem; }
.banner span { color:#059669; font-size:.85rem; font-weight:600; }
.rcard { border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem 1.1rem; background:#fff;
         height:100%; margin-bottom:1rem; }
.rcard h4 { margin:0 0 .5rem; font-size:1rem; font-weight:700; color:var(--ink); padding:0; }
.urdu { font-family:'Noto Nastaliq Urdu',serif; direction:rtl; text-align:right; font-size:1.25rem;
        line-height:2.7; color:var(--ink); }
.ptag { display:inline-block; background:#2563eb; color:#fff; border-radius:7px; padding:0 .5rem; margin-left:.25rem; }
/* token chips (piece + small-caps id, like entity tags) */
.chips { direction:rtl; text-align:right; line-height:3; }
.tok { display:inline-flex; align-items:baseline; gap:.3rem; margin:.12rem; padding:.05rem .5rem;
       border-radius:8px; border:1.5px solid; font-family:'Noto Nastaliq Urdu',serif; font-size:1.15rem; }
.tok small { font-family:'Source Sans 3',sans-serif; font-size:.62rem; font-weight:700; letter-spacing:.04em; }
.c0{border-color:#14b8a6;background:#f0fdfa;color:#0f766e}.c1{border-color:#f97316;background:#fff7ed;color:#c2410c}
.c2{border-color:#3b82f6;background:#eff6ff;color:#1d4ed8}.c3{border-color:#a855f7;background:#faf5ff;color:#7e22ce}
.c4{border-color:#ec4899;background:#fdf2f8;color:#be185d}
.idpill { display:inline-block; margin:.15rem; padding:.1rem .55rem; border-radius:7px; color:#fff;
          font-weight:700; font-size:.85rem; background:#475569; }
/* metrics */
.mgrid { display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin:.4rem 0 1rem; }
.mcard { border:1px solid var(--line); border-radius:14px; padding:.85rem 1rem; background:#fff; }
.mcard .k { color:var(--muted); font-size:.82rem; font-weight:600; }
.mcard .v { color:var(--ink); font-size:1.55rem; font-weight:800; }
.mcard .d { color:#059669; font-size:.8rem; font-weight:700; }
@media (max-width:700px){ .mgrid{grid-template-columns:repeat(2,1fr);} .banner{flex-direction:column;gap:.2rem;} }
.footer { text-align:center; color:var(--muted); font-size:.85rem; margin-top:2.6rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- loading
@st.cache_resource
def load_tokenizer():
    return spm.SentencePieceProcessor(model_file=str(ASSETS / "urdu_unigram.model"))


@st.cache_resource
def load_model():
    ckpt = torch.load(ASSETS / "best.pt", map_location=DEVICE)
    model = MiniGPT(Config()).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("step"), ckpt.get("val_loss")


@st.cache_data
def load_json(name):
    p = ASSETS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@torch.no_grad()
def generate(model, ids, max_new, temperature, top_k, rep_penalty, eos_id):
    idx = torch.tensor([ids], device=DEVICE)
    for _ in range(max_new):
        logits = model(idx[:, -model.cfg.block_size:])[0][:, -1, :]
        if rep_penalty != 1.0:  # down-weight tokens already in the text (stops "X X X" loops)
            seen = torch.unique(idx[0])
            vals = logits[0, seen]
            logits[0, seen] = torch.where(vals > 0, vals / rep_penalty, vals * rep_penalty)
        logits = logits / max(temperature, 1e-5)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
        if nxt.item() == eos_id:
            break
    return idx[0].tolist()


def chips(pieces, ids):
    out = "".join(f'<span class="tok c{i % 5}">{html.escape(p.replace("▁", "␣"))}<small>{t}</small></span>'
                  for i, (p, t) in enumerate(zip(pieces, ids)))
    return f'<div class="chips">{out}</div>'


def card(title, body):
    return f'<div class="rcard"><h4>{title}</h4>{body}</div>'


sp = load_tokenizer()
model, best_step, best_val = load_model()
acc = load_json("accuracy.json") or {}
test = load_json("test.json") or {}
training = load_json("training.json") or {}
n_params = sum(p.numel() for p in model.parameters())
cfg = model.cfg
test_top1 = acc.get("test", {}).get("top1_accuracy")
LOGO = (ASSETS / "logo.svg").read_text() if (ASSETS / "logo.svg").exists() else "🪶"

# ---------------------------------------------------------------- hero
st.markdown(f"""
<div class="hero">
  <div class="logo">{LOGO}</div>
  <h1>Urdu-MiniGPT</h1>
  <div class="sub">A tiny GPT that writes Urdu — trained from scratch. <span class="ur">اردو میں لکھیں</span></div>
  <span class="pill">🧩 SentencePiece Unigram · {sp.get_piece_size():,} tokens</span>
  <span class="pill">🧠 {n_params / 1e6:.1f}M parameters · no pretrained weights</span>
  <span class="pill">📊 {f"{test_top1:.1f}% next-token accuracy on test" if test_top1 else "test results"}</span>
</div>
""", unsafe_allow_html=True)

tab_try, tab_tok, tab_res, tab_how = st.tabs(["🪶 Try it", "🔤 Tokenizer", "📊 Results", "⚙️ How it works"])

# ---------------------------------------------------------------- try it
EXAMPLES = {"🇵🇰 Pakistan": "پاکستان", "🏏 Cricket": "کرکٹ میچ میں", "🏛️ Government": "حکومت نے",
            "🌆 Lahore": "لاہور میں آج", "🎤 Prime Minister": "وزیر اعظم نے کہا کہ"}


def _pick_example():
    choice = st.session_state.get("ex_pick")
    if choice:
        st.session_state["prompt"] = EXAMPLES[choice]


def _clear():
    st.session_state["prompt"] = ""
    st.session_state["ex_pick"] = None
    st.session_state.pop("outputs", None)


with tab_try:
    st.markdown("Try an example")
    st.pills("examples", list(EXAMPLES), key="ex_pick", on_change=_pick_example, label_visibility="collapsed")
    prompt = st.text_area("prompt", key="prompt", height=120, label_visibility="collapsed",
                          placeholder="یہاں اردو کا ابتدائی جملہ لکھیں… (Type an Urdu opening, or leave empty)")
    b1, b2, b3 = st.columns([2.1, 1, 1.45])
    go = b1.button("🪶 Write Urdu text", type="primary", width="stretch")
    b2.button("Clear", width="stretch", on_click=_clear)
    with b3.popover("⚙️ Settings", width="stretch"):
        temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05, help="Lower = safer, higher = more creative.")
        top_k = st.slider("Top-k", 0, 200, 40, 5, help="Sample only from the k most likely tokens (0 = all).")
        rep_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.15, 0.05, help="Stops repeating loops.")
        max_new = st.slider("Max new tokens", 10, 300, 100, 10)
        n_samples = st.slider("Samples", 1, 4, 2)
        seed = st.number_input("Seed", 0, 99999, 42, help="Same seed + settings = same text.")

    if go:
        torch.manual_seed(int(seed))
        ids = [sp.eos_id()] + sp.encode(prompt.strip())  # training docs always start after </s>
        t0 = time.time()
        with st.spinner("Writing…"):
            outs = [generate(model, ids, max_new, temperature, top_k, rep_penalty, sp.eos_id())
                    for _ in range(n_samples)]
        st.session_state["outputs"] = dict(prompt=prompt.strip(), n_prompt=len(ids), outs=outs,
                                           ms=(time.time() - t0) * 1000)

    res = st.session_state.get("outputs")
    if res:
        new_tok = sum(len(o) - res["n_prompt"] for o in res["outs"])
        st.markdown(f'<div class="banner"><b>✅ Wrote {len(res["outs"])} sample'
                    f'{"s" if len(res["outs"]) > 1 else ""} · {new_tok} new tokens</b>'
                    f'<span>MiniGPT {n_params / 1e6:.1f}M · {DEVICE.upper()} · {res["ms"]:.0f} ms</span></div>',
                    unsafe_allow_html=True)
        cols = st.columns(2)
        for i, out in enumerate(res["outs"]):
            cont = html.escape(sp.decode(out[res["n_prompt"]:]))
            head = f'<span class="ptag">{html.escape(res["prompt"])}</span> ' if res["prompt"] else ""
            cols[i % 2].markdown(card(f"📝 Sample {i + 1}", f'<div class="urdu">{head}{cont}</div>'),
                                 unsafe_allow_html=True)
        if res["prompt"]:
            p_ids = sp.encode(res["prompt"])
            st.markdown(card("🔤 How your prompt was tokenized",
                             chips(sp.encode(res["prompt"], out_type=str), p_ids)), unsafe_allow_html=True)
    else:
        st.markdown('<p class="hint">Pick an example above, or type your own Urdu opening and press the button.</p>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- tokenizer
with tab_tok:
    text = st.text_area("tok", "کھلاڑیوں نے بہت محنت کی اور پاکستان نے میچ جیت لیا۔", height=100,
                        label_visibility="collapsed")
    if text.strip():
        pieces, ids = sp.encode(text, out_type=str), sp.encode(text)
        ok = sp.decode(ids) == text
        st.markdown(f'<div class="banner"><b>✅ Split into {len(ids)} tokens</b>'
                    f'<span>{len(text.split())} words · {len(text) / max(len(ids), 1):.2f} chars per token · '
                    f'round-trip {"exact" if ok else "changed"}</span></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(card("🔍 Pieces <small style='color:#64748b;font-weight:500'>(␣ = start of a word)</small>",
                         chips(pieces, ids)), unsafe_allow_html=True)
        c2.markdown(card("🔢 Token IDs fed to the model",
                         "".join(f'<span class="idpill">{t}</span>' for t in ids)), unsafe_allow_html=True)

# ---------------------------------------------------------------- results
with tab_res:
    if acc:
        t = acc["test"]
        st.markdown(f"""<div class="mgrid">
          <div class="mcard"><div class="k">Test loss</div><div class="v">{t['loss']:.3f}</div></div>
          <div class="mcard"><div class="k">Test perplexity</div><div class="v">{t['perplexity']:.1f}</div></div>
          <div class="mcard"><div class="k">Top-1 accuracy</div><div class="v">{t['top1_accuracy']:.1f}%</div>
               <div class="d">random guess = 0.05%</div></div>
          <div class="mcard"><div class="k">Top-5 accuracy</div><div class="v">{t['top5_accuracy']:.1f}%</div></div>
        </div>""", unsafe_allow_html=True)
        df = pd.DataFrame(acc).T.rename(columns={"tokens": "Tokens", "loss": "Loss", "perplexity": "Perplexity",
                                                 "top1_accuracy": "Top-1 %", "top5_accuracy": "Top-5 %"})
        df.index = df.index.str.capitalize()
        st.dataframe(df, width="stretch")
    if training.get("history"):
        st.markdown("##### Training curve")
        h = pd.DataFrame(training["history"]).set_index("step")
        h.columns = ["Train loss", "Val loss"]
        st.line_chart(h, height=300, color=["#2563eb", "#f97316"])
        st.caption(f"Best val loss {training['best_val_loss']:.4f} · trained in "
                   f"{training['training_minutes']} min on an RTX 5070 Ti")
    rows = []
    for f in ("sweep.json", "sweep2.json"):
        for name, r in (load_json(f) or {}).items():
            rows.append({"Run": name, "Layers": r["n_layer"], "Dim": r["n_embd"], "Dropout": r["dropout"],
                         "Ctx": r["block_size"], "LR": r["lr"], "Weight decay": r.get("wd", 0.1),
                         "Tied": r.get("tie", False), "Val loss": r["val_loss"], "Val top-1 %": r["val_top1"]})
    if rows:
        st.markdown("##### Settings we tried (picked on validation only)")
        st.dataframe(pd.DataFrame(rows).sort_values("Val top-1 %", ascending=False),
                     width="stretch", hide_index=True)

# ---------------------------------------------------------------- how it works
with tab_how:
    steps = [
        ("1 · Clean", "The course's ALIF Urdu Corpus split (train / val / test) is read as UTF-8 and cleaned."),
        ("2 · Tokenize", "A SentencePiece <b>Unigram</b> tokenizer with 2,000 tokens is learned from train.txt only."),
        ("3 · Train", f"A {cfg.n_layer}-layer, {cfg.n_embd}-dim decoder-only Transformer, written from scratch, "
                      "learns to predict the next token. Dropout, weight decay and early stopping on val."),
        ("4 · Write", "Starting from your prompt, it predicts one token at a time and samples with "
                      "temperature, top-k and a repetition penalty."),
    ]
    cols = st.columns(4)
    for c, (title, body) in zip(cols, steps):
        c.markdown(card(title, f'<div style="color:#475569;font-size:.95rem">{body}</div>'),
                   unsafe_allow_html=True)
    st.caption("Token-level perplexity depends on the tokenizer, so it should not be used alone to compare "
               "BPE and Unigram models.")

st.markdown(f'<div class="footer">Runs on {DEVICE.upper()} · checkpoint step {best_step} · '
            'your text is processed in memory and not stored</div>', unsafe_allow_html=True)
