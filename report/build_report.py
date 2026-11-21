"""Fill report_template.tex with numbers, Urdu examples and samples taken directly from the result files,
then compile with XeLaTeX.  Run from the project root:  python report/build_report.py"""
import json
import re
import subprocess
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
REP = ROOT / "report"


def tex(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def tok_demo():
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizer/urdu_unigram.model"))
    sents = re.findall(r"^## \d+\. (.+)$", (ROOT / "results/tokenizer_demo.md").read_text(encoding="utf-8"), re.M)
    rows = [r"\begin{tabular}{r >{\raggedleft\arraybackslash}p{0.33\textwidth} >{\raggedleft\arraybackslash}p{0.43\textwidth} r}", r"\toprule",
            r"\# & Sentence & Pieces & Tokens \\", r"\midrule"]
    for i, s in enumerate(sents, 1):
        pieces = []
        for p in sp.encode(s, out_type=str):
            if p == "▁":
                continue
            pieces.append(r"\ws{" + tex(p[1:]) + "}" if p.startswith("▁") else tex(p))
        n = len(sp.encode(s))
        rows.append(f"{i} & \\ur{{{tex(s)}}} & \\ur{{{' | '.join(pieces)}}} & {n} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows)


def sweep():
    runs = {}
    for f in ("sweep.json", "sweep2.json"):
        runs.update(json.loads((ROOT / "results" / f).read_text()))
    rows = [r"\begin{tabular}{lrrrrrrcrr}", r"\toprule",
            r"Run & Layers & Dim & Dropout & Context & LR & WD & Tied & Val loss & Val top-1 \\", r"\midrule"]
    best = max(r["val_top1"] for r in runs.values())
    for name, r in sorted(runs.items(), key=lambda kv: -kv[1]["val_top1"]):
        top = f"{r['val_top1']:.2f}\\%"
        row = (f"{tex(name)} & {r['n_layer']} & {r['n_embd']} & {r['dropout']} & {r['block_size']} & "
               f"{r['lr']:.0e} & {r.get('wd', 0.1)} & {'yes' if r.get('tie') else 'no'} & {r['val_loss']:.3f} & {top}")
        rows.append((r"\textbf{" + row.replace(" & ", r"} & \textbf{") + "}" if r["val_top1"] == best else row)
                    + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows).replace("e-04", r"e-4").replace("e-03", r"e-3")


def acc():
    a = json.loads((ROOT / "results/accuracy.json").read_text())
    return "\n".join(f"{s} & {r['loss']:.3f} & {r['perplexity']:.1f} & {r['top1_accuracy']:.1f}\\% & "
                     f"{r['top5_accuracy']:.1f}\\% \\\\" for s, r in a.items())


def curve():
    h = json.loads((ROOT / "results/training.json").read_text())["history"]
    return (" ".join(f"({r['step']},{r['train_loss']:.3f})" for r in h),
            " ".join(f"({r['step']},{r['val_loss']:.3f})" for r in h))


def samples():
    md = (ROOT / "results/samples.md").read_text(encoding="utf-8")
    out = []
    for prompt, text in re.findall(r"## \d+\. Prompt: `(.*?)`\n\n(.+?)(?:\n\n|\Z)", md, re.S):
        label = r"(empty prompt)" if prompt == "(empty)" else r"prompt \ur{" + tex(prompt) + "}"
        out.append(r"\medskip\noindent\textbf{Sample " + str(len(out) + 1) + "} -- " + label + r"\par\nopagebreak" + "\n"
                   r"\begin{urdu}" + tex(text.strip()) + r"\end{urdu}" + "\n")
    return "\n".join(out)


t = (REP / "report_template.tex").read_text(encoding="utf-8")
tr, va = curve()
for key, val in {"%%TOKDEMO%%": tok_demo(), "%%SWEEP%%": sweep(), "%%ACC%%": acc(),
                 "%%TRAIN%%": tr, "%%VAL%%": va, "%%SAMPLES%%": samples()}.items():
    t = t.replace(key, val)
(REP / "report.tex").write_text(t, encoding="utf-8")
for _ in range(2):
    r = subprocess.run(["xelatex", "-interaction=nonstopmode", "report.tex"], cwd=REP,
                       capture_output=True, text=True)
print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("!", "Output written", "LaTeX Warning")))[-2500:])
