"""Round 2 sweep around round-1 winner D (6 layers, 256-dim, ctx 128, cosine lr). VAL ONLY, test never loaded.
Run from the project root:  PYTHONPATH=src:scripts python3 scripts/sweep2.py"""
import json
import math
import time
from pathlib import Path

import torch

import sweep
from model import MiniGPT

BASE = dict(n_layer=6, n_head=4, n_embd=256, dropout=0.3, block_size=128, lr=6e-4, cosine=True,
            wd=0.1, batch=32, tie=False)
RUNS = {
    "R2_D_repeat":        {},
    "R2_drop0.4":         dict(dropout=0.4),
    "R2_wd0.3":           dict(wd=0.3),
    "R2_tie":             dict(tie=True),
    "R2_tie_drop0.4":     dict(tie=True, dropout=0.4),
    "R2_tie_drop0.4_lr1e-3": dict(tie=True, dropout=0.4, lr=1e-3),
    "R2_batch64_lr8e-4":  dict(batch=64, lr=8e-4),
    "R2_8L_tie_drop0.4":  dict(n_layer=8, tie=True, dropout=0.4),
}


def run(name, p):
    torch.manual_seed(42)
    cfg = sweep.make_cfg(p)
    cfg.tie_weights = p["tie"]
    sweep.BATCH = p["batch"]
    model = MiniGPT(cfg).to(sweep.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["wd"])
    params = sum(x.numel() for x in model.parameters()) / 1e6
    best, best_state, best_step, bad, t0 = float("inf"), None, 0, 0, time.time()
    for step in range(1, sweep.STEPS + 1):
        lr = p["lr"] * min(1, step / sweep.WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / sweep.STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        _, loss = model(*sweep.batch(sweep.train_data, cfg.block_size))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % sweep.EVAL_EVERY == 0:
            vl, _, _ = sweep.full_val(model, cfg.block_size)
            if vl < best:
                best, best_step, bad = vl, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= sweep.PATIENCE:
                    break
    model.load_state_dict(best_state)
    vl, a1, a5 = sweep.full_val(model, cfg.block_size)
    res = dict(params_M=round(params, 2), best_step=best_step, val_loss=round(vl, 4),
               val_ppl=round(math.exp(vl), 1), val_top1=round(a1, 2), val_top5=round(a5, 2),
               minutes=round((time.time() - t0) / 60, 1), **p)
    print(name, json.dumps(res), flush=True)
    Path("checkpoints/sweep").mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": p, "step": best_step, "val_loss": vl},
               f"checkpoints/sweep/{name}.pt")
    return res


if __name__ == "__main__":
    results = {name: run(name, {**BASE, **over}) for name, over in RUNS.items()}
    Path("results/sweep2.json").write_text(json.dumps(results, indent=2))
