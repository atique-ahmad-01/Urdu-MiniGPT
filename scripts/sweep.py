"""Hyperparameter sweep. Selection uses VAL ONLY (loss + next-token accuracy); test is never loaded.
Run from the project root:  PYTHONPATH=src python3 scripts/sweep.py"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import Config, MiniGPT

DEVICE = "cuda"
train_data = torch.from_numpy(np.load("data/encoded/train.npy").astype(np.int64))
val_data = torch.from_numpy(np.load("data/encoded/val.npy").astype(np.int64))

RUNS = {
    "A_baseline":      dict(n_layer=4, n_head=4, n_embd=192, dropout=0.2, block_size=128, lr=3e-4, cosine=False),
    "B_cosine_lr1e-3": dict(n_layer=4, n_head=4, n_embd=192, dropout=0.2, block_size=128, lr=1e-3, cosine=True),
    "C_drop0.3":       dict(n_layer=4, n_head=4, n_embd=192, dropout=0.3, block_size=128, lr=1e-3, cosine=True),
    "D_6L_256":        dict(n_layer=6, n_head=4, n_embd=256, dropout=0.3, block_size=128, lr=6e-4, cosine=True),
    "E_6L_256_ctx256": dict(n_layer=6, n_head=4, n_embd=256, dropout=0.3, block_size=256, lr=6e-4, cosine=True),
    "F_6L_384_drop0.4": dict(n_layer=6, n_head=6, n_embd=384, dropout=0.4, block_size=256, lr=5e-4, cosine=True),
}
BATCH, STEPS, WARMUP, EVAL_EVERY, PATIENCE = 32, 5000, 200, 250, 4


def make_cfg(p):
    cfg = Config()
    for k in ("n_layer", "n_head", "n_embd", "dropout", "block_size"):
        setattr(cfg, k, p[k])
    return cfg


def batch(data, bs):
    s = torch.randint(len(data) - bs - 1, (BATCH,))
    x = torch.stack([data[i:i + bs] for i in s])
    y = torch.stack([data[i + 1:i + bs + 1] for i in s])
    return x.to(DEVICE), y.to(DEVICE)


@torch.no_grad()
def full_val(model, bs):
    model.eval()
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    for i in range(0, len(val_data) - 1, bs):
        y = val_data[i + 1:i + bs + 1].unsqueeze(0).to(DEVICE)
        x = val_data[i:i + y.size(1)].unsqueeze(0).to(DEVICE)
        logits, loss = model(x, y)
        loss_sum += loss.item() * y.size(1)
        b5 = logits[0].topk(5, -1).indices
        top1 += (b5[:, 0] == y[0]).sum().item()
        top5 += (b5 == y[0, :, None]).any(-1).sum().item()
        n += y.size(1)
    model.train()
    return loss_sum / n, 100 * top1 / n, 100 * top5 / n


def run(name, p):
    torch.manual_seed(42)
    cfg = make_cfg(p)
    model = MiniGPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=0.1)
    params = sum(x.numel() for x in model.parameters()) / 1e6
    best, best_state, best_step, bad, t0 = float("inf"), None, 0, 0, time.time()
    for step in range(1, STEPS + 1):
        if p["cosine"]:
            lr = p["lr"] * min(1, step / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
            for g in opt.param_groups:
                g["lr"] = lr
        _, loss = model(*batch(train_data, cfg.block_size))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            vl, _, _ = full_val(model, cfg.block_size)
            if vl < best:
                best, best_step, bad = vl, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    model.load_state_dict(best_state)
    vl, a1, a5 = full_val(model, cfg.block_size)
    res = dict(params_M=round(params, 2), best_step=best_step, val_loss=round(vl, 4),
               val_ppl=round(math.exp(vl), 1), val_top1=round(a1, 2), val_top5=round(a5, 2),
               minutes=round((time.time() - t0) / 60, 1), **p)
    print(name, json.dumps(res), flush=True)
    Path("checkpoints/sweep").mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "config": p, "step": best_step, "val_loss": vl},
               f"checkpoints/sweep/{name}.pt")
    return res


if __name__ == "__main__":
    results = {name: run(name, p) for name, p in RUNS.items()}
    Path("results/sweep.json").write_text(json.dumps(results, indent=2))
