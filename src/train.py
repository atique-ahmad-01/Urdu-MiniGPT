"""Step 5: train MiniGPT on train, use val for early stopping. Test is NOT touched here."""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import Config, MiniGPT

BATCH_SIZE = 32
LR = 6e-4           # peak learning rate (chosen on val, see results/sweep*.json)
WEIGHT_DECAY = 0.3  # chosen on val
WARMUP = 200        # linear warmup steps, then cosine decay down to 10% of LR
MAX_STEPS = 5000
EVAL_EVERY = 250
EVAL_BATCHES = 50
SAVE_EVERY = 500  # also keep a separate checkpoint every 500 steps (must be a multiple of EVAL_EVERY)
PATIENCE = 4  # stop after 4 checks (1000 steps) with no val improvement

torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = Config()
train_data = torch.from_numpy(np.load("data/encoded/train.npy").astype(np.int64))
val_data = torch.from_numpy(np.load("data/encoded/val.npy").astype(np.int64))


def get_batch(data):
    starts = torch.randint(len(data) - cfg.block_size - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + cfg.block_size] for i in starts])
    y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in starts])
    return x.to(DEVICE), y.to(DEVICE)


def lr_at(step):
    warm = min(1.0, step / WARMUP)
    cosine = 0.5 * (1 + math.cos(math.pi * step / MAX_STEPS))
    return LR * warm * (0.1 + 0.9 * cosine)


@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = [model(*get_batch(data))[1].item() for _ in range(EVAL_BATCHES)]
    model.train()
    return sum(losses) / len(losses)


def main():
    print(f"Training on {DEVICE}")
    model = MiniGPT(cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Path("checkpoints").mkdir(exist_ok=True)
    history, best_val, bad_checks = [], float("inf"), 0
    start = time.time()

    for step in range(1, MAX_STEPS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        _, loss = model(*get_batch(train_data))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0:
            train_loss = estimate_loss(model, train_data)
            val_loss = estimate_loss(model, val_data)
            history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
            print(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | {time.time() - start:.0f}s")
            if step % SAVE_EVERY == 0:
                torch.save({"model": model.state_dict(), "step": step, "val_loss": val_loss},
                           f"checkpoints/step_{step}.pt")
            if val_loss < best_val:
                best_val, bad_checks = val_loss, 0
                torch.save({"model": model.state_dict(), "step": step, "val_loss": val_loss}, "checkpoints/best.pt")
            else:
                bad_checks += 1
                if bad_checks >= PATIENCE:
                    print("Val loss stopped improving, stopping early.")
                    break

    minutes = (time.time() - start) / 60
    print(f"Best val loss {best_val:.4f}, training time {minutes:.1f} min")
    Path("results").mkdir(exist_ok=True)
    summary = {"best_val_loss": best_val, "training_minutes": round(minutes, 1), "history": history}
    Path("results/training.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
