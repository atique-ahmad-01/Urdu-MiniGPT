"""Next-token accuracy (top-1 / top-5), loss and perplexity of the final model on train / val / test.
Read-only: uses the finalized checkpoints/best.pt, nothing is tuned on these numbers."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from model import Config, MiniGPT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg = Config()
model = MiniGPT(cfg).to(DEVICE)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=DEVICE)["model"])
model.eval()


@torch.no_grad()
def evaluate(split):
    # same method as evaluate.py: non-overlapping 128-token windows over the whole split
    data = torch.from_numpy(np.load(f"data/encoded/{split}.npy").astype(np.int64))
    loss_sum, top1, top5, n = 0.0, 0, 0, 0
    for i in range(0, len(data) - 1, cfg.block_size):
        y = data[i + 1:i + cfg.block_size + 1].unsqueeze(0).to(DEVICE)
        x = data[i:i + y.size(1)].unsqueeze(0).to(DEVICE)
        logits, loss = model(x, y)
        loss_sum += loss.item() * y.size(1)
        best5 = logits[0].topk(5, dim=-1).indices
        top1 += (best5[:, 0] == y[0]).sum().item()
        top5 += (best5 == y[0, :, None]).any(-1).sum().item()
        n += y.size(1)
    loss = loss_sum / n
    return {"tokens": n, "loss": round(loss, 4), "perplexity": round(math.exp(loss), 2),
            "top1_accuracy": round(100 * top1 / n, 2), "top5_accuracy": round(100 * top5 / n, 2)}


results = {split: evaluate(split) for split in ["train", "val", "test"]}
lines = ["# Next-token accuracy (checkpoints/best.pt)", "",
         "| Split | Tokens | Loss | Perplexity | Top-1 acc % | Top-5 acc % |",
         "|---|---:|---:|---:|---:|---:|"]
for split, r in results.items():
    lines.append(f"| {split} | {r['tokens']:,} | {r['loss']} | {r['perplexity']} | "
                 f"{r['top1_accuracy']} | {r['top5_accuracy']} |")
print("\n".join(lines))
Path("results/accuracy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
Path("results/accuracy.json").write_text(json.dumps(results, indent=2))
