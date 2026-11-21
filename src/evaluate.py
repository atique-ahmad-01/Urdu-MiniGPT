"""Step 6: run ONCE after training is final. Test loss/perplexity + 5 generated Urdu samples."""
import json
import math
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from model import Config, MiniGPT

PROMPTS = ["پاکستان", "کرکٹ میچ میں", "حکومت نے", "", ""]

torch.manual_seed(42)
cfg = Config()
sp = spm.SentencePieceProcessor(model_file="tokenizer/urdu_unigram.model")
model = MiniGPT(cfg)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location="cpu")["model"])
model.eval()


@torch.no_grad()
def test_loss():
    # walk through the whole test set in non-overlapping 128-token windows
    data = torch.from_numpy(np.load("data/encoded/test.npy").astype(np.int64))
    total, count = 0.0, 0
    for i in range(0, len(data) - 1, cfg.block_size):
        x = data[i:i + cfg.block_size].unsqueeze(0)
        y = data[i + 1:i + cfg.block_size + 1].unsqueeze(0)
        x = x[:, :y.size(1)]
        _, loss = model(x, y)
        total += loss.item() * y.size(1)
        count += y.size(1)
    return total / count


def sample(prompt):
    ids = [sp.eos_id()] + sp.encode(prompt)  # in training a new document always follows </s>
    out = model.generate(torch.tensor([ids]), max_new_tokens=80, eos_id=sp.eos_id())
    return sp.decode(out[0, 1:].tolist())


loss = test_loss()
print(f"Test loss {loss:.4f} | perplexity {math.exp(loss):.2f}")

lines = ["# Generated samples", ""]
for i, prompt in enumerate(PROMPTS, 1):
    text = sample(prompt)
    lines += [f"## {i}. Prompt: `{prompt or '(empty)'}`", "", text, ""]
    print(f"\n[{i}] {text}")

Path("results/samples.md").write_text("\n".join(lines), encoding="utf-8")
Path("results/test.json").write_text(json.dumps({"test_loss": loss, "test_perplexity": math.exp(loss)}, indent=2))
