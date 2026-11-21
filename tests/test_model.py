"""The three model checks from docs/03_minigpt_plan.md."""
import sys

import torch

sys.path.insert(0, "src")
from model import Config, MiniGPT  # noqa: E402


def test_output_shape():
    model = MiniGPT()
    x = torch.randint(0, 2000, (2, 16))
    logits, _ = model(x)
    assert logits.shape == (2, 16, 2000)


def test_future_tokens_hidden():
    model = MiniGPT().eval()
    x = torch.randint(0, 2000, (1, 10))
    y = x.clone()
    y[0, 7:] = (y[0, 7:] + 1) % 2000  # change only the later tokens
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[0, :7], b[0, :7], atol=1e-5)


def test_can_memorize_one_batch():
    torch.manual_seed(0)
    cfg = Config()
    cfg.dropout = 0.0
    model = MiniGPT(cfg)
    x = torch.randint(0, 2000, (4, 32))
    y = torch.randint(0, 2000, (4, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
