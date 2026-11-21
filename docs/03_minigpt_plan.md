# 03 — MiniGPT Model

Our own decoder-only Transformer in PyTorch. No pretrained weights.

## Size

| Setting | Value |
|---------|------:|
| Vocab | 2,000 |
| Context length | 128 tokens |
| Layers | 4 |
| Heads | 4 |
| Embedding size | 192 |
| Dropout | 0.2 |
| Parameters | about 2M |

Kept small because we only have about 0.3M training tokens.

## Parts

| Part | Job |
|------|-----|
| Token + position embedding | Turns ids into vectors |
| Masked self-attention | Each token only sees earlier tokens |
| Feed-forward | Processes each token |
| Output layer | Predicts the next token |

## Checks

| Test | Expect |
|------|--------|
| Output shape | (batch, length, 2000) |
| Future tokens hidden | Yes |
| Can memorize 1 batch | Loss near 0 |
