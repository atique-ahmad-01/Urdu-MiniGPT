# 04 — Training, Testing & Report

## Training

| Setting | Value |
|---------|------:|
| Batch size | 32 |
| Learning rate | 3e-4 |
| Max steps | 5,000 |
| Check val every | 250 steps |
| Stop early | Val stops improving |
| Seed | 42 |

## Data use

| Split | Used for | How often |
|-------|----------|-----------|
| train | Training | Always |
| val | Tuning + early stopping | Many times |
| test | Final score | **Once** |

## Results to record

| Result | Value |
|--------|------:|
| Best val loss | |
| Test loss | |
| Test perplexity | |
| Training time | |

## Samples (5)

| # | Start text |
|---|------------|
| 1 | `پاکستان` |
| 2 | `کرکٹ میچ میں` |
| 3 | `حکومت نے` |
| 4 | (empty) |
| 5 | (empty) |

## Timeline

| Step | Time |
|------|-----:|
| Setup | 15 min |
| Clean data | 30 min |
| Tokenizer + demo | 45 min |
| Encode | 15 min |
| Model | 45 min |
| Train | 1–2 h |
| Test + samples | 30 min |
| Report | 1 h |
