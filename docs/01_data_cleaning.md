# 01 — Data Cleaning

## The data

| File | Characters | Documents |
|------|-----------:|----------:|
| train.txt | 800,016 | 547 |
| val.txt | 99,547 | 81 |
| test.txt | 100,884 | 70 |

One document per line.

## What we found

| Finding | Action |
|---------|--------|
| Urdu letters already correct | Nothing |
| No hidden characters or extra spaces | Nothing |
| 89 `آ` written in two parts | Fix (NFC) |
| Some `آ` missing in source (`غاز` → should be `آغاز`) | Keep, mention in report |
| Diacritics, `ﷺ`, some English words | Keep |
| **Long lines get skipped by SentencePiece (54% of train!)** | Split train into one sentence per line for tokenizer |

## Cleaning steps

| # | Step | Files |
|---|------|-------|
| 1 | Read as UTF-8 | all |
| 2 | Unicode NFC | all |
| 3 | Remove empty lines | all |
| 4 | Save to `data/clean/` | all |
| 5 | Split into sentences → `train_sentences.txt` | train only |
