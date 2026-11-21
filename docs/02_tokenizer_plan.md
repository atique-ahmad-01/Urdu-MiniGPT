# 02 — Tokenizer (Unigram)

## How Unigram works

| Step | What happens |
|------|--------------|
| 1 | Start with a very large list of possible pieces |
| 2 | Give each piece a probability |
| 3 | Remove the pieces that matter least |
| 4 | Repeat until 2,000 pieces are left |
| 5 | To encode, pick the most likely split |

## Settings

| Setting | Value | Why |
|---------|-------|-----|
| model_type | unigram | Assigned |
| vocab_size | 2000 | Required |
| input | `train_sentences.txt` | Train only |
| character_coverage | 1.0 | Keep all Urdu characters |
| normalization | identity | We already cleaned the data |
| split_digits | True | Numbers don't waste vocab slots |
| byte_fallback | False | Saves 256 vocab slots |
| special tokens | `<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3 | |

## Demo (5+ sentences)

| Shows | Example |
|-------|---------|
| Common words | `پاکستان کرکٹ ٹیم نے میچ جیت لیا۔` |
| Word + suffix | `کھلاڑیوں` |
| Long word | `یکطرفہ` |
| English word | `کورونا وائرس` |
| Numbers | `۲۰۱۹ میں ۴۶ رنز` |

## Encoding

| Split | Token count |
|-------|------------:|
| train | (fill in) |
| val | (fill in) |
| test | (fill in) |
