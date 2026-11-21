"""Step 4: encode train/val/test with the same tokenizer and record token counts."""
from pathlib import Path

import numpy as np
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="tokenizer/urdu_unigram.model")
out = Path("data/encoded")
out.mkdir(parents=True, exist_ok=True)

rows = ["| Split | Documents | Tokens |", "|---|---:|---:|"]
for split in ["train", "val", "test"]:
    docs = Path(f"data/clean/{split}.txt").read_text(encoding="utf-8").strip().split("\n")
    ids = []
    for doc in docs:
        ids += sp.encode(doc) + [sp.eos_id()]  # </s> marks the end of each document
    np.save(out / f"{split}.npy", np.array(ids, dtype=np.uint16))
    rows.append(f"| {split} | {len(docs)} | {len(ids):,} |")

Path("results").mkdir(exist_ok=True)
Path("results/token_counts.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("\n".join(rows))
