"""Step 1: clean the raw Urdu files (NFC, drop empty lines) and split train into sentences."""
import re
import unicodedata
from pathlib import Path

RAW = Path("data/raw/Urdu_MiniGPT_Dataset")
CLEAN = Path("data/clean")


def clean(text):
    text = unicodedata.normalize("NFC", text)  # joins the split آ (ا + ٓ) into one letter
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def split_sentences(text):
    # break after Urdu full stop, question mark, or ! so SentencePiece doesn't skip long lines
    parts = re.split(r"(?<=[۔؟!])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    for name in ["train.txt", "val.txt", "test.txt"]:
        text = clean((RAW / name).read_text(encoding="utf-8"))
        (CLEAN / name).write_text(text + "\n", encoding="utf-8")
        print(f"{name}: {len(text):,} chars, {text.count(chr(10)) + 1} docs")

    train = (CLEAN / "train.txt").read_text(encoding="utf-8")
    sentences = [s for line in train.split("\n") for s in split_sentences(line)]
    (CLEAN / "train_sentences.txt").write_text("\n".join(sentences) + "\n", encoding="utf-8")
    print(f"train_sentences.txt: {len(sentences):,} sentences, longest {max(map(len, sentences))} chars")


if __name__ == "__main__":
    main()
