"""Step 2 + 3: train the SentencePiece Unigram tokenizer on train only, then demo it on 5 sentences."""
from pathlib import Path

import sentencepiece as spm

MODEL_PREFIX = "tokenizer/urdu_unigram"

DEMO = [
    "پاکستان کرکٹ ٹیم نے میچ جیت لیا۔",
    "کھلاڑیوں نے بہت محنت کی۔",
    "یہ ایک یکطرفہ فیصلہ تھا۔",
    "کورونا وائرس سے بچنے کے لیے احتیاط کریں۔",
    "۲۰۱۹ میں اس نے ۴۶ رنز بنائے۔",
]


def train():
    Path("tokenizer").mkdir(exist_ok=True)
    spm.SentencePieceTrainer.train(
        input="data/clean/train_sentences.txt",
        model_prefix=MODEL_PREFIX,
        model_type="unigram",
        vocab_size=2000,
        character_coverage=1.0,
        normalization_rule_name="identity",
        split_digits=True,
        byte_fallback=False,
        max_sentence_length=32768,  # bytes; a few train lines have no full stop and are long
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        pad_piece="<pad>",
    )


def demo(sp):
    lines = ["# Tokenizer demo", "", f"SentencePiece Unigram, vocab size = {sp.get_piece_size()}", ""]
    for i, text in enumerate(DEMO, 1):
        pieces = sp.encode(text, out_type=str)
        ids = sp.encode(text)
        lines += [f"## {i}. {text}", "", f"- Pieces ({len(pieces)}): `{' | '.join(pieces)}`",
                  f"- IDs: `{ids}`", f"- Decoded: {sp.decode(ids)}", ""]
    Path("results").mkdir(exist_ok=True)
    Path("results/tokenizer_demo.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    train()
    demo(spm.SentencePieceProcessor(model_file=MODEL_PREFIX + ".model"))
