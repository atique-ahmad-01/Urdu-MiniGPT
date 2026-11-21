#!/bin/sh
# Copy the trained tokenizer, final checkpoint, model code and result files into this app.
set -e
cd "$(dirname "$0")"
cp ../src/model.py minigpt.py
cp ../tokenizer/urdu_unigram.model ../tokenizer/urdu_unigram.vocab assets/
cp ../checkpoints/best.pt assets/
for f in accuracy.json test.json training.json sweep.json sweep2.json; do
  [ -f "../results/$f" ] && cp "../results/$f" assets/
done
echo "assets synced"
