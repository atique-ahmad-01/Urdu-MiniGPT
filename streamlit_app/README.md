# Urdu-MiniGPT — Streamlit app

Self-contained demo of the trained model: generate Urdu text, inspect the Unigram tokenizer, and view results.

```bash
cd streamlit_app
pip install -r requirements.txt
./sync_assets.sh          # copy tokenizer, checkpoints/best.pt and results/*.json into assets/
streamlit run app.py
```

| Tab | Shows |
|---|---|
| ✍️ Generate | Urdu prompt → samples (temperature, top-k, repetition penalty, seed in the sidebar) |
| 🔤 Tokenizer | Pieces + token IDs as chips, token counts |
| 📊 Results | Test loss / perplexity / accuracy, training curve, hyperparameter sweep |
| ℹ️ About | Pipeline summary |

`minigpt.py` is a copy of `../src/model.py` (refreshed by `sync_assets.sh`) so the folder can be deployed on its own,
e.g. to Streamlit Community Cloud with `streamlit_app/app.py` as the entry point.
