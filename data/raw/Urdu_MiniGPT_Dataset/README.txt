Urdu MiniGPT — shared dataset for Groups 1–12
Source: https://huggingface.co/datasets/orature/ALIF_Urdu_Corpus_AUC
Random seed: 42
Sample: approximately 1,000,000 characters, selected as complete documents
Split: by document, targeting 80% train / 10% validation / 10% test by character count
Detected source text field: Data
Documents: 698
Actual characters (including separators): {'train': 800016, 'val': 99547, 'test': 100884}

train.txt: train your assigned tokenizer ONLY on this file; train GPT on this file.
val.txt: evaluate during training / choose hyperparameters; do not train tokenizer on it.
test.txt: final held-out evaluation only.
All groups: 2,000-token vocabulary (where supported), same files and seed.
Check the source dataset's license and attribution terms before redistributing.
