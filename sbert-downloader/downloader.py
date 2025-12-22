from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import sys

MODEL_NAME = "google-bert/bert-base-uncased"
OUT_DIR = Path("/models/bert-base-uncased")

# Simple existence check
if (OUT_DIR / "config.json").exists():
    print("BERT model already present, skipping download.")
    sys.exit(0)

OUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

tokenizer.save_pretrained(OUT_DIR)
model.save_pretrained(OUT_DIR)

print("BERT model downloaded.")
