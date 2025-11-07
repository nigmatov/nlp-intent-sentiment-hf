import argparse, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--text", required=True)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.ckpt)
mdl = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
inp = tok(args.text, return_tensors="pt")
with torch.no_grad():
    logits = mdl(**inp).logits
probs = logits.softmax(-1).numpy()[0].tolist()
print({"probs": probs, "label": int(np.argmax(probs))})
