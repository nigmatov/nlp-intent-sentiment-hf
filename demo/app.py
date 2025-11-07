import argparse, gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="outputs/best")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.ckpt)
mdl = AutoModelForSequenceClassification.from_pretrained(args.ckpt)

def predict(text):
    if not text.strip():
        return {"negative": 0.5, "positive": 0.5}, "Enter text"
    inp = tok(text, return_tensors="pt")
    probs = mdl(**inp).logits.softmax(-1).detach().numpy()[0]
    return {"negative": float(probs[0]), "positive": float(probs[1])}, ("positive" if probs[1] > probs[0] else "negative")

gr.Interface(fn=predict, inputs=gr.Textbox(), outputs=[gr.Label(num_top_classes=2), gr.Textbox()],
             examples=[["I love this product"], ["This is awful"]]).launch()
