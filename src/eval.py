import argparse, numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from evaluate import load as load_metric

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--dataset", default="sst2")
args = ap.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
metric = load_metric("glue", args.dataset)
ds = load_dataset("glue", args.dataset)

def tok(ex):
    return tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tok, batched=True)
ds = ds.rename_column("label", "labels")
ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

preds = []
labels = []
for batch in ds["validation"]:
    out = model(input_ids=batch["input_ids"].unsqueeze(0),
                attention_mask=batch["attention_mask"].unsqueeze(0))
    preds.append(out.logits.detach().numpy().argmax(axis=-1)[0])
    labels.append(int(batch["labels"]))

res = metric.compute(predictions=preds, references=labels)
print(res)
