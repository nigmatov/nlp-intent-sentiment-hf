import argparse, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from evaluate import load as load_metric

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--outdir", default="outputs/best")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    ds = load_dataset("glue", args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok(ex):
        return tokenizer(ex["sentence"], truncation=True, padding="max_length", max_length=128)

    ds = ds.map(tok, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    metric = load_metric("glue", args.dataset)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    ta = TrainingArguments(
        output_dir="outputs",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        fp16=False,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print("Saved to", args.outdir)

if __name__ == "__main__":
    main()
