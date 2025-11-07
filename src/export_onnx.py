import argparse, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="model.onnx")
args = ap.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
model = AutoModelForSequenceClassification.from_pretrained(args.ckpt)
dummy = tokenizer("hello world", return_tensors="pt")
torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"]),
                  args.out, input_names=["input_ids","attention_mask"],
                  output_names=["logits"], opset_version=17,
                  dynamic_axes={"input_ids":{0:"batch"}, "attention_mask":{0:"batch"}})
print("Exported", args.out)
