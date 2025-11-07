.PHONY: setup train eval demo export-onnx
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python src/train.py --model distilbert-base-uncased --dataset sst2 --epochs 1

eval:
	python src/eval.py --ckpt outputs/best

demo:
	python demo/app.py --ckpt outputs/best

export-onnx:
	python src/export_onnx.py --ckpt outputs/best
