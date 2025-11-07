from datasets import load_dataset
from sklearn.model_selection import train_test_split

def load_data(name: str="sst2", test_size: float=0.1, seed: int=42):
    if name.lower() == "sst2":
        ds = load_dataset("glue", "sst2")
        return ds["train"], ds["validation"]
    texts = ["good movie", "bad film", "awesome", "terrible"]
    labels = [1,0,1,0]
    X_tr, X_va, y_tr, y_va = train_test_split(texts, labels, test_size=test_size, random_state=seed)
    return {"sentence": X_tr, "label": y_tr}, {"sentence": X_va, "label": y_va}
