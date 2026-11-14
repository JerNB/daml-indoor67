"""Baseline: logistic regression on frozen features; validate on one fold of splits/folds.csv.
Writes one JSON per run into results/runs/ (unique filenames -> no merge conflicts).
Example: python src/linear_probe.py --model resnet50.a1_in1k --C 1.0 --name baseline"""
import argparse, json, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import data_root, set_seed, who, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet50.a1_in1k")
ap.add_argument("--C", type=float, default=1.0)
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--name", default="linear_probe")
a = ap.parse_args()
set_seed()

root = data_root()
X = np.load(root / "cache" / f"{a.model}_train.npy")
df = pd.read_csv(root / "train.csv").merge(pd.read_csv("splits/folds.csv"), on="id")
tr, va = (df["fold"] != a.fold).values, (df["fold"] == a.fold).values
sc = StandardScaler().fit(X[tr])
clf = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(X[tr]), df["label"][tr])
acc = float((clf.predict(sc.transform(X[va])) == df["label"][va]).mean())
print(f"val_acc (fold {a.fold}) = {acc:.4f}")

stamp = time.strftime("%Y%m%d-%H%M%S")
rec = dict(name=a.name, user=who(), model=a.model, C=a.C, fold=a.fold, seed=SEED,
           val_acc=acc, time=stamp)
fn = f"results/runs/{stamp}_{who()}_{a.name}.json"
json.dump(rec, open(fn, "w"), indent=1)
print("saved", fn)
