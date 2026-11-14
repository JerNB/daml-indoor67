"""Refit on ALL labeled train images, predict test, write + validate a Kaggle csv.
Example: python src/make_submission.py --model resnet50.a1_in1k --C 1.0 --tag baseline"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import data_root, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet50.a1_in1k")
ap.add_argument("--C", type=float, default=1.0)
ap.add_argument("--tag", default="baseline")
a = ap.parse_args()
set_seed()

root = data_root()
tr = pd.read_csv(root / "train.csv")
sample = pd.read_csv(root / "sample_submission.csv")
classes = set(pd.read_csv(root / "classes.csv")["label"])
Xtr = np.load(root / "cache" / f"{a.model}_train.npy")
Xte = np.load(root / "cache" / f"{a.model}_test.npy")
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(Xtr), tr["label"])
sub = pd.DataFrame({"id": sample["id"], "label": clf.predict(sc.transform(Xte))})

assert len(sub) == len(sample) and sub["id"].is_unique and set(sub["label"]) <= classes
Path("submissions").mkdir(exist_ok=True)
fn = f"submissions/{time.strftime('%Y%m%d-%H%M')}_{a.tag}.csv"
sub.to_csv(fn, index=False)
print("wrote", fn)
