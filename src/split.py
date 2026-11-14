"""Create the ONE shared stratified split. Run once; commit splits/folds.csv.
Everyone must use this file so validation scores are comparable.
5-fold stratified (fold 0..4). Default validation = fold 0."""
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from common import data_root, SEED

df = pd.read_csv(data_root() / "train.csv")
df["fold"] = -1
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for k, (_, va) in enumerate(skf.split(df, df["label"])):
    df.loc[va, "fold"] = k
os.makedirs("splits", exist_ok=True)
df[["id", "fold"]].to_csv("splits/folds.csv", index=False)  # ids only, no labels
print(df["fold"].value_counts().sort_index())
