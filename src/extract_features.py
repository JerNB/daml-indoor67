"""Frozen ImageNet-pretrained backbone -> feature vectors (cached as .npy, outside git).
Example: python src/extract_features.py --model resnet50.a1_in1k"""
import argparse
import numpy as np, pandas as pd, torch, timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from common import data_root, set_seed


class Imgs(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def run(model_name, bs, workers):
    root = data_root()
    set_seed()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(dev)
    tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=model))
    out = root / "cache"
    out.mkdir(exist_ok=True)
    sets = {
        "train": [root / "images" / p for p in pd.read_csv(root / "train.csv")["file_path"]],
        "test": [root / "images" / "test" / f"{i}.jpg" for i in pd.read_csv(root / "sample_submission.csv")["id"]],
    }
    for name, paths in sets.items():
        loader = DataLoader(Imgs(paths, tf), bs, num_workers=workers)
        feats = [model(x.to(dev)).cpu().numpy() for x in tqdm(loader, desc=name)]
        np.save(out / f"{model_name}_{name}.npy", np.concatenate(feats))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50.a1_in1k")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    run(a.model, a.bs, a.workers)
