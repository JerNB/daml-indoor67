"""Fine-tune an ImageNet-pretrained backbone end to end (all layers unfrozen).

Difference from the baseline: the baseline froze the network and only trained a
logistic regression on its features. Here the whole network keeps learning on
Indoor67, so the features themselves adapt to indoor scenes.

Examples:
  # model selection: train on folds 1-4, validate on fold 0
  python src/finetune.py --model convnext_base.fb_in1k --epochs 10 --name ft_convnext
  # final run: train on ALL labeled images, then write a Kaggle submission
  python src/finetune.py --model convnext_base.fb_in1k --epochs 10 --fold -1 --name ft_convnext_full
"""
import argparse, json, math, time
from pathlib import Path

import pandas as pd
import torch
import timm
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from common import data_root, set_seed, who, SEED

# ---------------------------------------------------------------------------
# 1. Hyperparameters (the knobs you will experiment with)
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="convnext_base.fb_in1k")
ap.add_argument("--epochs", type=int, default=10, help="passes over the training set")
ap.add_argument("--bs", type=int, default=32, help="images per training step")
ap.add_argument("--lr", type=float, default=1e-4, help="learning rate of the pretrained backbone")
ap.add_argument("--head_lr", type=float, default=1e-3, help="learning rate of the new 67-class layer")
ap.add_argument("--wd", type=float, default=0.05, help="weight decay: regularization, plays the role 1/C did")
ap.add_argument("--smoothing", type=float, default=0.1, help="label smoothing")
ap.add_argument("--img", type=int, default=224, help="input image size")
ap.add_argument("--fold", type=int, default=0, help="validation fold; -1 = train on all data + write submission")
ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--limit", type=int, default=0, help="debug only: use N random labeled images")
ap.add_argument("--name", default="finetune")
ap.add_argument("--tta", action="store_true", help="test-time augmentation: average original + flipped image")
ap.add_argument("--eval_ckpt", default="", help="skip training; load these weights and only evaluate / predict")
a = ap.parse_args()

set_seed()
dev = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = dev == "cuda"  # mixed precision: faster + less GPU memory, GPU only
root = data_root()

# ---------------------------------------------------------------------------
# 2. Data: labels -> integers, and the shared train/val split
# ---------------------------------------------------------------------------
classes = pd.read_csv(root / "classes.csv")["label"].tolist()
cls2idx = {c: i for i, c in enumerate(classes)}          # "bedroom" -> 7, etc.
df = pd.read_csv(root / "train.csv").merge(pd.read_csv("splits/folds.csv"), on="id")
df["y"] = df["label"].map(cls2idx)
if a.limit:
    df = df.sample(a.limit, random_state=SEED)
tr_df = df[df["fold"] != a.fold]
va_df = df[df["fold"] == a.fold]                          # empty when fold == -1

# ---------------------------------------------------------------------------
# 3. Model: pretrained backbone + a NEW classifier layer with 67 outputs
# ---------------------------------------------------------------------------
model = timm.create_model(a.model, pretrained=True, num_classes=len(classes)).to(dev)

# Preprocessing. Training uses random augmentation so the network sees a slightly
# different version of each image every epoch (fights overfitting on 5k images).
cfg = timm.data.resolve_data_config({}, model=model)
norm = T.Normalize(cfg["mean"], cfg["std"])               # same normalization as ImageNet training
train_tf = T.Compose([
    T.RandomResizedCrop(a.img, scale=(0.5, 1.0)),         # random zoom / crop
    T.RandomHorizontalFlip(),                             # a mirrored kitchen is still a kitchen
    T.ColorJitter(0.2, 0.2, 0.2),                         # small brightness/contrast/saturation changes
    T.ToTensor(), norm,
])
eval_tf = T.Compose([                                     # no randomness when evaluating
    T.Resize(int(a.img / 0.875)), T.CenterCrop(a.img),
    T.ToTensor(), norm,
])


class Imgs(Dataset):
    def __init__(self, paths, ys, tf):
        self.paths, self.ys, self.tf = list(paths), list(ys), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.ys[i]


def loader(frame, tf, shuffle):
    paths = [root / "images" / p for p in frame["file_path"]]
    g = torch.Generator().manual_seed(SEED)               # reproducible shuffling
    return DataLoader(Imgs(paths, frame["y"], tf), a.bs, shuffle=shuffle, drop_last=shuffle,
                      num_workers=a.workers, generator=g, pin_memory=use_amp)


train_loader = loader(tr_df, train_tf, shuffle=True)
val_loader = loader(va_df, eval_tf, shuffle=False) if len(va_df) else None

# ---------------------------------------------------------------------------
# 4. Optimizer: two learning rates. The backbone already knows how to see, so it
#    moves slowly; the new head starts from random numbers, so it moves faster.
# ---------------------------------------------------------------------------
head = list(model.get_classifier().parameters())
head_ids = {id(p) for p in head}
body = [p for p in model.parameters() if id(p) not in head_ids]
opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.head_lr}],
                        weight_decay=a.wd)

# Learning-rate schedule: 1 epoch of warmup (ramp up from ~0), then cosine decay to 0.
steps, warm = a.epochs * len(train_loader), len(train_loader)
def lr_factor(s):
    if s < warm:
        return (s + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

loss_fn = nn.CrossEntropyLoss(label_smoothing=a.smoothing)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


@torch.no_grad()
def predict(dl):
    model.eval()                                          # evaluation mode (no dropout etc.)
    out = []
    for x, _ in dl:
        x = x.to(dev)
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
            prob = model(x).float().softmax(1)            # 67 class probabilities per image
            if a.tta:                                     # also look at the mirrored image
                prob = prob + model(torch.flip(x, dims=[3])).float().softmax(1)
        out.append(prob.argmax(1).cpu())                  # highest probability = predicted class
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# 5. Training loop: for each epoch, go over every training batch once
# ---------------------------------------------------------------------------
ckpt_dir = root / "checkpoints"
ckpt_dir.mkdir(exist_ok=True)
ckpt = ckpt_dir / f"{a.name}.pt"
best_acc, best_ep, history = -1.0, 0, []

if a.eval_ckpt:                                           # reuse trained weights, no training
    model.load_state_dict(torch.load(a.eval_ckpt, map_location=dev))
    ckpt = Path(a.eval_ckpt)
    if val_loader is not None:
        best_acc = float((predict(val_loader) == va_df["y"].values).mean())
        print(f"eval {ckpt.name}  tta={a.tta}  val_acc {best_acc:.4f}")

for ep in range(1, 0 if a.eval_ckpt else a.epochs + 1):
    model.train()                                         # training mode
    t0, total, n = time.time(), 0.0, 0
    for x, y in train_loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(x), y)                   # forward pass: how wrong are we?
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()                     # backward pass: compute gradients
        scaler.step(opt)                                  # nudge every parameter downhill
        scaler.update()
        sched.step()
        total += loss.item() * len(y)
        n += len(y)

    msg = f"epoch {ep}/{a.epochs}  train_loss {total / n:.3f}"
    if val_loader is not None:
        acc = float((predict(val_loader) == va_df["y"].values).mean())
        history.append(round(acc, 4))
        msg += f"  val_acc {acc:.4f}"
        if acc > best_acc:                                # keep the best epoch's weights
            best_acc, best_ep = acc, ep
            torch.save(model.state_dict(), ckpt)
    print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)

# ---------------------------------------------------------------------------
# 6. Outputs
# ---------------------------------------------------------------------------
if val_loader is not None:
    print(f"best val_acc {best_acc:.4f} at epoch {best_ep}; weights -> {ckpt}")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rec = dict(name=a.name, user=who(), model=a.model, fold=a.fold, seed=SEED, val_acc=best_acc,
               best_epoch=best_ep, epochs=a.epochs, lr=a.lr, head_lr=a.head_lr, wd=a.wd,
               smoothing=a.smoothing, bs=a.bs, img=a.img, limit=a.limit, tta=a.tta,
               eval_ckpt=a.eval_ckpt, history=history, time=stamp)
    Path("results/runs").mkdir(parents=True, exist_ok=True)
    fn = f"results/runs/{stamp}_{who()}_{a.name}.json"
    json.dump(rec, open(fn, "w"), indent=1)
    print("saved", fn)
else:
    # fold == -1: trained on everything; save final weights and predict the test set
    if not a.eval_ckpt:
        torch.save(model.state_dict(), ckpt)
    sample = pd.read_csv(root / "sample_submission.csv")
    test = pd.DataFrame({"file_path": [f"test/{i}.jpg" for i in sample["id"]], "y": -1})
    pred = predict(loader(test, eval_tf, shuffle=False))
    sub = pd.DataFrame({"id": sample["id"], "label": [classes[i] for i in pred]})
    assert len(sub) == len(sample) and sub["id"].is_unique
    Path("submissions").mkdir(exist_ok=True)
    out = f"submissions/{time.strftime('%Y%m%d-%H%M')}_{a.name}.csv"
    sub.to_csv(out, index=False)
    print("weights ->", ckpt, "\nwrote", out)
