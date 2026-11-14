"""Shared helpers. Data layout under DATA_ROOT:
  train.csv  classes.csv  sample_submission.csv
  images/train/*.jpg   images/test/*.jpg
Set the DATA_ROOT env var (Colab: the Drive folder; local: e.g. ~/Downloads)."""
import os, random, getpass
from pathlib import Path
import numpy as np

SEED = 42


def data_root() -> Path:
    root = Path(os.environ.get("DATA_ROOT", "data")).expanduser()
    if not (root / "train.csv").exists():
        raise SystemExit(f"train.csv not found in {root}. Set DATA_ROOT to the data folder.")
    return root


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def who() -> str:
    return os.environ.get("RUN_USER", getpass.getuser())
