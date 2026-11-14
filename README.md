# DAML AITP Fall 2026 Challenge: MIT Indoor67

Classify indoor scene images into 67 categories (metric: accuracy).
Team collaboration process: see [TEAM_WORKFLOW.md](TEAM_WORKFLOW.md).

## Layout
```
src/            training / feature / submission code (shared helpers in common.py)
splits/         folds.csv  -- the single shared stratified 5-fold split (ids only)
results/runs/   one JSON per experiment run (val accuracy, config, seed, user)
notebooks/      colab_setup.ipynb -- Colab entry point
configs/        notes/configs per experiment idea
submissions/    generated Kaggle csv files (git-ignored)
```
Data (images, train.csv, classes.csv, sample_submission.csv) is **not** in this repo.
Set `DATA_ROOT` to the folder that contains `train.csv` and `images/`.

## Quickstart
```bash
pip install -r requirements.txt
export DATA_ROOT=/path/to/data            # Windows PowerShell: $env:DATA_ROOT="C:\path\to\data"
python src/extract_features.py --model resnet50.a1_in1k
python src/linear_probe.py --model resnet50.a1_in1k --name baseline
python src/summarize.py
python src/make_submission.py --model resnet50.a1_in1k --tag baseline
```

## Reproducibility
Seed fixed to 42 (`src/common.py`); validation = fold 0 of `splits/folds.csv`.
Only ImageNet-pretrained encoders are used (no Places / SUN / Indoor67 models).

## AI assistant use
Initial project scaffold was drafted with Claude Code; all code is reviewed by team members.
