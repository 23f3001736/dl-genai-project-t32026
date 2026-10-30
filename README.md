# DL GenAI Project

Name: Sripathi Dhruv Reddy  
Roll Number: 23f3001736

Project: Industrial Surface Defect Segmentation (Kaggle: SurfaceGuard AI)

Each strip image (128 x 800) can have four defect types (defect_a to defect_d). The task is to predict a
pixel mask per defect type and submit it as run-length encoding (1-based, column-major).

## Folders

- `data/` competition files go here (not committed, see `data/README.md`)
- `notebooks/` Kaggle notebooks for each milestone
- `scripts/` preprocessing, training and inference code
- `models/` saved weights (not committed)
- `reports/` milestone and final reports

## Setup

```bash
pip install -r requirements.txt
```

## How to run

```bash
python scripts/preprocess.py --out data/splits
python scripts/eda.py
wandb login   # once, experiments go to the W&B project 23f3001736-t32026
python scripts/train.py --model scratch_unet --epochs 15 --lr 1e-3
python scripts/inference.py --models scratch_unet:models/scratch_unet.pt --out submission.csv
```

Milestone 1 grayscale baseline: `python scripts/m1_baseline.py` and `python scripts/m1_inference.py`.

## Branches

One branch per milestone (`milestone-1`, `milestone-2`, ...), merged into `main` when the milestone is done.
