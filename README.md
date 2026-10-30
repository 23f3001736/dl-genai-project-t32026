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

## How to run (Milestone 1 baseline)

```bash
python scripts/preprocess.py --out data/splits
python scripts/train.py --epochs 5 --out models/baseline_tiny_unet.pt
python scripts/inference.py --weights models/baseline_tiny_unet.pt --out submission.csv
```

## Branches

One branch per milestone (`milestone-1`, `milestone-2`, ...), merged into `main` when the milestone is done.
