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

Milestone 4 final model: two EfficientNet-B3 FPN models (`effb3_fpn`) averaged, see `notebooks/milestone-4.ipynb`.

Milestone 5 final pipeline: models D + E with 4-flip TTA, averaged probabilities, per-class threshold and
minimum area, then RLE. See `notebooks/milestone-5.ipynb` (same as Kaggle notebook
`DL-23f3001736-notebook-t32026`, version 14). Public leaderboard: 0.66965, validation score: 0.7308.

## Deployment (Hugging Face)

- Live demo: https://huggingface.co/spaces/23f3001736/surfaceguard-defect-segmentation
- Model: https://huggingface.co/23f3001736/surfaceguard-defect-segmentation

The demo is a static Space (`deploy/hf-static/index.html`). It runs models D + E with 4-flip TTA in the
browser with onnxruntime-web, then the same per-class threshold and minimum area as the Kaggle submission.
`deploy/export_onnx.py` builds the ONNX file from the two PyTorch checkpoints. `deploy/hf-space/app.py` is
the same pipeline as a Gradio app (`python app.py` with the two `.pt` files next to it).

## Final report

`reports/final/23f3001736_DL_final_report.pdf` (source: `reports/final/report.md`).

## Branches

One branch per milestone (`milestone-1`, `milestone-2`, ...), merged into `main` when the milestone is done.
