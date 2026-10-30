# Scripts

Run everything from the repo root.

| Script | What it does |
|---|---|
| `utils.py` | constants, RLE encode/decode, seeding, data path |
| `preprocess.py` | one row per image + stratified 80/20 split into `data/splits/` |
| `eda.py` | class counts, co-occurrence, correlation and pixel overlap checks |
| `dataset.py` | PyTorch datasets (RGB image + 4 channel float32 mask) |
| `models.py` | model definitions (`build_model`) |
| `train.py` | trains one model, logs every epoch to Weights & Biases |
| `inference.py` | threshold / min area tuning on validation, test predictions, submission.csv |
| `m1_baseline.py`, `m1_inference.py` | the Milestone 1 grayscale baseline |
