# Data

The competition data is not stored in this repo. Download it from the Kaggle competition
(industrial-surface-defect-detection) and place it like this:

```
data/public/train.csv
data/public/sample_submission.csv
data/public/train/*.jpg
data/public/test/*.jpg
```

`scripts/preprocess.py` writes the train/validation split to `data/splits/`.
