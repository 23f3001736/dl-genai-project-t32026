"""Turn the long train.csv into one row per image and make a stratified 80/20 split.

Usage: python scripts/preprocess.py --out data/splits
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import CLASSES, SEED, find_data_root


def load_wide(root):
    long = pd.read_csv(root / "train.csv", keep_default_na=False)
    wide = long.pivot(index="image_id", columns="defect_type", values="mask_rle").fillna("")
    return wide.reindex(columns=CLASSES, fill_value="")


def presence_table(wide):
    return wide.apply(lambda col: col.str.strip() != "").astype(int)


def make_split(wide, seed=SEED, test_size=0.2):
    # stratify on the defect combination (e.g. "1010"), rare combos grouped so sklearn can split them
    combo = presence_table(wide).astype(str).agg("".join, axis=1)
    strata = combo.where(combo.map(combo.value_counts()) >= 2, "rare")
    return train_test_split(wide.index.to_numpy(), test_size=test_size, random_state=seed, stratify=strata)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/splits")
    args = parser.parse_args()
    root = find_data_root()
    wide = load_wide(root)
    pres = presence_table(wide)
    print("images:", len(wide))
    print("images per class:\n", pres.sum())
    print("clean images:", int((pres.sum(1) == 0).sum()))
    train_ids, val_ids = make_split(wide)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.Series(train_ids, name="image_id").to_csv(out / "train_ids.csv", index=False)
    pd.Series(val_ids, name="image_id").to_csv(out / "val_ids.csv", index=False)
    print("saved split:", len(train_ids), "train /", len(val_ids), "val ->", out)


if __name__ == "__main__":
    main()
