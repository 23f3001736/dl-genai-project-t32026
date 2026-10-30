"""Milestone 2 data checks: class prevalence, co-occurrence, correlation and pixel overlap.

Usage: python scripts/eda.py
"""
import numpy as np

from preprocess import load_wide, presence_table
from utils import CLASSES, find_data_root, rle_decode


def main():
    wide = load_wide(find_data_root())
    pres = presence_table(wide)
    counts = pres.sum()
    print("images per class:\n", counts, "\nprevalence %:\n", (100 * counts / len(pres)).round(2))

    X = pres.values
    C = X.T @ X  # co-occurrence: diagonal = class counts, off diagonal = images with both
    print("co-occurrence matrix:\n", C)
    print("phi correlation:\n", pres.corr().round(4))
    n = len(pres)
    for i in range(4):
        for j in range(i + 1, 4):
            expected = counts.iloc[i] * counts.iloc[j] / n
            print(f"{CLASSES[i]}-{CLASSES[j]}: observed {C[i, j]}, expected if independent {expected:.1f}")

    # do two different defects ever share a pixel?
    max_overlap = 0
    for image_id in pres.index[pres.sum(1) >= 2]:
        masks = [rle_decode(wide.loc[image_id, c]) for c in CLASSES if pres.loc[image_id, c]]
        for a in range(len(masks)):
            for b in range(a + 1, len(masks)):
                max_overlap = max(max_overlap, int((masks[a] & masks[b]).sum()))
    print("max pixel overlap between two defect classes:", max_overlap)


if __name__ == "__main__":
    main()
