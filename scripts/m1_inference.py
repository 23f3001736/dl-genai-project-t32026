"""Milestone 1: threshold sweep on the validation split and a test submission.

Usage: python scripts/m1_inference.py --weights models/baseline_tiny_unet.pt --out submission.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from preprocess import load_wide
from m1_baseline import DEVICE, GrayStripDataset, TinyUNet
from utils import CLASSES, find_data_root, rle_encode

THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


def official_style_score(prob, truth, tau):
    # 70% Dice on pairs with a defect + 20% false positive pixel score + 10% empty accuracy
    pred = prob >= tau
    has_defect = truth.sum(axis=(2, 3)) > 0
    pred_pos = pred.sum(axis=(2, 3)) > 0
    inter = (pred & (truth > 0)).sum(axis=(2, 3))
    denom = pred.sum(axis=(2, 3)) + truth.sum(axis=(2, 3))
    dice = np.where(denom > 0, 2 * inter / np.maximum(denom, 1), 1.0)
    fp_score = 1.0 - pred[~has_defect].mean()
    empty_acc = 1.0 - pred_pos[~has_defect].mean()
    return 0.7 * dice[has_defect].mean() + 0.2 * fp_score + 0.1 * empty_acc


def predict(model, loader):
    model.eval()
    probs, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(torch.sigmoid(model(x.to(DEVICE))).cpu().numpy().astype(np.float16))
            truths.append(y.numpy().astype(np.uint8))
    return np.concatenate(probs), np.concatenate(truths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="models/baseline_tiny_unet.pt")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    root = find_data_root()
    wide = load_wide(root)
    model = TinyUNet().to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    val_ids = pd.read_csv(f"{args.splits}/val_ids.csv").image_id
    val_loader = DataLoader(GrayStripDataset(val_ids, wide, root / "train"), batch_size=16)
    prob, truth = predict(model, val_loader)
    scores = {t: official_style_score(prob, truth, t) for t in THRESHOLDS}
    for t, s in scores.items():
        print(f"threshold {t}: official style score {s:.4f}")
    best = max(scores, key=scores.get)
    print("best threshold:", best)

    sample = pd.read_csv(root / "sample_submission.csv", keep_default_na=False)
    test_ids = sample.image_id.drop_duplicates().tolist()
    empty = pd.DataFrame("", index=test_ids, columns=CLASSES)
    test_loader = DataLoader(GrayStripDataset(test_ids, empty, root / "test"), batch_size=16)
    test_prob, _ = predict(model, test_loader)
    rles = {}
    for i, image_id in enumerate(test_ids):
        for c, name in enumerate(CLASSES):
            rles[(image_id, name)] = rle_encode(test_prob[i, c] >= best)
    sample["mask_rle"] = [rles[(i, c)] or " " for i, c in zip(sample.image_id, sample.defect_type)]
    sample.to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
