"""Tune thresholds on validation and write the Kaggle submission.

One or more trained models can be given; their probabilities (with horizontal flip TTA) are averaged.
Usage: python scripts/inference.py --models scratch_unet:models/scratch_unet.pt --out submission.csv
"""
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import SurfaceAnomalyDataset, TestDataset
from models import build_model
from preprocess import load_wide
from utils import CLASSES, find_data_root, rle_encode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_AREAS = [0, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 6000]


def tta_probs(models, x):
    total = 0
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
        for m in models:
            p = torch.sigmoid(m(x)).float()
            p_flip = torch.sigmoid(m(torch.flip(x, dims=[3]))).float()
            total = total + (p + torch.flip(p_flip, dims=[3])) / 2
    return total / len(models)


def tune_on_validation(models, loader):
    # pick a threshold and a minimum mask area per class with the competition style score
    pred_count, inter_count, gt_count = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        p = tta_probs(models, x)
        pc, ic = [], []
        for t in TAUS:
            pm = p >= t
            pc.append(pm.sum((2, 3)))
            ic.append((pm & (y > 0.5)).sum((2, 3)))
        pred_count.append(torch.stack(pc, -1).cpu())
        inter_count.append(torch.stack(ic, -1).cpu())
        gt_count.append(y.sum((2, 3)).cpu())
    pred_count, inter_count, gt_count = (torch.cat(a).numpy() for a in (pred_count, inter_count, gt_count))
    positive = gt_count > 0
    n_pos, n_empty = positive.sum(), (~positive).sum()
    cfg, total = {}, 0.0
    for c, name in enumerate(CLASSES):
        options = []
        for ti, tau in enumerate(TAUS):
            for area in MIN_AREAS:
                keep = pred_count[:, c, ti] >= max(area, 1)
                pos = positive[:, c]
                dice = np.where(keep, 2 * inter_count[:, c, ti] / np.maximum(pred_count[:, c, ti] + gt_count[:, c], 1), 0.0)
                options.append((0.7 * dice[pos].sum() / n_pos + 0.3 * (~keep[~pos]).sum() / n_empty, tau, area))
        cfg[name] = max(options)
        total += cfg[name][0]
    return total, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, help="name:path pairs, e.g. scratch_unet:models/scratch_unet.pt")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    root = find_data_root()
    models = []
    for spec in args.models:
        name, path = spec.split(":", 1)
        m = build_model(name, pretrained=False).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        models.append(m)

    wide = load_wide(root).reset_index()
    val_ids = set(pd.read_csv(f"{args.splits}/val_ids.csv").image_id)
    val_loader = DataLoader(SurfaceAnomalyDataset(wide[wide.image_id.isin(val_ids)], root / "train"), batch_size=16)
    score, cfg = tune_on_validation(models, val_loader)
    print(f"tuned validation score {score:.4f}")
    for name in CLASSES:
        print(name, "threshold", cfg[name][1], "min_area", cfg[name][2])

    sample = pd.read_csv(root / "sample_submission.csv", keep_default_na=False)
    test_ids = sample.image_id.drop_duplicates().tolist()
    rles = {}
    for x, ids in DataLoader(TestDataset(test_ids, root / "test"), batch_size=16):
        p = tta_probs(models, x.to(DEVICE)).cpu().numpy()
        for i, image_id in enumerate(ids):
            for c, name in enumerate(CLASSES):
                _, tau, area = cfg[name]
                m = (p[i, c] >= tau).astype(np.uint8)
                rles[(image_id, name)] = rle_encode(m) if m.sum() >= max(area, 1) else ""
    sample["mask_rle"] = [rles[(i, c)] or " " for i, c in zip(sample.image_id, sample.defect_type)]
    sample.to_csv(args.out, index=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
