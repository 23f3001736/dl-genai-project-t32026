"""Shared helpers for the project: constants, RLE encode/decode, seeding and data paths."""
import random
from pathlib import Path

import numpy as np

H, W = 128, 800
CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
SEED = 1736


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def find_data_root():
    # works on Kaggle (/kaggle/input/...) and locally (<repo>/data/public/train.csv)
    local = Path(__file__).resolve().parent.parent / "data" / "public"
    if (local / "train.csv").exists():
        return local
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.glob("**/public/train.csv"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError("train.csv not found, put the competition files in data/public/")


def rle_decode(rle, shape=(H, W)):
    """Competition RLE (1-based starts, column-major order) -> binary mask."""
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    text = str(rle).strip() if isinstance(rle, str) else ""
    if text:
        pairs = np.asarray(text.split(), dtype=np.int64).reshape(-1, 2)
        for start, length in pairs:
            mask[start - 1:start - 1 + length] = 1
    return mask.reshape(shape, order="F")


def rle_encode(mask):
    """Binary mask -> competition RLE string ('' for an empty mask)."""
    pixels = np.asarray(mask, dtype=np.uint8).reshape(-1, order="F")
    padded = np.concatenate([[0], pixels, [0]])
    runs = np.where(padded[1:] != padded[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))
