"""PyTorch datasets for the strip images (Paradigm 1: one 4-channel mask per image)."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import CLASSES, rle_decode


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class SurfaceAnomalyDataset(Dataset):
    """Returns (image [3, 128, 800] in 0..1, masks [4, 128, 800]) both float32."""

    def __init__(self, image_df, image_dir, augment=False):
        self.df = image_df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_rgb(self.image_dir / f"{row['image_id']}.jpg").transpose(2, 0, 1)
        masks = np.stack([rle_decode(row[c]) for c in CLASSES]).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                image, masks = image[:, :, ::-1], masks[:, :, ::-1]
            if random.random() < 0.5:
                image, masks = image[:, ::-1, :], masks[:, ::-1, :]
            if random.random() < 0.3:
                # small brightness / contrast change, only on the image
                image = np.clip(image * random.uniform(0.85, 1.15) + random.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)), torch.from_numpy(np.ascontiguousarray(masks))


class TestDataset(Dataset):
    """Test images only, returns (image, image_id)."""

    def __init__(self, image_ids, image_dir):
        self.ids = list(image_ids)
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = load_rgb(self.image_dir / f"{self.ids[idx]}.jpg").transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(image)), self.ids[idx]
