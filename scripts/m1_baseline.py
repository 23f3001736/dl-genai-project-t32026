"""Milestone 1 baseline: small U-Net on grayscale strips, BCE + Dice loss, 5 epochs.

Usage: python scripts/train.py --epochs 5 --out models/baseline_tiny_unet.pt
(run scripts/preprocess.py first to create data/splits)
"""
import argparse
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from preprocess import load_wide
from utils import CLASSES, H, W, find_data_root, rle_decode, seed_everything

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GrayStripDataset(Dataset):
    def __init__(self, image_ids, wide, image_dir, augment=False):
        self.ids, self.wide, self.image_dir, self.augment = list(image_ids), wide, image_dir, augment

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = cv2.imread(str(self.image_dir / f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.25
        mask = np.stack([rle_decode(self.wide.loc[image_id, c]) for c in CLASSES]).astype(np.float32)
        if self.augment and random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(image[None])), torch.from_numpy(np.ascontiguousarray(mask))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, base=8):
        super().__init__()
        self.e1, self.e2, self.e3 = ConvBlock(1, base), ConvBlock(base, base * 2), ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(base * 4, base * 8)
        self.u3, self.d3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2), ConvBlock(base * 8, base * 4)
        self.u2, self.d2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2), ConvBlock(base * 4, base * 2)
        self.u1, self.d1 = nn.ConvTranspose2d(base * 2, base, 2, 2), ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, len(CLASSES), 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        x = self.mid(self.pool(e3))
        x = self.d3(torch.cat([self.u3(x), e3], 1))
        x = self.d2(torch.cat([self.u2(x), e2], 1))
        x = self.d1(torch.cat([self.u1(x), e1], 1))
        return self.out(x)


def bce_dice(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3)
    dice = (2 * (prob * target).sum(dims) + 1) / (prob.sum(dims) + target.sum(dims) + 1)
    return bce + (1 - dice.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--out", default="models/baseline_tiny_unet.pt")
    args = parser.parse_args()

    seed_everything()
    root = find_data_root()
    wide = load_wide(root)
    train_ids = pd.read_csv(f"{args.splits}/train_ids.csv").image_id
    loader = DataLoader(GrayStripDataset(train_ids, wide, root / "train", augment=True),
                        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    model = TinyUNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = bce_dice(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch}/{args.epochs} train_loss={total / len(loader):.5f}")
    torch.save(model.state_dict(), args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
