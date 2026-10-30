"""Train one segmentation model and track it with Weights & Biases.

Usage:
  python scripts/train.py --model scratch_unet --epochs 15 --lr 1e-3
  python scripts/train.py --model baseline_unet --epochs 5 --loss bce --no-wandb
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dataset import SurfaceAnomalyDataset
from models import build_model
from preprocess import load_wide
from utils import find_data_root, seed_everything

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bce_dice(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = (0, 2, 3)
    dice = (2 * (prob * target).sum(dims) + 1) / (prob.sum(dims) + target.sum(dims) + 1)
    return bce + (1 - dice.mean())


def evaluate(model, loader, loss_fn):
    # val loss, Dice on pairs with a defect, accuracy and macro F1 of "is this defect present"
    model.eval()
    total, count, inter, pred_px, gt_px = 0.0, 0, [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                logits = model(x).float()
            total += loss_fn(logits, y).item() * x.size(0)
            count += x.size(0)
            pred = torch.sigmoid(logits) > 0.5
            inter.append((pred & (y > 0.5)).sum((2, 3)).cpu())
            pred_px.append(pred.sum((2, 3)).cpu())
            gt_px.append(y.sum((2, 3)).cpu())
    inter, pred_px, gt_px = torch.cat(inter).numpy(), torch.cat(pred_px).numpy(), torch.cat(gt_px).numpy()
    has_defect, said_defect = gt_px > 0, pred_px > 0
    dice = 2 * inter / np.maximum(pred_px + gt_px, 1)
    return {
        "val_loss": total / count,
        "val_dice": float(dice[has_defect].mean()),
        "val_accuracy": float((has_defect == said_defect).mean()),
        "val_f1": float(np.mean([f1_score(has_defect[:, c], said_defect[:, c], zero_division=0) for c in range(4)])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="scratch_unet")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loss", choices=["bce", "bce_dice"], default="bce_dice")
    parser.add_argument("--splits", default="data/splits")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--wandb-project", default="23f3001736-t32026")
    parser.add_argument("--wandb-entity", default="23f3001736-indian-institute-of-technology-madras")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    seed_everything()
    root = find_data_root()
    wide = load_wide(root).reset_index()
    train_ids = set(pd.read_csv(f"{args.splits}/train_ids.csv").image_id)
    val_ids = set(pd.read_csv(f"{args.splits}/val_ids.csv").image_id)
    train_loader = DataLoader(SurfaceAnomalyDataset(wide[wide.image_id.isin(train_ids)], root / "train", augment=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(SurfaceAnomalyDataset(wide[wide.image_id.isin(val_ids)], root / "train"),
                            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    import wandb
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.model,
                     mode="disabled" if args.no_wandb else "online", config=vars(args))

    loss_fn = bce_dice if args.loss == "bce_dice" else nn.BCEWithLogitsLoss()
    model = build_model(args.model).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best, history = -1.0, []
    out_path = f"{args.out_dir}/{args.model}.pt"
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        metrics = {"epoch": epoch, "train_loss": running / len(train_loader), **evaluate(model, val_loader, loss_fn)}
        history.append(metrics)
        wandb.log(metrics)
        print(f"epoch {epoch}: train {metrics['train_loss']:.4f} val {metrics['val_loss']:.4f} "
              f"dice {metrics['val_dice']:.4f} acc {metrics['val_accuracy']:.4f} f1 {metrics['val_f1']:.4f} "
              f"({time.time() - start:.0f}s)")
        if metrics["val_dice"] > best:
            best = metrics["val_dice"]
            torch.save(model.state_dict(), out_path)
    pd.DataFrame(history).to_csv(f"{args.out_dir}/{args.model}_history.csv", index=False)
    for k, v in max(history, key=lambda h: h["val_dice"]).items():
        run.summary["best_" + k] = v
    run.finish()
    print("best val dice", round(best, 4), "saved", out_path)


if __name__ == "__main__":
    main()
