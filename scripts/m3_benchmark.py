# Milestone 3 official protocol: Q2 parameter audit, Q8 and Q9 five epoch runs, Q9 memory check
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.8.0", "torchvision==0.23.0",
                "segmentation-models-pytorch==0.5.0"], check=True)

import hashlib, json, platform, random, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

OUT = Path("/kaggle/working")
H, W = 128, 800
CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EPOCHS = 16, 3e-4, 1e-2, 5

roots = list(Path("/kaggle/input").glob("**/public/train.csv")) or list(Path("/kaggle/input").glob("**/train.csv"))
TRAIN_DIR = roots[0].parent / "train"
train_long = pd.read_csv(roots[0], keep_default_na=False)


def rle_decode(rle):
    mask = np.zeros(H * W, dtype=np.uint8)
    text = str(rle).strip()
    if text:
        values = np.asarray(text.split(), dtype=np.int64).reshape(-1, 2)
        for start, length in values:
            mask[start - 1:start - 1 + length] = 1
    return mask.reshape((H, W), order="F")


# same 80/20 split as milestone 2 (combination stratified, seed 1736)
wide = train_long.pivot(index="image_id", columns="defect_type", values="mask_rle").fillna("")
wide = wide.reindex(columns=CLASSES, fill_value="")
combo = wide.apply(lambda col: col.str.strip() != "").astype(int).astype(str).agg("".join, axis=1)
strata = combo.where(combo.map(combo.value_counts()) >= 2, "rare")
train_ids, val_ids = train_test_split(wide.index.to_numpy(), test_size=0.20, random_state=1736, stratify=strata)
split_hash = hashlib.md5(",".join(sorted(val_ids)).encode()).hexdigest()
print("split:", len(train_ids), len(val_ids), "val hash", split_hash, flush=True)


class StripDataset(Dataset):
    # grayscale float32 in [0,1], no augmentation and no extra normalisation
    def __init__(self, ids):
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image = cv2.imread(str(TRAIN_DIR / f"{image_id}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = np.stack([rle_decode(wide.loc[image_id, c]) for c in CLASSES]).astype(np.float32)
        return torch.from_numpy(image[None]), torch.from_numpy(mask)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class CompoundLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        return self.alpha * self.bce(logits, targets) + self.beta * self.dice(logits, targets)


def make_model(encoder, weights):
    torch.manual_seed(42)
    return smp.Unet(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=4, encoder_depth=5,
                    decoder_channels=(256, 128, 64, 32, 16), decoder_use_norm="batchnorm",
                    decoder_interpolation="nearest", activation=None)


def set_seeds():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def evaluate(model, ids, criterion, metrics=False):
    # eval mode, fixed order, loss weighted by batch size, hard masks at 0.5 with no area filter
    model.eval()
    loader = DataLoader(StripDataset(ids), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False)
    loss_sum, seen = 0.0, 0
    dice_sum, n_pos, fp_sum, empty_ok, n_empty = 0.0, 0, 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * x.size(0)
            seen += x.size(0)
            if metrics:
                pred = torch.sigmoid(logits) >= 0.5
                truth = y > 0.5
                inter = (pred & truth).sum((2, 3)).float()
                pred_px = pred.sum((2, 3)).float()
                true_px = truth.sum((2, 3)).float()
                pos = true_px > 0
                dice = 2 * inter / (pred_px + true_px).clamp(min=1)
                dice_sum += dice[pos].sum().item()
                n_pos += pos.sum().item()
                fp_sum += (pred_px[~pos] / (H * W)).sum().item()
                empty_ok += (pred_px[~pos] == 0).sum().item()
                n_empty += (~pos).sum().item()
    out = {"loss": loss_sum / seen}
    if metrics:
        out["foreground_dice"] = dice_sum / n_pos
        out["false_positive_score"] = 1 - fp_sum / n_empty
        out["empty_accuracy"] = empty_ok / n_empty
        out["score"] = 0.7 * out["foreground_dice"] + 0.2 * out["false_positive_score"] + 0.1 * out["empty_accuracy"]
    return out


def train_run(name, model):
    set_seeds()
    gen = torch.Generator()
    gen.manual_seed(42)
    model = model.to(DEVICE)
    criterion = CompoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(StripDataset(train_ids), batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                        drop_last=False, generator=gen)
    history = []
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        online, seen = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            online += loss.item() * x.size(0)
            seen += x.size(0)
        train_eval = evaluate(model, train_ids, criterion)
        val_eval = evaluate(model, val_ids, criterion, metrics=True)
        row = {"model": name, "epoch": epoch, "train_online_loss": online / seen, "train_eval_loss": train_eval["loss"],
               "val_eval_loss": val_eval["loss"], "gap": abs(val_eval["loss"] - train_eval["loss"]),
               "foreground_dice": val_eval["foreground_dice"], "false_positive_score": val_eval["false_positive_score"],
               "empty_accuracy": val_eval["empty_accuracy"], "score": val_eval["score"], "seconds": time.time() - start}
        history.append(row)
        print(json.dumps(row), flush=True)
        pd.DataFrame(history).to_csv(OUT / f"m3_{name}_history.csv", index=False)
    del model, optimizer
    torch.cuda.empty_cache()
    return history


# Q2 parameter audit
resnet = make_model("resnet34", None)
effnet = make_model("efficientnet-b0", None)
res_total, eff_total = count_params(resnet), count_params(effnet)
for p in resnet.encoder.parameters():
    p.requires_grad = False
res_trainable = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
results = {"split_val_hash": split_hash, "n_train": len(train_ids), "n_val": len(val_ids),
           "resnet_total": res_total, "efficientnet_total": eff_total, "param_diff": res_total - eff_total,
           "resnet_trainable": res_trainable, "trainable_pct": 100 * res_trainable / res_total,
           "param_ratio": res_total / eff_total,
           "versions": {"python": platform.python_version(), "torch": torch.__version__, "smp": smp.__version__,
                        "cuda": torch.version.cuda, "numpy": np.__version__}}
print(json.dumps(results), flush=True)
del resnet, effnet
(OUT / "m3_results.json").write_text(json.dumps(results, indent=1))

# Q9 memory: fresh process for each model on the same GPU, this process has not used CUDA yet
first = DataLoader(StripDataset(train_ids), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
x0, y0 = next(iter(first))
torch.save({"x": x0, "y": y0}, OUT / "mem_batch.pt")
MEM = r'''
import json, sys, torch, torch.nn as nn, segmentation_models_pytorch as smp
encoder = sys.argv[1]
b = torch.load("/kaggle/working/mem_batch.pt")
x, y = b["x"].cuda(), b["y"].cuda()
torch.manual_seed(42)
model = smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=1, classes=4, encoder_depth=5,
                 decoder_channels=(256, 128, 64, 32, 16), decoder_use_norm="batchnorm",
                 decoder_interpolation="nearest", activation=None).cuda()
bce = nn.BCEWithLogitsLoss()
def crit(l, t):
    p = torch.sigmoid(l).reshape(-1); f = t.reshape(-1)
    return 0.5 * bce(l, t) + 0.5 * (1 - (2 * (p * f).sum() + 1) / (p.sum() + f.sum() + 1))
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
model.train()
def step():
    opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
step(); step()
torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
step()
torch.cuda.synchronize()
print("PEAK", json.dumps({"encoder": encoder, "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                          "gpu": torch.cuda.get_device_name(0)}))
'''
(OUT / "mem.py").write_text(MEM)
for enc in ["resnet34", "efficientnet-b0"]:
    r = subprocess.run([sys.executable, str(OUT / "mem.py"), enc], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("PEAK")]
    results["mem_" + enc] = json.loads(line[0][5:]) if line else ("oom" if "out of memory" in r.stderr else r.stderr[-400:])
    print(enc, results["mem_" + enc], flush=True)
(OUT / "mem_batch.pt").unlink()
(OUT / "m3_results.json").write_text(json.dumps(results, indent=1))

# Q8 and Q9 runs, same decoder and head start for the two resnet models
scratch = make_model("resnet34", None)
pretrained = make_model("resnet34", "imagenet")
pretrained.decoder.load_state_dict(scratch.decoder.state_dict())
pretrained.segmentation_head.load_state_dict(scratch.segmentation_head.state_dict())
results["scratch"] = train_run("resnet34_scratch", scratch)
results["pretrained"] = train_run("resnet34_pretrained", pretrained)
(OUT / "m3_results.json").write_text(json.dumps(results, indent=1))
results["efficientnet"] = train_run("efficientnet_b0_pretrained", make_model("efficientnet-b0", "imagenet"))
(OUT / "m3_results.json").write_text(json.dumps(results, indent=1))
print("done", flush=True)
