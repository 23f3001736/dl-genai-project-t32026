# Final report numbers: models D + E (4 flip TTA, final per-class config) on the same validation split
# as the main notebook. Computes the official-style score, per-class Dice, presence macro F1 and
# saves a figure of failure examples.
import json, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

SEED, H, W = 1736, 128, 800
CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
FINAL_CFG = {"defect_a": (0.2, 0), "defect_b": (0.2, 100), "defect_c": (0.2, 50), "defect_d": (0.2, 0)}
DEVICE = torch.device("cuda")
OUT = Path("/kaggle/working")

train_csv = (list(Path("/kaggle/input").glob("**/public/train.csv")) or list(Path("/kaggle/input").glob("**/train.csv")))[0]
TRAIN_DIR = train_csv.parent / "train"
train_long = pd.read_csv(train_csv, keep_default_na=False)


def rle_decode(rle):
    mask = np.zeros(H * W, dtype=np.uint8)
    text = str(rle).strip()
    if text:
        v = np.asarray(text.split(), dtype=np.int64).reshape(-1, 2)
        for s, l in v:
            mask[s - 1:s - 1 + l] = 1
    return mask.reshape((H, W), order="F")


# same combination-stratified 80/20 split as the notebook (cell 5)
wide = train_long.pivot(index="image_id", columns="defect_type", values="mask_rle").fillna("")
wide = wide.reindex(columns=CLASSES, fill_value="")
combo = wide.apply(lambda col: col.map(lambda x: bool(str(x).strip()))).astype(int).astype(str).agg("".join, axis=1)
strata = combo.where(combo.map(combo.value_counts()) >= 2, "rare")
train_ids, val_ids = train_test_split(wide.index.to_numpy(), test_size=0.20, random_state=SEED, stratify=strata)
val_ids = sorted(val_ids, key=list(wide.index).index)  # keep notebook order
print("val images:", len(val_ids))

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Normalized(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def load(name):
    path = list(Path("/kaggle/input").glob(f"**/{name}"))[0]
    net = Normalized(smp.FPN("efficientnet-b3", encoder_weights=None, in_channels=3, classes=4))
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(DEVICE).eval()


models = [load("m4_D_effb3_fpn.pt"), load("m4_E_effb3_fpn_seed7.pt")]


@torch.no_grad()
def tta4(model, x):
    outs = []
    with torch.amp.autocast("cuda"):
        for dims in ([], [3], [2], [2, 3]):
            xx = torch.flip(x, dims=dims) if dims else x
            p = torch.sigmoid(model(xx)).float()
            outs.append(torch.flip(p, dims=dims) if dims else p)
    return sum(outs) / 4


rows, keep = [], {}
for start in range(0, len(val_ids), 16):
    ids = val_ids[start:start + 16]
    imgs = [cv2.cvtColor(cv2.imread(str(TRAIN_DIR / f"{i}.jpg")), cv2.COLOR_BGR2RGB) for i in ids]
    x = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(DEVICE)
    prob = (sum(tta4(m, x) for m in models) / len(models)).cpu().numpy()
    for b, iid in enumerate(ids):
        for c, name in enumerate(CLASSES):
            tau, area = FINAL_CFG[name]
            pred = (prob[b, c] >= tau).astype(np.uint8)
            if pred.sum() < max(area, 1):
                pred[:] = 0
            gt = rle_decode(wide.loc[iid, name])
            inter, ps, gs = int((pred & gt).sum()), int(pred.sum()), int(gt.sum())
            dice = 2 * inter / (ps + gs) if gs > 0 else np.nan
            rows.append(dict(image_id=iid, defect=name, gt_px=gs, pred_px=ps, dice=dice))
            keep[(iid, name)] = (imgs[b], gt, pred)

df = pd.DataFrame(rows)
pos, neg = df[df.gt_px > 0], df[df.gt_px == 0]
dice_pos = pos.dice.mean()
fp_pixel = (neg.pred_px / (H * W)).mean()
empty_acc = (neg.pred_px == 0).mean()
score = 0.7 * dice_pos + 0.2 * (1 - fp_pixel) + 0.1 * empty_acc
res = {"val_images": len(val_ids), "dice_pos": dice_pos, "fp_pixel": fp_pixel, "empty_acc": empty_acc, "score": score, "per_class": {}}
f1s = []
for name in CLASSES:
    d = df[df.defect == name]
    y, p = d.gt_px > 0, d.pred_px > 0
    f1 = f1_score(y, p, zero_division=0)
    f1s.append(f1)
    res["per_class"][name] = {"positives": int(y.sum()), "dice_pos": float(d[y].dice.mean()), "presence_f1": float(f1),
                              "precision": float((y & p).sum() / max(p.sum(), 1)), "recall": float((y & p).sum() / max(y.sum(), 1)),
                              "missed": int((y & ~p).sum()), "false_alarms": int((~y & p).sum())}
res["presence_macro_f1"] = float(np.mean(f1s))
res = json.loads(json.dumps(res, default=float))
print(json.dumps(res, indent=2))
(OUT / "final_eval.json").write_text(json.dumps(res, indent=2))
df.to_csv(OUT / "final_eval_pairs.csv", index=False)

# failure examples: biggest missed defect, biggest false alarm, lowest Dice on a found defect
miss = pos[pos.pred_px == 0].sort_values("gt_px", ascending=False).head(1)
fa = neg[neg.pred_px > 0].sort_values("pred_px", ascending=False).head(1)
low = pos[(pos.pred_px > 0) & (pos.gt_px > 2000)].sort_values("dice").head(1)
cases = [("Missed defect", miss), ("False alarm", fa), ("Poor overlap", low)]
fig, axes = plt.subplots(len(cases), 3, figsize=(15, 2.2 * len(cases)))
for r, (title, sel) in enumerate(cases):
    row = sel.iloc[0]
    img, gt, pred = keep[(row.image_id, row.defect)]
    d = "" if np.isnan(row.dice) else f", Dice {row.dice:.2f}"
    for k, (im, t) in enumerate([(img, f"{title}: {row.defect}{d}"), (gt, f"Ground truth ({row.gt_px} px)"), (pred, f"Prediction ({row.pred_px} px)")]):
        ax = axes[r, k]
        ax.imshow(im, cmap=None if k == 0 else "gray", aspect="auto")
        ax.set_title(t, fontsize=9)
        ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "failure_examples.png", dpi=110)
print("cases:", [(t, s.iloc[0].image_id, s.iloc[0].defect) for t, s in cases])
