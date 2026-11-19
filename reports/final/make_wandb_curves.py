# Training curves for the report, pulled from my W&B project 23f3001736-t32026.
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

api = wandb.Api(timeout=60)
runs = api.runs("23f3001736-indian-institute-of-technology-madras/23f3001736-t32026", order="-created_at")
names = ["A_scratch_unet", "B_resnet34_unet", "C_effb2_fpn", "D_effb3_fpn", "E_effb3_fpn_seed7"]
hist = {}
for r in runs:
    if r.name in names and r.name not in hist:
        hist[r.name] = r.history(pandas=True, samples=1000).sort_values("epoch")

labels = {"A_scratch_unet": "A scratch U-Net", "B_resnet34_unet": "B ResNet34 U-Net", "C_effb2_fpn": "C EffNet-B2 FPN",
          "D_effb3_fpn": "D EffNet-B3 FPN", "E_effb3_fpn_seed7": "E EffNet-B3 FPN (seed 7, BCE w=8 on b)"}
cols = {"A_scratch_unet": "#1f77b4", "B_resnet34_unet": "#ff7f0e", "C_effb2_fpn": "#2ca02c", "D_effb3_fpn": "#d62728", "E_effb3_fpn_seed7": "#9467bd"}
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for n in names[:3]:
    h = hist[n]
    ax[0].plot(h.epoch, h.train_loss, "--", color=cols[n], lw=1)
    ax[0].plot(h.epoch, h.val_loss, "-", color=cols[n], lw=1.6, label=labels[n])
    ax[1].plot(h.epoch, h.val_dice, "-", color=cols[n], lw=1.6, label=labels[n] + " Dice")
    ax[1].plot(h.epoch, h.val_f1, ":", color=cols[n], lw=1.6)
ax[0].set_title("Milestone 3: train (dashed) and val (solid) loss", fontsize=10)
ax[1].set_title("Milestone 3: val Dice (solid) and presence F1 (dotted)", fontsize=10)
for n in names[3:]:
    h = hist[n]
    ax[2].plot(h.epoch, h.score, "-", color=cols[n], lw=1.6, label=labels[n])
ax[2].set_title("Milestone 4: official-style val score (tau 0.4, area 100)", fontsize=10)
for a in ax:
    a.set_xlabel("epoch"); a.grid(alpha=.3); a.legend(fontsize=7)
plt.tight_layout()
plt.savefig(sys.argv[1], dpi=120)
