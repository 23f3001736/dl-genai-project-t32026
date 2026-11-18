# Draws the final pipeline / architecture diagram used in the report.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 110); ax.set_ylim(-2, 44); ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#444", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444", lw=1))


box(1, 17, 12, 8, "Input image\n3 x 128 x 800\nRGB in [0, 1]", "#e8eef7")
box(16.5, 17, 12, 8, "4-flip TTA\nnone, H, V, H+V", "#fdf3d8")
for y, name in [(27, "Model D: seed 1736, plain BCE + Dice"), (3, "Model E: seed 7, BCE pos_weight 8 on defect_b")]:
    box(31, y, 43, 12, "", "#eaf5ea")
    ax.text(52.5, y + 10.3, name, ha="center", va="center", fontsize=8.5, weight="bold")
    for i, t in enumerate(["ImageNet\nnormalise", "EfficientNet-B3\nencoder\n(ImageNet)", "FPN\ndecoder\n(multi-scale)", "4 x sigmoid\n(multi-label)"]):
        box(32.5 + i * 10.4, y + 1, 9.2, 7.6, t, "#fff")
        if i:
            arrow(32.5 + i * 10.4 - 1.2, y + 4.8, 32.5 + i * 10.4, y + 4.8)
for y in (33, 9):
    arrow(28.8, 21, 31, y)
box(76, 17, 11, 8, "Average\nprobabilities\n(flip back first)", "#fdf3d8")
for y in (33, 9):
    arrow(74.8, y, 76, 21)
box(91, 17, 17, 8, "Per class threshold 0.2\n+ min area\n(a 0, b 100, c 50, d 0)\n-> RLE", "#f7e8ee")
arrow(13.8, 21, 17, 21); arrow(87.8, 21, 91, 21)
ax.text(55, 42, "Final pipeline: D + E ensemble, 4-flip TTA, per-class post-processing", ha="center", fontsize=10, weight="bold")
ax.text(55, -1, "Training: BCE + soft Dice, AdamW (wd 1e-4), OneCycle lr 5e-4, 40 epochs, batch 16, AMP, EMA weights, "
        "rare-class oversampling (a x2, b x3), flips + brightness/contrast", ha="center", fontsize=7.5, color="#333")
plt.savefig(__import__("sys").argv[1], dpi=130, bbox_inches="tight")
