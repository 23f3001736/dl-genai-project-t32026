# Architecture diagram of model A, my 3-stage U-Net built from scratch (UNet3, base width b = 32).
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 110); ax.set_ylim(-4, 46); ax.axis("off")


def box(x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=fc, ec="#444", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.6)


def arrow(x1, y1, x2, y2, style="->", color="#444", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle=style, color=color, lw=1, ls=ls))


enc = [(4, 33, "Input 3 x 128 x 800\n-> block(3, 32)\n32 x 128 x 800", "#e8eef7"),
       (16, 23, "maxpool 2 -> block(32, 64)\n64 x 64 x 400", "#e8eef7"),
       (28, 13, "maxpool 2 -> block(64, 128)\n128 x 32 x 200", "#e8eef7")]
dec = [(80, 33, "up-conv -> concat e1\n-> block(64, 32)\n32 x 128 x 800", "#eaf5ea"),
       (68, 23, "up-conv -> concat e2\n-> block(128, 64)\n64 x 64 x 400", "#eaf5ea"),
       (56, 13, "up-conv -> concat e3\n-> block(256, 128)\n128 x 32 x 200", "#eaf5ea")]
W_, H_ = 22, 8
for x, y, t, fc in enc + dec:
    box(x, y, W_, H_, t, fc)
box(42, 2, 22, 7, "maxpool 2 -> block(128, 256)\nbottleneck 256 x 16 x 100", "#fdf3d8")
box(94, 22, 14, 8, "1x1 conv -> 4 logits\nsigmoid per class\n4 x 128 x 800", "#f7e8ee")
for (x1, y1, *_), (x2, y2, *_) in zip(enc, enc[1:]):
    arrow(x1 + 11, y1, x2 + 11, y2 + H_)
arrow(39, 13, 50, 9); arrow(56, 9, 64, 13)
for (x1, y1, *_), (x2, y2, *_) in zip(dec[::-1], dec[::-1][1:]):
    arrow(x1 + 11, y1 + H_, x2 + 11, y2)
for (xe, ye, *_), (xd, yd, *_) in zip(enc, dec):
    arrow(xe + W_, ye + H_ / 2, xd, yd + H_ / 2, color="#888", ls="--")
arrow(102, 37, 101, 30)
ax.text(55, 44.5, "Model A: 3-stage U-Net built from scratch (1.93 M parameters, random init)", ha="center", fontsize=10, weight="bold")
ax.text(55, -3, "block(c_in, c_out) = [3x3 conv -> BatchNorm -> ReLU] x 2 (padding 1).  Dashed = skip connection (copy and concatenate).  "
        "up-conv = 2x2 transposed conv.", ha="center", fontsize=7.5, color="#333")
plt.savefig(sys.argv[1], dpi=130, bbox_inches="tight")
