# SurfaceGuard AI demo: same inference pipeline as the final Kaggle notebook (Milestone 5)
# image -> models D and E with 4 flip TTA -> average probabilities -> per class threshold and min area -> masks + RLE
import cv2
import gradio as gr
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

H, W = 128, 800
CLASSES = ["defect_a", "defect_b", "defect_c", "defect_d"]
# (threshold, min area in pixels) per class, tuned on the validation split in the notebook
FINAL_CFG = {"defect_a": (0.2, 0), "defect_b": (0.2, 100), "defect_c": (0.2, 50), "defect_d": (0.2, 0)}
# overlay colour per class (RGB)
COLOURS = {"defect_a": (255, 60, 60), "defect_b": (60, 200, 255), "defect_c": (255, 200, 0), "defect_d": (180, 80, 255)}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Normalized(nn.Module):
    # the pretrained encoders expect ImageNet normalised input
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", MEAN.clone())
        self.register_buffer("std", STD.clone())

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def load_model(path):
    # FPN with EfficientNet-B3 encoder, weights come from the EMA checkpoint saved in the notebook
    net = Normalized(smp.FPN("efficientnet-b3", encoder_weights=None, in_channels=3, classes=4))
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval()


MODELS = [load_model("m4_D_effb3_fpn.pt"), load_model("m4_E_effb3_fpn_seed7.pt")]


def tta4(model, x):
    # average over no flip, horizontal flip, vertical flip and both, each output flipped back
    outs = []
    for dims in ([], [3], [2], [2, 3]):
        xx = torch.flip(x, dims=dims) if dims else x
        p = torch.sigmoid(model(xx))
        outs.append(torch.flip(p, dims=dims) if dims else p)
    return sum(outs) / 4


def rle_encode(mask):
    # competition RLE: 1-based starts, column-major (Fortran) order
    pixels = np.concatenate([[0], mask.flatten(order="F"), [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


@torch.no_grad()
def predict(image):
    if image is None:
        raise gr.Error("Please upload an image first.")
    rgb = np.asarray(image.convert("RGB"))
    orig_h, orig_w = rgb.shape[:2]
    # the models were trained at native 128 x 800
    if (orig_h, orig_w) != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    prob = (sum(tta4(m, x) for m in MODELS) / len(MODELS))[0].numpy()

    overlay = rgb.astype(np.float32).copy()
    rows, rles = [], []
    for c, name in enumerate(CLASSES):
        tau, min_area = FINAL_CFG[name]
        mask = (prob[c] >= tau).astype(np.uint8)
        area = int(mask.sum())
        # drop the whole mask if it is smaller than the tuned minimum area
        found = area >= max(min_area, 1)
        if found:
            colour = np.array(COLOURS[name], dtype=np.float32)
            overlay[mask > 0] = 0.45 * overlay[mask > 0] + 0.55 * colour
        rows.append([name, "yes" if found else "no", area if found else 0, round(float(prob[c].max()), 3)])
        rles.append(f"{name}: {rle_encode(mask) if found else '(empty)'}")
    note = "" if (orig_h, orig_w) == (H, W) else f"Input was {orig_w} x {orig_h}, resized to 800 x 128 for the model.\n\n"
    return overlay.clip(0, 255).astype(np.uint8), rows, note + "\n".join(rles)


LEGEND = " | ".join(f"<span style='color:rgb{COLOURS[c]}'>&#9632;</span> {c}" for c in CLASSES)

with gr.Blocks(title="SurfaceGuard AI - Defect Segmentation") as demo:
    gr.Markdown(
        "# SurfaceGuard AI: Industrial Surface Defect Segmentation\n"
        "DL GenAI Project (IITM BS, Term 3 2026) by **Sripathi Dhruv Reddy (23f3001736)**.\n\n"
        "Upload a 800 x 128 grayscale surface strip image. Two EfficientNet-B3 FPN models with 4-flip "
        "test time augmentation are averaged, then a per-class threshold and minimum area are applied, "
        "exactly as in the final Kaggle submission (public leaderboard 0.66965).\n\n" + LEGEND
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Input image")
    btn = gr.Button("Detect defects", variant="primary")
    out_img = gr.Image(label="Predicted defects (overlay)")
    out_tab = gr.Dataframe(headers=["defect", "found", "area (pixels)", "max probability"], label="Per class result")
    out_rle = gr.Textbox(label="Run-length encoding (competition format)", lines=5)
    btn.click(predict, inputs=inp, outputs=[out_img, out_tab, out_rle])

if __name__ == "__main__":
    demo.launch()
