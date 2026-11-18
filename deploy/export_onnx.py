# Export models D + E with 4 flip TTA into one ONNX graph for the browser demo.
# Input: (1, 3, 128, 800) float32 RGB in [0, 1]. Output: (1, 4, 128, 800) averaged probabilities.
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent / "hf-space"))
import app  # noqa: E402  (loads D and E exactly as the Gradio app does)


class Ensemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        out = 0
        for m in self.models:
            out = out + app.tta4(m, x)
        return out / len(self.models)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    model = Ensemble(app.MODELS).eval()
    x = torch.rand(1, 3, app.H, app.W)
    torch.onnx.export(model, (x,), str(out), input_names=["image"], output_names=["prob"],
                      opset_version=17, dynamo=False)
    print("saved", out, round(out.stat().st_size / 1e6, 1), "MB")
