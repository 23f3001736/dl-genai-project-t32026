---
title: SurfaceGuard AI Defect Segmentation
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Industrial surface defect segmentation (IITM DL GenAI)
models:
  - 23f3001736/surfaceguard-defect-segmentation
---

# SurfaceGuard AI: Industrial Surface Defect Segmentation

DL GenAI Project, IITM BS, Term 3 2026, by Sripathi Dhruv Reddy (23f3001736).

Upload a 800 x 128 surface strip image. The page runs the final Kaggle pipeline in the browser with onnxruntime-web:
models D + E (FPN, EfficientNet-B3), 4-flip TTA, averaged probabilities, per-class threshold and minimum area,
then a coloured overlay, a per-class table and the competition RLE.

Model: [23f3001736/surfaceguard-defect-segmentation](https://huggingface.co/23f3001736/surfaceguard-defect-segmentation).
Public leaderboard 0.66965, validation score 0.7308.
