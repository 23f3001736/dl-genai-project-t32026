---
title: SurfaceGuard AI Defect Segmentation
emoji: 🔍
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Industrial surface defect segmentation (IITM DL GenAI)
---

# SurfaceGuard AI: Industrial Surface Defect Segmentation

DL GenAI Project, IITM BS, Term 3 2026, by Sripathi Dhruv Reddy (23f3001736).

Upload a 800 x 128 surface strip image. The app runs the same pipeline as the final Kaggle submission:

1. two FPN models with an EfficientNet-B3 encoder (D and E, trained 40 epochs with EMA)
2. 4-flip test time augmentation for each model, probabilities averaged
3. per-class threshold and minimum area (tuned on validation)
4. coloured mask overlay, per-class table and competition RLE

Public leaderboard: 0.66965. Validation score: 0.7308.

Code: GitHub `23f3001736/dl-genai-project-t32026`.
