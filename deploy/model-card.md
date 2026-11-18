---
license: mit
library_name: onnx
pipeline_tag: image-segmentation
tags: [segmentation, defect-detection, fpn, efficientnet, onnx]
---

# SurfaceGuard AI defect segmentation (D + E ensemble)

DL GenAI Project, IITM BS, Term 3 2026, by Sripathi Dhruv Reddy (23f3001736).

- `surfaceguard_ensemble.onnx`: models D and E (FPN, EfficientNet-B3 encoder) with 4-flip TTA in one graph.
  Input `image` (1, 3, 128, 800) float32 RGB in [0, 1]. Output `prob` (1, 4, 128, 800), sigmoid probabilities
  for defect_a to defect_d. ImageNet normalisation is inside the graph.
- `m4_D_effb3_fpn.pt`, `m4_E_effb3_fpn_seed7.pt`: PyTorch EMA state dicts (`Normalized(smp.FPN("efficientnet-b3", classes=4))`).

Post-processing used for the Kaggle submission: threshold 0.2 for every class, minimum area
defect_a 0, defect_b 100, defect_c 50, defect_d 0 pixels. Public leaderboard 0.66965, validation score 0.7308.
