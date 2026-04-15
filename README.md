# Few-Shot Segmentation — Crack Detection in Radiographic Images

## Overview

Research-grade framework for **few-shot segmentation of cracks** in radiographic images.

The system learns to segment cracks from very few annotated examples (1–5 shots),
using a **Siamese Encoder + Prototype Matching + U-Net Decoder** architecture.

---

## Project Structure

```
fewshot/
├── config/
│   ├── __init__.py
│   └── base_config.py          ← All configuration dataclasses
│
├── datasets/
│   ├── __init__.py
│   ├── episode_dataset.py      ← Episodic dataset (Step 7)
│   └── preprocessing.py        ← 3-channel preprocessing pipeline
│
├── models/
│   ├── __init__.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   └── resnet_encoder.py   ← ResNet backbone wrapper (Step 2)
│   ├── fewshot/
│   │   ├── __init__.py
│   │   ├── prototype.py        ← Masked average pooling (Step 3)
│   │   └── similarity.py       ← Cosine similarity maps (Step 4)
│   └── decoders/
│       ├── __init__.py
│       └── unet_decoder.py     ← U-Net style decoder (Step 5)
│
├── training/
│   ├── __init__.py
│   ├── trainer.py              ← Training loop (Step 8)
│   └── losses.py               ← Dice + BCE loss (Step 8)
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py              ← IoU, Dice score evaluation
│   └── visualization.py        ← Episode visualization helpers
│
├── experiments/
│   ├── __init__.py
│   ├── baseline.py             ← Baseline experiment runner
│   └── configs/
│       └── baseline.py         ← Baseline experiment config
│
├── README.md
└── requirements.txt
```

---

## Input Preprocessing

Each radiographic image is converted to a 3-channel tensor:

| Channel | Description             | Method                    |
|---------|-------------------------|---------------------------|
| 1       | Normalized radiograph   | Percentile clipping 1–99  |
| 2       | Edge enhancement        | Unsharp mask              |
| 3       | High-frequency filter   | Difference of Gaussians   |

Final tensor shape: `3 × H × W`

---

## Training Paradigm

Training uses **episodic few-shot learning**.

Each episode contains:
- `support_image` + `support_mask` → used to compute the crack prototype
- `query_image` + `query_mask` → the loss is computed **only** on this branch

> **Critical rule:** The support mask is used **only** for prototype computation.
> The loss is **never** computed on the support prediction.

---

## Architecture

```
Support image ──→ Encoder ──→ Support features ──→ Prototype (crack + background)
                                                           │
Query image ───→ Encoder ──→ Query features ──→ Similarity maps ──→ Decoder ──→ Mask
                    │                                                     ↑
                    └──────────── skip connections ─────────────────────┘
```

---

## Development Steps

| Step | Module              | Status  |
|------|---------------------|---------|
| 1    | Project structure   | ✅ Done  |
| 2    | Encoder wrapper     | ⬜ TODO  |
| 3    | Prototype module    | ⬜ TODO  |
| 4    | Similarity module   | ⬜ TODO  |
| 5    | Decoder             | ⬜ TODO  |
| 6    | Full model          | ⬜ TODO  |
| 7    | Episodic dataset    | ⬜ TODO  |
| 8    | Training pipeline   | ⬜ TODO  |

---

## Requirements

```
torch>=2.0
torchvision>=0.15
numpy
opencv-python
scikit-image
albumentations
```

---

## Usage (future)

```python
from config.base_config import FewShotConfig
from experiments.baseline import run_experiment

cfg = FewShotConfig()
cfg.encoder.backbone = "resnet34"
cfg.training.epochs = 100

run_experiment(cfg)
```
