# Few-Shot Segmentation — Crack Detection in Radiographic Images

## Project Structure
```
fewshot/
├── config/
│   └── base_config.py          ✅ done
├── datasets/
│   ├── episode_dataset.py      ← Step 7
│   └── preprocessing.py        ← futuro
├── models/
│   ├── encoders/
│   │   └── resnet_encoder.py   ✅ done
│   ├── fewshot/
│   │   ├── prototype.py        ✅ done
│   │   └── similarity.py       ← Step 4
│   └── decoders/
│       └── unet_decoder.py     ← Step 5
├── training/
│   ├── trainer.py              ← Step 8
│   └── losses.py               ← Step 8
├── utils/
│   ├── metrics.py              ← Step 8
│   └── visualization.py        ← Step 8
├── experiments/
│   └── baseline.py             ✅ done  
└── train.py                    ← Step 8
```

## Steps

| Step | Module | Status |
|------|--------|--------|
| 1 | Project structure
| 2 | Encoder wrapper
| 3 | Prototype module 
| 4 | Similarity module 
| 5 | Decoder 
| 6 | Full model
| 7 | Episodic dataset
| 8 | Training pipeline 