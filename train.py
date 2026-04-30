"""
train.py

Punto de entrada para entrenar el modelo de segmentación few-shot.
Instancia todos los componentes desde la config y lanza el loop de entrenamiento.

Uso:
    python train.py                        # config baseline
    python train.py --backbone resnet50    # cambiar backbone
    python train.py --epochs 200           # cambiar épocas
    python train.py --data data/custom/    # cambiar ruta de datos
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.base_config import FewShotConfig
from datasets.episode_dataset import EpisodicDataset
from datasets.episode_dataset_png import EpisodicDatasetPNG
from experiments.baseline import get_baseline_config
from models.fewshot_model import FewShotModel
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos para los ajustes más comunes."""
    parser = argparse.ArgumentParser(description="Train few-shot crack segmentation model.")

    parser.add_argument("--backbone",  type=str,   default=None, help="resnet18 | resnet34 | resnet50 | resnet101")
    parser.add_argument("--epochs",    type=int,   default=None, help="Number of training epochs.")
    parser.add_argument("--lr",        type=float, default=None, help="Learning rate.")
    parser.add_argument("--k_shot",    type=int,   default=None, help="Number of support images per episode.")
    parser.add_argument("--data",      type=str,   default=None, help="Path to data root directory.")
    parser.add_argument("--device",    type=str,   default=None, help="cuda | cpu")
    parser.add_argument("--workers",   type=int,   default=4,    help="DataLoader num_workers.")
    parser.add_argument("--batch_size",  type=int,   default=None, help="Batch size (episodes per gradient update).")
    parser.add_argument("--frozen_layers", type=str, default=None, help="Capas a congelar separadas por coma. Ej: layer1 o layer1,layer2")

    return parser.parse_args()


def apply_overrides(cfg: FewShotConfig, args: argparse.Namespace) -> FewShotConfig:
    """Aplica los argumentos de CLI sobre el objeto config. Solo sobreescribe los que se pasaron explícitamente."""
    # Args con valor None significan "usar el default de la config"
    if args.backbone is not None:
        cfg.encoder.backbone = args.backbone
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.k_shot is not None:
        cfg.dataset.k_shot = args.k_shot
    if args.data is not None:
        cfg.dataset.data_root = args.data
    if args.device is not None:
        cfg.training.device = args.device
    if args.frozen_layers is not None:
        cfg.encoder.frozen_layers = args.frozen_layers.split(",")

    return cfg


def set_seed(seed: int) -> None:
    """Fija las semillas aleatorias en Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(cfg: FewShotConfig, split: str, workers: int) -> DataLoader:
    """Crea el EpisodicDataset y lo envuelve en un DataLoader."""
    dataset = EpisodicDatasetPNG(cfg.dataset, split=split)

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=(cfg.training.device == "cuda"),
        drop_last=(split == "train"),
    )


def main() -> None:
    args = parse_args()
    cfg = get_baseline_config()
    cfg = apply_overrides(cfg, args)
    set_seed(cfg.training.seed)

    print(f"Experiment: {cfg.experiment_name}")
    print(f"Backbone:   {cfg.encoder.backbone}")
    print(f"k_shot:     {cfg.dataset.k_shot}")
    print(f"Image size: {cfg.dataset.image_size}")
    print(f"Device:     {cfg.training.device}")
    print(f"Epochs:     {cfg.training.epochs}")
    print()

    train_loader = build_dataloader(cfg, split="train", workers=args.workers)
    val_loader   = build_dataloader(cfg, split="val",   workers=args.workers)

    print(f"Train episodes: {len(train_loader.dataset)}")
    print(f"Val episodes:   {len(val_loader.dataset)}")
    print()

    model = FewShotModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    print()

    trainer = Trainer(model, cfg, train_loader, val_loader)
    trainer.fit()


if __name__ == "__main__":
    main()
