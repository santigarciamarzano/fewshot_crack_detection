"""
visualize_predictions.py

Visualización de predicciones del modelo sobre episodios del val set.

Genera una grilla PNG con N filas, una por episodio:
    support image | query image | ground truth | prediction

Útil para inspección visual rápida del modelo durante desarrollo.
No requiere imágenes completas — trabaja directamente sobre el val set
de parches, igual que el entrenamiento.

Uso:
    python visualize_predictions.py \
        --checkpoint checkpoints/baseline/best_model.pt \
        --data data/ \
        --n_episodes 8 \
        --threshold 0.5 \
        --output results/viz.png \
        --format png
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # sin display — funciona en servidores sin GUI
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.episode_dataset import EpisodicDataset
from datasets.episode_dataset_png import EpisodicDatasetPNG
from models.fewshot_model import FewShotModel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualiza predicciones del modelo sobre el val set."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path al checkpoint .pt.")
    parser.add_argument("--data",       type=str, required=True,
                        help="Path al data root (debe contener val/).")
    parser.add_argument("--n_episodes", type=int, default=8,
                        help="Número de episodios a visualizar. Default: 8")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Umbral para binarizar predicción. Default: 0.5")
    parser.add_argument("--output",     type=str, default="results/viz.png",
                        help="Path de salida para el PNG. Default: results/viz.png")
    parser.add_argument("--format",     type=str, default="png",
                        choices=["png", "tiff"],
                        help="Formato del dataset: png o tiff. Default: png")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Semilla para reproducibilidad. Default: 42")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers de conversión tensor → numpy para matplotlib
# ---------------------------------------------------------------------------

def tensor_to_img(t: torch.Tensor) -> np.ndarray:
    """Convierte tensor 3 × H × W a array H × W × 3 para matplotlib."""
    return t.permute(1, 2, 0).cpu().numpy()


def tensor_to_mask(t: torch.Tensor) -> np.ndarray:
    """Convierte tensor 1 × H × W a array 2D para matplotlib."""
    return t.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Cargando checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    cfg = checkpoint["config"]
    cfg.dataset.data_root = args.data
    cfg.dataset.augment_support = False  # sin augmentation en visualización
    cfg.dataset.augment_query   = False

    model = FewShotModel(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Modelo cargado — epoch {checkpoint['epoch']}, val_iou {checkpoint['val_iou']:.4f}")
    print(f"Device: {device}")

    dataset_cls = EpisodicDatasetPNG if args.format == "png" else EpisodicDataset
    dataset = dataset_cls(cfg.dataset, split="val")
    n_episodes = min(args.n_episodes, len(dataset))
    indices = random.sample(range(len(dataset)), n_episodes)
    print(f"Visualizando {n_episodes} episodios del val set ({args.format})")

    # 4 columnas: support | query | ground truth | prediction
    n_cols = 4
    fig, axes = plt.subplots(
        n_episodes, n_cols,
        figsize=(n_cols * 3, n_episodes * 3),
    )

    if n_episodes == 1:
        axes = axes[np.newaxis, :]  # garantizar que axes sea siempre 2D

    col_titles = ["Support", "Query", "Ground Truth", "Prediction"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    with torch.no_grad():
        for row, idx in enumerate(indices):
            support_imgs, support_masks, query_img, query_mask = dataset[idx]
            # support_imgs:  K × 3 × H × W
            # query_img:     3 × H × W

            support_img  = support_imgs[0].unsqueeze(0).to(device)   # 1 × 3 × H × W
            support_mask = support_masks[0].unsqueeze(0).to(device)  # 1 × 1 × H × W
            query_input  = query_img.unsqueeze(0).to(device)         # 1 × 3 × H × W

            logits = model(support_img, support_mask, query_input)   # 1 × 1 × H × W
            pred   = (torch.sigmoid(logits) > args.threshold).float().squeeze()

            axes[row, 0].imshow(tensor_to_img(support_imgs[0]), cmap="gray")
            axes[row, 0].contourf(
                tensor_to_mask(support_masks[0]),
                levels=[0.5, 1.5],
                colors=["red"],
                alpha=0.35,
            )
            axes[row, 1].imshow(tensor_to_img(query_img), cmap="gray")
            axes[row, 2].imshow(tensor_to_mask(query_mask), cmap="hot", vmin=0, vmax=1)
            axes[row, 3].imshow(pred.cpu().numpy(), cmap="hot", vmin=0, vmax=1)

            axes[row, 0].set_ylabel(f"ep {idx}", fontsize=9)
            for col in range(n_cols):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])

    plt.suptitle(
        f"Predicciones val set — threshold={args.threshold} — "
        f"epoch {checkpoint['epoch']} — IoU {checkpoint['val_iou']:.4f}",
        fontsize=11,
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nGrilla guardada en: {output_path}")
    print(f"Tamaño: {n_episodes} episodios × {n_cols} columnas")


if __name__ == "__main__":
    main()