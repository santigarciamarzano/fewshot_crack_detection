"""
training/metrics.py

Métricas de evaluación para segmentación binaria de grietas.

A diferencia de las losses, las métricas trabajan sobre predicciones
binarizadas {0, 1}, no sobre logits. Son solo para monitoring, nunca para backprop.

Shapes de referencia:
    preds:   B × 1 × 256 × 256  — binario {0, 1} tras aplicar threshold
    targets: B × 1 × 256 × 256  — máscara ground truth {0, 1}
"""

import torch


def binary_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """IoU medio sobre el batch para segmentación binaria.

    IoU = intersección / (pred_area + target_area - intersección)
    """
    preds = (torch.sigmoid(logits) > threshold).float()

    preds   = preds.view(preds.shape[0], -1)    # B × N
    targets = targets.view(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou   = (intersection + eps) / (union + eps)

    return iou.mean()


def binary_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Dice coefficient medio sobre el batch para segmentación binaria.

    Dice = 2 * intersección / (pred_area + target_area)

    Nota: esta es la métrica de evaluación (predicciones binarias duras),
    no la Dice loss (que usa probabilidades continuas para ser diferenciable).
    """
    preds = (torch.sigmoid(logits) > threshold).float()

    preds   = preds.view(preds.shape[0], -1)    # B × N
    targets = targets.view(targets.shape[0], -1)

    intersection = (preds * targets).sum(dim=1)
    dice = (2.0 * intersection + eps) / (
        preds.sum(dim=1) + targets.sum(dim=1) + eps
    )

    return dice.mean()
