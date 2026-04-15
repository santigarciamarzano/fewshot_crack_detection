"""
training/metrics.py

Evaluation metrics for binary crack segmentation.

Unlike losses, metrics operate on thresholded binary predictions {0, 1},
not on raw logits or probabilities. They are not differentiable and are
used only for monitoring — never for backprop.

Reference shapes (baseline):
    preds:   B × 1 × 256 × 256  — binary {0, 1} after threshold
    targets: B × 1 × 256 × 256  — binary {0, 1} ground truth
"""

import torch


def binary_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean Intersection over Union for binary segmentation.

    Applies sigmoid + threshold to convert logits to binary predictions,
    then computes IoU per image and averages over the batch.

    IoU = intersection / (pred_area + target_area - intersection)

    Args:
        logits:    Raw model output — B × 1 × H × W, no activation applied.
        targets:   Binary ground truth — B × 1 × H × W, values in {0.0, 1.0}.
        threshold: Probability threshold to binarize predictions. Default 0.5.
        eps:       Smoothing to avoid division by zero on empty masks.

    Returns:
        Scalar mean IoU over the batch.
    """
    # preds: B × 1 × H × W — binary {0, 1}
    preds = (torch.sigmoid(logits) > threshold).float()

    # Flatten spatial dims.
    # preds:   B × N
    # targets: B × N
    preds   = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # intersection: B
    intersection = (preds * targets).sum(dim=1)

    # union: B
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    # iou: B
    iou = (intersection + eps) / (union + eps)

    return iou.mean()


def binary_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean Dice coefficient for binary segmentation.

    Applies sigmoid + threshold to convert logits to binary predictions,
    then computes Dice per image and averages over the batch.

    Dice = 2 * intersection / (pred_area + target_area)

    Note: this is the evaluation metric, not the Dice loss. The loss uses
    soft probabilities to stay differentiable; this uses hard binary preds.

    Args:
        logits:    Raw model output — B × 1 × H × W, no activation applied.
        targets:   Binary ground truth — B × 1 × H × W, values in {0.0, 1.0}.
        threshold: Probability threshold to binarize predictions. Default 0.5.
        eps:       Smoothing to avoid division by zero on empty masks.

    Returns:
        Scalar mean Dice coefficient over the batch.
    """
    # preds: B × 1 × H × W — binary {0, 1}
    preds = (torch.sigmoid(logits) > threshold).float()

    # Flatten spatial dims.
    # preds:   B × N
    # targets: B × N
    preds   = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # intersection: B
    intersection = (preds * targets).sum(dim=1)

    # dice: B
    dice = (2.0 * intersection + eps) / (
        preds.sum(dim=1) + targets.sum(dim=1) + eps
    )

    return dice.mean()
