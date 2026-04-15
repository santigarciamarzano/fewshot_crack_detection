"""
training/losses.py

Combined Dice + BCE loss for binary crack segmentation.

Both loss components operate on raw logits — sigmoid is applied internally.
This is numerically more stable than applying sigmoid before the loss.

Reference shapes (baseline):
    logits:  B × 1 × 256 × 256  — raw model output, no activation
    targets: B × 1 × 256 × 256  — binary mask, values in {0.0, 1.0}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.base_config import LossConfig


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross-Entropy loss for binary segmentation.

    Dice loss handles class imbalance — critical when crack pixels are
    a small fraction of the image. BCE provides stable gradients everywhere.
    Together they complement each other's weaknesses.

    Args:
        cfg: LossConfig with dice_weight, bce_weight, and dice_smooth fields.

    Example:
        cfg = LossConfig(dice_weight=1.0, bce_weight=1.0, dice_smooth=1.0)
        criterion = DiceBCELoss(cfg)

        loss = criterion(logits, targets)
        # logits:  B × 1 × 256 × 256
        # targets: B × 1 × 256 × 256
    """

    def __init__(self, cfg: LossConfig) -> None:
        super().__init__()
        self.dice_weight = cfg.dice_weight
        self.bce_weight  = cfg.bce_weight
        self.smooth      = cfg.dice_smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined Dice + BCE loss.

        Args:
            logits:  Raw model output — B × 1 × 256 × 256, no activation applied.
            targets: Binary ground truth — B × 1 × 256 × 256, values in {0.0, 1.0}.

        Returns:
            Scalar loss tensor.
        """
        # logits:  B × 1 × 256 × 256
        # targets: B × 1 × 256 × 256

        bce  = self._bce(logits, targets)
        dice = self._dice(logits, targets)

        return self.bce_weight * bce + self.dice_weight * dice

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # def _bce(
    #     self,
    #     logits: torch.Tensor,
    #     targets: torch.Tensor,
    # ) -> torch.Tensor:
    #     """Binary cross-entropy loss computed on raw logits.

    #     Uses F.binary_cross_entropy_with_logits which applies sigmoid
    #     internally for numerical stability (log-sum-exp trick).

    #     Args:
    #         logits:  B × 1 × 256 × 256
    #         targets: B × 1 × 256 × 256

    #     Returns:
    #         Scalar BCE loss.
    #     """
    #     return F.binary_cross_entropy_with_logits(logits, targets)
    
    def _bce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # FOCAL LOSS: Castiga duramente los falsos positivos
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # pt es la probabilidad de acertar. Si el modelo falla, pt es bajo.
        pt = torch.exp(-bce_loss) 
        
        # alpha=0.25 y gamma=2.0 son los valores estándar de la industria
        focal_loss = 1.0 * ((1 - pt) ** 1.3) * bce_loss
        
        return focal_loss.mean()

    def _dice(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Soft Dice loss computed on sigmoid probabilities.

        "Soft" means we use continuous probabilities instead of thresholded
        binary predictions — this makes the loss differentiable everywhere.

        Dice = 1 - (2 * intersection + smooth) / (pred_sum + target_sum + smooth)

        Args:
            logits:  B × 1 × 256 × 256
            targets: B × 1 × 256 × 256

        Returns:
            Scalar Dice loss.
        """
        # probs: B × 1 × 256 × 256
        probs = torch.sigmoid(logits)

        # Flatten spatial dims for dot product computation.
        # probs:   B × N  (N = 1 × H × W)
        # targets: B × N
        probs   = probs.view(probs.shape[0], -1)
        targets = targets.view(targets.shape[0], -1)

        # intersection: B
        intersection = (probs * targets).sum(dim=1)

        # dice_score: B
        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        # Average over batch, return as loss (1 - dice_score).
        return 1.0 - dice_score.mean()
