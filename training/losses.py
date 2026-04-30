"""
training/losses.py

Loss combinada Dice + BCE para segmentación binaria de grietas.

Ambos componentes trabajan sobre logits crudos (sigmoid se aplica internamente).

Shapes de referencia (baseline):
    logits:  B × 1 × 256 × 256  — salida raw del modelo, sin activación
    targets: B × 1 × 256 × 256  — máscara binaria, valores en {0.0, 1.0}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.base_config import LossConfig


class DiceBCELoss(nn.Module):
    """Loss combinada Dice + BCE para segmentación binaria.

    Dice maneja el desbalanceo de clases (píxeles de grieta son minoría).
    BCE aporta gradientes estables en toda la imagen. Se complementan bien.

    Args:
        cfg: LossConfig con los campos dice_weight, bce_weight y dice_smooth.
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
        """Calcula la loss combinada Dice + BCE."""
        bce  = self._bce(logits, targets)
        dice = self._dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _bce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        # Focal loss: pesa más los ejemplos difíciles (falsos positivos/negativos)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)                     # probabilidad de acertar en cada píxel
        focal_loss = 1.0 * ((1 - pt) ** 1.3) * bce_loss
        return focal_loss.mean()

    def _dice(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Soft Dice loss sobre probabilidades sigmoid (diferenciable en todo punto)."""
        probs = torch.sigmoid(logits)

        # Aplanamos las dimensiones espaciales para calcular el producto escalar
        probs   = probs.view(probs.shape[0], -1)    # B × N
        targets = targets.view(targets.shape[0], -1)

        intersection = (probs * targets).sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        return 1.0 - dice_score.mean()
