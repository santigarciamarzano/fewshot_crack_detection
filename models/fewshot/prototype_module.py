"""
models/fewshot/prototype_module.py

Cálculo de prototipos mediante Masked Average Pooling (MAP).

Dado el mapa de características del support y su máscara de segmentación,
calcula dos vectores prototipo:
    - prototipo de grieta:   promedio de features DENTRO de la máscara
    - prototipo de fondo:    promedio de features FUERA de la máscara

Shapes (baseline: ResNet34, entrada 256×256, k_shot=1):
    features:  B × 512 × 8 × 8
    mask:      B × 1   × 256 × 256  →  downsampled a  B × 1 × 8 × 8
    prototype: B × 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config.base_config import PrototypeConfig


class PrototypeModule(nn.Module):
    """Calcula los prototipos de grieta y fondo mediante Masked Average Pooling.

    No tiene parámetros aprendibles: es una transformación funcional pura.

    Args:
        cfg: PrototypeConfig con los campos normalize_features y eps.
    """

    def __init__(self, cfg: PrototypeConfig) -> None:
        super().__init__()
        self.normalize_features = cfg.normalize_features
        self.eps = cfg.eps

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcula los prototipos de grieta y fondo a partir del support.

        La máscara se redimensiona automáticamente para coincidir con la
        resolución espacial del mapa de características.
        """
        _, _, h, w = features.shape

        mask_downsampled = F.interpolate(
            mask.float(),
            size=(h, w),
            mode="nearest",  # nearest preserva valores binarios (bilinear introduciría valores intermedios)
        )  # B × 1 × H × W

        mask_bg = 1.0 - mask_downsampled

        proto_crack = self._masked_average_pool(features, mask_downsampled)
        proto_bg    = self._masked_average_pool(features, mask_bg)

        if self.normalize_features:
            proto_crack = F.normalize(proto_crack, p=2, dim=1)  # L2-normalize para similitud coseno
            proto_bg    = F.normalize(proto_bg,    p=2, dim=1)

        return proto_crack, proto_bg

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _masked_average_pool(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Promedia features ponderado por la máscara. Solo contribuyen los píxeles donde mask > 0."""
        masked_features = features * mask  # mask broadcast: B×1×H×W → B×C×H×W
        summed = masked_features.sum(dim=(2, 3))         # B × C
        area   = mask.sum(dim=(2, 3)) + self.eps         # B × 1  (eps evita división por cero)
        return summed / area