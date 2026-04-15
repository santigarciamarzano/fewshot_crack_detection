"""
models/fewshot/prototype_module.py

Prototype computation via Masked Average Pooling (MAP).

Given a support feature map and its binary segmentation mask, this module
computes two prototype vectors:
    - crack prototype:      average of features INSIDE the mask
    - background prototype: average of features OUTSIDE the mask

These prototypes compress a single support image into two vectors that
represent "what a crack looks like" and "what background looks like"
in feature space. They are the core of the few-shot branch.

Shapes (baseline: ResNet34, 256×256 input, k_shot=1):
    features:  B × 512 × 8 × 8
    mask:      B × 1   × 256 × 256  →  downsampled to  B × 1 × 8 × 8
    prototype: B × 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from config.base_config import PrototypeConfig


class PrototypeModule(nn.Module):
    """Computes crack and background prototypes via Masked Average Pooling.

    No learnable parameters — this module is a pure functional transformation.
    The only behavior controlled by config is optional L2 normalization.

    Args:
        cfg: PrototypeConfig with normalize_features and eps fields.

    Example:
        cfg = PrototypeConfig(normalize_features=True, eps=1e-6)
        module = PrototypeModule(cfg)

        features = encoder(support_img)["layer4"]  # B × 512 × 8 × 8
        mask = ...                                  # B × 1 × 256 × 256

        proto_crack, proto_bg = module(features, mask)
        # proto_crack: B × 512
        # proto_bg:    B × 512
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
        """Compute crack and background prototypes from support data.

        The mask is automatically downsampled to match the spatial resolution
        of the feature map. No assumptions are made about input resolution.

        Args:
            features: Support feature map of shape B × C × H × W.
                      Typically layer4 output: B × 512 × 8 × 8.
            mask:     Binary segmentation mask of shape B × 1 × H_in × W_in.
                      Will be resized to match (H, W) of features.
                      Values expected in {0, 1} or [0, 1] range.

        Returns:
            Tuple of (proto_crack, proto_bg), each of shape B × C.
                proto_crack: prototype for crack class (foreground).
                proto_bg:    prototype for background class.
        """
        # features: B × C × H × W
        # mask:     B × 1 × H_in × W_in
        _, _, h, w = features.shape

        # mode="nearest" preserves binary values — bilinear would introduce
        # fractional values like 0.3, 0.7 that corrupt the masked pooling.
        # mask_downsampled: B × 1 × H × W
        mask_downsampled = F.interpolate(
            mask.float(),
            size=(h, w),
            mode="nearest",
        )

        # mask_bg: B × 1 × H × W
        mask_bg = 1.0 - mask_downsampled

        # proto_crack: B × C
        # proto_bg:    B × C
        proto_crack = self._masked_average_pool(features, mask_downsampled)
        proto_bg    = self._masked_average_pool(features, mask_bg)

        if self.normalize_features:
            # L2-normalize along channel dim before cosine similarity downstream.
            proto_crack = F.normalize(proto_crack, p=2, dim=1)
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
        """Average pool features weighted by a binary spatial mask.

        Only spatial locations where mask > 0 contribute to the average.
        This is fundamentally different from F.avg_pool2d, which averages
        all spatial locations uniformly regardless of mask shape.

        Args:
            features: B × C × H × W
            mask:     B × 1 × H × W — values in [0, 1].

        Returns:
            B × C prototype vector.
        """
        # mask broadcasts across C: B×1×H×W → B×C×H×W
        # masked_features: B × C × H × W
        masked_features = features * mask

        # summed: B × C
        summed = masked_features.sum(dim=(2, 3))

        # eps guards against all-zero masks (support image with no crack pixels).
        # area: B × 1
        area = mask.sum(dim=(2, 3)) + self.eps

        # area broadcasts across C: B×1 → B×C
        # prototype: B × C
        return summed / area