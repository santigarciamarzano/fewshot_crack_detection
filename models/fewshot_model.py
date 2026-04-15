"""
models/fewshot_model.py

Full few-shot segmentation model.

Assembles the encoder, prototype module, similarity module, and decoder
into a single end-to-end model. This module contains no logic of its own —
its only responsibility is to orchestrate the forward pass in the correct order.

Data flow:
    support_img  → encoder → layer4 → PrototypeModule → proto_crack, proto_bg
    query_img    → encoder → layer4 → SimilarityModule (+ prototypes) → sim_map
                          → layer1..layer3 → skip connections
    cat(query layer4, sim_map) → UNetDecoder (+ skips) → mask logits

The encoder is Siamese: support and query share the exact same weights.
Loss is computed externally — this module returns raw logits only.

Shapes (baseline: ResNet34, 256×256 input):
    support_img:   B × 3 × 256 × 256
    support_mask:  B × 1 × 256 × 256
    query_img:     B × 3 × 256 × 256
    mask_logits:   B × 1 × 256 × 256
"""

import torch
import torch.nn as nn
from typing import Dict

from config.base_config import FewShotConfig
from models.encoders.resnet_encoder import ResNetEncoder
from models.fewshot.prototype_module import PrototypeModule
from models.fewshot.similarity import SimilarityModule
from models.decoders.unet_decoder import UNetDecoder


# Skip connection channels for each backbone, ordered layer3 → layer2 → layer1.
# These are fixed by the ResNet architecture — not configurable.
_SKIP_CHANNELS: Dict[str, list] = {
    "resnet18": [256, 128, 64],
    "resnet34": [256, 128, 64],
    "resnet50": [1024, 512, 256],
    "resnet101": [1024, 512, 256],
}


class FewShotModel(nn.Module):
    """End-to-end few-shot segmentation model.

    Composes encoder, prototype, similarity, and decoder modules.
    The encoder is shared (Siamese) between support and query branches.

    Args:
        cfg: FewShotConfig root config object. All sub-configs are read
             from here — no values are hardcoded in this class.

    Example:
        cfg = get_baseline_config()
        model = FewShotModel(cfg)

        logits = model(support_img, support_mask, query_img)
        # logits: B × 1 × 256 × 256

        probs = torch.sigmoid(logits)   # for inference
    """

    def __init__(self, cfg: FewShotConfig) -> None:
        super().__init__()

        # --- Encoder (Siamese — shared between support and query) ----------
        self.encoder = ResNetEncoder(cfg.encoder)

        # --- Few-shot branch -----------------------------------------------
        self.prototype = PrototypeModule(cfg.prototype)
        self.similarity = SimilarityModule(cfg.similarity)

        # --- Decoder ---------------------------------------------------------
        # bottleneck_channels: layer4 output channels + 2 similarity channels.
        # skip_channels: fixed by backbone architecture, not by config.
        backbone = cfg.encoder.backbone
        bottleneck_channels = self.encoder.out_channels + 2
        skip_channels = _SKIP_CHANNELS[backbone]

        self.decoder = UNetDecoder(
            cfg=cfg.decoder,
            bottleneck_channels=bottleneck_channels,
            skip_channels=skip_channels,
        )

    def forward(
        self,
        support_img: torch.Tensor,
        support_mask: torch.Tensor,
        query_img: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full few-shot segmentation forward pass.

        Args:
            support_img:  Support image — B × 3 × H × W.
            support_mask: Binary support mask — B × 1 × H × W.
                          Values in {0, 1}: 1 = crack, 0 = background.
            query_img:    Query image to segment — B × 3 × H × W.

        Returns:
            Segmentation logits — B × 1 × H × W.
            Apply sigmoid externally for probabilities.
        """
        # --- Support branch -------------------------------------------------
        # support_features["layer4"]: B × 512 × 8 × 8
        support_features = self.encoder(support_img)

        # proto_crack: B × 512
        # proto_bg:    B × 512
        proto_crack, proto_bg = self.prototype(
            support_features["layer4"],
            support_mask,
        )

        # --- Query branch ---------------------------------------------------
        # query_features["layer1"]: B × 64  × 64 × 64
        # query_features["layer2"]: B × 128 × 32 × 32
        # query_features["layer3"]: B × 256 × 16 × 16
        # query_features["layer4"]: B × 512 × 8  × 8
        query_features = self.encoder(query_img)

        # sim_map: B × 2 × 8 × 8
        sim_map = self.similarity(
            query_features["layer4"],
            proto_crack,
            proto_bg,
        )

        # --- Decoder --------------------------------------------------------
        # Concatenate query layer4 with similarity map to form the bottleneck.
        # bottleneck: B × 514 × 8 × 8  (512 + 2)
        bottleneck = torch.cat([query_features["layer4"], sim_map], dim=1)

        # mask_logits: B × 1 × 256 × 256
        mask_logits = self.decoder(
            bottleneck,
            skips={
                "layer3": query_features["layer3"],
                "layer2": query_features["layer2"],
                "layer1": query_features["layer1"],
            },
        )

        return mask_logits
