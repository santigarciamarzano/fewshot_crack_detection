"""
models/fewshot_model.py

Modelo completo de segmentación few-shot.

Ensambla el encoder, el módulo de prototipos, el módulo de similitud y el decoder
en un único modelo end-to-end. El encoder es Siamés: support y query comparten pesos.

Flujo de datos:
    support_img  → encoder → layer4 → PrototypeModule → proto_crack, proto_bg
    query_img    → encoder → layer4 → SimilarityModule (+ prototipos) → sim_map
                          → layer1..layer3 → skip connections
    cat(query layer4, sim_map) → UNetDecoder (+ skips) → logits de máscara

Shapes (baseline: ResNet34, entrada 256×256):
    support_img:   B × 3 × 256 × 256
    support_mask:  B × 1 × 256 × 256
    query_img:     B × 3 × 256 × 256
    mask_logits:   B × 1 × 256 × 256
"""

import torch
import torch.nn as nn
from typing import Dict

from config.base_config import FewShotConfig
from models.encoders.encoder_factory import build_encoder
from models.fewshot.prototype_module import PrototypeModule
from models.fewshot.similarity import SimilarityModule
from models.decoders.unet_decoder import UNetDecoder

class FewShotModel(nn.Module):
    """Modelo de segmentación few-shot end-to-end.

    Combina encoder (Siamés), módulo de prototipos, módulo de similitud y decoder.

    Args:
        cfg: Objeto FewShotConfig raíz. Todos los sub-configs se leen desde aquí.

    Ejemplo:
        cfg = get_baseline_config()
        model = FewShotModel(cfg)
        logits = model(support_img, support_mask, query_img)  # B × 1 × 256 × 256
    """

    def __init__(self, cfg: FewShotConfig) -> None:
        super().__init__()

        self.encoder   = build_encoder(cfg.encoder)
        self.prototype = PrototypeModule(cfg.prototype)
        self.similarity = SimilarityModule(cfg.similarity)

        # bottleneck = layer4 channels + 2 canales del mapa de similitud
        bottleneck_channels = self.encoder.out_channels + 2
        skip_channels = self.encoder.skip_channels

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
        """Forward pass completo. Devuelve logits de segmentación B × 1 × H × W."""
        # Rama support: extrae características y calcula prototipos de grieta y fondo
        support_features = self.encoder(support_img)
        proto_crack, proto_bg = self.prototype(
            support_features["layer4"],
            support_mask,
        )

        # Rama query: extrae características a múltiples escalas
        # layer1..layer3 van como skip connections al decoder
        query_features = self.encoder(query_img)

        # Mapa de similitud coseno entre query y los dos prototipos: B × 2 × H/32 × W/32
        sim_map = self.similarity(
            query_features["layer4"],
            proto_crack,
            proto_bg,
        )

        # Bottleneck: concatenación de layer4 + sim_map
        bottleneck = torch.cat([query_features["layer4"], sim_map], dim=1)

        mask_logits = self.decoder(
            bottleneck,
            skips={
                "layer3": query_features["layer3"],
                "layer2": query_features["layer2"],
                "layer1": query_features["layer1"],
            },
        )

        return mask_logits
