"""
models/encoders/resnet_encoder.py

Wrapper de backbone ResNet para extraer features a múltiples escalas.

Expone los mapas de features intermedios de los cuatro bloques residuales:
    - layer4 (más profundo) → cálculo de prototipos en la rama few-shot
    - layer1..layer3         → skip connections en el decoder U-Net

El mismo encoder se comparte (Siamés) entre support y query.

Shapes (con entrada 3 × 256 × 256):
    layer1: B × 64  × 64 × 64
    layer2: B × 128 × 32 × 32
    layer3: B × 256 × 16 × 16
    layer4: B × 512 ×  8 ×  8   (ResNet18 / ResNet34)
           B × 2048 ×  8 ×  8   (ResNet50)
"""

from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)

from config.base_config import EncoderConfig
from models.encoders.base_encoder import BaseEncoder


# Mapping from backbone name → (model constructor, pretrained weights)
_BACKBONE_REGISTRY = {
    "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V2),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
}

# Output channels for layer4, per backbone
BACKBONE_OUT_CHANNELS: Dict[str, int] = {
    "resnet18": 512,
    "resnet34": 512,
    "resnet50": 2048,
    "resnet101": 2048,
}

_RESNET_SKIP_CHANNELS: Dict[str, List[int]] = {
    "resnet18":  [256, 128, 64],
    "resnet34":  [256, 128, 64],
    "resnet50":  [1024, 512, 256],
    "resnet101": [1024, 512, 256],
}


class ResNetEncoder(BaseEncoder):
    """Encoder ResNet multi-escala para segmentación few-shot.

    Extrae mapas de features en cuatro escalas y opcionalmente congela capas.

    Args:
        cfg: EncoderConfig con backbone, pretrained, in_channels y frozen_layers.
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()

        if cfg.backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{cfg.backbone}'. "
                f"Valid options: {list(_BACKBONE_REGISTRY.keys())}"
            )

        constructor, weights = _BACKBONE_REGISTRY[cfg.backbone]
        backbone = constructor(weights=weights if cfg.pretrained else None)

        if cfg.in_channels != 3:
            # Reemplazamos la primera conv preservando kernel, stride y padding
            backbone.conv1 = nn.Conv2d(
                cfg.in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False,
            )

        # Extraemos el stem y los cuatro bloques residuales; descartamos avgpool y fc
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 64 ch,       stride 4
        self.layer2 = backbone.layer2  # 128 ch,      stride 8
        self.layer3 = backbone.layer3  # 256 ch,      stride 16
        self.layer4 = backbone.layer4  # 512/2048 ch, stride 32

        self._out_channels = BACKBONE_OUT_CHANNELS[cfg.backbone]
        self._backbone_name = cfg.backbone
        self._freeze_layers(cfg.frozen_layers)

    # ------------------------------------------------------------------
    # BaseEncoder contract
    # ------------------------------------------------------------------

    @property
    def out_channels(self) -> int:
        """Canales en la salida de layer4: 512 para ResNet18/34, 2048 para ResNet50/101."""
        return self._out_channels
    
    @property
    def skip_channels(self) -> List[int]:
        return _RESNET_SKIP_CHANNELS[self._backbone_name]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extrae features multi-escala. Devuelve dict con keys layer1..layer4."""
        x  = self.stem(x)    # B × 64  × H/4  × W/4
        f1 = self.layer1(x)  # B × 64  × H/4  × W/4
        f2 = self.layer2(f1) # B × 128 × H/8  × W/8
        f3 = self.layer3(f2) # B × 256 × H/16 × W/16
        f4 = self.layer4(f3) # B × 512 × H/32 × W/32
        return {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}

    def _freeze_layers(self, layer_names: list[str]) -> None:
        """Congela los parámetros de las capas indicadas (excluye de backprop)."""
        for name in layer_names:
            module = getattr(self, name, None)
            if module is None:
                raise ValueError(
                    f"Cannot freeze '{name}': layer not found in encoder. "
                    f"Valid layers: stem, layer1, layer2, layer3, layer4."
                )
            for param in module.parameters():
                param.requires_grad = False
