"""
models/encoders/resnet_encoder.py

ResNet backbone wrapper for feature extraction at multiple scales.

The encoder wraps a torchvision ResNet and exposes intermediate feature maps
from all four residual blocks. These features are used in two ways:

    - layer4 (deepest) → prototype computation in the few-shot branch
    - layer1..layer3   → skip connections in the U-Net decoder

The same encoder instance is shared (Siamese) between support and query:
    both images pass through the same weights.

Shapes (with input 3 × 256 × 256):
    layer1: B × 64  × 64 × 64
    layer2: B × 128 × 32 × 32
    layer3: B × 256 × 16 × 16
    layer4: B × 512 ×  8 ×  8   (ResNet18 / ResNet34)
           B × 2048 ×  8 ×  8   (ResNet50)
"""

from typing import Dict

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


class ResNetEncoder(nn.Module):
    """Multi-scale ResNet encoder for few-shot segmentation.

    Wraps a torchvision ResNet and extracts feature maps at four scales.
    The first convolution layer is replaced to accept `in_channels` input
    (default 3 for our preprocessed radiographic images).

    Args:
        cfg: EncoderConfig with backbone, pretrained, in_channels,
             and frozen_layers fields.

    Example:
        cfg = EncoderConfig(backbone="resnet34", pretrained=True, in_channels=3)
        encoder = ResNetEncoder(cfg)
        features = encoder(x)          # x: B × 3 × 256 × 256
        features["layer4"]             # B × 512 × 8 × 8
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()

        if cfg.backbone not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{cfg.backbone}'. "
                f"Valid options: {list(_BACKBONE_REGISTRY.keys())}"
            )

        constructor, weights = _BACKBONE_REGISTRY[cfg.backbone]

        # Load backbone (with or without pretrained weights)
        weights = weights if cfg.pretrained else None
        backbone = constructor(weights=weights)

        # Replace the first conv if in_channels != 3.
        # We preserve kernel_size, stride and padding so spatial dims are
        # identical to the standard ResNet stem.
        if cfg.in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                cfg.in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        # Extract the stem and the four residual blocks.
        # We discard avgpool and fc (classification head).
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1  # 64 channels,  stride 4 from input
        self.layer2 = backbone.layer2  # 128 channels, stride 8
        self.layer3 = backbone.layer3  # 256 channels, stride 16
        self.layer4 = backbone.layer4  # 512/2048 ch,  stride 32

        self.out_channels = BACKBONE_OUT_CHANNELS[cfg.backbone]

        # Freeze requested layers
        self._freeze_layers(cfg.frozen_layers)


    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from input image tensor.

        Args:
            x: Input tensor of shape B × C × H × W.

        Returns:
            Dictionary with keys "layer1" … "layer4", each mapping to
            the corresponding feature tensor:
                "layer1": B × 64  × (H/4)  × (W/4)
                "layer2": B × 128 × (H/8)  × (W/8)
                "layer3": B × 256 × (H/16) × (W/16)
                "layer4": B × 512 × (H/32) × (W/32)   [ResNet18/34]
                          B × 2048 × (H/32) × (W/32)  [ResNet50]
        """
        x = self.stem(x)        # B × 64 × H/4  × W/4

        f1 = self.layer1(x)     # B × 64  × H/4  × W/4
        f2 = self.layer2(f1)    # B × 128 × H/8  × W/8
        f3 = self.layer3(f2)    # B × 256 × H/16 × W/16
        f4 = self.layer4(f3)    # B × 512 × H/32 × W/32

        return {
            "layer1": f1,
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
        }

    def _freeze_layers(self, layer_names: list[str]) -> None:
        """Freeze parameters in the specified layers.

        Frozen parameters are excluded from gradient computation and will
        not be updated during training. Useful for fine-tuning experiments
        where we want to keep early layers fixed.

        Args:
            layer_names: List of attribute names to freeze.
                Valid values: "stem", "layer1", "layer2", "layer3", "layer4".
        """
        for name in layer_names:
            module = getattr(self, name, None)
            if module is None:
                raise ValueError(
                    f"Cannot freeze '{name}': layer not found in encoder. "
                    f"Valid layers: stem, layer1, layer2, layer3, layer4."
                )
            for param in module.parameters():
                param.requires_grad = False
