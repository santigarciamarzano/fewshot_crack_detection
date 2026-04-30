"""
models/encoders/base_encoder.py

Clase base abstracta para todos los encoders backbone.

Define el contrato que cada encoder debe cumplir para que el resto del sistema
(PrototypeModule, SimilarityModule, UNetDecoder, FewShotModel) funcione
con cualquier backbone sin modificaciones.

Contrato:
    1. forward(x) debe devolver un dict con claves "layer1".."layer4",
       cada una mapeando al tensor de features de esa escala.
    2. out_channels debe exponer el número de canales en "layer4".
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Clase base abstracta para todos los encoders backbone.

    Las subclases deben implementar:
        - forward():     extrae features multi-escala desde un tensor de entrada.
        - out_channels:  número de canales en el mapa de features más profundo (layer4).
        - skip_channels: canales de las skip connections, ordenados layer3 → layer2 → layer1.

    Formato de salida esperado de forward():
        {
            "layer1": B × C1 × (H/4)  × (W/4),
            "layer2": B × C2 × (H/8)  × (W/8),
            "layer3": B × C3 × (H/16) × (W/16),
            "layer4": B × C4 × (H/32) × (W/32),
        }
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Canales en el mapa de features layer4 (más profundo).

        Usado por FewShotModel para calcular el tamaño del bottleneck:
            bottleneck_channels = encoder.out_channels + 2  (+ mapa de similitud)
        """
        ...

    @property
    @abstractmethod
    def skip_channels(self) -> List[int]:
        """Channels for skip connections, ordered layer3 → layer2 → layer1."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from an input image tensor.

        Args:
            x: Input tensor of shape B × C × H × W.
               C is typically 3 for RGB or preprocessed radiographic images.

        Returns:
            Dictionary with exactly four keys:
                "layer1": B × C1 × (H/4)  × (W/4)
                "layer2": B × C2 × (H/8)  × (W/8)
                "layer3": B × C3 × (H/16) × (W/16)
                "layer4": B × C4 × (H/32) × (W/32)
        """
        ...