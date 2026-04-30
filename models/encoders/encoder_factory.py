"""
models/encoders/encoder_factory.py

Función factory para construir el encoder a partir de la config.

Es el único punto de contacto entre FewShotModel y las implementaciones de encoder.
FewShotModel llama a build_encoder(cfg) y recibe un BaseEncoder, sin importar
directamente ResNetEncoder ni SwinEncoder.

Para añadir un nuevo backbone:
    - Si es un ResNet de torchvision: añadirlo a ResNetEncoder._BACKBONE_REGISTRY.
    - Si es un modelo timm con features_only: configurar el nombre en config,
      SwinEncoder lo maneja automáticamente.
    - Si necesita una clase nueva: crearla, heredar de BaseEncoder,
      y añadir un branch en build_encoder().
"""

from config.base_config import EncoderConfig
from models.encoders.base_encoder import BaseEncoder
from models.encoders.resnet_encoder import ResNetEncoder, _BACKBONE_REGISTRY as _RESNET_REGISTRY
from models.encoders.swin_encoder import SwinEncoder


def build_encoder(cfg: EncoderConfig) -> BaseEncoder:
    """Instancia el encoder correcto para la config dada.

    - backbone en el registry de ResNet → ResNetEncoder (torchvision)
    - cualquier otra cosa              → SwinEncoder (timm)
    """
    if cfg.backbone in _RESNET_REGISTRY:
        return ResNetEncoder(cfg)

    # Todo lo demás pasa por timm. SwinEncoder soporta cualquier backbone con features_only.
    return SwinEncoder(cfg)