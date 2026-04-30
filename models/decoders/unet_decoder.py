"""
models/decoders/unet_decoder.py

Decoder estilo U-Net para segmentación few-shot.

Recibe el bottleneck concatenado (query layer4 + mapa de similitud) y
reconstruye la máscara a resolución completa mediante upsampling progresivo,
incorporando skip connections del encoder de la query en cada escala.

Shapes (baseline: ResNet34, entrada 256×256):
    Entrada:         B × 514 × 8   × 8    (512 de layer4 + 2 de sim_map)
    Tras stage 1:    B × 256 × 16  × 16   (+ skip de layer3: 256ch)
    Tras stage 2:    B × 128 × 32  × 32   (+ skip de layer2: 128ch)
    Tras stage 3:    B × 64  × 64  × 64   (+ skip de layer1: 64ch)
    Tras stage 4:    B × 32  × 128 × 128  (sin skip)
    Salida:          B × 1   × 256 × 256  (logits, sin activación)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from config.base_config import DecoderConfig


class DecoderBlock(nn.Module):
    """Un stage del decoder: upsample → concat skip opcional → bloque conv.

    Args:
        in_channels:   Canales de entrada (antes del concat con skip).
        out_channels:  Canales de salida tras el bloque conv.
        skip_channels: Canales de la skip connection. 0 si no hay skip.
        dropout_rate:  Probabilidad de dropout tras el bloque conv.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)  # bilinear evita artefactos de checkerboard

        conv_in = in_channels + skip_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Upsample, concatena skip si existe y aplica el bloque conv."""
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return self.dropout(x)


class UNetDecoder(nn.Module):
    """Decoder U-Net que reconstruye la máscara de segmentación desde el bottleneck.

    Cuatro DecoderBlocks: stages 1-3 usan skips de layer3, layer2, layer1;
    stage 4 no tiene skip. Una conv 1×1 final produce el canal de salida (logits).

    Args:
        cfg:                 DecoderConfig con canales y dropout.
        bottleneck_channels: Canales entrantes al decoder (layer4_ch + 2).
        skip_channels:       Lista de canales de skip, de más profundo a más superficial:
                             [layer3, layer2, layer1]. Para ResNet34: [256, 128, 64].
    """

    def __init__(
        self,
        cfg: DecoderConfig,
        bottleneck_channels: int,
        skip_channels: List[int],
    ) -> None:
        super().__init__()

        if len(cfg.decoder_channels) != 4:
            raise ValueError(f"decoder_channels must have exactly 4 elements, got {len(cfg.decoder_channels)}.")
        if len(skip_channels) != 3:
            raise ValueError(f"skip_channels must have exactly 3 elements (layer3, layer2, layer1), got {len(skip_channels)}.")

        d  = cfg.decoder_channels   # [256, 128, 64, 32]
        dr = cfg.dropout_rate

        self.stage1 = DecoderBlock(bottleneck_channels, d[0], skip_channels[0], dr)  # 8×8   → 16×16,  skip layer3
        self.stage2 = DecoderBlock(d[0],                d[1], skip_channels[1], dr)  # 16×16 → 32×32,  skip layer2
        self.stage3 = DecoderBlock(d[1],                d[2], skip_channels[2], dr)  # 32×32 → 64×64,  skip layer1
        self.stage4 = DecoderBlock(d[2],                d[3], skip_channels=0,  dropout_rate=dr)  # 64×64 → 128×128

        # Upsample final + conv 1×1 a un único canal (logits, sin activación)
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(d[3], 1, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruye la máscara desde el bottleneck y las skip connections.
        Devuelve logits B × 1 × 256 × 256.
        """
        x = self.stage1(bottleneck, skips["layer3"])  # B × 256 × 16  × 16
        x = self.stage2(x,          skips["layer2"])  # B × 128 × 32  × 32
        x = self.stage3(x,          skips["layer1"])  # B × 64  × 64  × 64
        x = self.stage4(x,          skip=None)        # B × 32  × 128 × 128
        x = self.final_upsample(x)                    # B × 32  × 256 × 256
        x = self.final_conv(x)                        # B × 1   × 256 × 256
        return x
