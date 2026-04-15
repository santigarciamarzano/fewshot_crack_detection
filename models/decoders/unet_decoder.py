"""
models/decoders/unet_decoder.py

U-Net style decoder for few-shot segmentation.

Takes the concatenated bottleneck (query layer4 + similarity map) and
reconstructs a full-resolution segmentation mask through progressive
upsampling, incorporating skip connections from the query encoder at
each spatial scale.

The decoder has no access to support features — all few-shot information
enters through the similarity map concatenated at the bottleneck.

Shapes (baseline: ResNet34, 256×256 input):
    Input:          B × 514 × 8   × 8    (512 from layer4 + 2 from sim_map)
    After stage 1:  B × 256 × 16  × 16   (+ skip from layer3: 256ch)
    After stage 2:  B × 128 × 32  × 32   (+ skip from layer2: 128ch)
    After stage 3:  B × 64  × 64  × 64   (+ skip from layer1: 64ch)
    After stage 4:  B × 32  × 128 × 128  (no skip)
    Output:         B × 1   × 256 × 256  (logits, no activation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from config.base_config import DecoderConfig


class DecoderBlock(nn.Module):
    """Single decoder stage: upsample → optional skip concat → conv block.

    Used at every stage of the decoder. The skip connection is optional
    because the last upsampling stage (128→256) has no corresponding
    encoder feature map.

    Args:
        in_channels:  Number of input channels (after concat with skip).
        out_channels: Number of output channels after the conv block.
        skip_channels: Channels from the skip connection. 0 if no skip.
        dropout_rate: Dropout probability after the conv block.

    Example (stage 1):
        block = DecoderBlock(in_channels=514, out_channels=256, skip_channels=256)
        # x:    B × 514 × 8  × 8
        # skip: B × 256 × 16 × 16
        # out:  B × 256 × 16 × 16
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # Bilinear upsample avoids checkerboard artifacts from ConvTranspose2d.
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # After upsample + skip concat, input to conv is in_channels + skip_channels.
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
        """Upsample, concatenate skip, and apply conv block.

        Args:
            x:    Input feature map — B × C_in × H × W.
            skip: Skip connection from query encoder — B × C_skip × H*2 × W*2.
                  None if this stage has no skip connection.

        Returns:
            B × C_out × H*2 × W*2
        """
        # x: B × C_in × H × W  →  B × C_in × H*2 × W*2
        x = self.upsample(x)

        if skip is not None:
            # x: B × (C_in + C_skip) × H*2 × W*2
            x = torch.cat([x, skip], dim=1)

        # x: B × C_out × H*2 × W*2
        x = self.conv_block(x)
        x = self.dropout(x)

        return x


class UNetDecoder(nn.Module):
    """U-Net decoder that reconstructs a segmentation mask from the bottleneck.

    Composes four DecoderBlocks with skip connections (stages 1-3 use skips
    from query encoder layer3, layer2, layer1; stage 4 has no skip) plus a
    final 1×1 conv that produces the single-channel output logits.

    The input channels at the bottleneck depend on the encoder backbone and
    the similarity map. For ResNet34: 512 (layer4) + 2 (sim_map) = 514.

    Args:
        cfg:              DecoderConfig controlling channels and dropout.
        bottleneck_channels: Channels entering the decoder (layer4_ch + 2).
        skip_channels:    List of skip connection channel counts, ordered
                          from deepest to shallowest: [layer3, layer2, layer1].
                          For ResNet34: [256, 128, 64].

    Example:
        cfg = DecoderConfig(decoder_channels=[256, 128, 64, 32], dropout_rate=0.1)
        decoder = UNetDecoder(cfg, bottleneck_channels=514, skip_channels=[256, 128, 64])

        bottleneck = torch.cat([query_layer4, sim_map], dim=1)  # B × 514 × 8 × 8
        skips = {
            "layer3": query_features["layer3"],  # B × 256 × 16 × 16
            "layer2": query_features["layer2"],  # B × 128 × 32 × 32
            "layer1": query_features["layer1"],  # B × 64  × 64 × 64
        }
        mask_logits = decoder(bottleneck, skips)  # B × 1 × 256 × 256
    """

    def __init__(
        self,
        cfg: DecoderConfig,
        bottleneck_channels: int,
        skip_channels: List[int],
    ) -> None:
        super().__init__()

        if len(cfg.decoder_channels) != 4:
            raise ValueError(
                f"decoder_channels must have exactly 4 elements, "
                f"got {len(cfg.decoder_channels)}."
            )
        if len(skip_channels) != 3:
            raise ValueError(
                f"skip_channels must have exactly 3 elements "
                f"(layer3, layer2, layer1), got {len(skip_channels)}."
            )

        d = cfg.decoder_channels          # [256, 128, 64, 32]
        dr = cfg.dropout_rate

        # Stage 1: 8×8   → 16×16  — skip from layer3
        # Stage 2: 16×16 → 32×32  — skip from layer2
        # Stage 3: 32×32 → 64×64  — skip from layer1
        # Stage 4: 64×64 → 128×128 — no skip
        self.stage1 = DecoderBlock(bottleneck_channels, d[0], skip_channels[0], dr)
        self.stage2 = DecoderBlock(d[0],                d[1], skip_channels[1], dr)
        self.stage3 = DecoderBlock(d[1],                d[2], skip_channels[2], dr)
        self.stage4 = DecoderBlock(d[2],                d[3], skip_channels=0,  dropout_rate=dr)

        # Final upsample 128×128 → 256×256 + 1×1 conv to single-channel logits.
        # No activation — loss (BCEWithLogitsLoss) applies sigmoid internally.
        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(d[3], 1, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Reconstruct segmentation mask from bottleneck and skip connections.

        Args:
            bottleneck: Concatenated query layer4 + sim_map — B × 514 × 8 × 8.
            skips:      Dict with keys "layer1", "layer2", "layer3", each
                        mapping to the corresponding query feature map.

        Returns:
            Segmentation logits of shape B × 1 × 256 × 256.
            Apply sigmoid externally for probabilities.
        """
        # bottleneck: B × 514 × 8 × 8
        x = self.stage1(bottleneck, skips["layer3"])  # B × 256 × 16  × 16
        x = self.stage2(x,          skips["layer2"])  # B × 128 × 32  × 32
        x = self.stage3(x,          skips["layer1"])  # B × 64  × 64  × 64
        x = self.stage4(x,          skip=None)        # B × 32  × 128 × 128

        x = self.final_upsample(x)                    # B × 32  × 256 × 256
        x = self.final_conv(x)                        # B × 1   × 256 × 256

        return x
