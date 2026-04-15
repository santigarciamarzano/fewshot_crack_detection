"""
tests/test_smoke.py

Smoke tests for the few-shot segmentation model.

These tests verify that:
    1. Each module instantiates without errors.
    2. Forward passes complete without errors.
    3. Output shapes match expectations.

No real data is needed — inputs are random tensors with the correct shapes.
Run with: python -m pytest tests/test_smoke.py -v
"""

import torch
import torch.nn as nn
import pytest

from config.base_config import (
    FewShotConfig,
    EncoderConfig,
    PrototypeConfig,
    SimilarityConfig,
    DecoderConfig,
)
from models.encoders.resnet_encoder import ResNetEncoder
from models.fewshot.prototype_module import PrototypeModule
from models.fewshot.similarity import SimilarityModule
from models.decoders.unet_decoder import UNetDecoder
from models.fewshot_model import FewShotModel
from experiments.baseline import get_baseline_config


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B = 2          # batch size — use >1 to catch bugs that only appear in batches
H, W = 256, 256
C_IMG = 3


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TestResNetEncoder:

    def test_resnet34_output_shapes(self):
        """layer1..layer4 shapes for ResNet34 with 256×256 input."""
        cfg = EncoderConfig(backbone="resnet34", pretrained=False, in_channels=3)
        encoder = ResNetEncoder(cfg)
        encoder.eval()

        x = torch.randn(B, C_IMG, H, W)  # B × 3 × 256 × 256

        with torch.no_grad():
            features = encoder(x)

        assert features["layer1"].shape == (B, 64,  64, 64), features["layer1"].shape
        assert features["layer2"].shape == (B, 128, 32, 32), features["layer2"].shape
        assert features["layer3"].shape == (B, 256, 16, 16), features["layer3"].shape
        assert features["layer4"].shape == (B, 512,  8,  8), features["layer4"].shape

    def test_resnet50_output_shapes(self):
        """layer4 has 2048 channels for ResNet50."""
        cfg = EncoderConfig(backbone="resnet50", pretrained=False, in_channels=3)
        encoder = ResNetEncoder(cfg)
        encoder.eval()

        x = torch.randn(B, C_IMG, H, W)  # B × 3 × 256 × 256

        with torch.no_grad():
            features = encoder(x)

        assert features["layer4"].shape == (B, 2048, 8, 8), features["layer4"].shape

    def test_out_channels_attribute(self):
        """encoder.out_channels must match actual layer4 channels."""
        for backbone, expected_ch in [("resnet34", 512), ("resnet50", 2048)]:
            cfg = EncoderConfig(backbone=backbone, pretrained=False)
            encoder = ResNetEncoder(cfg)
            assert encoder.out_channels == expected_ch

    def test_invalid_backbone_raises(self):
        """Unknown backbone name must raise ValueError immediately."""
        cfg = EncoderConfig(backbone="resnet999", pretrained=False)
        with pytest.raises(ValueError, match="resnet999"):
            ResNetEncoder(cfg)


# ---------------------------------------------------------------------------
# Prototype module
# ---------------------------------------------------------------------------

class TestPrototypeModule:

    def test_output_shapes(self):
        """Both prototypes must be B × C."""
        cfg = PrototypeConfig(normalize_features=True, eps=1e-6)
        module = PrototypeModule(cfg)

        features = torch.randn(B, 512, 8, 8)          # B × 512 × 8 × 8
        mask = torch.randint(0, 2, (B, 1, 256, 256)).float()  # B × 1 × 256 × 256

        proto_crack, proto_bg = module(features, mask)

        assert proto_crack.shape == (B, 512), proto_crack.shape
        assert proto_bg.shape    == (B, 512), proto_bg.shape

    def test_normalized_prototypes_have_unit_norm(self):
        """With normalize_features=True, prototypes must have L2 norm ≈ 1."""
        cfg = PrototypeConfig(normalize_features=True)
        module = PrototypeModule(cfg)

        features = torch.randn(B, 512, 8, 8)
        mask = torch.randint(0, 2, (B, 1, 256, 256)).float()

        proto_crack, proto_bg = module(features, mask)

        norms_crack = proto_crack.norm(dim=1)  # B
        norms_bg    = proto_bg.norm(dim=1)     # B

        assert torch.allclose(norms_crack, torch.ones(B), atol=1e-5), norms_crack
        assert torch.allclose(norms_bg,    torch.ones(B), atol=1e-5), norms_bg

    def test_empty_mask_does_not_crash(self):
        """All-zero mask (no crack pixels) must not produce NaN or inf."""
        cfg = PrototypeConfig(normalize_features=True, eps=1e-6)
        module = PrototypeModule(cfg)

        features = torch.randn(B, 512, 8, 8)
        mask = torch.zeros(B, 1, 256, 256)      # no crack pixels at all

        proto_crack, proto_bg = module(features, mask)

        assert not torch.isnan(proto_crack).any(), "NaN in proto_crack with empty mask"
        assert not torch.isnan(proto_bg).any(),    "NaN in proto_bg with empty mask"

    def test_mask_downsampled_to_feature_resolution(self):
        """Module must handle any input mask resolution, not just 256×256."""
        cfg = PrototypeConfig(normalize_features=False)
        module = PrototypeModule(cfg)

        features = torch.randn(B, 512, 8, 8)
        mask = torch.randint(0, 2, (B, 1, 512, 512)).float()  # different resolution

        proto_crack, proto_bg = module(features, mask)

        assert proto_crack.shape == (B, 512)
        assert proto_bg.shape    == (B, 512)


# ---------------------------------------------------------------------------
# Similarity module
# ---------------------------------------------------------------------------

class TestSimilarityModule:

    def test_output_shape(self):
        """sim_map must be B × 2 × H × W matching query spatial dims."""
        cfg = SimilarityConfig(temperature=1.0, normalize_query=True)
        module = SimilarityModule(cfg)

        query_features = torch.randn(B, 512, 8, 8)  # B × 512 × 8 × 8
        proto_crack    = torch.randn(B, 512)          # B × 512
        proto_bg       = torch.randn(B, 512)          # B × 512

        sim_map = module(query_features, proto_crack, proto_bg)

        assert sim_map.shape == (B, 2, 8, 8), sim_map.shape

    def test_similarity_values_in_valid_range(self):
        """With temperature=1.0, cosine similarity values must be in [-1, 1]."""
        cfg = SimilarityConfig(temperature=1.0, normalize_query=True)
        module = SimilarityModule(cfg)

        query_features = torch.randn(B, 512, 8, 8)
        proto_crack    = F_normalize(torch.randn(B, 512))
        proto_bg       = F_normalize(torch.randn(B, 512))

        sim_map = module(query_features, proto_crack, proto_bg)

        assert sim_map.min() >= -1.0 - 1e-5, sim_map.min()
        assert sim_map.max() <=  1.0 + 1e-5, sim_map.max()

    def test_temperature_scales_output(self):
        """Output with temperature=2.0 must be exactly 2× the temperature=1.0 output."""
        query_features = torch.randn(B, 512, 8, 8)
        proto_crack    = torch.randn(B, 512)
        proto_bg       = torch.randn(B, 512)

        module_t1 = SimilarityModule(SimilarityConfig(temperature=1.0))
        module_t2 = SimilarityModule(SimilarityConfig(temperature=2.0))

        sim_t1 = module_t1(query_features, proto_crack, proto_bg)
        sim_t2 = module_t2(query_features, proto_crack, proto_bg)

        assert torch.allclose(sim_t2, sim_t1 * 2.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TestUNetDecoder:

    def test_output_shape_resnet34(self):
        """Decoder output must be B × 1 × 256 × 256 for ResNet34 bottleneck."""
        cfg = DecoderConfig(decoder_channels=[256, 128, 64, 32], dropout_rate=0.0)
        decoder = UNetDecoder(
            cfg=cfg,
            bottleneck_channels=514,       # 512 + 2
            skip_channels=[256, 128, 64],  # layer3, layer2, layer1
        )
        decoder.eval()

        bottleneck = torch.randn(B, 514, 8, 8)   # B × 514 × 8  × 8
        skips = {
            "layer3": torch.randn(B, 256, 16, 16),
            "layer2": torch.randn(B, 128, 32, 32),
            "layer1": torch.randn(B, 64,  64, 64),
        }

        with torch.no_grad():
            logits = decoder(bottleneck, skips)

        assert logits.shape == (B, 1, 256, 256), logits.shape

    def test_output_shape_resnet50(self):
        """Decoder output must be B × 1 × 256 × 256 for ResNet50 bottleneck."""
        cfg = DecoderConfig(decoder_channels=[256, 128, 64, 32], dropout_rate=0.0)
        decoder = UNetDecoder(
            cfg=cfg,
            bottleneck_channels=2050,          # 2048 + 2
            skip_channels=[1024, 512, 256],    # layer3, layer2, layer1 for ResNet50
        )
        decoder.eval()

        bottleneck = torch.randn(B, 2050, 8, 8)
        skips = {
            "layer3": torch.randn(B, 1024, 16, 16),
            "layer2": torch.randn(B, 512,  32, 32),
            "layer1": torch.randn(B, 256,  64, 64),
        }

        with torch.no_grad():
            logits = decoder(bottleneck, skips)

        assert logits.shape == (B, 1, 256, 256), logits.shape

    def test_invalid_decoder_channels_raises(self):
        """decoder_channels with wrong length must raise ValueError."""
        cfg = DecoderConfig(decoder_channels=[256, 128, 64])  # only 3 elements
        with pytest.raises(ValueError, match="4 elements"):
            UNetDecoder(cfg=cfg, bottleneck_channels=514, skip_channels=[256, 128, 64])

    def test_logits_have_no_activation(self):
        """Output logits must not be constrained to [0, 1] — no sigmoid applied.

        Checks the decoder's structure directly instead of relying on output
        values, which are unstable due to BatchNorm dampening amplified inputs.
        """
        cfg = DecoderConfig(decoder_channels=[256, 128, 64, 32], dropout_rate=0.0)
        decoder = UNetDecoder(cfg=cfg, bottleneck_channels=514, skip_channels=[256, 128, 64])

        # final_conv must be a plain Conv2d with no activation attached.
        assert isinstance(decoder.final_conv, nn.Conv2d), (
            f"final_conv must be nn.Conv2d, got {type(decoder.final_conv)}"
        )

        # No Sigmoid anywhere in the decoder — not in final_conv, not in any block.
        sigmoid_modules = [
            name for name, mod in decoder.named_modules()
            if isinstance(mod, (nn.Sigmoid, nn.Hardsigmoid))
        ]
        assert len(sigmoid_modules) == 0, (
            f"Decoder must not contain sigmoid activations, found: {sigmoid_modules}"
        )


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TestFewShotModel:

    def test_baseline_forward_pass(self):
        """Full forward pass with baseline config must produce correct output shape."""
        cfg = get_baseline_config()
        cfg.encoder.pretrained = False  # avoid downloading weights in CI
        model = FewShotModel(cfg)
        model.eval()

        support_img  = torch.randn(B, 3, H, W)                        # B × 3 × 256 × 256
        support_mask = torch.randint(0, 2, (B, 1, H, W)).float()      # B × 1 × 256 × 256
        query_img    = torch.randn(B, 3, H, W)                        # B × 3 × 256 × 256

        with torch.no_grad():
            logits = model(support_img, support_mask, query_img)

        assert logits.shape == (B, 1, H, W), logits.shape

    def test_resnet50_forward_pass(self):
        """Swapping backbone to ResNet50 must work without touching any other config."""
        cfg = get_baseline_config()
        cfg.encoder.backbone   = "resnet50"
        cfg.encoder.pretrained = False
        model = FewShotModel(cfg)
        model.eval()

        support_img  = torch.randn(B, 3, H, W)
        support_mask = torch.randint(0, 2, (B, 1, H, W)).float()
        query_img    = torch.randn(B, 3, H, W)

        with torch.no_grad():
            logits = model(support_img, support_mask, query_img)

        assert logits.shape == (B, 1, H, W), logits.shape

    def test_output_has_no_nan_or_inf(self):
        """Forward pass must not produce NaN or inf values."""
        cfg = get_baseline_config()
        cfg.encoder.pretrained = False
        model = FewShotModel(cfg)
        model.eval()

        support_img  = torch.randn(B, 3, H, W)
        support_mask = torch.randint(0, 2, (B, 1, H, W)).float()
        query_img    = torch.randn(B, 3, H, W)

        with torch.no_grad():
            logits = model(support_img, support_mask, query_img)

        assert not torch.isnan(logits).any(), "NaN in model output"
        assert not torch.isinf(logits).any(), "Inf in model output"

    def test_support_and_query_use_same_encoder_weights(self):
        """Encoder must be Siamese — same weights for both branches."""
        cfg = get_baseline_config()
        cfg.encoder.pretrained = False
        model = FewShotModel(cfg)

        # The model has a single encoder attribute — not two separate ones.
        # If it were not Siamese, there would be encoder_support and encoder_query.
        assert hasattr(model, "encoder"), "Model must have a single shared encoder"
        assert not hasattr(model, "encoder_support"), "Encoder must not be duplicated"
        assert not hasattr(model, "encoder_query"),   "Encoder must not be duplicated"


# ---------------------------------------------------------------------------
# Helper — local F.normalize to avoid importing torch.nn.functional in tests
# ---------------------------------------------------------------------------

def F_normalize(x: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    return F.normalize(x, p=2, dim=1)
