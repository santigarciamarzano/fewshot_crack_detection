"""
config/base_config.py

Central configuration dataclasses for the Few-Shot Segmentation framework.

Each module of the system has its own config dataclass.
All configs compose into a single FewShotConfig root object.

Usage:
    from config.base_config import FewShotConfig
    cfg = FewShotConfig()
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """Configuration for the backbone encoder.

    Attributes:
        backbone: ResNet variant to use as backbone.
            Options: "resnet18", "resnet34", "resnet50".
        pretrained: Whether to load ImageNet pretrained weights.
        in_channels: Number of input channels. 3 for our preprocessed
            radiographic images (normalized, edge, high-freq).
        frozen_layers: List of layer names to freeze during training.
            Empty list means all layers are trainable.
            Example: ["layer1", "layer2"] to freeze early layers.
    """
    backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101"] = "resnet34"
    pretrained: bool = True
    in_channels: int = 3
    frozen_layers: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prototype
# ---------------------------------------------------------------------------

@dataclass
class PrototypeConfig:
    """Configuration for the prototype computation module.

    Attributes:
        normalize_features: Whether to L2-normalize features before
            computing prototypes. Strongly recommended for cosine similarity.
        eps: Small epsilon to avoid division by zero in masked average pooling.
            Used when a support mask is all-zero (no crack pixels).
    """
    normalize_features: bool = True
    eps: float = 1e-6


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

@dataclass
class SimilarityConfig:
    """Configuration for the similarity computation module.

    Attributes:
        temperature: Scaling factor applied to cosine similarities before
            concatenation. Higher values sharpen the similarity map.
        normalize_query: Whether to L2-normalize query features before
            similarity computation.
    """
    temperature: float = 1.0
    normalize_query: bool = True


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    """Configuration for the U-Net style decoder.

    Attributes:
        use_skip_connections: Whether to use skip connections from the
            query encoder. Disabling this is useful for ablation studies.
        decoder_channels: Number of channels at each decoder stage.
            Length must match the number of upsampling stages (4).
        dropout_rate: Dropout probability applied after each decoder block.
            Set to 0.0 to disable.
    """
    use_skip_connections: bool = True
    decoder_channels: List[int] = field(
        default_factory=lambda: [256, 128, 64, 32]
    )
    dropout_rate: float = 0.1


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

@dataclass
class LossConfig:
    """Configuration for the combined Dice + BCE loss.

    Attributes:
        dice_weight: Relative weight of the Dice loss component.
        bce_weight: Relative weight of the Binary Cross-Entropy loss component.
        dice_smooth: Smoothing factor for the Dice numerator/denominator
            to avoid zero-division on empty masks.
    """
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    dice_smooth: float = 1.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for the episodic dataset.

    Attributes:
        data_root: Path to the root directory containing images and masks.
        image_size: Spatial size (H, W) to resize images and masks.
        n_way: Number of classes per episode.
            Fixed to 1 for our binary (crack / no-crack) setting.
        k_shot: Number of support images per episode.
        n_query: Number of query images per episode.
        augment_support: Whether to apply augmentation to support images.
        augment_query: Whether to apply augmentation to query images.
    """
    data_root: str = "data/"
    image_size: Tuple[int, int] = (256, 256)
    n_way: int = 1
    k_shot: int = 1
    n_query: int = 1
    augment_support: bool = True
    augment_query: bool = True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Configuration for the training loop.

    Attributes:
        epochs: Total number of training epochs.
        episodes_per_epoch: Number of episodes sampled per epoch.
        batch_size: Number of episodes per gradient update.
        learning_rate: Initial learning rate for Adam optimizer.
        weight_decay: L2 regularization weight.
        lr_scheduler: Learning rate scheduler type.
            Options: "cosine", "step", "none".
        grad_clip: Max norm for gradient clipping. None to disable.
        device: Compute device. "cuda" or "cpu".
        seed: Random seed for reproducibility.
        checkpoint_dir: Directory to save model checkpoints.
        log_every_n_episodes: Logging frequency (in episodes).
    """
    epochs: int = 100
    episodes_per_epoch: int = 200
    batch_size: int = 4
    optimizer: Literal["adam", "adamw"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: Literal["cosine", "step", "none"] = "cosine"
    grad_clip: Optional[float] = 1.0
    device: Literal["cuda", "cpu"] = "cuda"
    seed: int = 42
    checkpoint_dir: str = "checkpoints/"
    log_every_n_episodes: int = 50


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class FewShotConfig:
    """Root configuration object for the entire framework.

    All module configs are composed here.
    Pass this object around instead of individual sub-configs.

    Example:
        cfg = FewShotConfig()
        cfg.encoder.backbone = "resnet50"
        cfg.training.epochs = 200
    """
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = "baseline"
    notes: str = ""
