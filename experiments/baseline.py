"""
experiments/baseline.py

Configuration for the baseline experiment.

This is the reference experiment: ResNet50 encoder, 1-shot, 512x512.
All future experiments should derive from or compare against this config.
"""

from config.base_config import (
    FewShotConfig,
    EncoderConfig,
    PrototypeConfig,
    SimilarityConfig,
    DecoderConfig,
    LossConfig,
    DatasetConfig,
    TrainingConfig,
)


def get_baseline_config() -> FewShotConfig:
    """Return the baseline experiment configuration.

    Baseline settings:
        - Backbone: ResNet50, ImageNet pretrained
        - 1-shot episodic training
        - Image size: 512 × 512
        - Loss: Dice + BCE (equal weights)
        - Optimizer: Adam, lr=1e-4, cosine annealing

    Returns:
        FewShotConfig: Fully configured root config object.
    """
    return FewShotConfig(
        experiment_name="baseline_resnet50_1shot",
        notes="Reference experiment. ResNet50 backbone, 1-shot, 512x512.",

        encoder=EncoderConfig(
            backbone="resnet50",
            pretrained=True,
            in_channels=3,
            frozen_layers=[],
        ),

        prototype=PrototypeConfig(
            normalize_features=True,
            eps=1e-6,
        ),

        similarity=SimilarityConfig(
            temperature=1.0,
            normalize_query=True,
        ),

        decoder=DecoderConfig(
            use_skip_connections=True,
            decoder_channels=[256, 128, 64, 32],
            dropout_rate=0.15,
        ),

        loss=LossConfig(
            dice_weight=1.0,
            bce_weight=1.0,
            dice_smooth=1.0,
        ),

        dataset=DatasetConfig(
            data_root="data/",
            image_size=(512, 512),
            n_way=1,
            k_shot=1,
            n_query=1,
            augment_support=True,
            augment_query=True,
        ),

        training=TrainingConfig(
            epochs=100,
            episodes_per_epoch=200,
            batch_size=4,
            optimizer="adamw",
            learning_rate=1e-4,
            weight_decay=2e-4,
            lr_scheduler="cosine",
            grad_clip=1.0,
            device="cuda",
            seed=42,
            checkpoint_dir="checkpoints/baseline/",
            log_every_n_episodes=50,
        ),
    )

