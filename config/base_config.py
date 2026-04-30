"""
config/base_config.py

Dataclasses de configuración para el framework de segmentación few-shot.

Cada módulo del sistema tiene su propio dataclass de config.
Todos componen en un único objeto raíz FewShotConfig.

Uso:
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
    """Configuración del encoder backbone.

    Atributos:
        backbone:       Variante de ResNet. Opciones: resnet18, resnet34, resnet50, resnet101.
        pretrained:     Si cargar pesos preentrenados en ImageNet.
        in_channels:    Número de canales de entrada. 3 para nuestras imágenes.
        frozen_layers:  Capas a congelar durante el entrenamiento. Lista vacía = todas entrenables.
    """
    backbone: Literal["resnet18", "resnet34", "resnet50", "resnet101"] = "resnet34"
    pretrained: bool = True
    in_channels: int = 3
    frozen_layers: List[str] = field(default_factory=list)
    img_size: int = 256  # solo usado para Swin, ignorado en ResNet


# ---------------------------------------------------------------------------
# Prototype
# ---------------------------------------------------------------------------

@dataclass
class PrototypeConfig:
    """Configuración del módulo de cálculo de prototipos.

    Atributos:
        normalize_features: Si normalizar features con L2 antes de calcular prototipos.
        eps:                Epsilon para evitar división por cero en masked average pooling.
    """
    normalize_features: bool = True
    eps: float = 1e-6


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

@dataclass
class SimilarityConfig:
    """Configuración del módulo de similitud.

    Atributos:
        temperature:     Factor de escala aplicado a las similitudes coseno.
        normalize_query: Si normalizar las features de la query con L2 antes de la similitud.
    """
    temperature: float = 1.0
    normalize_query: bool = True


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

@dataclass
class DecoderConfig:
    """Configuración del decoder estilo U-Net.

    Atributos:
        use_skip_connections: Si usar skip connections del encoder de la query.
        decoder_channels:     Canales en cada stage del decoder. Longitud fija: 4.
        dropout_rate:         Probabilidad de dropout. 0.0 para desactivar.
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
    """Configuración de la loss combinada Dice + BCE.

    Atributos:
        dice_weight: Peso relativo del componente Dice.
        bce_weight:  Peso relativo del componente BCE.
        dice_smooth: Factor de suavizado para evitar división por cero en máscaras vacías.
    """
    dice_weight: float = 1.0
    bce_weight: float = 1.0
    dice_smooth: float = 1.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuración del dataset episódico.

    Atributos:
        data_root:        Ruta al directorio raíz con train/ y val/.
        image_size:       Tamaño espacial (H, W) al que se redimensionan imágenes y máscaras.
        n_way:            Número de clases por episodio. Fijo a 1 (binario: grieta / fondo).
        k_shot:           Número de imágenes de support por episodio.
        n_query:          Número de imágenes de query por episodio.
        augment_support:  Si aplicar augmentation a las imágenes de support.
        augment_query:    Si aplicar augmentation a las imágenes de query.
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
    """Configuración del loop de entrenamiento.

    Atributos:
        epochs:                  Total de épocas de entrenamiento.
        episodes_per_epoch:      Episodios por época.
        batch_size:              Episodios por actualización de gradiente.
        learning_rate:           Learning rate inicial.
        weight_decay:            Regularización L2.
        lr_scheduler:            Tipo de scheduler. Opciones: cosine, step, none.
        grad_clip:               Norma máxima para gradient clipping. None para desactivar.
        device:                  Dispositivo de cómputo: cuda o cpu.
        seed:                    Semilla aleatoria para reproducibilidad.
        checkpoint_dir:          Directorio donde se guardan los checkpoints.
        log_every_n_episodes:    Frecuencia de logging (en episodios).
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
    """Objeto raíz de configuración para todo el framework.

    Todos los sub-configs se componen aquí.
    Pasar este objeto en vez de sub-configs individuales.

    Ejemplo:
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

    experiment_name: str = "baseline"
    notes: str = ""
