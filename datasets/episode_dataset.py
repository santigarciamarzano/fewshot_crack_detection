"""
datasets/episode_dataset.py

Episodic dataset for few-shot segmentation.

Each item returned by this dataset is a complete self-contained episode:
    - K support images + masks  ← "this is what a crack looks like"
    - 1 query image  + mask     ← "segment this image"

Support and query are always distinct images sampled without replacement.
The query mask is used externally for loss computation — the model never
sees it during the forward pass.

Expected directory structure:
    data/
    ├── train/
    │   ├── images/   crack_001.tiff, crack_002.tiff, ...
    │   └── masks/    crack_001.tiff, crack_002.tiff, ...
    └── val/
        ├── images/
        └── masks/

Image format:
    Images:  TIFF 16-bit, 3 channels (normalized + edge + high-freq)
    Masks:   TIFF 16-bit, 1 channel, binary {0, 65535}

Both are normalized to float32 in [0, 1] before returning.

Shapes returned per episode:
    support_imgs:   K × 3 × H × W
    support_masks:  K × 1 × H × W
    query_img:      3 × H × W
    query_mask:     1 × H × W
"""

import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tifffile

from config.base_config import DatasetConfig


class EpisodicDataset(Dataset):
    """Dataset episódico para segmentación few-shot 1-way K-shot.

    Cada llamada a __getitem__ muestrea un episodio fresco: K imágenes de support
    y 1 de query, siempre distintas. El augmentation se aplica de forma independiente
    a support y query.

    Args:
        cfg:   DatasetConfig con data_root, image_size, k_shot, augment_support, augment_query.
        split: "train" o "val".
    """

    # Normalization divisors — use _NORM_16BIT for current TIFF pipeline.
    # Switch to _NORM_8BIT if preprocessing outputs 8-bit images instead.
    _NORM_16BIT: float = 65535.0
    _NORM_8BIT:  float = 255.0

    def __init__(self, cfg: DatasetConfig, split: str) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got '{split}'.")

        self.cfg   = cfg
        self.split = split

        root          = Path(cfg.data_root) / split
        self.img_dir  = root / "images"
        self.mask_dir = root / "masks"

        self._validate_directories()

        # Escaneamos el directorio una sola vez al inicio; ningún acceso a disco durante el entrenamiento
        self.samples: List[Tuple[Path, Path]] = self._build_index()

        if len(self.samples) < cfg.k_shot + 1:
            raise ValueError(
                f"Dataset at '{root}' has {len(self.samples)} samples, "
                f"but k_shot={cfg.k_shot} requires at least {cfg.k_shot + 1}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Muestrea un episodio completo con idx como query.

        La query es siempre la imagen en la posición idx.
        Las imágenes de support se muestrean aleatoriamente del resto.

        Returns:
            Tuple (support_imgs, support_masks, query_img, query_mask):
                support_imgs:  K × 3 × H × W  — float32, [0, 1]
                support_masks: K × 1 × H × W  — float32, {0.0, 1.0}
                query_img:     3 × H × W       — float32, [0, 1]
                query_mask:    1 × H × W       — float32, {0.0, 1.0}
        """
        all_indices   = list(range(len(self.samples)))
        support_pool  = [i for i in all_indices if i != idx]
        support_idxs  = random.sample(support_pool, self.cfg.k_shot)

        query_img, query_mask = self._load_sample(idx, augment=self.cfg.augment_query)

        support_imgs  = []
        support_masks = []
        for s_idx in support_idxs:
            s_img, s_mask = self._load_sample(s_idx, augment=self.cfg.augment_support)
            support_imgs.append(s_img)
            support_masks.append(s_mask)

        support_imgs  = torch.stack(support_imgs,  dim=0)  # K × 3 × H × W
        support_masks = torch.stack(support_masks, dim=0)  # K × 1 × H × W

        return support_imgs, support_masks, query_img, query_mask

    def _load_sample(
        self,
        idx: int,
        augment: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Carga, redimensiona y opcionalmente augmenta un par imagen+máscara."""
        img_path, mask_path = self.samples[idx]

        img  = tifffile.imread(str(img_path))   # H × W × 3, uint16
        mask = tifffile.imread(str(mask_path))  # H × W,     uint16

        img_tensor  = self._to_tensor_img(img)
        mask_tensor = self._to_tensor_mask(mask)

        h, w = self.cfg.image_size
        img_tensor  = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0),
            size=(h, w),
            mode="nearest",  # nearest preserva valores binarios
        ).squeeze(0)

        if augment:
            img_tensor, mask_tensor = self._augment(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

    def _to_tensor_img(self, img_np) -> torch.Tensor:
        """Convert uint16 numpy array H×W×C to float32 tensor C×H×W in [0,1].

        Args:
            img_np: numpy array of shape H × W × 3, dtype uint16.

        Returns:
            Float32 tensor of shape 3 × H × W, values in [0, 1].
        """
        # H × W × 3  →  3 × H × W
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return tensor / self._NORM_16BIT

    def _to_tensor_mask(self, mask_np) -> torch.Tensor:
        """Convert uint16 numpy array H×W to binary float32 tensor 1×H×W.

        Input values are {0, 65535}. Output values are {0.0, 1.0}.

        Args:
            mask_np: numpy array of shape H × W, dtype uint16.

        Returns:
            Float32 tensor of shape 1 × H × W, values in {0.0, 1.0}.
        """
        # H × W  →  1 × H × W
        tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        return tensor / self._NORM_16BIT

    def _augment(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aplica augmentation espacial aleatoria de forma conjunta a imagen y máscara."""
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(std_range=(0.008, 0.02), p=0.4),
        ])

        img_np  = img.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy()

        transformed = transform(image=img_np, mask=mask_np)

        img_tensor  = torch.from_numpy(transformed['image']).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(transformed['mask']).permute(2, 0, 1)

        return img_tensor, mask_tensor

    def _build_index(self) -> List[Tuple[Path, Path]]:
        """Scan image directory and pair each image with its mask.

        Images and masks are matched by filename — both directories must
        contain files with identical names.

        Returns:
            Sorted list of (image_path, mask_path) tuples.

        Raises:
            FileNotFoundError: If a mask file is missing for any image.
        """
        img_paths = sorted(self.img_dir.glob("*.tiff"))

        if len(img_paths) == 0:
            raise FileNotFoundError(
                f"No .tiff files found in '{self.img_dir}'."
            )

        samples = []
        for img_path in img_paths:
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Mask not found for image '{img_path.name}'. "
                    f"Expected at '{mask_path}'."
                )
            samples.append((img_path, mask_path))

        return samples

    def _validate_directories(self) -> None:
        """Raise clear errors if required directories are missing."""
        for path in (self.img_dir, self.mask_dir):
            if not path.exists():
                raise FileNotFoundError(
                    f"Required directory not found: '{path}'. "
                    f"Expected structure: {self.cfg.data_root}/{self.split}/images/ "
                    f"and {self.cfg.data_root}/{self.split}/masks/"
                )
