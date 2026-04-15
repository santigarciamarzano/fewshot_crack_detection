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
    """Episodic dataset for 1-way K-shot few-shot segmentation.

    Each __getitem__ call samples a fresh episode: K support images and
    1 query image, all distinct. Augmentation is applied independently
    to support and query to avoid leaking transformation information.

    Args:
        cfg:   DatasetConfig with data_root, image_size, k_shot,
               augment_support, augment_query fields.
        split: "train" or "val" — selects the subdirectory to load from.

    Example:
        cfg = DatasetConfig(data_root="data/", k_shot=1, image_size=(256, 256))
        dataset = EpisodicDataset(cfg, split="train")

        support_imgs, support_masks, query_img, query_mask = dataset[0]
        # support_imgs:  K × 3 × 256 × 256
        # support_masks: K × 1 × 256 × 256
        # query_img:     3 × 256 × 256
        # query_mask:    1 × 256 × 256
    """

    # Normalization divisors — use _NORM_16BIT for current TIFF pipeline.
    # Switch to _NORM_8BIT if preprocessing outputs 8-bit images instead.
    _NORM_16BIT: float = 65535.0
    _NORM_8BIT:  float = 255.0

    def __init__(self, cfg: DatasetConfig, split: str) -> None:
        if split not in ("train", "val"):
            raise ValueError(
                f"split must be 'train' or 'val', got '{split}'."
            )

        self.cfg   = cfg
        self.split = split

        # Resolve image and mask directories
        root        = Path(cfg.data_root) / split
        self.img_dir  = root / "images"
        self.mask_dir = root / "masks"

        self._validate_directories()

        # Build index: list of (image_path, mask_path) pairs.
        # Scanned once at init — no filesystem calls during training.
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
        """Sample a fresh episode centered on idx as the query.

        The query is always the image at position idx.
        Support images are sampled randomly from the remaining indices.
        This ensures every image gets to be a query when iterating sequentially,
        while support is always fresh and random.

        Args:
            idx: Index of the query image.

        Returns:
            Tuple of (support_imgs, support_masks, query_img, query_mask):
                support_imgs:  K × 3 × H × W  — float32, [0, 1]
                support_masks: K × 1 × H × W  — float32, {0.0, 1.0}
                query_img:     3 × H × W       — float32, [0, 1]
                query_mask:    1 × H × W       — float32, {0.0, 1.0}
        """
        # Sample k_shot support indices, excluding the query index
        all_indices   = list(range(len(self.samples)))
        support_pool  = [i for i in all_indices if i != idx]
        support_idxs  = random.sample(support_pool, self.cfg.k_shot)

        # Load query
        query_img, query_mask = self._load_sample(
            idx,
            augment=self.cfg.augment_query,
        )
        # query_img:  3 × H × W
        # query_mask: 1 × H × W

        # Load K support images
        support_imgs  = []
        support_masks = []

        for s_idx in support_idxs:
            s_img, s_mask = self._load_sample(
                s_idx,
                augment=self.cfg.augment_support,
            )
            support_imgs.append(s_img)    # 3 × H × W
            support_masks.append(s_mask)  # 1 × H × W

        # Stack along new leading dimension → K × C × H × W
        support_imgs  = torch.stack(support_imgs,  dim=0)  # K × 3 × H × W
        support_masks = torch.stack(support_masks, dim=0)  # K × 1 × H × W

        return support_imgs, support_masks, query_img, query_mask

    def _load_sample(
        self,
        idx: int,
        augment: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load, resize, optionally augment, and normalize one image+mask pair.

        Augmentation is applied jointly to image and mask so spatial
        transforms remain consistent within a single sample. Support and
        query are augmented independently from each other.

        Args:
            idx:     Index into self.samples.
            augment: Whether to apply random augmentation.

        Returns:
            Tuple of (img_tensor, mask_tensor):
                img_tensor:  3 × H × W — float32, [0, 1]
                mask_tensor: 1 × H × W — float32, {0.0, 1.0}
        """
        img_path, mask_path = self.samples[idx]

        # Load from disk → numpy uint16 arrays
        img  = tifffile.imread(str(img_path))   # H × W × 3  uint16
        mask = tifffile.imread(str(mask_path))  # H × W      uint16

        # Convert to float32 tensors and normalize to [0, 1]
        # img:  3 × H × W
        # mask: 1 × H × W
        img_tensor  = self._to_tensor_img(img)
        mask_tensor = self._to_tensor_mask(mask)

        # Resize to configured spatial size
        # Bilinear for images, nearest for masks (preserve binary values)
        h, w = self.cfg.image_size
        img_tensor  = F.interpolate(
            img_tensor.unsqueeze(0),   # 1 × 3 × H × W
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                   # 3 × H × W

        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0),  # 1 × 1 × H × W
            size=(h, w),
            mode="nearest",
        ).squeeze(0)                   # 1 × H × W

        # Apply augmentation jointly to image and mask
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
        """Apply random spatial augmentation jointly to image and mask.

        Augmentation is intentionally conservative for radiographic images:
        only flips, no color jitter or rotations that could corrupt features.
        The same transform is applied to both img and mask to keep alignment.

        Args:
            img:  3 × H × W — float32
            mask: 1 × H × W — float32

        Returns:
            Augmented (img, mask) with same shapes.
        """
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(std_range=(0.008, 0.02), p=0.4),
        ])

        # Convertimos de Tensor a Numpy para que Albumentations trabaje
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.permute(1, 2, 0).numpy()

        # Aplicamos la transformada
        transformed = transform(image=img_np, mask=mask_np)
        
        # Convertimos de vuelta a Tensor de PyTorch
        img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1)
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
