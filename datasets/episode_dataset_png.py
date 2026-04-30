"""
datasets/episode_dataset_png.py

Variante PNG del EpisodicDataset para imágenes radiográficas preprocesadas.

Hereda toda la lógica episódica de EpisodicDataset y solo sobreescribe
los métodos de carga de disco. Lee PNG RGB de 8 bits en vez de TIFF de 16 bits.

Uso:
    from datasets.episode_dataset_png import EpisodicDatasetPNG

    cfg = DatasetConfig(data_root="data_patches/", k_shot=1, image_size=(256, 256))
    dataset = EpisodicDatasetPNG(cfg, split="train")

    support_imgs, support_masks, query_img, query_mask = dataset[0]
    # support_imgs:  K × 3 × 256 × 256  — float32, [0, 1]
    # support_masks: K × 1 × 256 × 256  — float32, {0.0, 1.0}
    # query_img:     3 × 256 × 256
    # query_mask:    1 × 256 × 256
"""
 
from pathlib import Path
from typing import List, Tuple
 
import numpy as np
import torch
from PIL import Image
 
from config.base_config import DatasetConfig
from datasets.episode_dataset import EpisodicDataset
 
 
class EpisodicDatasetPNG(EpisodicDataset):
    """EpisodicDataset para imágenes PNG RGB de 8 bits.

    Sobreescribe únicamente los métodos de lectura de disco:
        - _build_index:    busca *.png en vez de *.tiff
        - _to_tensor_img:  normaliza uint8 [0,255] → float32 [0,1]
        - _to_tensor_mask: normaliza uint8 {0,255} → float32 {0.0,1.0}
        - _load_sample:    usa PIL en vez de tifffile

    Args:
        cfg:   DatasetConfig. data_root debe apuntar a data_patches/ o equivalente.
        split: "train" o "val".
    """
 
    def __init__(self, cfg: DatasetConfig, split: str) -> None:
        super().__init__(cfg, split)
 
    # ------------------------------------------------------------------
    # Métodos sobreescritos
    # ------------------------------------------------------------------
 
    def _build_index(self) -> List[Tuple[Path, Path]]:
        """Escanea el directorio buscando *.png en vez de *.tiff."""
        img_paths = sorted(self.img_dir.glob("*.png"))

        if len(img_paths) == 0:
            raise FileNotFoundError(f"No se encontraron archivos .png en '{self.img_dir}'.")

        samples = []
        for img_path in img_paths:
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Máscara no encontrada para '{img_path.name}'. "
                    f"Esperada en '{mask_path}'."
                )
            samples.append((img_path, mask_path))

        return samples
 
    def _to_tensor_img(self, img_np: np.ndarray) -> torch.Tensor:
        """Convierte imagen PNG RGB uint8 a tensor float32 en [0, 1]."""
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()  # H×W×3 → 3×H×W
        return tensor / self._NORM_8BIT

    def _to_tensor_mask(self, mask_np: np.ndarray) -> torch.Tensor:
        """Convierte máscara PNG escala de grises uint8 {0,255} a tensor float32 {0.0,1.0}."""
        tensor = torch.from_numpy(mask_np).unsqueeze(0).float()  # H×W → 1×H×W
        return tensor / self._NORM_8BIT
 
    # ------------------------------------------------------------------
    # Sobreescribir _load_sample para usar PIL en vez de tifffile
    # ------------------------------------------------------------------
 
    def _load_sample(self, idx: int, augment: bool):
        """Carga un par imagen+máscara PNG desde disco."""
        import torch.nn.functional as F

        img_path, mask_path = self.samples[idx]

        img_np  = np.array(Image.open(img_path).convert("RGB"))   # H × W × 3, uint8
        mask_np = np.array(Image.open(mask_path).convert("L"))    # H × W,     uint8

        img_tensor  = self._to_tensor_img(img_np)
        mask_tensor = self._to_tensor_mask(mask_np)

        h, w = self.cfg.image_size
        img_tensor = F.interpolate(
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
 