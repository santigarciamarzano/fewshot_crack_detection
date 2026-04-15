"""
datasets/episode_dataset_png.py
 
Variante PNG del EpisodicDataset para imágenes radiográficas preprocesadas.
 
Hereda toda la lógica episódica de EpisodicDataset y solo sobreescribe
los métodos de carga de disco. El resto — episodic sampling, augmentation,
resize, validación de directorios — es idéntico.
 
Diferencias respecto al dataset original:
    - Lee PNG en vez de TIFF
    - Imágenes RGB uint8 [0, 255] → float32 [0, 1]
    - Máscaras escala de grises uint8 {0, 255} → float32 {0.0, 1.0}
 
Cuando tengas los TIFF de 16 bits, solo cambiás EpisodicDatasetPNG
por EpisodicDataset en train.py — ningún otro cambio necesario.
 
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
 
    Todo lo demás se hereda de EpisodicDataset sin modificación.
 
    Args:
        cfg:   DatasetConfig — mismos campos que el dataset original.
               data_root debe apuntar a data_patches/ o equivalente.
        split: "train" o "val".
    """
 
    def __init__(self, cfg: DatasetConfig, split: str) -> None:
        # Llamamos al __init__ del padre — valida directorios,
        # construye el índice y verifica que hay suficientes muestras.
        super().__init__(cfg, split)
 
    # ------------------------------------------------------------------
    # Métodos sobreescritos
    # ------------------------------------------------------------------
 
    def _build_index(self) -> List[Tuple[Path, Path]]:
        """Escanea el directorio buscando *.png en vez de *.tiff.
 
        Idéntica lógica al padre — empareja imagen con máscara por nombre.
        Solo cambia el patrón de glob.
 
        Returns:
            Lista ordenada de (image_path, mask_path).
 
        Raises:
            FileNotFoundError: Si no hay PNG o falta alguna máscara.
        """
        img_paths = sorted(self.img_dir.glob("*.png"))
 
        if len(img_paths) == 0:
            raise FileNotFoundError(
                f"No se encontraron archivos .png en '{self.img_dir}'."
            )
 
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
        """Convierte imagen PNG RGB uint8 a tensor float32 en [0, 1].
 
        A diferencia del padre (que divide por 65535 para uint16),
        aquí dividimos por 255 para uint8.
 
        Args:
            img_np: Array H × W × 3, dtype uint8, valores [0, 255].
                    Cargado con PIL.Image.convert("RGB").
 
        Returns:
            Tensor float32 de shape 3 × H × W, valores en [0, 1].
        """
        # H × W × 3  →  3 × H × W
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return tensor / self._NORM_8BIT  # _NORM_8BIT = 255.0, definido en padre
 
    def _to_tensor_mask(self, mask_np: np.ndarray) -> torch.Tensor:
        """Convierte máscara PNG escala de grises uint8 a tensor float32 binario.
 
        Valores de entrada: {0, 255}. Valores de salida: {0.0, 1.0}.
 
        Args:
            mask_np: Array H × W, dtype uint8, valores {0, 255}.
                     Cargado con PIL.Image.convert("L").
 
        Returns:
            Tensor float32 de shape 1 × H × W, valores en {0.0, 1.0}.
        """
        # H × W  →  1 × H × W
        tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
        return tensor / self._NORM_8BIT  # {0, 255} → {0.0, 1.0}
 
    # ------------------------------------------------------------------
    # Sobreescribir _load_sample para usar PIL en vez de tifffile
    # ------------------------------------------------------------------
 
    def _load_sample(
        self,
        idx: int,
        augment: bool,
    ):
        """Carga un par imagen+máscara PNG desde disco.
 
        Sobreescribe el método del padre para usar PIL.Image en vez de
        tifffile. El resto del procesamiento (resize, augmentation)
        es idéntico al padre y se reutiliza llamando a super().
 
        Args:
            idx:     Índice en self.samples.
            augment: Si aplicar augmentation aleatoria.
 
        Returns:
            Tuple (img_tensor, mask_tensor):
                img_tensor:  3 × H × W — float32, [0, 1]
                mask_tensor: 1 × H × W — float32, {0.0, 1.0}
        """
        import torch.nn.functional as F
 
        img_path, mask_path = self.samples[idx]
 
        # Cargar con PIL — mucho más simple que tifffile para PNG
        img_np  = np.array(Image.open(img_path).convert("RGB"))   # H × W × 3, uint8
        mask_np = np.array(Image.open(mask_path).convert("L"))    # H × W,     uint8
 
        # Convertir a tensores float32 normalizados
        img_tensor  = self._to_tensor_img(img_np)    # 3 × H × W, [0, 1]
        mask_tensor = self._to_tensor_mask(mask_np)  # 1 × H × W, {0.0, 1.0}
 
        # Resize al tamaño configurado
        # Bilinear para imagen, nearest para máscara (preserva valores binarios)
        h, w = self.cfg.image_size
 
        img_tensor = F.interpolate(
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
 
        # Augmentation conjunta imagen+máscara
        if augment:
            img_tensor, mask_tensor = self._augment(img_tensor, mask_tensor)
 
        return img_tensor, mask_tensor
 