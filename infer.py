"""
infer.py
 
Inferencia few-shot sobre imágenes completas.
 
El support es un parche 256×256 con grieta visible.
La query es una imagen de cualquier tamaño — se divide automáticamente
en parches 256×256, se infiere sobre cada uno usando el mismo support,
y las máscaras se reensamblan en la posición original.
 
Uso:
    python infer.py --support_img imagenes_inferencia/support_img.png --support_mask imagenes_inferencia/support_mask.png --query_img imagenes_inferencia/query_img.png --checkpoint checkpoints/baseline/best_model.pt --output results/ --threshold 0.99
 
Salida en results/:
    query.png     → imagen query original sin modificar
    pred_mask.png → máscara predicha binaria {0, 255} al tamaño original
"""
 
import argparse
from pathlib import Path
from typing import Tuple
 
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
 
from models.fewshot_model import FewShotModel
 
 
PATCH_SIZE = 512  # debe coincidir con image_size del entrenamiento
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia few-shot sobre imagen completa con parcheo automático."
    )
    parser.add_argument("--support_img",  type=str, required=True, help="Parche 256×256 de support (imagen).")
    parser.add_argument("--support_mask", type=str, required=True, help="Parche 256×256 de support (máscara).")
    parser.add_argument("--query_img",    type=str, required=True, help="Imagen query de cualquier tamaño.")
    parser.add_argument("--checkpoint",   type=str, required=True, help="Path al checkpoint .pt.")
    parser.add_argument("--output",       type=str, default="results/", help="Carpeta de salida. Default: results/")
    parser.add_argument("--threshold",    type=float, default=0.5, help="Umbral para binarizar. Default: 0.5")
    return parser.parse_args()
 
 
# ---------------------------------------------------------------------------
# Carga de imágenes
# ---------------------------------------------------------------------------
 
def load_support_image(path: str) -> torch.Tensor:
    """Carga el parche support RGB y devuelve un tensor 1 × 3 × PATCH_SIZE × PATCH_SIZE."""
    img = np.array(Image.open(path).convert("RGB"))   # H × W × 3, uint8
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # 3 × H × W
 
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(PATCH_SIZE, PATCH_SIZE),
        mode="bilinear",
        align_corners=False,
    )  # 1 × 3 × PATCH_SIZE × PATCH_SIZE
 
    return tensor
 
 
def load_support_mask(path: str) -> torch.Tensor:
    """Carga la máscara support y devuelve un tensor binario 1 × 1 × PATCH_SIZE × PATCH_SIZE."""
    mask = np.array(Image.open(path).convert("L"))    # H × W, uint8
    tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() / 255.0
 
    tensor = F.interpolate(
        tensor,
        size=(PATCH_SIZE, PATCH_SIZE),
        mode="nearest",
    )  # 1 × 1 × PATCH_SIZE × PATCH_SIZE
 
    return tensor
 
 
def load_query_image(path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Carga la imagen query completa. Devuelve el tensor y el tamaño original."""
    img = np.array(Image.open(path).convert("RGB"))   # H × W × 3, uint8
    H_orig, W_orig = img.shape[:2]
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # 3 × H × W
 
    return tensor, (H_orig, W_orig)
 
 
# ---------------------------------------------------------------------------
# Parcheo y reensamblado
# ---------------------------------------------------------------------------
 
def pad_to_multiple(
    img: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Rellena la imagen con ceros para que H y W sean múltiplos de patch_size."""
    _, H, W = img.shape
 
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
 
    img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)  # orden: left, right, top, bottom
 
    return img_padded, (pad_h, pad_w)
 
 
def extract_patches(
    img: torch.Tensor,
    patch_size: int,
) -> Tuple[torch.Tensor, int, int]:
    """Divide la imagen en una grilla de parches sin solapamiento. La imagen debe ser divisible por patch_size."""
    _, H, W = img.shape
    n_rows = H // patch_size
    n_cols = W // patch_size
 
    patches = []
    for row in range(n_rows):
        for col in range(n_cols):
            y0, y1 = row * patch_size, (row + 1) * patch_size
            x0, x1 = col * patch_size, (col + 1) * patch_size
            patches.append(img[:, y0:y1, x0:x1])  # 3 × patch_size × patch_size
 
    patches = torch.stack(patches, dim=0)  # N × 3 × patch_size × patch_size
    return patches, n_rows, n_cols
 
 
def reassemble_mask(
    pred_patches: torch.Tensor,
    n_rows: int,
    n_cols: int,
    patch_size: int,
    orig_size: Tuple[int, int],
) -> np.ndarray:
    """Reensambla los parches predichos en la máscara completa al tamaño original."""
    H_pad = n_rows * patch_size
    W_pad = n_cols * patch_size
 
    full_mask = torch.zeros(H_pad, W_pad)
 
    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            y0, y1 = row * patch_size, (row + 1) * patch_size
            x0, x1 = col * patch_size, (col + 1) * patch_size
            full_mask[y0:y1, x0:x1] = pred_patches[idx]
            idx += 1
 
    H_orig, W_orig = orig_size
    full_mask = full_mask[:H_orig, :W_orig]  # recortar el padding
 
    return (full_mask.numpy() * 255).astype(np.uint8)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    args = parse_args()
 
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print(f"Cargando checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
 
    cfg = checkpoint["config"]
    model = FewShotModel(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Modelo cargado — epoch {checkpoint['epoch']}, val_iou {checkpoint['val_iou']:.4f}")
    print(f"Device: {device}")
    print()
 
    support_img  = load_support_image(args.support_img).to(device)   # 1 × 3 × PATCH_SIZE × PATCH_SIZE
    support_mask = load_support_mask(args.support_mask).to(device)   # 1 × 1 × PATCH_SIZE × PATCH_SIZE
 
    query_tensor, orig_size = load_query_image(args.query_img)       # 3 × H × W
    H_orig, W_orig = orig_size
 
    print(f"Support: {args.support_img}")
    print(f"Query:   {args.query_img}  ({W_orig}×{H_orig} px)")
    print()
 
    query_padded, (pad_h, pad_w) = pad_to_multiple(query_tensor, PATCH_SIZE)
    patches, n_rows, n_cols = extract_patches(query_padded, PATCH_SIZE)
    n_patches = n_rows * n_cols
    print(f"Parches: {n_rows} filas × {n_cols} cols = {n_patches} parches de {PATCH_SIZE}×{PATCH_SIZE}")
    print()
 
    pred_patches = []
 
    with torch.no_grad():
        for i, patch in enumerate(patches):
            patch_input = patch.unsqueeze(0).to(device)              # 1 × 3 × PATCH_SIZE × PATCH_SIZE
            logits = model(support_img, support_mask, patch_input)   # 1 × 1 × PATCH_SIZE × PATCH_SIZE
 
            pred = (torch.sigmoid(logits) > args.threshold).float().squeeze()
            pred_patches.append(pred.cpu())
 
            if (i + 1) % 10 == 0 or (i + 1) == n_patches:
                print(f"  Parche {i+1}/{n_patches}")
 
    pred_patches = torch.stack(pred_patches, dim=0)  # N × PATCH_SIZE × PATCH_SIZE
    full_mask = reassemble_mask(pred_patches, n_rows, n_cols, PATCH_SIZE, orig_size)
 
    query_dst = output_dir / "query.png"
    Image.open(args.query_img).convert("RGB").save(query_dst)
 
    pred_dst = output_dir / "pred_mask.png"
    Image.fromarray(full_mask, mode="L").save(pred_dst)
 
    # overlay: grietas en rojo sobre la query original
    query_np = np.array(Image.open(args.query_img).convert("RGB"))
    overlay  = query_np.copy()
    crack_pixels = full_mask > 0
    overlay[crack_pixels, 0] = 255
    overlay[crack_pixels, 1] = 0
    overlay[crack_pixels, 2] = 0
    overlay_dst = output_dir / "overlay.png"
    Image.fromarray(overlay, mode="RGB").save(overlay_dst)
 
    n_crack = int((full_mask > 0).sum())
    total   = full_mask.size
    print()
    print(f"Resultados guardados en '{output_dir}':")
    print(f"  query.png     → {W_orig}×{H_orig} px")
    print(f"  pred_mask.png → {W_orig}×{H_orig} px")
    print(f"  overlay.png   → grietas en rojo sobre la query")
    print(f"Píxeles de grieta predichos: {n_crack} / {total} ({n_crack/total*100:.1f}%)")
 
 
if __name__ == "__main__":
    main()