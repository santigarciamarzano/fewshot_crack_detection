"""
generate_support_bank.py

Busca grietas en imágenes grandes y extrae parches centrados en ellas.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Genera un banco de parches Support centrados en grietas.")
    parser.add_argument("--src_imgs", type=str, required=True, help="Carpeta con imágenes grandes originales.")
    parser.add_argument("--src_masks", type=str, required=True, help="Carpeta con máscaras grandes originales.")
    parser.add_argument("--dst", type=str, default="support_bank/", help="Carpeta destino.")
    parser.add_argument("--size", type=int, default=256, help="Tamaño del parche (ej. 256 o 512).")
    return parser.parse_args()

def get_valid_crop(c, max_val, size):
    """Calcula coordenadas de recorte evitando salirse de los bordes."""
    start = int(c - size // 2)
    end = start + size
    if start < 0:
        start = 0
        end = size
    elif end > max_val:
        end = max_val
        start = max_val - size
    return start, end

def main():
    args = parse_args()
    src_imgs = Path(args.src_imgs)
    src_masks = Path(args.src_masks)
    
    img_out = Path(args.dst) / "images"
    mask_out = Path(args.dst) / "masks"
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    # Buscamos png, tiff y tif
    mask_paths = []
    for ext in ["*.png", "*.tiff", "*.tif"]:
        mask_paths.extend(src_masks.glob(ext))
    
    print(f"Buscando grietas en {len(mask_paths)} máscaras para tamaño {args.size}x{args.size}...")
    
    count = 0
    for mask_path in mask_paths:
        # Buscamos la imagen correspondiente probando varias extensiones
        img_path = src_imgs / mask_path.name
        if not img_path.exists():
            img_path = src_imgs / (mask_path.stem + ".png")
        if not img_path.exists():
            img_path = src_imgs / (mask_path.stem + ".tiff")
            
        if not img_path.exists():
            print(f"⚠️ No se encontró la imagen para la máscara {mask_path.name}")
            continue

        # Leer imagen y máscara con OpenCV (IMREAD_UNCHANGED respeta los bits originales)
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        if mask is None or img is None:
            continue

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # escala de grises → RGB
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR (OpenCV por defecto) → RGB

        if len(mask.shape) > 2:
            mask = mask[:, :, 0]  # si la máscara tiene varios canales (ej. RGBA), nos quedamos con el primero

        # Binarizar la máscara de forma segura para OpenCV (0 y 1)
        binary_mask = (mask > 0).astype(np.uint8)

        if binary_mask.max() == 0:
            print(f"  - Saltando {mask_path.name}: La máscara es completamente negra.")
            continue

        H, W = binary_mask.shape
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        for i in range(1, num_labels):  # el label 0 es el fondo
            cx, cy = centroids[i]

            y0, y1 = get_valid_crop(cy, H, args.size)
            x0, x1 = get_valid_crop(cx, W, args.size)

            img_patch = img[y0:y1, x0:x1]
            mask_patch = binary_mask[y0:y1, x0:x1] * 255

            patch_name = f"{mask_path.stem}_crack{i}.png"
            cv2.imwrite(str(img_out / patch_name), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))  # OpenCV guarda en BGR
            cv2.imwrite(str(mask_out / patch_name), mask_patch)

            count += 1

    print(f"\n¡Listo! Se generaron {count} parches perfectos en '{args.dst}'.")

if __name__ == "__main__":
    main()