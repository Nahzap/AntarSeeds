"""
Reorganiza splits train/val/test previniendo Data Leakage por Planos Focales (Z-stack).
Las imágenes de una misma coordenada física (ej. _fp0, _fp1, _fp2) se agrupan
para garantizar que vayan todas al mismo split (train, val o test).

Uso:
    python scripts/reorganize_splits_grouped.py --preset standard --seed 42
"""
import sys
import random
import shutil
import re
from pathlib import Path
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PRESETS = {
    "balanced": (0.8, 0.1, 0.1),
    "standard": (0.7, 0.15, 0.15),
    "research": (0.6, 0.2, 0.2),
    "production": (0.9, 0.05, 0.05)
}

def get_group_key(filename):
    """
    Extrae la clave del grupo (la coordenada física) ignorando el plano focal.
    Ejemplo:
        Quillaja_R05_X1250_Y3250_fp0.jpg -> Quillaja_R05_X1250_Y3250
        Quillaja_R05_X1250_Y3250.jpg -> Quillaja_R05_X1250_Y3250
    """
    # Buscar el patron _fp seguido de numeros al final del nombre
    match = re.sub(r'_fp\d+$', '', filename)
    return match

def reorganize_splits_grouped(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, dry_run=False, 
                     processed_root="data/processed", annotation_root="data/annotations"):
    random.seed(seed)
    
    processed_root = Path(processed_root)
    annotation_root = Path(annotation_root)
    
    if not annotation_root.exists():
        print(f"X {annotation_root} no existe")
        return False
    
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        print(f"X Los ratios deben sumar 1.0 (actual: {total:.3f})")
        return False
    
    print(f"\n{'='*60}")
    print(f"REORGANIZACIÓN AGRUPADA SIN DATA LEAKAGE")
    print(f"{'='*60}")
    print(f"Ratios (por grupo): train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No se moverán archivos\n")
    
    classes = sorted([d.name for d in annotation_root.iterdir() if d.is_dir()])
    if not classes:
        print(f"X No se encontraron clases en {annotation_root}")
        return False
    
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    
    for class_name in classes:
        print(f"[{class_name}]")
        
        # Agrupar TODAS las imágenes de esta clase por su clave (coordenada física)
        groups = defaultdict(list)
        total_images = 0
        
        for split in ["train", "val", "test"]:
            split_dir = processed_root / split / class_name
            if split_dir.exists():
                for ext in IMAGE_EXTENSIONS:
                    for img_path in split_dir.glob(f"*{ext}"):
                        group_key = get_group_key(img_path.stem)
                        groups[group_key].append(img_path)
                        total_images += 1
        
        if total_images == 0:
            print(f"  -  No se encontraron imagenes para {class_name}")
            continue
            
        group_keys = list(groups.keys())
        # Shuffle aleatorio de los GRUPOS, no de las imágenes individuales
        random.shuffle(group_keys)
        
        n_groups = len(group_keys)
        n_train = int(n_groups * train_ratio)
        n_val = int(n_groups * val_ratio)
        
        train_keys = group_keys[:n_train]
        val_keys = group_keys[n_train:n_train + n_val]
        test_keys = group_keys[n_train + n_val:]
        
        new_splits = {"train": [], "val": [], "test": []}
        
        for key in train_keys: new_splits["train"].extend(groups[key])
        for key in val_keys: new_splits["val"].extend(groups[key])
        for key in test_keys: new_splits["test"].extend(groups[key])
        
        if not dry_run:
            for split_name in ["train", "val", "test"]:
                split_dir = processed_root / split_name / class_name
                split_dir.mkdir(parents=True, exist_ok=True)
            
            for split_name, images in new_splits.items():
                split_dir = processed_root / split_name / class_name
                for img in images:
                    dest = split_dir / img.name
                    if img.resolve() != dest.resolve():
                        shutil.move(str(img), str(dest))
        
        print(f"  Grupos físicos: {n_groups} -> Imágenes: {total_images}")
        print(f"  Reorganized (Imgs): {len(new_splits['train'])} train, "
              f"{len(new_splits['val'])} val, {len(new_splits['test'])} test")
    
    print(f"\n{'='*60}")
    if not dry_run:
        print(f"OK REORGANIZACION COMPLETA SIN LEAKAGE")
    else:
        print(f"INFO Esto fue un DRY RUN. Ejecuta sin --dry-run para reorganizar.")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reorganizar agrupando por planos focales")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="standard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    tr, vr, te = PRESETS[args.preset]
    success = reorganize_splits_grouped(tr, vr, te, seed=args.seed, dry_run=args.dry_run)
    sys.exit(0 if success else 1)
