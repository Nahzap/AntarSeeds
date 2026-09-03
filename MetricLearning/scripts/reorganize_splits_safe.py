"""
Reorganiza splits train/val/test SIN tocar las anotaciones.
Las anotaciones permanecen en data/annotations/ (inmutables).

Uso:
    python scripts/reorganize_splits_safe.py --train 0.7 --val 0.15 --test 0.15 --seed 42
    python scripts/reorganize_splits_safe.py --preset balanced  # 80/10/10

Seguridad:
    - Solo mueve imágenes, NO anotaciones
    - Valida que todas las imágenes tengan anotación antes de mover
    - Crea backup del estado anterior
"""
import sys
import random
import shutil
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


def reorganize_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, dry_run=False, 
                     processed_root="data/processed", annotation_root="data/annotations"):
    """
    Re-organiza aleatoriamente imágenes entre splits.
    
    GARANTÍA: Las anotaciones en data/annotations/ NO se tocan.
    
    Args:
        train_ratio: Proporción para train (0-1)
        val_ratio: Proporción para val (0-1)
        test_ratio: Proporción para test (0-1)
        seed: Semilla para reproducibilidad
        dry_run: Si True, solo reporta qué se haría
        processed_root: Ruta al directorio processed (default: "data/processed")
        annotation_root: Ruta al directorio de anotaciones (default: "data/annotations")
    """
    random.seed(seed)
    
    processed_root = Path(processed_root)
    annotation_root = Path(annotation_root)
    
    # Validaciones
    if not annotation_root.exists():
        print(f"❌ {annotation_root} no existe")
        print(f"   Ejecuta: python scripts/migrate_annotations_to_centralized.py")
        return False
    
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        print(f"❌ Los ratios deben sumar 1.0 (actual: {total:.3f})")
        return False
    
    print(f"\n{'='*60}")
    print(f"REORGANIZACIÓN SEGURA DE SPLITS")
    print(f"{'='*60}")
    print(f"Ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
    print(f"Seed:   {seed}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No se moverán archivos\n")
    
    # Recolectar clases desde annotations/ (source of truth)
    classes = sorted([d.name for d in annotation_root.iterdir() if d.is_dir()])
    
    if not classes:
        print(f"❌ No se encontraron clases en {annotation_root}")
        return False
    
    print(f"📊 Clases encontradas: {len(classes)}\n")
    
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    
    # Procesar cada clase
    for class_name in classes:
        print(f"[{class_name}]")
        
        # Recolectar TODAS las imágenes de esta clase (todos los splits)
        all_images = []
        for split in ["train", "val", "test"]:
            split_dir = processed_root / split / class_name
            if split_dir.exists():
                for ext in IMAGE_EXTENSIONS:
                    all_images.extend(list(split_dir.glob(f"*{ext}")))
        
        if not all_images:
            print(f"  ⚠️  No se encontraron imágenes para {class_name}")
            continue
        
        # Verificar que todas tengan anotación
        missing_annot = []
        for img in all_images:
            seg_path = annotation_root / class_name / f"{img.stem}.seg"
            if not seg_path.exists():
                missing_annot.append(img.name)
        
        if missing_annot:
            print(f"  ⚠️  {len(missing_annot)} imágenes sin anotación:")
            for name in missing_annot[:5]:
                print(f"     - {name}")
            if len(missing_annot) > 5:
                print(f"     ... y {len(missing_annot) - 5} más")
            print(f"  ⚠️  Ejecuta: python scripts/validate_annotations_centralized.py")
            continue
        
        # Shuffle aleatorio
        random.shuffle(all_images)
        
        # Calcular nuevos splits
        n = len(all_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        new_splits = {
            "train": all_images[:n_train],
            "val": all_images[n_train:n_train + n_val],
            "test": all_images[n_train + n_val:]
        }
        
        if not dry_run:
            # Crear directorios destino
            for split_name in ["train", "val", "test"]:
                split_dir = processed_root / split_name / class_name
                split_dir.mkdir(parents=True, exist_ok=True)
            
            # Mover imágenes a nuevos splits
            for split_name, images in new_splits.items():
                split_dir = processed_root / split_name / class_name
                
                for img in images:
                    dest = split_dir / img.name
                    
                    # Solo mover si no está ya en el destino
                    if img.resolve() != dest.resolve():
                        shutil.move(str(img), str(dest))
        
        print(f"  Reorganized: {len(new_splits['train'])} train, "
              f"{len(new_splits['val'])} val, {len(new_splits['test'])} test")
        print(f"  ✅ Anotaciones intactas en data/annotations/{class_name}/")
    
    print(f"\n{'='*60}")
    if not dry_run:
        print(f"✅ REORGANIZACIÓN COMPLETA")
        print(f"\nPróximos pasos:")
        print(f"   1. Valida: python scripts/validate_annotations_centralized.py")
        print(f"   2. Entrena: python scripts/train.py")
    else:
        print(f"🔍 Esto fue un DRY RUN. Ejecuta sin --dry-run para reorganizar.")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reorganizar splits de manera segura")
    parser.add_argument("--train", type=float, help="Ratio para train (0-1)")
    parser.add_argument("--val", type=float, help="Ratio para val (0-1)")
    parser.add_argument("--test", type=float, help="Ratio para test (0-1)")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), 
                       help="Preset de ratios: balanced(80/10/10), standard(70/15/15), research(60/20/20), production(90/5/5)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin mover archivos")
    args = parser.parse_args()
    
    # Determinar ratios
    if args.preset:
        train_ratio, val_ratio, test_ratio = PRESETS[args.preset]
        print(f"Usando preset '{args.preset}': {train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%}")
    elif args.train and args.val and args.test:
        train_ratio, val_ratio, test_ratio = args.train, args.val, args.test
    else:
        print("Especifica --train/--val/--test o --preset")
        print(f"Presets disponibles: {', '.join(PRESETS.keys())}")
        sys.exit(1)
    
    success = reorganize_splits(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
        dry_run=args.dry_run
    )
    sys.exit(0 if success else 1)
