"""
Valida que todas las imágenes en splits tengan su anotación en data/annotations/.

Uso:
    python scripts/validate_annotations_centralized.py
    
Checks:
    1. Cada imagen en train/val/test tiene su .seg en data/annotations/
    2. No hay .seg huérfanos (sin imagen correspondiente)
    3. Todos los .seg son parseables
"""
import sys
from pathlib import Path
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate_annotations(processed_root="data/processed", annotation_root="data/annotations"):
    """Verifica integridad entre imágenes y anotaciones centralizadas"""
    
    processed_root = Path(processed_root)
    annotation_root = Path(annotation_root)
    
    if not annotation_root.exists():
        print(f"❌ {annotation_root} no existe")
        print(f"   Ejecuta: python scripts/migrate_annotations_to_centralized.py")
        return False
    
    print(f"\n{'='*60}")
    print(f"VALIDACIÓN DE ANOTACIONES CENTRALIZADAS")
    print(f"{'='*60}\n")
    
    # Estadísticas
    stats = {
        "total_images": 0,
        "with_annotation": 0,
        "missing_annotation": [],
        "orphan_annotations": [],
        "corrupt_annotations": [],
        "by_split": defaultdict(int),
        "by_class": defaultdict(int)
    }
    
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    
    # 1. Verificar que cada imagen tenga su anotación
    print("🔍 Verificando imágenes → anotaciones...\n")
    
    for split in ["train", "val", "test"]:
        split_dir = processed_root / split
        if not split_dir.exists():
            continue
        
        for img_path in split_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            
            stats["total_images"] += 1
            stats["by_split"][split] += 1
            
            class_name = img_path.parent.name
            stats["by_class"][class_name] += 1
            
            # Buscar anotación en data/annotations/clase/
            seg_path = annotation_root / class_name / f"{img_path.stem}.seg"
            
            if seg_path.exists():
                stats["with_annotation"] += 1
                
                # Verificar que sea parseable
                try:
                    from src.grain_detection.seg_format import SegFileReader
                    result = SegFileReader.read(str(seg_path))
                    if not result.get("grains"):
                        stats["corrupt_annotations"].append(str(seg_path.relative_to(annotation_root)))
                except Exception as e:
                    stats["corrupt_annotations"].append(f"{seg_path.relative_to(annotation_root)} ({e})")
            else:
                stats["missing_annotation"].append(
                    f"{split}/{class_name}/{img_path.name}"
                )
    
    # 2. Verificar .seg huérfanos (sin imagen correspondiente)
    print("🔍 Verificando anotaciones huérfanas...\n")
    
    # Recolectar todos los nombres de imagen existentes
    existing_images = set()
    for split in ["train", "val", "test"]:
        split_dir = processed_root / split
        if not split_dir.exists():
            continue
        
        for img_path in split_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                class_name = img_path.parent.name
                existing_images.add((class_name, img_path.stem))
    
    # Verificar cada .seg en annotations/
    for seg_path in annotation_root.rglob("*.seg"):
        class_name = seg_path.parent.name
        img_stem = seg_path.stem
        
        if (class_name, img_stem) not in existing_images:
            try:
                stats["orphan_annotations"].append(
                    str(seg_path.relative_to(annotation_root))
                )
            except ValueError:
                stats["orphan_annotations"].append(str(seg_path))
    
    # 3. Reportar resultados
    print(f"{'='*60}")
    print(f"RESULTADOS DE VALIDACIÓN")
    print(f"{'='*60}\n")
    
    print(f"📊 Estadísticas Generales:")
    print(f"   Total imágenes:          {stats['total_images']}")
    print(f"   Con anotación:           {stats['with_annotation']}")
    print(f"   Sin anotación:           {len(stats['missing_annotation'])}")
    print(f"   Anotaciones corruptas:   {len(stats['corrupt_annotations'])}")
    print(f"   Anotaciones huérfanas:   {len(stats['orphan_annotations'])}")
    
    print(f"\n📊 Por Split:")
    for split in ["train", "val", "test"]:
        print(f"   {split:5s}: {stats['by_split'][split]:4d} imágenes")
    
    print(f"\n📊 Por Clase:")
    for class_name in sorted(stats['by_class'].keys()):
        count = stats['by_class'][class_name]
        print(f"   {class_name:30s}: {count:4d} imágenes")
    
    # Detalles de problemas
    if stats["missing_annotation"]:
        print(f"\n⚠️  Imágenes SIN anotación ({len(stats['missing_annotation'])}):")
        for path in stats["missing_annotation"][:10]:
            print(f"   - {path}")
        if len(stats["missing_annotation"]) > 10:
            print(f"   ... y {len(stats['missing_annotation']) - 10} más")
    
    if stats["corrupt_annotations"]:
        print(f"\n⚠️  Anotaciones CORRUPTAS ({len(stats['corrupt_annotations'])}):")
        for path in stats["corrupt_annotations"][:10]:
            print(f"   - {path}")
        if len(stats["corrupt_annotations"]) > 10:
            print(f"   ... y {len(stats['corrupt_annotations']) - 10} más")
    
    if stats["orphan_annotations"]:
        print(f"\n⚠️  Anotaciones HUÉRFANAS ({len(stats['orphan_annotations'])}):")
        for path in stats["orphan_annotations"][:10]:
            print(f"   - {path}")
        if len(stats["orphan_annotations"]) > 10:
            print(f"   ... y {len(stats['orphan_annotations']) - 10} más")
        print(f"\n   Nota: Anotaciones huérfanas pueden eliminarse con:")
        print(f"   find data/annotations -name '*.seg' | xargs -I {{}} sh -c 'test ! -f data/processed/*/*/`basename {{}} .seg`.png && rm {{}}'")
    
    # Veredicto final
    print(f"\n{'='*60}")
    if (len(stats["missing_annotation"]) == 0 and 
        len(stats["corrupt_annotations"]) == 0):
        print(f"✅ VALIDACIÓN EXITOSA")
        print(f"   Todas las {stats['total_images']} imágenes tienen anotaciones válidas")
        return True
    else:
        print(f"❌ VALIDACIÓN FALLIDA")
        print(f"   {len(stats['missing_annotation'])} imágenes sin anotación")
        print(f"   {len(stats['corrupt_annotations'])} anotaciones corruptas")
        return False


if __name__ == "__main__":
    success = validate_annotations()
    sys.exit(0 if success else 1)
