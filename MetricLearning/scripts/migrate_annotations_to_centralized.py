"""
Migra todas las anotaciones .seg existentes a data/annotations/.
Ejecutar UNA VEZ para migrar del sistema antiguo (anotaciones en splits) al nuevo.

Uso:
    python scripts/migrate_annotations_to_centralized.py
    
Seguridad:
    - Crea backups de .seg antes de mover
    - Detecta y resuelve duplicados
    - No elimina archivos hasta validar la migración
"""
import sys
import shutil
from pathlib import Path
from collections import defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def migrate_annotations(dry_run=False, processed_root="data/processed", annotation_root="data/annotations"):
    """
    Mueve todos los .seg de train/val/test/clase/ a data/annotations/clase/
    
    Args:
        dry_run: Si True, solo reporta qué se haría sin mover archivos
        processed_root: Ruta al directorio processed (default: "data/processed")
        annotation_root: Ruta al directorio de anotaciones (default: "data/annotations")
    """
    annotation_root = Path(annotation_root)
    processed_root = Path(processed_root)
    
    if not processed_root.exists():
        print(f"❌ {processed_root} no existe")
        return False
    
    # Estadísticas
    stats = {
        "moved": 0,
        "duplicates": 0,
        "conflicts": 0,
        "errors": 0
    }
    
    # Recolectar todos los .seg existentes
    seg_files = defaultdict(list)  # filename -> [paths]
    
    for split in ["train", "val", "test"]:
        split_dir = processed_root / split
        if not split_dir.exists():
            continue
        
        for seg_path in split_dir.rglob("*.seg"):
            class_name = seg_path.parent.name
            key = (class_name, seg_path.name)
            seg_files[key].append(seg_path)
    
    total_files = sum(len(paths) for paths in seg_files.values())
    print(f"\n{'='*60}")
    print(f"MIGRACIÓN DE ANOTACIONES: {len(seg_files)} archivos únicos")
    print(f"Total .seg encontrados: {total_files}")
    print(f"{'='*60}\n")
    
    if dry_run:
        print("🔍 DRY RUN MODE - No se moverán archivos\n")
    
    # Procesar cada archivo único
    for (class_name, filename), paths in seg_files.items():
        dest_dir = annotation_root / class_name
        dest_path = dest_dir / filename
        
        if len(paths) == 1:
            # Solo existe en un split - mover directo
            source = paths[0]
            
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(dest_path))
                print(f"✓ Migrado: {source.relative_to(processed_root)} → {dest_path.relative_to(annotation_root)}")
            else:
                print(f"[DRY] Migraría: {source.relative_to(processed_root)} → {dest_path.relative_to(annotation_root)}")
            
            stats["moved"] += 1
        
        else:
            # Existe en múltiples splits - resolver duplicado
            print(f"\n⚠ Duplicado: {class_name}/{filename} existe en {len(paths)} splits:")
            
            # Comparar contenidos
            contents = [p.read_text() for p in paths]
            unique_contents = set(contents)
            
            if len(unique_contents) == 1:
                # Contenido idéntico - tomar cualquiera
                source = max(paths, key=lambda p: p.stat().st_mtime)  # más reciente
                
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(source), str(dest_path))
                    print(f"  → Contenido idéntico, tomando versión de {source.parent.parent.name}/")
                else:
                    print(f"  [DRY] Tomaría versión de {source.parent.parent.name}/")
                
                stats["duplicates"] += len(paths) - 1
                stats["moved"] += 1
            
            else:
                # Contenidos diferentes - CONFLICTO
                print(f"  ⚠️  CONFLICTO: Contenidos diferentes entre splits")
                for i, path in enumerate(paths):
                    mtime = path.stat().st_mtime
                    size = path.stat().st_size
                    print(f"    [{i+1}] {path.relative_to(processed_root)} (size={size}B, mtime={mtime})")
                
                # Tomar el más reciente
                source = max(paths, key=lambda p: p.stat().st_mtime)
                
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(source), str(dest_path))
                    print(f"  → Tomando versión más reciente: {source.parent.parent.name}/")
                    
                    # Backup de versiones conflictivas
                    for i, path in enumerate(paths):
                        if path != source:
                            backup_path = dest_path.with_suffix(f".seg.conflict_{path.parent.parent.name}")
                            shutil.copy2(str(path), str(backup_path))
                            print(f"  → Backup conflicto: {backup_path.name}")
                else:
                    print(f"  [DRY] Tomaría versión más reciente: {source.parent.parent.name}/")
                
                stats["conflicts"] += 1
                stats["moved"] += 1
    
    # Resumen
    print(f"\n{'='*60}")
    print(f"RESUMEN DE MIGRACIÓN")
    print(f"{'='*60}")
    print(f"Archivos únicos migrados: {stats['moved']}")
    print(f"Duplicados (idénticos):   {stats['duplicates']}")
    print(f"Conflictos resueltos:     {stats['conflicts']}")
    print(f"Errores:                  {stats['errors']}")
    
    if not dry_run:
        print(f"\n✅ Anotaciones migradas a: {annotation_root}")
        print(f"\n⚠️  IMPORTANTE:")
        print(f"   1. Ejecuta: python scripts/validate_annotations_centralized.py")
        print(f"   2. Si todo OK, elimina .seg de splits con:")
        print(f"      find data/processed -name '*.seg' -delete")
    else:
        print(f"\n🔍 Esto fue un DRY RUN. Ejecuta sin --dry-run para migrar.")
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Migrar anotaciones a directorio centralizado")
    parser.add_argument("--dry-run", action="store_true", help="Simular migración sin mover archivos")
    args = parser.parse_args()
    
    success = migrate_annotations(dry_run=args.dry_run)
    sys.exit(0 if success else 1)
