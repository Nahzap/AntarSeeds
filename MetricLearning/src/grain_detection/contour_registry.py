"""
ContourRegistry — Registro persistente de segmentaciones de contornos.

Guarda un JSON por dataset (train/val/test) con:
- Parámetros de detección usados
- Lista de imágenes con sus .seg, granos por imagen, symlink target
- Estadísticas globales y por clase
- Timestamp

Esto permite restaurar el estado del Análisis de Contornos sin re-escanear
ni re-ejecutar U²-Net.

Ubicación: data/processed/{split}/contour_registry.json
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "contour_registry.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class ContourRegistry:
    """Registro persistente de segmentaciones generadas."""

    @staticmethod
    def save(
        root_dir: str,
        detection_params: Dict[str, Any],
        annotation_stats: Optional[Dict] = None,
        annotation_root: Optional[str] = None,
    ) -> Path:
        """
        Escanea el dataset y guarda un registro completo de todas las
        segmentaciones existentes.

        Args:
            root_dir: Raíz del dataset (e.g. data/processed/train)
            detection_params: Parámetros de detección usados
            annotation_stats: Stats de la última anotación (opcional)
            annotation_root: Directorio centralizado de anotaciones (e.g. data/annotations)

        Returns:
            Path al archivo de registro guardado
        """
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {root}")

        from src.grain_detection.seg_format import SegFileReader

        images_list = []
        total_grains = 0
        grains_per_class = {}
        images_with_seg = 0
        
        # Determinar dónde buscar .seg
        if annotation_root:
            annot_root = Path(annotation_root)
            logger.info(f"[ContourRegistry] Buscando .seg en: {annot_root} (centralizado)")
        else:
            annot_root = None
            logger.info(f"[ContourRegistry] Buscando .seg junto a imágenes (legacy)")

        class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_grains = 0

            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Buscar .seg en ubicación correcta
                if annot_root:
                    seg_path = annot_root / class_name / f"{f.stem}.seg"
                else:
                    seg_path = f.with_suffix(".seg")
                entry = {
                    "path": str(f.relative_to(root)),
                    "class": class_name,
                }

                # Resolver symlink target
                try:
                    if f.is_symlink():
                        entry["symlink_target"] = str(Path(os.readlink(str(f))))
                except (OSError, ValueError):
                    pass

                if seg_path.exists():
                    # Guardar ruta relativa al annotation_root si existe, sino relativa a root
                    if annot_root:
                        entry["seg_path"] = str(seg_path.relative_to(annot_root))
                    else:
                        entry["seg_path"] = str(seg_path.relative_to(root))
                    
                    try:
                        grains = SegFileReader.read_grains(str(seg_path))
                        n_grains = len(grains)
                    except Exception:
                        n_grains = 0
                    entry["grain_count"] = n_grains
                    class_grains += n_grains
                    images_with_seg += 1
                else:
                    entry["seg_path"] = None
                    entry["grain_count"] = 0

                images_list.append(entry)

            total_grains += class_grains
            if class_grains > 0:
                grains_per_class[class_name] = class_grains

        total_images = len(images_list)

        registry = {
            "version": 1,
            "timestamp": datetime.now().isoformat(),
            "dataset_root": str(root),
            "annotation_root": str(annot_root) if annot_root else None,
            "detection_params": detection_params,
            "summary": {
                "total_images": total_images,
                "images_with_seg": images_with_seg,
                "images_without_seg": total_images - images_with_seg,
                "total_grains": total_grains,
                "grains_per_image": round(
                    total_grains / max(1, images_with_seg), 2
                ),
                "classes": sorted(grains_per_class.keys()),
                "num_classes": len(grains_per_class),
                "grains_per_class": grains_per_class,
            },
            "images": images_list,
        }

        # Incluir stats de la última anotación si se proporcionan
        if annotation_stats:
            registry["last_annotation"] = {
                "annotated": annotation_stats.get("annotated", 0),
                "skipped": annotation_stats.get("skipped", 0),
                "errors": annotation_stats.get("errors", 0),
                "elapsed_ms": annotation_stats.get("elapsed_ms", 0),
            }

        out_path = root / REGISTRY_FILENAME
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[ContourRegistry] Guardado: {out_path} "
            f"({total_images} imgs, {images_with_seg} con .seg, "
            f"{total_grains} granos)"
        )
        return out_path

    @staticmethod
    def load(root_dir: str) -> Optional[Dict]:
        """
        Carga el registro desde disco.

        Args:
            root_dir: Raíz del dataset

        Returns:
            Dict con el registro o None si no existe
        """
        reg_path = Path(root_dir) / REGISTRY_FILENAME
        if not reg_path.exists():
            return None

        with open(reg_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        logger.info(
            f"[ContourRegistry] Cargado: {reg_path} "
            f"({registry['summary']['total_images']} imgs, "
            f"{registry['summary']['total_grains']} granos)"
        )
        return registry

    @staticmethod
    def exists(root_dir: str) -> bool:
        """Verifica si existe un registro guardado."""
        return (Path(root_dir) / REGISTRY_FILENAME).exists()

    @staticmethod
    def get_path(root_dir: str) -> Path:
        """Retorna la ruta del archivo de registro."""
        return Path(root_dir) / REGISTRY_FILENAME

    @staticmethod
    def is_stale(root_dir: str) -> bool:
        """
        Verifica si el registro está desactualizado comparando
        el timestamp del registro con los .seg más recientes.

        Returns:
            True si hay .seg más nuevos que el registro
        """
        reg_path = Path(root_dir) / REGISTRY_FILENAME
        if not reg_path.exists():
            return True

        reg_mtime = reg_path.stat().st_mtime

        root = Path(root_dir)
        for seg_file in root.rglob("*.seg"):
            if seg_file.stat().st_mtime > reg_mtime:
                return True

        return False

    @staticmethod
    def delete(root_dir: str) -> bool:
        """Elimina el registro."""
        reg_path = Path(root_dir) / REGISTRY_FILENAME
        if reg_path.exists():
            reg_path.unlink()
            logger.info(f"[ContourRegistry] Eliminado: {reg_path}")
            return True
        return False
