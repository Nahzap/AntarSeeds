"""
SegmentationAnnotator — Generación batch de archivos .seg

Escanea un directorio de imágenes (estructura ImageFolder), ejecuta
PollenGrainDetector sobre cada imagen y escribe archivos .seg.

Con object_finder=roi_seed (default), el lote usa las mismas semillas
automáticas que el click de ROI: picos de saliencia → recorte → contorno.

Uso:
    annotator = SegmentationAnnotator(detector)
    stats = annotator.annotate_directory("data/processed/train")
"""

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from .grain_detector import PollenGrainDetector
from .seg_format import SegFileWriter, get_seg_path, has_seg_file

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def count_annotations(
    annotation_root: str, class_filter: Optional[List[str]] = None
) -> Dict[str, int]:
    """Cuenta .seg por clase en el banco de anotaciones. No toca nada."""
    root = Path(annotation_root)
    if not root.exists():
        return {}
    wanted = set(class_filter) if class_filter else None
    counts: Dict[str, int] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if wanted is not None and class_dir.name not in wanted:
            continue
        n = len(list(class_dir.glob("*.seg")))
        if n:
            counts[class_dir.name] = n
    return counts


def delete_annotations(
    annotation_root: str,
    class_filter: Optional[List[str]] = None,
    remove_backups: bool = True,
) -> Dict:
    """
    Borra .seg del banco de anotaciones para empezar limpio.

    Args:
        annotation_root: Raíz centralizada (e.g. data/annotations)
        class_filter: Clases a limpiar. None = todas.
        remove_backups: También borra .seg.bak. Un backup cuyo .seg ya no existe
            solo estorba, así que por defecto se va con él.

    Returns:
        Dict con seg_removed, bak_removed, per_class y errors.
    """
    root = Path(annotation_root)
    result = {"seg_removed": 0, "bak_removed": 0, "per_class": {}, "errors": 0}
    if not root.exists():
        logger.info(f"[Limpieza] No existe {root}: nada que borrar")
        return result

    wanted = set(class_filter) if class_filter else None
    alcance = ", ".join(sorted(wanted)) if wanted else "TODAS las clases"
    logger.info(f"[Limpieza] Borrando .seg en {root} | alcance: {alcance}")

    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if wanted is not None and class_dir.name not in wanted:
            continue

        removed = 0
        patterns = ["*.seg", "*.seg.bak"] if remove_backups else ["*.seg"]
        for pattern in patterns:
            for f in sorted(class_dir.glob(pattern)):
                try:
                    f.unlink()
                    if f.name.endswith(".seg.bak"):
                        result["bak_removed"] += 1
                    else:
                        result["seg_removed"] += 1
                        removed += 1
                except OSError as e:
                    result["errors"] += 1
                    logger.error(f"[Limpieza] No se pudo borrar {f.name}: {e}")

        if removed:
            result["per_class"][class_dir.name] = removed
            logger.info(f"[Limpieza] {class_dir.name}: {removed} .seg borrados")

        # Carpeta de clase vacía tras limpiar: se retira para no dejar restos.
        try:
            if not any(class_dir.iterdir()):
                class_dir.rmdir()
        except OSError:
            pass

    logger.info(
        f"[Limpieza] Completado: {result['seg_removed']} .seg y "
        f"{result['bak_removed']} backups borrados, {result['errors']} errores"
    )
    return result


class SegmentationAnnotator:
    """
    Genera archivos .seg para un directorio de imágenes.

    Escanea la estructura ImageFolder (root/class/image.ext),
    detecta granos con U²-Net, y escribe .seg junto a cada imagen.
    """

    def __init__(self, detector: Optional[PollenGrainDetector] = None, config: Optional[Dict] = None):
        """
        Args:
            detector: PollenGrainDetector preconfigurado. Si None, se crea lazy al anotar.
            config: Dict de configuración para crear el detector (si detector es None).
        """
        self._detector = detector
        self._config = config

    @property
    def detector(self) -> PollenGrainDetector:
        """Lazy-load del detector: se crea solo cuando se necesita (no para validar)."""
        if self._detector is None:
            self._detector = PollenGrainDetector(config=self._config)
        return self._detector

    def annotate_directory(
        self,
        root_dir: str,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        class_filter: Optional[List[str]] = None,
        annotation_root: Optional[str] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict:
        """
        Genera .seg para todas las imágenes en un directorio ImageFolder.

        Args:
            root_dir: Ruta raíz (e.g. "data/processed/train")
            overwrite: Si True, regenera .seg existentes
            progress_callback: Callable(current, total, message) para reportar progreso
            class_filter: Lista de nombres de clase a procesar. None = todas.
            cancel_check: Callable() -> bool. Si devuelve True, corta el lote entre
                imágenes y conserva lo ya escrito (stats["cancelled"] = True).

        Returns:
            Dict con estadísticas:
                total_images, annotated, skipped, errors, cancelled,
                total_grains, grains_per_image, grains_per_class, elapsed_ms
        """
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {root}")

        # Descubrir imágenes (filtradas por clase si se especifica)
        image_paths = self._discover_images(root, class_filter=class_filter)
        total = len(image_paths)

        if total == 0:
            logger.warning(f"[Annotator] No se encontraron imágenes en {root}")
            return self._empty_stats()

        clases = sorted({p.parent.name for p in image_paths})
        logger.info(
            f"[Annotator] {total} imágenes en {root} | "
            f"clases ({len(clases)}): {', '.join(clases)} | "
            f"overwrite={overwrite}"
        )
        
        # Configurar directorio de anotaciones
        if annotation_root is not None:
            annot_root = Path(annotation_root)
            annot_root.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Annotator] Anotaciones se guardarán en: {annot_root}")
        else:
            annot_root = None
            logger.info(f"[Annotator] Anotaciones se guardarán junto a las imágenes (legacy mode)")

        stats = {
            "total_images": total,
            "annotated": 0,
            "skipped": 0,
            "errors": 0,
            "total_grains": 0,
            "cancelled": False,
            "grains_per_class": {},
        }

        t_start = time.perf_counter()

        for i, img_path in enumerate(image_paths):
            # Cancelación entre imágenes: nunca deja un .seg a medio escribir.
            if cancel_check is not None and cancel_check():
                stats["cancelled"] = True
                logger.info(f"[Annotator] Cancelado por el usuario en {i}/{total}")
                break

            class_name = img_path.parent.name
            
            # Determinar ruta .seg (centralizada o junto a imagen)
            if annot_root is not None:
                class_annot_dir = annot_root / class_name
                class_annot_dir.mkdir(parents=True, exist_ok=True)
                seg_path = class_annot_dir / f"{img_path.stem}.seg"
            else:
                seg_path = Path(get_seg_path(str(img_path)))

            # Skip si ya existe y no overwrite
            if not overwrite and seg_path.exists():
                stats["skipped"] += 1
                logger.debug(f"[{i + 1}/{total}] {img_path.name}: ya tiene .seg, omitido")
                if progress_callback:
                    progress_callback(i + 1, total, f"[skip] {img_path.name}")
                continue

            # Log when overwriting existing .seg (backup is handled by SegFileWriter)
            if overwrite and seg_path.exists():
                logger.info(f"[{i + 1}/{total}] sobrescribiendo {seg_path.name} (backup .seg.bak)")

            try:
                logger.info(
                    f"[{i + 1}/{total}] {class_name}/{img_path.name} — segmentando..."
                )
                t_img = time.perf_counter()
                n_grains = self.annotate_single(str(img_path), seg_output_path=str(seg_path))
                dt = (time.perf_counter() - t_img) * 1000
                stats["annotated"] += 1
                stats["total_grains"] += n_grains
                stats["grains_per_class"][class_name] = (
                    stats["grains_per_class"].get(class_name, 0) + n_grains
                )
                acumulado = stats["total_grains"]
                logger.info(
                    f"[{i + 1}/{total}] {class_name}/{img_path.name} -> "
                    f"{n_grains} cuerpo{'s' if n_grains != 1 else ''} en {dt:.0f} ms "
                    f"| .seg: {seg_path.name} | acumulado: {acumulado} granos"
                )
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"[{i + 1}/{total}] ERROR en {img_path.name}: {e}")

            if progress_callback:
                progress_callback(i + 1, total, f"{img_path.name}")

        elapsed = (time.perf_counter() - t_start) * 1000
        stats["elapsed_ms"] = elapsed

        processed = stats["annotated"] + stats["skipped"]
        if processed > 0:
            stats["grains_per_image"] = stats["total_grains"] / max(1, stats["annotated"])
        else:
            stats["grains_per_image"] = 0.0

        logger.info(
            f"[Annotator] {'Cancelado' if stats['cancelled'] else 'Completado'}: "
            f"{stats['annotated']} anotados, "
            f"{stats['skipped']} omitidos, {stats['errors']} errores, "
            f"{stats['total_grains']} granos totales ({elapsed:.0f}ms)"
        )

        return stats

    def annotate_single(self, image_path: str, seg_output_path: Optional[str] = None) -> int:
        """
        Genera .seg para una sola imagen.

        Args:
            image_path: Ruta a la imagen
            seg_output_path: Ruta de salida del .seg (opcional).
                           Si es None, usa get_seg_path(image_path).

        Returns:
            Número de granos detectados.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {img_path}")

        # Cargar imagen
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {img_path}")

        h, w = image.shape[:2]

        # Detectar granos (roi_seed = mismo algoritmo que el click de ROI)
        _, grains = self.detector.detect(image)

        # Determinar class_index del directorio padre
        class_name = img_path.parent.name
        # class_index se determina por contexto; default 0
        class_index = 0

        # Convertir DetectedGrain a dicts para el writer
        detections = []
        for grain in grains:
            detections.append({
                "index": grain.index,
                "class_index": class_index,
                "bbox": grain.bbox,
                "saliency": grain.saliency_prob,
                "contour": grain.contour,
            })

        # Escribir .seg
        if seg_output_path is None:
            seg_path = get_seg_path(str(img_path))
        else:
            seg_path = seg_output_path
        
        params = self.detector.get_parameters()

        SegFileWriter.write(
            seg_path=seg_path,
            image_name=img_path.name,
            image_dims=(h, w),
            detections=detections,
            params=params,
        )

        return len(grains)

    def validate_annotations(self, root_dir: str, annotation_root: Optional[str] = None) -> Dict:
        """
        Verifica integridad de pares imagen/.seg en un directorio.

        Args:
            root_dir: Directorio ImageFolder (e.g. data/processed/train)
            annotation_root: Directorio centralizado de anotaciones (e.g. data/annotations)

        Returns:
            Dict con estadísticas de validación
        """
        root = Path(root_dir)
        image_paths = self._discover_images(root)
        if not root.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {root}")
        
        # Determinar dónde buscar .seg
        if annotation_root:
            annot_root = Path(annotation_root)
            logger.info(f"[Validator] Buscando .seg en: {annot_root} (centralizado)")
            seg_paths = set(annot_root.rglob("*.seg"))
        else:
            annot_root = None
            logger.info(f"[Validator] Buscando .seg junto a imágenes (legacy)")
            seg_paths = set(root.rglob("*.seg"))

        valid = 0
        missing_seg = []
        corrupt_seg = []
        orphan_seg = []

        # Verificar que cada imagen tenga .seg
        expected_segs = set()
        for img_path in image_paths:
            # Determinar ubicación esperada del .seg
            if annot_root:
                class_name = img_path.parent.name
                seg = annot_root / class_name / f"{img_path.stem}.seg"
            else:
                seg = img_path.with_suffix(".seg")
            
            expected_segs.add(seg)

            if not seg.exists():
                missing_seg.append(str(img_path))
                continue

            # Verificar parseo
            try:
                from .seg_format import SegFileReader
                result = SegFileReader.read(str(seg))
                if result["grains"] is not None:
                    valid += 1
                else:
                    corrupt_seg.append(str(seg))
            except Exception:
                corrupt_seg.append(str(seg))

        # Encontrar .seg huérfanos
        for seg in seg_paths:
            if seg not in expected_segs:
                orphan_seg.append(str(seg))

        return {
            "total_images": len(image_paths),
            "valid": valid,
            "missing_seg": missing_seg,
            "corrupt_seg": corrupt_seg,
            "orphan_seg": orphan_seg,
        }

    def _discover_images(self, root: Path, class_filter: Optional[List[str]] = None) -> List[Path]:
        """Descubre imágenes en subdirectorios de root, opcionalmente filtradas por clase."""
        images = []
        if class_filter is not None:
            # Solo buscar en las clases especificadas
            for cls_name in class_filter:
                cls_dir = root / cls_name
                if not cls_dir.is_dir():
                    continue
                for ext in IMAGE_EXTENSIONS:
                    images.extend(cls_dir.glob(f"*{ext}"))
        else:
            for ext in IMAGE_EXTENSIONS:
                images.extend(root.rglob(f"*{ext}"))
        # Ordenar para reproducibilidad
        images.sort()
        return images

    @staticmethod
    def _empty_stats() -> Dict:
        return {
            "total_images": 0,
            "annotated": 0,
            "skipped": 0,
            "errors": 0,
            "total_grains": 0,
            "grains_per_image": 0.0,
            "cancelled": False,
            "grains_per_class": {},
            "elapsed_ms": 0.0,
        }
