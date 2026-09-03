"""
seg_format — Parser/Writer del formato de anotación .seg

Formato de archivo .seg:
    Líneas con '#' son comentarios/metadata.
    Datos: grain_index | class_index | bbox_x,bbox_y,bbox_w,bbox_h | saliency | contour_x1,y1,x2,y2,...

Ejemplo:
    # Generado por: PollenGrainDetector (U2-NETP)
    # Imagen: Acaena_sp_00001.png
    # Dimensiones: 1920x1080
    # Threshold: 0.30 | Adaptive_k: 0.30 | Min_area: 200
    0 | 0 | 234,156,87,92 | 0.891 | 234,180,238,168,245,160
    1 | 0 | 512,340,65,71 | 0.847 | 512,360,516,348,524,342
"""

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SegFileWriter:
    """
    Escritor de archivos .seg.
    
    Genera archivos de anotación de segmentación a partir de detecciones
    de PollenGrainDetector. Incluye header con metadata de parámetros
    para reproducibilidad.
    """

    @staticmethod
    def write(
        seg_path: str,
        image_name: str,
        image_dims: Tuple[int, int],
        detections: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Escribe un archivo .seg con las detecciones.

        Args:
            seg_path: Ruta de salida del archivo .seg
            image_name: Nombre del archivo de imagen asociado
            image_dims: (height, width) de la imagen original
            detections: Lista de dicts con keys:
                - index (int)
                - class_index (int)
                - bbox (tuple x,y,w,h)
                - saliency (float)
                - contour (np.ndarray Nx2)
            params: Dict opcional con parámetros de segmentación usados

        Returns:
            Ruta del archivo escrito.
        """
        seg_path = Path(seg_path)
        seg_path.parent.mkdir(parents=True, exist_ok=True)

        # Safety: backup existing .seg before overwriting
        if seg_path.exists():
            bak_path = seg_path.with_suffix(".seg.bak")
            try:
                shutil.copy2(str(seg_path), str(bak_path))
                logger.debug(f"[SegWriter] Backup: {seg_path.name} → {bak_path.name}")
            except Exception as e:
                logger.warning(f"[SegWriter] No se pudo crear backup de {seg_path.name}: {e}")

        h, w = image_dims
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        lines = []

        # Header
        lines.append(f"# Generado por: PollenGrainDetector")
        lines.append(f"# Fecha: {timestamp}")
        lines.append(f"# Imagen: {image_name}")
        lines.append(f"# Dimensiones: {w}x{h}")

        if params:
            param_str = " | ".join(f"{k}: {v}" for k, v in params.items())
            lines.append(f"# Params: {param_str}")

        lines.append("#")
        lines.append(
            "# Formato: grain_index | class_index | bbox_x,bbox_y,bbox_w,bbox_h"
            " | saliency | contour_x1,y1,x2,y2,..."
        )

        # Data lines
        for det in detections:
            idx = det["index"]
            cls = det["class_index"]
            bx, by, bw, bh = det["bbox"]
            sal = det["saliency"]
            contour = det["contour"]

            bbox_str = f"{bx},{by},{bw},{bh}"
            sal_str = f"{sal:.3f}"

            if contour is not None and len(contour) > 0:
                pts = contour.reshape(-1)
                contour_str = ",".join(str(int(v)) for v in pts)
            else:
                contour_str = ""

            lines.append(f"{idx} | {cls} | {bbox_str} | {sal_str} | {contour_str}")

        with open(seg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logger.debug(f"[SegWriter] {seg_path.name}: {len(detections)} granos")
        return str(seg_path)


class SegFileReader:
    """
    Lector y parser de archivos .seg.
    
    Retorna detecciones parseadas y metadata del header.
    """

    @staticmethod
    def read(seg_path: str) -> Dict[str, Any]:
        """
        Lee y parsea un archivo .seg completo.

        Args:
            seg_path: Ruta al archivo .seg

        Returns:
            Dict con:
                - 'metadata': dict con info del header
                - 'grains': lista de dicts con detecciones parseadas
        """
        seg_path = Path(seg_path)
        if not seg_path.exists():
            raise FileNotFoundError(f"Archivo .seg no encontrado: {seg_path}")

        metadata = {}
        grains = []

        with open(seg_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("#"):
                    SegFileReader._parse_header_line(line, metadata)
                    continue

                grain = SegFileReader._parse_data_line(line)
                if grain is not None:
                    grains.append(grain)

        return {"metadata": metadata, "grains": grains}

    @staticmethod
    def read_grains(seg_path: str) -> List[Dict[str, Any]]:
        """
        Lee solo las detecciones de un .seg (sin metadata).

        Args:
            seg_path: Ruta al archivo .seg

        Returns:
            Lista de dicts con keys: index, class_index, bbox, saliency, contour
        """
        result = SegFileReader.read(seg_path)
        return result["grains"]

    @staticmethod
    def count_grains(seg_path: str) -> int:
        """Retorna el número de granos en un .seg sin parsear contornos completos."""
        count = 0
        with open(seg_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    count += 1
        return count

    @staticmethod
    def _parse_header_line(line: str, metadata: dict):
        """Extrae metadata de una línea de header."""
        content = line.lstrip("# ").strip()
        if not content:
            return

        if content.startswith("Formato:"):
            return

        if ":" in content:
            key, _, value = content.partition(":")
            key = key.strip().lower()
            value = value.strip()

            if key == "imagen":
                metadata["image_name"] = value
            elif key == "dimensiones":
                metadata["dimensions"] = value
            elif key == "fecha":
                metadata["timestamp"] = value
            elif key == "params":
                params = {}
                for part in value.split("|"):
                    part = part.strip()
                    if ":" in part:
                        pk, _, pv = part.partition(":")
                        params[pk.strip()] = pv.strip()
                metadata["params"] = params

    @staticmethod
    def _parse_data_line(line: str) -> Optional[Dict[str, Any]]:
        """Parsea una línea de datos del .seg."""
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            logger.warning(f"[SegReader] Línea malformada (< 4 campos): {line[:80]}")
            return None

        try:
            grain_index = int(parts[0])
            class_index = int(parts[1])

            bbox_vals = [int(v) for v in parts[2].split(",")]
            if len(bbox_vals) != 4:
                logger.warning(f"[SegReader] BBox inválido: {parts[2]}")
                return None
            bbox = tuple(bbox_vals)

            saliency = float(parts[3])

            contour = None
            if len(parts) >= 5 and parts[4]:
                contour_vals = [int(v) for v in parts[4].split(",")]
                if len(contour_vals) >= 4 and len(contour_vals) % 2 == 0:
                    contour = np.array(contour_vals, dtype=np.int32).reshape(-1, 2)

            return {
                "index": grain_index,
                "class_index": class_index,
                "bbox": bbox,
                "saliency": saliency,
                "contour": contour,
            }

        except (ValueError, IndexError) as e:
            logger.warning(f"[SegReader] Error parseando línea: {e} — {line[:80]}")
            return None


def get_seg_path(image_path: str) -> str:
    """Retorna la ruta .seg correspondiente a una imagen."""
    return str(Path(image_path).with_suffix(".seg"))


def has_seg_file(image_path: str) -> bool:
    """Verifica si existe un archivo .seg para la imagen dada."""
    return Path(image_path).with_suffix(".seg").exists()
