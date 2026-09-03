"""
Utilidades para construir ROI manual por polígonos include/exclude.

Este módulo encapsula lógica pura (sin dependencias de GUI) para:
- rasterizar polígonos de inclusión y exclusión;
- construir máscara final include - exclude;
- extraer contornos resultantes para persistencia en .seg.
"""

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
Polygon = Sequence[Point]


def _as_contour(points: Polygon) -> np.ndarray | None:
    """Convierte una lista de puntos a contorno OpenCV Nx1x2 int32."""
    if points is None or len(points) < 3:
        return None
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    return pts


def rasterize_polygons(
    image_shape: Tuple[int, int],
    polygons: Iterable[Polygon],
) -> np.ndarray:
    """
    Rasteriza polígonos en una máscara binaria uint8 (0/255).

    Args:
        image_shape: (h, w) de la imagen.
        polygons: iterable de polígonos (cada uno: lista de (x, y)).
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    contours = []
    for poly in polygons:
        contour = _as_contour(poly)
        if contour is not None:
            contours.append(contour)
    if contours:
        cv2.fillPoly(mask, contours, 255)
    return mask


def build_manual_roi_mask(
    image_shape: Tuple[int, int],
    include_polygons: Iterable[Polygon],
    exclude_polygons: Iterable[Polygon],
) -> np.ndarray:
    """
    Construye la máscara final ROI como include - exclude.

    Returns:
        mask_final uint8 (0/255)
    """
    include_mask = rasterize_polygons(image_shape, include_polygons)
    exclude_mask = rasterize_polygons(image_shape, exclude_polygons)
    # include AND NOT exclude
    keep_mask = cv2.bitwise_and(include_mask, cv2.bitwise_not(exclude_mask))
    return keep_mask


def extract_contours_from_mask(
    mask: np.ndarray,
    min_area: float = 1.0,
) -> List[np.ndarray]:
    """
    Extrae contornos externos desde máscara binaria.

    Usa CHAIN_APPROX_NONE para conservar detalle de borde.
    Retorna lista de contornos Nx2 ordenados por área descendente.
    """
    if mask is None or mask.size == 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    parsed = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            parsed.append(cnt.reshape(-1, 2))
    parsed.sort(key=lambda c: cv2.contourArea(c.reshape(-1, 1, 2)), reverse=True)
    return parsed


def contour_bbox(contour: np.ndarray) -> Tuple[int, int, int, int]:
    """Calcula bbox (x, y, w, h) para contorno Nx2."""
    if contour is None or len(contour) < 3:
        return (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(contour.reshape(-1, 1, 2).astype(np.int32))
    return (int(x), int(y), int(w), int(h))

