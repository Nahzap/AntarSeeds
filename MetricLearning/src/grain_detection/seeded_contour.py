"""
Buscador canónico de objetos: contorno sembrado (ROI).

Click manual, preview («Probar aquí») y lote («Segmentar») usan el mismo
camino: ``SeededParams.from_config`` → ``resolve_seeded_object`` /
``propose_objects``. No hay rama de demostración ni segundo detector.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .saliency_grain_extraction import bbox_iou

# Proveedor de saliencia para un recorte: (imagen completa, (x1, y1, x2, y2)) -> mapa
# del tamaño del recorte. Permite que el click use U²-Net sobre el recorte y que el
# lote reuse un mapa ya calculado, sin duplicar el resto del algoritmo.
SaliencyProvider = Callable[[np.ndarray, Tuple[int, int, int, int]], np.ndarray]

logger = logging.getLogger(__name__)


@dataclass
class SeededParams:
    min_area: int = 800
    max_area: int = 800000
    max_area_frac: float = 0.50
    min_circularity: float = 0.12
    max_bbox_side_frac: float = 0.85
    max_aspect_ratio: float = 8.0
    morph_kernel_size: int = 3
    saliency_threshold: float = 0.30
    adaptive_k: float = 0.20
    crop_radius: int = 300
    max_crop_fill: float = 0.70
    peak_max: int = 48
    # Semillas en contacto: un componente conectado puede contener varios cuerpos.
    split_touching: bool = True
    seed_core_frac: float = 0.45
    # Cintura: un corte solo es real si el istmo es más delgado que los cuerpos.
    waist_frac: float = 0.80
    # Bordes: un contorno cortado por el recorte tiene un lado recto artificial.
    drop_border_objects: bool = False

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]] = None) -> "SeededParams":
        """Única fábrica de parámetros: click, preview y lote leen el mismo dict."""
        c = cfg or {}
        return cls(
            min_area=int(c.get("min_area", 800)),
            max_area=int(c.get("max_area", 800000)),
            max_area_frac=float(c.get("max_area_frac", 0.50)),
            min_circularity=float(c.get("min_circularity", 0.12)),
            max_bbox_side_frac=float(c.get("max_bbox_side_frac", 0.85)),
            max_aspect_ratio=float(c.get("max_aspect_ratio", 8.0)),
            morph_kernel_size=int(c.get("morph_kernel_size", 3)),
            saliency_threshold=float(c.get("saliency_threshold", 0.30)),
            adaptive_k=float(c.get("adaptive_k", 0.20)),
            crop_radius=int(c.get("crop_radius", 300)),
            max_crop_fill=float(c.get("max_crop_fill", 0.70)),
            peak_max=int(c.get("peak_max", 48)),
            split_touching=bool(c.get("split_touching", True)),
            seed_core_frac=float(c.get("seed_core_frac", 0.45)),
            waist_frac=float(c.get("waist_frac", 0.80)),
            drop_border_objects=bool(c.get("drop_border_objects", False)),
        )


@dataclass(frozen=True)
class ObjectBackground:
    """Medición objeto/fondo del recorte. Un umbral; click y lote leen el mismo."""

    seed_sal: float
    object_sal: float
    background_sal: float
    gap: float
    threshold: float
    object_L: Optional[float]
    background_L: Optional[float]
    L_gap: Optional[float]
    separated: bool
    reason: str

    def describe(self) -> str:
        L = ""
        if self.L_gap is not None:
            L = (
                f" | L objeto {self.object_L:.1f} fondo {self.background_L:.1f} "
                f"ΔL {self.L_gap:.1f}"
            )
        estado = "separado" if self.separated else "SIN SEPARACIÓN"
        extra = f" — {self.reason}" if self.reason else ""
        return (
            f"{estado}: sal objeto {self.object_sal:.3f} fondo {self.background_sal:.3f} "
            f"hueco {self.gap:.3f} umbral {self.threshold:.3f}{L}{extra}"
        )

    @property
    def seed_on_object(self) -> bool:
        """La semilla está del lado alto del mapa: se puede cerrar el contorno."""
        if self.seed_sal <= 0:
            return False
        return self.object_sal >= self.background_sal and self.seed_sal >= self.threshold


def compare_object_background(
    saliency: np.ndarray,
    seed: Tuple[int, int],
    image_bgr: Optional[np.ndarray] = None,
) -> ObjectBackground:
    """
    Objeto = componente de saliencia alta que contiene la semilla.
    Fondo = el resto del recorte. El umbral es el punto medio de las medianas.
    """
    if saliency is None or saliency.size == 0:
        return ObjectBackground(
            0.0, 0.0, 0.0, 0.0, 0.5, None, None, None, False, "saliencia vacía"
        )

    h, w = saliency.shape[:2]
    sx = int(np.clip(seed[0], 0, w - 1))
    sy = int(np.clip(seed[1], 0, h - 1))
    seed_sal = float(saliency[sy, sx])
    crop_med = float(np.median(saliency))
    mid = 0.5 * (seed_sal + crop_med)

    n_lbl, labels = cv2.connectedComponents((saliency >= mid).astype(np.uint8), 8)
    seed_lbl = int(labels[sy, sx]) if n_lbl else 0
    if seed_lbl <= 0:
        core = np.zeros((h, w), dtype=bool)
        core[sy, sx] = True
    else:
        core = labels == seed_lbl
    bg = ~core

    object_vals = saliency[core]
    bg_vals = saliency[bg]
    object_sal = float(np.median(object_vals)) if object_vals.size else seed_sal
    background_sal = float(np.median(bg_vals)) if bg_vals.size else crop_med
    gap = object_sal - background_sal
    threshold = float(np.clip(0.5 * (object_sal + background_sal), 0.0, 1.0))
    spread = (
        (float(np.std(object_vals)) if object_vals.size else 0.0)
        + (float(np.std(bg_vals)) if bg_vals.size else 0.0)
    )

    object_L = background_L = L_gap = None
    if image_bgr is not None and image_bgr.size and image_bgr.shape[:2] == (h, w):
        lum = (
            image_bgr.astype(np.float32)
            if image_bgr.ndim == 2
            else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        )
        oL, bL = lum[core], lum[bg]
        if oL.size and bL.size:
            object_L = float(np.median(oL))
            background_L = float(np.median(bL))
            L_gap = abs(object_L - background_L)

    if not np.any(bg):
        reason = "el objeto llena el recorte (no hay fondo)"
        separated = False
    elif gap <= 0:
        reason = "el fondo es igual o más saliente que el objeto"
        separated = False
    elif gap <= spread:
        reason = f"hueco {gap:.3f} ≤ dispersión {spread:.3f}"
        separated = False
    else:
        reason = ""
        separated = True

    return ObjectBackground(
        seed_sal=seed_sal,
        object_sal=object_sal,
        background_sal=background_sal,
        gap=gap,
        threshold=threshold,
        object_L=object_L,
        background_L=background_L,
        L_gap=L_gap,
        separated=separated,
        reason=reason,
    )


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Rellena huecos. El borde de 1 px evita invertir el encuadre si el objeto toca (0,0)."""
    if mask is None or mask.size == 0:
        return mask
    m = mask.copy().astype(np.uint8)
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(ff, flood_mask, (0, 0), 255)
    filled = cv2.bitwise_or(padded, cv2.bitwise_not(ff))
    return filled[1:-1, 1:-1]


def covers_full_frame(contour_xy: np.ndarray, image_h: int, image_w: int) -> bool:
    """El encuadre entero no es un objeto."""
    if contour_xy is None or len(contour_xy) < 3:
        return False
    _x, _y, bw, bh = cv2.boundingRect(contour_xy.reshape(-1, 1, 2).astype(np.int32))
    return bw >= image_w and bh >= image_h


def green_area(contour_xy: np.ndarray) -> float:
    if contour_xy is None or len(contour_xy) < 3:
        return 0.0
    pts = contour_xy.astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def single_external_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    if mask is None or mask.size == 0:
        return None
    filled = fill_holes(mask)
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    best = max(contours, key=lambda c: abs(green_area(c.reshape(-1, 2))))
    best = best.reshape(-1, 2).astype(np.int32)
    peri = cv2.arcLength(best.reshape(-1, 1, 2), True)
    # En cuerpos grandes 0.0025*perímetro borra el perfil (T_officinale ~20 px).
    eps = max(0.6, min(2.5, 0.0007 * peri))
    approx = cv2.approxPolyDP(best.reshape(-1, 1, 2), eps, True)
    return approx.reshape(-1, 2).astype(np.int32)


def distance_peaks(mask: np.ndarray, params: SeededParams) -> np.ndarray:
    """
    Máximos locales de la transformada de distancia: un núcleo por cuerpo.

    Un umbral global sobre la distancia no sirve: en dos semillas que se solapan,
    el cuello puede quedar por encima del umbral y los núcleos siguen unidos
    (con solape de 20 px el cuello está al 51 % del máximo). Los máximos locales
    sí distinguen los dos centros, que es donde un humano haría click.

    ``seed_core_frac`` actúa como prominencia mínima: descarta picos poco
    profundos (rebabas del borde) frente al punto más interior del componente.
    """
    empty = np.zeros(mask.shape[:2], dtype=np.uint8)
    if mask is None or np.count_nonzero(mask) == 0:
        return empty

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dmax = float(dist.max())
    if dmax <= 0.0:
        return empty

    # Ventana de supresión ligada al grosor del cuerpo: dos centros separados
    # por más de este ancho sobreviven como picos distintos.
    side = int(min(mask.shape[:2]))
    win = max(3, int(round(dmax * 0.8)) | 1)
    win = min(win, side if side % 2 else max(3, side - 1))
    local_max = cv2.dilate(dist, np.ones((win, win), np.uint8))

    frac = float(np.clip(params.seed_core_frac, 0.10, 0.90))
    peaks = (
        (dist >= local_max - 1e-3) & (dist >= frac * dmax) & (mask > 0)
    ).astype(np.uint8) * 255
    # Los picos son mesetas finas; se engordan para que watershed tenga marcador.
    return cv2.dilate(peaks, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))


def _deepest_point(dist: np.ndarray, region: np.ndarray) -> Tuple[int, int]:
    """Punto más interior de la región: donde un humano haría click."""
    masked = np.where(region, dist, -1.0)
    idx = int(np.argmax(masked))
    y, x = np.unravel_index(idx, masked.shape)
    return int(x), int(y)


def distance_cores(
    component_mask: np.ndarray, params: SeededParams
) -> List[Tuple[int, int, int]]:
    """
    Núcleos de un componente conectado: [(área, cx, cy), ...].

    Si los picos sugieren varios cuerpos, el corte se valida con la cintura; los
    cortes espurios se descartan y el componente vuelve a ser un solo cuerpo.
    """
    if component_mask is None or np.count_nonzero(component_mask) == 0:
        return []

    peaks = distance_peaks(component_mask, params)
    if np.count_nonzero(peaks) == 0:
        return []

    dist = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
    n_cores, _labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, 8)
    crudos: List[Tuple[int, int, int]] = []
    for lbl in range(1, n_cores):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < 4:
            continue
        cx, cy = centroids[lbl]
        crudos.append((area, int(round(cx)), int(round(cy))))

    if len(crudos) <= 1:
        return crudos

    markers = watershed_regions(component_mask, params)
    if markers is None:
        # Todos los cortes eran espurios: un cuerpo, semilla en el punto más interior.
        x, y = _deepest_point(dist, component_mask > 0)
        return [(int(np.count_nonzero(component_mask)), x, y)]

    out: List[Tuple[int, int, int]] = []
    for lbl in sorted({int(v) for v in np.unique(markers) if v > 1}):
        region = (markers == lbl) & (component_mask > 0)
        area = int(np.count_nonzero(region))
        if area < 4:
            continue
        x, y = _deepest_point(dist, region)
        out.append((area, x, y))
    return out


def merge_shallow_splits(
    markers: np.ndarray, binary: np.ndarray, dist: np.ndarray, params: SeededParams
) -> np.ndarray:
    """
    Deshace cortes sin cintura real.

    Un corte entre dos cuerpos solo es legítimo si el istmo es claramente más
    delgado que ellos. Se compara la altura del punto de silla de la
    transformada de distancia en la frontera con el pico de cada región:

        silla <= waist_frac * min(pico_a, pico_b)  ->  corte real

    Dos semillas tangentes dan silla/pico ~0.5. Una semilla alargada con un
    máximo doble espurio da ~0.95, y ese es el corte recto que atravesaba un
    cuerpo entero.
    """
    labels = sorted({int(v) for v in np.unique(markers) if v > 1})
    if len(labels) < 2:
        return markers

    peak = {}
    for lbl in labels:
        vals = dist[markers == lbl]
        peak[lbl] = float(vals.max()) if vals.size else 0.0

    parent = {lbl: lbl for lbl in labels}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grown = {
        lbl: cv2.dilate((markers == lbl).astype(np.uint8), kernel, iterations=2)
        for lbl in labels
    }
    inside = binary > 0
    waist = float(np.clip(params.waist_frac, 0.05, 0.99))

    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            frontier = (grown[a] > 0) & (grown[b] > 0) & inside
            if not frontier.any():
                continue
            saddle = float(dist[frontier].max())
            limit = waist * min(peak[a], peak[b])
            if saddle > limit:
                logger.debug(
                    "    corte %d|%d deshecho: cintura %.1f > %.1f "
                    "(picos %.1f/%.1f) — sin istmo real",
                    a, b, saddle, limit, peak[a], peak[b],
                )
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    out = markers.copy()
    for lbl in labels:
        root = find(lbl)
        if root != lbl:
            out[markers == lbl] = root
    return out


def watershed_regions(
    binary: np.ndarray, params: SeededParams
) -> Optional[np.ndarray]:
    """
    Separa cuerpos en contacto dentro de una máscara binaria.

    Devuelve el mapa de etiquetas de watershed, o None si no queda nada que
    separar. Los marcadores son los máximos de distancia y los cortes sin
    cintura real se deshacen antes de devolver el resultado.
    """
    if binary is None or np.count_nonzero(binary) == 0:
        return None

    sure_fg = distance_peaks(binary, params)
    if np.count_nonzero(sure_fg) == 0:
        return None

    n_markers, markers = cv2.connectedComponents(sure_fg)
    if n_markers <= 2:
        return None  # un único núcleo: no hay contacto que deshacer

    markers = markers.astype(np.int32) + 1
    unknown = cv2.subtract(binary, sure_fg)
    markers[unknown > 0] = 0

    # cv2.watershed exige 3 canales; el gradiente de la propia máscara basta
    # porque la frontera a cortar es el istmo entre los dos cuerpos.
    guide = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.watershed(guide, markers)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    markers = merge_shallow_splits(markers, binary, dist, params)
    if len({int(v) for v in np.unique(markers) if v > 1}) < 2:
        return None  # todos los cortes eran espurios
    return markers


def region_containing(
    markers: np.ndarray, binary: np.ndarray, sx: int, sy: int
) -> Optional[np.ndarray]:
    """Máscara de la región de watershed que contiene (sx, sy)."""
    if markers is None:
        return None
    h, w = markers.shape[:2]
    sx = int(np.clip(sx, 0, w - 1))
    sy = int(np.clip(sy, 0, h - 1))

    label = int(markers[sy, sx])
    if label <= 1:
        # La semilla cayó en la frontera o en el fondo: se busca la etiqueta
        # válida más cercana dentro de la máscara.
        ys, xs = np.where((markers > 1) & (binary > 0))
        if len(xs) == 0:
            return None
        k = int(np.argmin((xs - sx) ** 2 + (ys - sy) ** 2))
        label = int(markers[ys[k], xs[k]])

    region = ((markers == label) & (binary > 0)).astype(np.uint8) * 255
    if np.count_nonzero(region) == 0:
        return None
    return region


def is_valid_seeded_contour(
    contour_xy: np.ndarray,
    h: int,
    w: int,
    sx: int,
    sy: int,
    max_ratio: float,
    reject_frame: bool = True,
) -> bool:
    if contour_xy is None or len(contour_xy) < 3:
        return False
    cnt = contour_xy.reshape(-1, 1, 2).astype(np.int32)
    area = abs(cv2.contourArea(cnt))
    ratio = float(area) / float(max(1, h * w))
    if ratio > max_ratio:
        return False
    if reject_frame and covers_full_frame(contour_xy, h, w):
        return False
    inside = cv2.pointPolygonTest(cnt, (float(sx), float(sy)), True)
    return inside >= -6.0


def circularity(contour_xy: np.ndarray) -> float:
    cnt = contour_xy.reshape(-1, 1, 2).astype(np.int32)
    area = abs(cv2.contourArea(cnt))
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def validity_reason(
    contour_xy: np.ndarray, image_h: int, image_w: int, params: SeededParams
) -> Optional[str]:
    """
    None si el contorno es un cuerpo válido; si no, qué filtro lo descartó.

    Devolver el motivo permite que la terminal explique por qué se perdió un
    candidato en vez de solo decir que no encontró nada.
    """
    if contour_xy is None or len(contour_xy) < 3:
        return "contorno vacío"

    cnt = contour_xy.reshape(-1, 1, 2).astype(np.int32)
    area = abs(cv2.contourArea(cnt))
    area_cap = min(params.max_area, int(params.max_area_frac * image_h * image_w))
    if area < params.min_area:
        return f"área {area:.0f} < min_area {params.min_area}"
    if area > area_cap:
        return f"área {area:.0f} > tope {area_cap} (max_area/max_area_frac)"

    x, y, bw, bh = cv2.boundingRect(cnt)
    side = max(bw, bh)
    side_cap = params.max_bbox_side_frac * min(image_h, image_w)
    if side > side_cap:
        return f"lado {side} > {side_cap:.0f} (max_bbox_side_frac)"

    short = max(1, min(bw, bh))
    aspect = side / short
    if aspect > params.max_aspect_ratio:
        return f"aspecto {aspect:.2f} > {params.max_aspect_ratio} (max_aspect_ratio)"

    circ = circularity(contour_xy)
    if circ < params.min_circularity:
        return f"circularidad {circ:.3f} < {params.min_circularity} (min_circularity)"

    return None


def passes_validity_filters(
    contour_xy: np.ndarray, image_h: int, image_w: int, params: SeededParams
) -> bool:
    """Filtros de cuerpo válido en el marco completo (click y lote)."""
    return validity_reason(contour_xy, image_h, image_w, params) is None


def u2net_seeded_contour(
    saliency: np.ndarray,
    seed_x: int,
    seed_y: int,
    params: Optional[SeededParams] = None,
    image_bgr: Optional[np.ndarray] = None,
    separation: Optional[ObjectBackground] = None,
    seeds: Optional[Sequence[Tuple[int, int]]] = None,
    allow_crop_fill: bool = False,
) -> Optional[np.ndarray]:
    """Contorno de mayor puntaje U²-Net que cubre la semilla (o el trazo)."""
    params = params or SeededParams()
    if saliency is None or saliency.size == 0:
        return None

    h, w = saliency.shape[:2]
    pts = [(int(x), int(y)) for x, y in (seeds or ((seed_x, seed_y),))]
    if not pts:
        pts = [(int(seed_x), int(seed_y))]
    sx = int(np.clip(pts[0][0], 0, w - 1))
    sy = int(np.clip(pts[0][1], 0, h - 1))
    cmp = separation if separation is not None else compare_object_background(
        saliency, (sx, sy), image_bgr
    )
    if not cmp.seed_on_object:
        logger.info("  objeto/fondo: %s", cmp.describe())
        return None
    if not cmp.separated:
        logger.info("  objeto/fondo: %s (cierro por puntaje)", cmp.describe())

    min_area = 1
    max_ratio = 1.0 if allow_crop_fill else float(np.clip(params.max_crop_fill, 0.0, 1.0))
    max_area = max(min_area + 1, int(max_ratio * h * w))
    k = max(1, params.morph_kernel_size | 1)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = (saliency >= cmp.threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_k)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    if num_labels <= 1:
        return None

    hit_labels = []
    for px, py in pts:
        lx = int(np.clip(px, 0, w - 1))
        ly = int(np.clip(py, 0, h - 1))
        lbl = int(labels[ly, lx])
        if lbl > 0:
            hit_labels.append(lbl)
    if not hit_labels:
        return None

    def _score(lbl: int) -> float:
        sel = labels == lbl
        return float(np.mean(saliency[sel])) if np.any(sel) else -1.0

    best_lbl = max(set(hit_labels), key=_score)
    for px, py in pts:
        lx = int(np.clip(px, 0, w - 1))
        ly = int(np.clip(py, 0, h - 1))
        if int(labels[ly, lx]) == best_lbl:
            sx, sy = lx, ly
            break

    def contour_from_label(lbl: int) -> Optional[np.ndarray]:
        if lbl <= 0:
            return None
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            return None
        comp = (labels == lbl).astype(np.uint8) * 255

        if params.split_touching:
            markers = watershed_regions(comp, params)
            region = region_containing(markers, comp, sx, sy) if markers is not None else None
            if region is not None:
                r_area = int(np.count_nonzero(region))
                if r_area >= min_area:
                    logger.debug(
                        "    componente %d separado por watershed: %d -> %d px",
                        lbl, area, r_area,
                    )
                    comp = region

        # El borde es el isocontorno del umbral objeto/fondo (sin dilatación).
        # Un recorte local lleno es textura interior; el encuadre completo no es objeto.
        if not allow_crop_fill and int(np.count_nonzero(comp)) >= h * w:
            logger.info("  componente = encuadre (%d px) — no es un objeto", h * w)
            return None
        comp = fill_holes(comp)
        if not allow_crop_fill and int(np.count_nonzero(comp)) >= h * w:
            logger.info("  fill_holes = encuadre (%d px) — no es un objeto", h * w)
            return None
        cnt = single_external_contour(comp)
        if cnt is None or len(cnt) < 3:
            return None
        if not is_valid_seeded_contour(
            cnt, h, w, sx, sy, max_ratio=max_ratio, reject_frame=not allow_crop_fill
        ):
            return None
        return cnt

    return contour_from_label(best_lbl)


def peak_seeds(saliency: np.ndarray, params: SeededParams) -> List[Tuple[int, int]]:
    """
    Clicks automáticos: una semilla por cuerpo, no una por componente.

    El centroide de un componente con dos semillas en contacto cae en la unión y
    produce un cuerpo fusionado. Con ``split_touching`` cada componente se abre
    en núcleos de distancia, así que la semilla queda en el centro de cada
    cuerpo — donde un humano haría click.
    """
    if saliency is None or saliency.size == 0:
        return []
    h, w = saliency.shape[:2]
    mean_val = float(np.mean(saliency))
    std_val = float(np.std(saliency))
    thr = max(params.saliency_threshold, mean_val + params.adaptive_k * std_val)
    binary = (saliency >= thr).astype(np.uint8) * 255
    k = max(1, params.morph_kernel_size | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    if num_labels <= 1:
        logger.info(
            "  0 semillas: la saliencia no supera %.2f en ningún punto "
            "(threshold %.2f + k %.2f). Baja Threshold o Adaptive k.",
            thr, params.saliency_threshold, params.adaptive_k,
        )
        return []

    seeds: List[Tuple[int, int, int]] = []
    n_split = 0
    n_small = 0
    biggest_small = 0
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < params.min_area:
            n_small += 1
            biggest_small = max(biggest_small, area)
            continue

        if params.split_touching:
            comp = (labels == lbl).astype(np.uint8) * 255
            cores = distance_cores(comp, params)
            if len(cores) > 1:
                n_split += 1
                logger.debug(
                    "  componente %d (%d px): %d cuerpos en contacto",
                    lbl, area, len(cores),
                )
                # El área del núcleo solo ordena; el peso del cuerpo es el componente.
                seeds.extend((area, cx, cy) for _, cx, cy in cores)
                continue

        cx, cy = int(round(centroids[lbl][0])), int(round(centroids[lbl][1]))
        seeds.append((area, cx, cy))

    seeds.sort(reverse=True)
    if n_split:
        logger.debug("  %d componente(s) abiertos en cuerpos separados", n_split)

    if not seeds and n_small:
        # El caso que deja el lote en cero al cambiar de especie: el mínimo de
        # área viene calibrado para semillas más grandes.
        logger.info(
            "  0 semillas: %d cuerpo(s) descartados por Min área. El mayor mide "
            "%d px² y Min área está en %d px². Baja Min área a ~%d px².",
            n_small, biggest_small, params.min_area, int(biggest_small * 0.5),
        )
    elif n_small:
        logger.debug(
            "  %d cuerpo(s) por debajo de Min área (mayor %d px²)", n_small, biggest_small
        )
    return [(cx, cy) for _, cx, cy in seeds[: params.peak_max]]


def measure_bodies(saliency: np.ndarray, params: SeededParams) -> List[int]:
    """
    Áreas de los cuerpos del encuadre, **sin** aplicar los filtros de área.

    Sirve para calibrar: dice de qué tamaño son las semillas de esta especie,
    aunque los topes actuales las estén descartando todas.
    """
    if saliency is None or saliency.size == 0:
        return []

    h, w = saliency.shape[:2]
    mean_val = float(np.mean(saliency))
    std_val = float(np.std(saliency))
    thr = max(params.saliency_threshold, mean_val + params.adaptive_k * std_val)
    binary = (saliency >= thr).astype(np.uint8) * 255
    k = max(1, params.morph_kernel_size | 1)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )

    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(binary, 8)
    # Piso mínimo para ignorar ruido de un puñado de píxeles.
    floor = max(50, int(0.00002 * h * w))
    areas = [
        int(stats[lbl, cv2.CC_STAT_AREA])
        for lbl in range(1, n)
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= floor
    ]
    return sorted(areas)


def suggest_area_bounds(areas: Sequence[int]) -> Optional[Dict[str, int]]:
    """
    Propone Min/Max área a partir de los cuerpos medidos en el encuadre.

    Margen amplio a propósito: el objetivo es dejar de descartar la especie,
    no ajustar al píxel. El usuario afina después con la prueba en pantalla.
    """
    valid = [int(a) for a in areas if a > 0]
    if not valid:
        return None
    smallest, largest = min(valid), max(valid)
    return {
        "min_area": max(50, int(smallest * 0.5)),
        "max_area": int(largest * 2.5),
        "measured": len(valid),
        "smallest": smallest,
        "largest": largest,
    }


def _crop_box(cx: int, cy: int, r: int, h: int, w: int) -> Tuple[int, int, int, int]:
    x1 = max(0, cx - r)
    y1 = max(0, cy - r)
    x2 = min(w, cx + r)
    y2 = min(h, cy + r)
    return x1, y1, x2, y2


def _seeds_crop_box(
    seeds: Sequence[Tuple[int, int]], r: int, h: int, w: int
) -> Tuple[int, int, int, int]:
    xs = [int(s[0]) for s in seeds]
    ys = [int(s[1]) for s in seeds]
    return (
        max(0, min(xs) - r),
        max(0, min(ys) - r),
        min(w, max(xs) + r),
        min(h, max(ys) + r),
    )


def contour_touches_border(
    contour_local: np.ndarray, h: int, w: int, margin: int = 2
) -> bool:
    """El contorno llega al borde del recorte: probablemente está truncado."""
    if contour_local is None or len(contour_local) < 3:
        return True
    x = contour_local[:, 0]
    y = contour_local[:, 1]
    return bool(
        np.any(x <= margin)
        or np.any(y <= margin)
        or np.any(x >= (w - 1 - margin))
        or np.any(y >= (h - 1 - margin))
    )


def touches_interior_crop_border(
    contour_local: np.ndarray,
    box: Tuple[int, int, int, int],
    image_h: int,
    image_w: int,
    margin: int = 2,
) -> bool:
    """
    El contorno se apoya en un lado del recorte que NO es borde de la imagen.

    Ese lado es una línea recta inventada por el recorte, no el perfil del
    objeto. Si el lado coincide con el borde del encuadre, el corte es real
    (el cuerpo sale de la imagen) y no se considera artificio.
    """
    if contour_local is None or len(contour_local) < 3:
        return False
    x1, y1, x2, y2 = box
    cw, ch = x2 - x1, y2 - y1
    xs, ys = contour_local[:, 0], contour_local[:, 1]

    if x1 > 0 and np.any(xs <= margin):
        return True
    if y1 > 0 and np.any(ys <= margin):
        return True
    if x2 < image_w and np.any(xs >= cw - 1 - margin):
        return True
    if y2 < image_h and np.any(ys >= ch - 1 - margin):
        return True
    return False


def touches_image_border(
    contour_xy: np.ndarray, image_h: int, image_w: int, margin: int = 2
) -> bool:
    """El cuerpo sale del encuadre: es una semilla parcial."""
    if contour_xy is None or len(contour_xy) < 3:
        return False
    xs, ys = contour_xy[:, 0], contour_xy[:, 1]
    return bool(
        np.any(xs <= margin)
        or np.any(ys <= margin)
        or np.any(xs >= image_w - 1 - margin)
        or np.any(ys >= image_h - 1 - margin)
    )


def memoized_provider(provider: SaliencyProvider) -> SaliencyProvider:
    """Una caja = una inferencia. El encuadre completo se reutiliza entre semillas."""
    cache: Dict[Tuple[int, int, int, int], Optional[np.ndarray]] = {}

    def _cached(
        image_bgr: np.ndarray, box: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        key = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        if key not in cache:
            cache[key] = provider(image_bgr, box)
        return cache[key]

    return _cached


def crop_saliency_provider(saliency_full: np.ndarray) -> SaliencyProvider:
    """Provider que recorta un mapa ya calculado (sin volver a ejecutar la red)."""

    def provider(_image_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return saliency_full[y1:y2, x1:x2]

    return provider


def contour_to_grain_dict(
    contour_xy: np.ndarray,
    saliency: np.ndarray,
    index: int = 0,
    origin: Tuple[int, int] = (0, 0),
) -> Dict[str, Any]:
    """
    Grano en coordenadas de ``contour_xy``.

    ``origin`` es el desplazamiento del mapa de saliencia respecto al contorno:
    (0, 0) si el mapa cubre la imagen completa, (x1, y1) si es un recorte.
    """
    cnt = contour_xy.reshape(-1, 1, 2).astype(np.int32)
    bx, by, bw, bh = cv2.boundingRect(cnt)
    local = contour_xy.astype(np.int32).copy()
    local[:, 0] -= int(origin[0])
    local[:, 1] -= int(origin[1])
    local_mask = np.zeros(saliency.shape[:2], dtype=np.uint8)
    cv2.drawContours(local_mask, [local.reshape(-1, 1, 2)], -1, 255, -1)
    if np.any(local_mask > 0):
        sal_prob = float(np.mean(saliency[local_mask > 0]))
    else:
        sal_prob = 0.0
    return {
        "index": index,
        "class_index": 0,
        "bbox": (int(bx), int(by), int(bw), int(bh)),
        "saliency": sal_prob,
        "contour": contour_xy.astype(np.int32),
    }


def nms_grains(grains: Sequence[Dict[str, Any]], iou: float = 0.35) -> List[Dict[str, Any]]:
    ordered = sorted(grains, key=lambda g: float(g.get("saliency", 0.0)), reverse=True)
    kept: List[Dict[str, Any]] = []
    for g in ordered:
        box = tuple(g["bbox"])
        if all(bbox_iou(box, k["bbox"]) < iou for k in kept):
            kept.append(g)
    for i, g in enumerate(kept):
        g["index"] = i
    return kept


def resolve_seeded_object(
    image_bgr: np.ndarray,
    seed: Tuple[int, int],
    params: SeededParams,
    saliency_provider: SaliencyProvider,
    index: int = 0,
    reject_log: Optional[Counter] = None,
    seeds: Optional[Sequence[Tuple[int, int]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Semilla → cuerpo válido en coordenadas de la imagen completa.

    Camino único del buscador ROI: lo usan el click manual y el lote débil.
    Amplía el recorte mientras el contorno toque el borde (objeto truncado).
    ``reject_log`` acumula motivos de descarte para el resumen en terminal.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None

    h, w = image_bgr.shape[:2]
    stroke = [(int(x), int(y)) for x, y in (seeds or (seed,))]
    if not stroke:
        stroke = [(int(seed[0]), int(seed[1]))]
    stroke = [
        (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in stroke
    ]
    sx, sy = stroke[0]
    provider = memoized_provider(saliency_provider)
    fallback: Optional[Dict[str, Any]] = None
    last_reason = "sin contorno en ningún radio"
    frame = max(h, w)
    r = int(max(1, min(frame, params.crop_radius)))
    tried: set = set()

    # Como máximo dos inferencias: el recorte del trazo y, si truncó, el encuadre.
    for _ in range(2):
        r = int(max(1, min(frame, r)))
        if r in tried:
            break
        tried.add(r)
        box = _seeds_crop_box(stroke, r, h, w)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            break
        allow_crop_fill = (x2 - x1) < w or (y2 - y1) < h

        crop_saliency = provider(image_bgr, box)
        if crop_saliency is None or crop_saliency.size == 0:
            last_reason = "saliencia vacía en el recorte"
            logger.info("  semilla (%d,%d) r=%d: %s", sx, sy, r, last_reason)
            if fallback is None and r < frame:
                r = frame
                continue
            break

        crop_bgr = image_bgr[y1:y2, x1:x2]
        local_seeds = [(x - x1, y - y1) for x, y in stroke]
        cmp = compare_object_background(crop_saliency, local_seeds[0], crop_bgr)
        logger.info("  semilla (%d,%d) r=%d | %s", sx, sy, r, cmp.describe())

        local = u2net_seeded_contour(
            crop_saliency, local_seeds[0][0], local_seeds[0][1], params,
            separation=cmp, seeds=local_seeds, allow_crop_fill=allow_crop_fill,
        )
        if local is None or len(local) < 3:
            last_reason = cmp.reason or "sin contorno sembrado"
            logger.info("  semilla (%d,%d) r=%d: sin cuerpo — %s", sx, sy, r, last_reason)
            if fallback is not None:
                break
            if r < frame:
                r = frame
                continue
            break

        mapped = local.copy()
        mapped[:, 0] += x1
        mapped[:, 1] += y1
        if covers_full_frame(mapped, h, w):
            last_reason = "el encuadre no es un objeto"
            logger.info("  semilla (%d,%d) r=%d: %s", sx, sy, r, last_reason)
            break
        reason = validity_reason(mapped, h, w, params)
        if reason is not None:
            last_reason = reason
            logger.debug("  semilla (%d,%d) r=%d: descartada — %s", sx, sy, r, reason)
            if fallback is not None:
                break
            if r < frame:
                r = frame
                continue
            break

        if params.drop_border_objects and touches_image_border(mapped, h, w):
            last_reason = "cuerpo parcial en el borde del encuadre"
            logger.debug("  semilla (%d,%d) r=%d: %s", sx, sy, r, last_reason)
            if fallback is not None:
                break
            if r < frame:
                r = frame
                continue
            break

        grain = contour_to_grain_dict(mapped, crop_saliency, index=index, origin=(x1, y1))
        bx, by, bw, bh = grain["bbox"]
        grain["trace"] = cmp.describe()
        if not touches_interior_crop_border(local, box, h, w):
            logger.info(
                "  semilla (%d,%d) r=%d: OK bbox=(%d,%d,%d,%d) sal=%.3f | %s",
                sx, sy, r, bx, by, bw, bh, grain["saliency"], cmp.describe(),
            )
            return grain

        fallback = grain
        last_reason = "truncado por el recorte (borde recto)"
        logger.info(
            "  semilla (%d,%d) r=%d: truncado por el recorte, amplío", sx, sy, r
        )
        if r >= frame:
            if covers_full_frame(mapped, h, w):
                last_reason = "el encuadre no es un objeto"
                logger.info("  semilla (%d,%d): %s", sx, sy, last_reason)
                break
            logger.info(
                "  semilla (%d,%d): OK truncado por el encuadre bbox=(%d,%d,%d,%d)",
                sx, sy, bx, by, bw, bh,
            )
            return grain
        r = frame

    if fallback is not None and not covers_full_frame(fallback["contour"], h, w):
        logger.info(
            "  semilla (%d,%d): acepto el recorte anterior (ampliar perdió el objeto)",
            sx, sy,
        )
        return fallback
    if fallback is not None:
        logger.info("  semilla (%d,%d): fallback era el encuadre — descartado", sx, sy)

    if reject_log is not None:
        reject_log[last_reason] += 1
    logger.info("  semilla (%d,%d): sin cuerpo — %s", sx, sy, last_reason)
    return None


def propose_objects(
    saliency: np.ndarray,
    image_bgr: np.ndarray,
    params: Optional[SeededParams] = None,
    extra_seeds: Optional[Iterable[Tuple[int, int]]] = None,
    saliency_provider: Optional[SaliencyProvider] = None,
) -> List[Dict[str, Any]]:
    """
    Propone cuerpos con el mismo algoritmo que el click de ROI.

    ``saliency`` solo elige las semillas (picos = clicks automáticos). Cada semilla
    se resuelve con ``resolve_seeded_object``; si se pasa ``saliency_provider``, el
    lote ejecuta la red por recorte igual que el click.
    """
    params = params or SeededParams()
    if saliency is None or image_bgr is None:
        return []

    t0 = time.perf_counter()
    provider = memoized_provider(saliency_provider or crop_saliency_provider(saliency))
    seeds = list(peak_seeds(saliency, params))
    n_auto = len(seeds)
    if extra_seeds:
        seeds.extend((int(x), int(y)) for x, y in extra_seeds)

    logger.debug(
        "buscador ROI: %d semillas automáticas (+%d extra) | threshold=%.2f k=%.2f radio=%d",
        n_auto, len(seeds) - n_auto, params.saliency_threshold,
        params.adaptive_k, params.crop_radius,
    )

    grains: List[Dict[str, Any]] = []
    rejects: Counter = Counter()
    seen = set()
    n_dup = 0
    for sx, sy in seeds:
        key = (sx // 8, sy // 8)
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        grain = resolve_seeded_object(
            image_bgr, (sx, sy), params, provider,
            index=len(grains), reject_log=rejects,
        )
        if grain is not None:
            grains.append(grain)

    kept = nms_grains(grains)
    elapsed = (time.perf_counter() - t0) * 1000
    detalle = ", ".join(f"{r} x{n}" for r, n in rejects.most_common(3))
    logger.info(
        "buscador ROI: %d semillas -> %d cuerpos (%d fusionados por NMS, "
        "%d semillas repetidas) en %.0f ms%s",
        len(seeds), len(kept), len(grains) - len(kept), n_dup, elapsed,
        f" | descartes: {detalle}" if detalle else "",
    )
    return kept
