"""
Calibración por demostración: el etiquetado manual fija los parámetros.

Los topes del buscador ROI (área, circularidad, aspecto, lado, radio) dependen
de la especie: lo que sirve para `C_Quitensis` (~290.000 px² por grano) descarta
por completo a `J_Bufonius` (~60.000 px²). Configurarlos a mano para cada clase
es trabajo perdido y una fuente de lotes vacíos.

Aquí se mide lo que el usuario ya aceptó como cuerpo válido y se derivan los
topes que lo admiten. Click, preview y lote usan `SeededParams.from_config`
sin rama paralela: etiquetar calibra el mismo algoritmo automático.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Márgenes sobre lo observado. Amplios a propósito: el objetivo es no descartar
# la especie, no ajustar al píxel.
MIN_AREA_MARGIN = 0.45
MAX_AREA_MARGIN = 2.50
CIRC_MARGIN = 0.75
ASPECT_MARGIN = 1.35
SIDE_MARGIN = 1.25

ENVELOPE_KEYS = (
    "min_area", "max_area", "min_circularity",
    "max_aspect_ratio", "max_bbox_side_frac", "max_side_px",
)


def measure_contour(contour_xy: np.ndarray, image_h: int, image_w: int) -> Optional[Dict[str, float]]:
    """Propiedades de un cuerpo aceptado por el usuario."""
    if contour_xy is None or len(contour_xy) < 3:
        return None

    cnt = np.asarray(contour_xy, dtype=np.int32).reshape(-1, 1, 2)
    area = abs(cv2.contourArea(cnt))
    if area <= 0:
        return None

    peri = cv2.arcLength(cnt, True)
    circ = float(4.0 * np.pi * area / (peri * peri)) if peri > 1e-6 else 0.0
    _x, _y, bw, bh = cv2.boundingRect(cnt)
    side = max(bw, bh)
    short = max(1, min(bw, bh))
    short_side = max(1, min(image_h, image_w))

    return {
        "area": float(area),
        "circularity": circ,
        "aspect_ratio": float(side) / float(short),
        "side_frac": float(side) / float(short_side),
        "side_px": float(side),
    }


def envelope_from_contours(
    contours: Iterable[np.ndarray], image_h: int, image_w: int
) -> Optional[Dict[str, float]]:
    """Envolvente de los cuerpos medidos: los extremos que hay que admitir."""
    medidas = [
        m for m in (measure_contour(c, image_h, image_w) for c in contours)
        if m is not None
    ]
    if not medidas:
        return None

    return {
        "n": float(len(medidas)),
        "min_area": min(m["area"] for m in medidas),
        "max_area": max(m["area"] for m in medidas),
        "min_circularity": min(m["circularity"] for m in medidas),
        "max_aspect_ratio": max(m["aspect_ratio"] for m in medidas),
        "max_bbox_side_frac": max(m["side_frac"] for m in medidas),
        "max_side_px": max(m["side_px"] for m in medidas),
    }


def merge_envelopes(
    a: Optional[Dict[str, float]], b: Optional[Dict[str, float]]
) -> Optional[Dict[str, float]]:
    """Acumula observaciones: la envolvente crece con cada cuerpo etiquetado."""
    if a is None:
        return dict(b) if b else None
    if b is None:
        return dict(a)

    return {
        "n": a.get("n", 0.0) + b.get("n", 0.0),
        "min_area": min(a["min_area"], b["min_area"]),
        "max_area": max(a["max_area"], b["max_area"]),
        "min_circularity": min(a["min_circularity"], b["min_circularity"]),
        "max_aspect_ratio": max(a["max_aspect_ratio"], b["max_aspect_ratio"]),
        "max_bbox_side_frac": max(a["max_bbox_side_frac"], b["max_bbox_side_frac"]),
        "max_side_px": max(a["max_side_px"], b["max_side_px"]),
    }


def params_from_envelope(
    envelope: Optional[Dict[str, float]], frame_area: int
) -> Optional[Dict[str, Any]]:
    """
    Topes del buscador derivados de la envolvente observada.

    Un solo cuerpo ya sirve: con márgenes amplios, el automático deja de
    descartar la especie y el usuario afina con la prueba en pantalla.
    """
    if not envelope:
        return None

    min_area = max(50, int(envelope["min_area"] * MIN_AREA_MARGIN))
    max_area = int(min(frame_area, envelope["max_area"] * MAX_AREA_MARGIN))
    if max_area <= min_area:
        max_area = min(frame_area, min_area * 4)

    return {
        "min_area": min_area,
        "max_area": max_area,
        "min_circularity": round(max(0.02, envelope["min_circularity"] * CIRC_MARGIN), 2),
        "max_aspect_ratio": round(min(20.0, envelope["max_aspect_ratio"] * ASPECT_MARGIN), 1),
        "max_bbox_side_frac": round(min(1.0, envelope["max_bbox_side_frac"] * SIDE_MARGIN), 2),
    }


# Política de mezcla con lo que ya está configurado.
#
# El área SÍ se re-escala: es lo que cambia de especie a especie y lo que dejaba
# el lote en cero. Los límites de forma solo se relajan, nunca se aprietan: con
# uno o dos cuerpos observados, un círculo perfecto llevaría la circularidad
# mínima a 0.75 y excluiría las semillas menos redondas de la misma clase.
RESCALED_KEYS = ("min_area", "max_area")
RELAX_DOWN_KEYS = ("min_circularity",)
RELAX_UP_KEYS = ("max_aspect_ratio", "max_bbox_side_frac")


def blend_with_current(
    learned: Optional[Dict[str, Any]], current: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Combina lo aprendido con lo configurado sin volverse más restrictivo.

    Aprender nunca puede empezar a rechazar cuerpos que antes se aceptaban.
    """
    if not learned:
        return None
    current = current or {}
    out: Dict[str, Any] = {}

    for key in RESCALED_KEYS:
        out[key] = learned[key]

    for key in RELAX_DOWN_KEYS:
        actual = current.get(key)
        out[key] = learned[key] if actual is None else min(float(actual), learned[key])

    for key in RELAX_UP_KEYS:
        actual = current.get(key)
        out[key] = learned[key] if actual is None else max(type(learned[key])(actual), learned[key])

    return out


def describe_envelope(envelope: Optional[Dict[str, float]]) -> str:
    """Resumen legible para la terminal y la interfaz."""
    if not envelope:
        return "sin cuerpos medidos"
    return (
        f"{int(envelope['n'])} cuerpo(s): área "
        f"{int(envelope['min_area']):,}–{int(envelope['max_area']):,} px², "
        f"circ. mín {envelope['min_circularity']:.2f}, "
        f"aspecto máx {envelope['max_aspect_ratio']:.2f}, "
        f"lado máx {int(envelope['max_side_px'])} px"
    )


def contours_from_grains(grains: Sequence[Dict[str, Any]]) -> List[np.ndarray]:
    """Extrae los contornos de una lista de granos de .seg / edición."""
    out: List[np.ndarray] = []
    for g in grains or []:
        cnt = g.get("contour") if isinstance(g, dict) else None
        if cnt is None:
            continue
        arr = np.asarray(cnt)
        if arr.ndim == 3:
            arr = arr.reshape(-1, 2)
        if arr.ndim == 2 and len(arr) >= 3:
            out.append(arr.astype(np.int32))
    return out


ENVELOPES_FILENAME = "class_envelopes.json"


def envelopes_path(annotation_root: Union[str, Path]) -> Path:
    return Path(annotation_root) / ENVELOPES_FILENAME


def save_envelopes(
    annotation_root: Union[str, Path], envelopes: Dict[str, Dict[str, float]]
) -> Path:
    """Persiste la envolvente por clase junto al banco de anotaciones."""
    path = envelopes_path(annotation_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "classes": {
            name: {k: float(v) for k, v in env.items()}
            for name, env in envelopes.items()
            if env
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_envelopes(annotation_root: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """Recupera las envolventes aprendidas. Vacío si aún no hay archivo."""
    path = envelopes_path(annotation_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("[Calibración] no se pudo leer %s: %s", path, exc)
        return {}
    classes = data.get("classes") if isinstance(data, dict) else None
    if not isinstance(classes, dict):
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for name, env in classes.items():
        if not isinstance(env, dict):
            continue
        try:
            out[str(name)] = {k: float(env[k]) for k in ("n",) + ENVELOPE_KEYS}
        except (KeyError, TypeError, ValueError):
            continue
    return out
