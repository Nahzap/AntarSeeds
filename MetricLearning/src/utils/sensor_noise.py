"""
Sensor noise calibration from training crops (Phase E-03).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def estimate_gaussian_noise_std(
    images: List[np.ndarray],
    *,
    blur_ksize: int = 5,
    max_pixels: int = 2_000_000,
) -> Tuple[float, float]:
    """
    Estimate residual noise std (min, max) in [0,255] space via high-pass residual.

    Returns (p10, p90) of per-image median absolute residual per channel.
    """
    if not images:
        raise ValueError("Sin imágenes para calibrar ruido")

    import cv2

    residuals = []
    for img in images:
        arr = img.astype(np.float32)
        if arr.max() <= 1.0:
            arr *= 255.0
        if arr.ndim != 3:
            continue
        blurred = cv2.GaussianBlur(arr, (blur_ksize, blur_ksize), 0)
        resid = np.abs(arr - blurred)
        residuals.append(float(np.median(resid)))

    if not residuals:
        return 10.0, 50.0

    p10, p90 = np.percentile(residuals, [10, 90])
    return float(max(1.0, p10)), float(max(p10 + 1.0, p90))


def save_noise_model(path: str | Path, var_limit: Tuple[float, float], meta: Optional[dict] = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "var_limit": [float(var_limit[0]), float(var_limit[1])],
        "meta": meta or {},
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Sensor noise model guardado: {out}")
    return out


def load_noise_model(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def resolve_gauss_noise_var_limit(config: dict) -> Tuple[float, float]:
    """Return var_limit in [0,255] units for GaussNoise."""
    aug = config.get("augmentation") or {}
    micro = aug.get("microscopy_specific") or {}
    gn = micro.get("gaussian_noise") or {}
    default = tuple(gn.get("var_limit") or [10.0, 50.0])

    model_cfg = aug.get("sensor_noise_model") or {}
    if not model_cfg.get("enabled", False):
        return float(default[0]), float(default[1])

    calibrated = model_cfg.get("var_limit")
    if calibrated and len(calibrated) == 2:
        scale = float(model_cfg.get("scale_factor", 1.0))
        return float(calibrated[0]) * scale, float(calibrated[1]) * scale

    path = model_cfg.get("stats_path")
    if path and Path(path).exists():
        loaded = load_noise_model(path)
        vl = loaded.get("var_limit", default)
        scale = float(model_cfg.get("scale_factor", 1.0))
        return float(vl[0]) * scale, float(vl[1]) * scale

    return float(default[0]), float(default[1])
