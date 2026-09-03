"""
Dataset normalization utilities (Phase E).

Supports ImageNet stats, dataset-computed stats, and optional stain normalization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def imagenet_neutral_rgb_uint8(mean: Optional[List[float]] = None) -> Tuple[int, int, int]:
    """
    RGB uint8 values that map to ~0 after ImageNet Normalize(mean, std).

    Used for *zero activation* background fill before normalization: background
    patches do not inject offset activations into ViT attention (author design).
    """
    m = mean or IMAGENET_MEAN
    return tuple(int(round(v * 255)) for v in m)


def imagenet_neutral_bgr_uint8(mean: Optional[List[float]] = None) -> Tuple[int, int, int]:
    r, g, b = imagenet_neutral_rgb_uint8(mean)
    return (b, g, r)


def load_norm_stats(stats_path: str | Path) -> Dict[str, Any]:
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Norm stats no encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "mean" not in data or "std" not in data:
        raise ValueError(f"stats_path inválido (falta mean/std): {path}")
    return data


def save_norm_stats(
    stats_path: str | Path,
    mean: List[float],
    std: List[float],
    *,
    meta: Optional[dict] = None,
) -> Path:
    path = Path(stats_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Norm stats guardadas en {path}")
    return path


def resolve_normalization_config(config: dict) -> dict:
    """
    Resolve effective normalize block for transforms (E-01).

    Priority:
      data.normalize.mode + stats_path → mean/std
      fallback → augmentation.normalize
    """
    data_norm = (config.get("data") or {}).get("normalize") or {}
    flat_norm = config.get("normalize") or {}
    if not data_norm and isinstance(flat_norm, dict) and (
        "mode" in flat_norm or "stats_path" in flat_norm
    ):
        data_norm = flat_norm
    aug_norm = (config.get("augmentation") or {}).get("normalize") or flat_norm or {}

    mode = str(data_norm.get("mode", "imagenet")).lower()
    stain_method = data_norm.get("stain_method")
    stats_path = data_norm.get("stats_path")

    mean = list(data_norm.get("mean") or aug_norm.get("mean") or IMAGENET_MEAN)
    std = list(data_norm.get("std") or aug_norm.get("std") or IMAGENET_STD)

    if mode == "dataset":
        if stats_path:
            loaded = load_norm_stats(stats_path)
            mean = list(loaded["mean"])
            std = list(loaded["std"])
        else:
            logger.warning(
                "data.normalize.mode=dataset sin stats_path; usando mean/std del YAML"
            )
    elif mode == "imagenet":
        mean = list(data_norm.get("mean") or IMAGENET_MEAN)
        std = list(data_norm.get("std") or IMAGENET_STD)
    elif mode == "macenko":
        stain_method = stain_method or "macenko"
        mean = list(data_norm.get("mean") or IMAGENET_MEAN)
        std = list(data_norm.get("std") or IMAGENET_STD)
    else:
        raise ValueError(
            f"data.normalize.mode debe ser imagenet|dataset|macenko, recibido: {mode!r}"
        )

    return {
        "mode": mode,
        "mean": mean,
        "std": std,
        "stain_method": stain_method,
        "stats_path": stats_path,
    }


def compute_channel_mean_std(
    images: List[np.ndarray],
    *,
    max_pixels: int = 5_000_000,
) -> Tuple[List[float], List[float]]:
    """Compute per-channel mean/std in [0,1] from RGB uint8 or float images."""
    if not images:
        raise ValueError("Lista de imágenes vacía para compute_channel_mean_std")

    sums = np.zeros(3, dtype=np.float64)
    sumsq = np.zeros(3, dtype=np.float64)
    count = 0

    for img in images:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Imagen debe ser HxWx3, recibido {img.shape}")
        arr = img.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        flat = arr.reshape(-1, 3)
        if count + flat.shape[0] > max_pixels:
            flat = flat[: max(1, max_pixels - count)]
        sums += flat.sum(axis=0)
        sumsq += (flat ** 2).sum(axis=0)
        count += flat.shape[0]
        if count >= max_pixels:
            break

    mean = sums / count
    var = sumsq / count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.tolist(), std.tolist()
