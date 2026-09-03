"""Presets for morphological grain extraction (sensitive / balanced / strict)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

PROFILES: Dict[str, Dict[str, Any]] = {
    "sensitive": {
        "adaptive_k": 0.18,
        "min_area": 250,
        "max_area": 120000,
        "max_area_frac": 0.15,
        "min_circularity": 0.25,
        "max_bbox_side_frac": 0.45,
        "max_aspect_ratio": 3.0,
        "morph_kernel_size": 2,
    },
    "balanced": {
        "adaptive_k": 0.22,
        "min_area": 400,
        "max_area": 100000,
        "min_circularity": 0.32,
        "max_bbox_side_frac": 0.40,
        "max_aspect_ratio": 2.8,
        "morph_kernel_size": 3,
    },
    "strict": {
        "adaptive_k": 0.30,
        "min_area": 1000,
        "max_area": 80000,
        "min_circularity": 0.45,
        "max_bbox_side_frac": 0.35,
        "max_aspect_ratio": 2.5,
        "morph_kernel_size": 3,
    },
    "seed": {
        "adaptive_k": 0.20,
        "min_area": 800,
        "max_area": 800000,
        "max_area_frac": 0.50,
        "min_circularity": 0.12,
        "max_bbox_side_frac": 0.85,
        "max_aspect_ratio": 8.0,
        "morph_kernel_size": 3,
        "crop_radius": 300,
        "object_finder": "roi_seed",
        "split_touching": True,
        "seed_core_frac": 0.45,
        "waist_frac": 0.80,
    },
}


# Filtros que extract_grains_from_saliency honra al decidir si un contorno es un
# objeto válido. Si el llamador omite alguno, el preset lo rellena en silencio,
# que es exactamente cómo el perfil de polen anulaba los ajustes de la GUI.
FILTER_KEYS = (
    "adaptive_k",
    "min_area",
    "max_area",
    "max_area_frac",
    "min_circularity",
    "max_bbox_side_frac",
    "max_aspect_ratio",
    "morph_kernel_size",
)


def missing_filter_keys(grain_cfg: dict | None) -> tuple[str, ...]:
    """Filtros que el llamador deja al preset en vez de declarar explícitamente."""
    cfg = grain_cfg or {}
    return tuple(k for k in FILTER_KEYS if cfg.get(k) is None)


def resolve_grain_detection_config(grain_cfg: dict | None) -> dict:
    """Merge detection_profile preset into grain_detection config."""
    cfg = deepcopy(grain_cfg or {})
    profile = str(cfg.pop("detection_profile", "balanced")).strip().lower()
    preset = PROFILES.get(profile, PROFILES["balanced"])
    for key, value in preset.items():
        cfg.setdefault(key, value)
    cfg["detection_profile"] = profile if profile in PROFILES else "balanced"
    cfg.setdefault("object_finder", "roi_seed")
    cfg.setdefault("crop_radius", 300)
    return cfg
