"""Canonical image resolution for the full MetricLearning pipeline."""

from __future__ import annotations

from typing import Tuple


def resolve_effective_image_size(config: dict) -> int:
    """
    Single source of truth for crop / transform / HDF5 cache resolution.

    Training and validation always use ``data.image_size``.
    ``grain_detection.crop_size`` is kept in YAML for legacy/GUI mirror only.
    """
    data_cfg = config.get("data", {}) or {}
    if "image_size" in data_cfg and data_cfg["image_size"] is not None:
        return int(data_cfg["image_size"])
    grain_cfg = config.get("grain_detection", {}) or {}
    return int(grain_cfg.get("crop_size", 224))


def sync_resolution_config(config: dict) -> Tuple[dict, bool]:
    """
    Mirror ``data.image_size`` into dependent keys before save/train.

    Returns (config, changed).
    """
    size = resolve_effective_image_size(config)
    changed = False

    data_cfg = config.setdefault("data", {})
    if data_cfg.get("image_size") != size:
        data_cfg["image_size"] = size
        changed = True
    if data_cfg.get("resize_size") != size:
        data_cfg["resize_size"] = size
        changed = True

    grain_cfg = config.setdefault("grain_detection", {})
    if grain_cfg.get("crop_size") != size:
        grain_cfg["crop_size"] = size
        changed = True

    return config, changed
