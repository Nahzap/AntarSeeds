"""Canonical resolution for ViT-dense pollen localization (train + inference)."""

from __future__ import annotations

from typing import Tuple


def resolve_detector_patch_size(config: dict) -> int:
    det_cfg = config.get("detection_training", {}) or {}
    return int(det_cfg.get("patch_size", 14))


def resolve_detector_input_size(config: dict) -> int:
    """
    Single source of truth for detector spatial resolution.

    Training, validation, inference and ViT patch grid alignment use this value.
    Must be a multiple of ``patch_size`` (default 14 for DINOv2 ViT-S/14).
    """
    det_cfg = config.get("detection_training", {}) or {}
    if "input_size" in det_cfg and det_cfg["input_size"] is not None:
        return int(det_cfg["input_size"])
    grain_cfg = config.get("grain_detection", {}) or {}
    return int(grain_cfg.get("max_load_side", 504))


def sync_detector_resolution_config(config: dict) -> Tuple[dict, bool]:
    """
    Mirror ``detection_training.input_size`` into ``grain_detection.max_load_side``.

    Returns (config, changed).
    """
    size = resolve_detector_input_size(config)
    changed = False

    det_cfg = config.setdefault("detection_training", {})
    if det_cfg.get("input_size") != size:
        det_cfg["input_size"] = size
        changed = True
    if det_cfg.get("patch_size", 14) != 14:
        det_cfg.setdefault("patch_size", 14)

    grain_cfg = config.setdefault("grain_detection", {})
    if grain_cfg.get("max_load_side") != size:
        grain_cfg["max_load_side"] = size
        changed = True

    return config, changed
