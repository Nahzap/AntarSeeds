"""Tests for detector resolution config."""

from src.utils.detector_resolution_config import (
    resolve_detector_input_size,
    resolve_detector_patch_size,
    sync_detector_resolution_config,
)


def test_resolve_detector_input_size():
    cfg = {"detection_training": {"input_size": 504, "patch_size": 14}}
    assert resolve_detector_input_size(cfg) == 504
    assert resolve_detector_patch_size(cfg) == 14


def test_sync_detector_resolution():
    cfg = {
        "detection_training": {"input_size": 504},
        "grain_detection": {"max_load_side": 512},
    }
    out, changed = sync_detector_resolution_config(cfg)
    assert changed
    assert out["grain_detection"]["max_load_side"] == 504
