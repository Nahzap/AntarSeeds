"""Tests for detection profile presets."""

from src.grain_detection.detection_profiles import PROFILES, resolve_grain_detection_config


def test_sensitive_profile_lowers_min_area():
    cfg = resolve_grain_detection_config({"detection_profile": "sensitive"})
    assert cfg["min_area"] == PROFILES["sensitive"]["min_area"]
    assert cfg["min_area"] < PROFILES["strict"]["min_area"]


def test_explicit_override_wins():
    cfg = resolve_grain_detection_config({"detection_profile": "strict", "min_area": 500})
    assert cfg["min_area"] == 500


def test_balanced_default():
    cfg = resolve_grain_detection_config({})
    assert cfg["detection_profile"] == "balanced"
