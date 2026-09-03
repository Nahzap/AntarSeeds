"""Tests for localization detector resolution."""

from src.grain_detection.full_image_classifier import resolve_localization_detector


def test_resolve_from_active_detector():
    cfg = {
        "active_detector": {"backend": "vit_dense", "checkpoint": "x.pth"},
        "localization_detector": "u2net",
    }
    assert resolve_localization_detector(cfg) == "vit_dense"


def test_resolve_legacy_fallback():
    cfg = {"localization_detector": "u2net"}
    assert resolve_localization_detector(cfg) == "u2net"
