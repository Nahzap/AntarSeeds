"""Tests for DetectorRegistry."""

from src.grain_detection.detector_registry import (
    build_active_detector_payload,
    get_active_detector_config,
    normalize_backend,
    validate_active_detector,
)


def test_normalize_backend_aliases():
    assert normalize_backend("u2-net") == "u2net"
    assert normalize_backend("vit_dense") == "vit_dense"


def test_validate_u2net_ok():
    cfg = {
        "grain_detection": {
            "active_detector": {"backend": "u2net", "checkpoint": None},
        },
        "data": {"crop_source": "seg"},
    }
    assert validate_active_detector(cfg) == []


def test_validate_vit_dense_requires_checkpoint():
    cfg = {
        "grain_detection": {"active_detector": {"backend": "vit_dense"}},
        "data": {"crop_source": "detector"},
    }
    errs = validate_active_detector(cfg)
    assert any("checkpoint" in e for e in errs)


def test_build_payload():
    p = build_active_detector_payload("u2net")
    assert p["backend"] == "u2net"
    assert p["loaded_at"] is not None
