"""
DetectorRegistry — unified active detector loading for inference and DML crops.

Supports:
  - u2net: PollenGrainDetector (pretrained U²-Net SOD)
  - vit_dense: PollenViTDetector (trained ViT-dense checkpoint)
  - vit: ModelGrainDetector (legacy ViT attention; requires DML model at inference)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

BACKEND_ALIASES = {
    "pollen": "u2net",
    "u2-net": "u2net",
    "u²-net": "u2net",
    "model": "vit",
    "analogynet": "vit",
    "vit_attention": "vit",
    "vit-dense": "vit_dense",
    "vit_dense": "vit_dense",
}

VALID_BACKENDS = ("u2net", "vit_dense", "vit")


@dataclass
class DetectorCheckpointInfo:
    path: str
    run_dir: str
    mtime: float
    label: str


def normalize_backend(backend: str) -> str:
    b = str(backend or "u2net").strip().lower()
    b = BACKEND_ALIASES.get(b, b)
    if b not in VALID_BACKENDS:
        raise ValueError(f"Backend de detector inválido: {backend!r}")
    return b


def get_active_detector_config(config: dict) -> dict:
    grain_cfg = config.get("grain_detection", {}) or {}
    active = grain_cfg.get("active_detector")
    if isinstance(active, dict) and active.get("backend"):
        return active
    # Legacy fallback
    legacy = str(grain_cfg.get("localization_detector", "u2net")).strip().lower()
    return {
        "backend": normalize_backend(legacy),
        "checkpoint": grain_cfg.get("detector_checkpoint"),
        "label": legacy,
        "loaded_at": None,
    }


def list_trained_checkpoints(runs_dir: str = "runs") -> List[DetectorCheckpointInfo]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    found: List[DetectorCheckpointInfo] = []
    for ckpt in root.glob("*/detector/best_detector.pth"):
        mtime = ckpt.stat().st_mtime
        found.append(
            DetectorCheckpointInfo(
                path=str(ckpt.resolve()),
                run_dir=str(ckpt.parent.parent.resolve()),
                mtime=mtime,
                label=f"{ckpt.parent.parent.name} / best_detector.pth",
            )
        )
    for ckpt in root.glob("*/detector/last_detector.pth"):
        if any(f.path == str(ckpt.resolve()) for f in found):
            continue
        found.append(
            DetectorCheckpointInfo(
                path=str(ckpt.resolve()),
                run_dir=str(ckpt.parent.parent.resolve()),
                mtime=ckpt.stat().st_mtime,
                label=f"{ckpt.parent.parent.name} / last_detector.pth",
            )
        )
    found.sort(key=lambda x: x.mtime, reverse=True)
    return found


def validate_active_detector(config: dict) -> List[str]:
    errors: List[str] = []
    active = get_active_detector_config(config)
    backend = normalize_backend(active.get("backend", "u2net"))
    ckpt = active.get("checkpoint")
    if backend == "vit_dense":
        if not ckpt:
            errors.append("active_detector.backend=vit_dense requiere checkpoint")
        elif not Path(str(ckpt)).exists():
            errors.append(f"Checkpoint detector no encontrado: {ckpt}")
    crop_source = str(config.get("data", {}).get("crop_source", "seg")).lower()
    if crop_source == "detector" and backend == "vit_dense" and not ckpt:
        errors.append("data.crop_source=detector con vit_dense requiere checkpoint cargado")
    return errors


def build_active_detector_payload(
    backend: str,
    checkpoint: Optional[str] = None,
    label: Optional[str] = None,
) -> dict:
    b = normalize_backend(backend)
    return {
        "backend": b,
        "checkpoint": checkpoint if b == "vit_dense" else None,
        "label": label or _default_label(b, checkpoint),
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
    }


def _default_label(backend: str, checkpoint: Optional[str]) -> str:
    if backend == "u2net":
        return "U²-Net (preentrenado)"
    if backend == "vit_dense" and checkpoint:
        return f"ViT-denso @ {Path(checkpoint).parent.parent.name}"
    if backend == "vit":
        return "ViT atención (legacy)"
    return backend


def load_active_detector(
    config: dict,
    *,
    device: Optional[str] = None,
    dml_model=None,
    transforms=None,
    slice_classifier=None,
) -> Any:
    """
    Instantiate the active localization backend from config.

    Returns PollenGrainDetector, PollenViTDetector, or ModelGrainDetector.
    """
    grain_cfg = config.get("grain_detection", {}) or {}
    active = get_active_detector_config(config)
    backend = normalize_backend(active["backend"])
    det_config = dict(grain_cfg)
    if device:
        det_config["device"] = device

    if backend == "u2net":
        from src.grain_detection.grain_detector import PollenGrainDetector
        return PollenGrainDetector(config=det_config)

    if backend == "vit_dense":
        from src.grain_detection.vit_dense_detector import PollenViTDetector
        ckpt = active.get("checkpoint")
        if not ckpt or not Path(str(ckpt)).exists():
            raise FileNotFoundError(
                "vit_dense activo sin checkpoint válido. Entrene o cargue un detector."
            )
        merged = {**det_config, **(config.get("detection_training") or {})}
        return PollenViTDetector(str(ckpt), config=merged, device=device)

    if backend == "vit":
        from src.grain_detection.model_grain_detector import ModelGrainDetector
        if dml_model is None:
            raise ValueError("ModelGrainDetector (vit) requiere modelo DML en inferencia")
        import torch
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return ModelGrainDetector(
            model=dml_model,
            transforms=transforms,
            device=dev,
            config=det_config,
            slice_classifier=slice_classifier,
        )

    raise ValueError(f"Backend no soportado: {backend}")
