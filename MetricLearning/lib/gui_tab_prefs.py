"""Persistencia de formularios GUI (pestañas Detección y Entrenamiento DML) en config.yaml."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.grain_detection.detector_registry import build_active_detector_payload


def resolve_config_path(config_path: str | Path) -> Path:
    p = Path(config_path)
    if not p.is_absolute():
        root = Path(__file__).resolve().parent.parent
        if (root / p).exists():
            return root / p
        return root / p
    return p


def load_config(config_path: str | Path) -> dict:
    p = resolve_config_path(config_path)
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def save_config_merged(config_path: str | Path, updates: dict, *, backup: bool = True) -> Path:
    p = resolve_config_path(config_path)
    cfg = load_config(p)
    _deep_update(cfg, updates)
    if backup and p.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(p, p.with_name(f"{p.stem}.backup_{ts}{p.suffix}"))
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return p


def save_config_full(config_path: str | Path, cfg: dict, *, backup: bool = True) -> Path:
    p = resolve_config_path(config_path)
    if backup and p.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(p, p.with_name(f"{p.stem}.backup_{ts}{p.suffix}"))
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return p


def detector_tab_updates_from_widgets(tab, *, include_active_detector: bool = True) -> dict:
    """Build detection_training + gui.detector_training patch from DetectionTrainingTab."""
    det: Dict[str, Any] = {
        "epochs": int(tab.epochs_spin.value()),
        "batch_size": int(tab.batch_spin.value()),
        "val_batch_size": int(tab.batch_spin.value()),
        "input_size": int(tab.input_spin.value()),
        "num_workers": int(tab.workers_spin.value()),
        "max_train_images": int(tab.max_train_spin.value()),
        "max_val_images": int(tab.max_val_spin.value()),
        "learning_rate": float(tab.lr_spin.value()),
        "num_unfrozen_blocks": int(tab.unfrozen_spin.value()),
        "val_fast": bool(tab.val_fast_check.isChecked()),
        "val_det_f1_max_images": int(tab.val_f1_spin.value()),
        "validation_metric": tab.metric_combo.currentData(),
        "loss": {"type": tab.loss_combo.currentData()},
    }
    gui_det = {
        "resume": bool(tab.resume_check.isChecked()),
        "checkpoint": tab.checkpoint_path.text().strip(),
    }
    out: Dict[str, Any] = {"detection_training": det, "gui": {"detector_training": gui_det}}
    if include_active_detector and hasattr(tab, "detector_widget"):
        widget = tab.detector_widget
        backend = widget._backend_key()
        ckpt = widget.checkpoint_combo.currentData()
        payload = build_active_detector_payload(
            backend,
            ckpt if backend == "vit_dense" else None,
        )
        out.setdefault("grain_detection", {})
        out["grain_detection"]["active_detector"] = payload
        out["grain_detection"]["localization_detector"] = backend
    return out


def apply_detector_tab_to_widgets(tab, cfg: dict) -> None:
    det = cfg.get("detection_training", {}) or {}
    gui_det = (cfg.get("gui", {}) or {}).get("detector_training", {}) or {}

    tab.epochs_spin.setValue(int(det.get("epochs", tab.epochs_spin.value())))
    tab.batch_spin.setValue(int(det.get("batch_size", tab.batch_spin.value())))
    tab.input_spin.setValue(int(det.get("input_size", tab.input_spin.value())))
    tab.workers_spin.setValue(int(det.get("num_workers", tab.workers_spin.value())))
    tab.max_train_spin.setValue(int(det.get("max_train_images", tab.max_train_spin.value())))
    tab.max_val_spin.setValue(int(det.get("max_val_images", tab.max_val_spin.value())))
    tab.lr_spin.setValue(float(det.get("learning_rate", tab.lr_spin.value())))
    tab.unfrozen_spin.setValue(int(det.get("num_unfrozen_blocks", tab.unfrozen_spin.value())))
    tab.val_fast_check.setChecked(bool(det.get("val_fast", tab.val_fast_check.isChecked())))
    tab.val_f1_spin.setValue(int(det.get("val_det_f1_max_images", tab.val_f1_spin.value())))

    loss_type = str((det.get("loss") or {}).get("type", "bce_dice"))
    idx = tab.loss_combo.findData(loss_type)
    if idx >= 0:
        tab.loss_combo.setCurrentIndex(idx)
    val_metric = str(det.get("validation_metric", "det_f1"))
    if val_metric.lower() == "fm":
        val_metric = "Fm"
    midx = tab.metric_combo.findData(val_metric)
    if midx >= 0:
        tab.metric_combo.setCurrentIndex(midx)

    tab.resume_check.setChecked(bool(gui_det.get("resume", False)))
    ckpt = gui_det.get("checkpoint", "")
    if ckpt:
        tab.checkpoint_path.setText(str(ckpt))
    if hasattr(tab, "detector_widget"):
        tab.detector_widget.set_config_path(tab.config_path.text())
        tab.detector_widget.restore_from_config()


def dml_tab_updates_from_widgets(tab) -> dict:
    """Solo preferencias UI del entrenamiento DML (siempre crop_source=seg)."""
    return {
        "data": {"crop_source": "seg"},
        "gui": {
            "dml_training": {
                "resume": bool(tab.resume_check.isChecked()),
                "checkpoint": tab.checkpoint_path.text().strip(),
            }
        },
    }


def apply_dml_tab_to_widgets(tab, cfg: dict) -> None:
    gui_dml = (cfg.get("gui", {}) or {}).get("dml_training", {}) or {}
    tab.resume_check.setChecked(bool(gui_dml.get("resume", False)))
    ckpt = gui_dml.get("checkpoint", "")
    if ckpt:
        tab.checkpoint_path.setText(str(ckpt))


SAVE_BUTTON_STYLE = """
    QPushButton {
        background-color: #0e639c;
        color: white;
        font-weight: bold;
        padding: 8px 16px;
        border-radius: 5px;
    }
    QPushButton:hover {
        background-color: #1177bb;
    }
"""
