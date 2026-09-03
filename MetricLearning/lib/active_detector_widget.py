"""
ActiveDetectorWidget — selector unificado de detector (U²-Net / ViT-denso / legacy).

Persiste la selección en config.yaml y restaura el último cargado al arranque.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout,
)
from PyQt5.QtCore import pyqtSignal

from src.grain_detection.detector_registry import (
    build_active_detector_payload,
    get_active_detector_config,
    list_trained_checkpoints,
    normalize_backend,
)
from src.utils.detector_resolution_config import sync_detector_resolution_config


class ActiveDetectorWidget(QWidget):
    detectorLoaded = pyqtSignal(dict)

    BACKEND_ITEMS = [
        ("u2net", "U²-Net (preentrenado)"),
        ("vit_dense", "ViT-denso (entrenado)"),
        ("vit", "ViT atención (legacy)"),
    ]

    def __init__(self, parent=None, config_path: str = "config.yaml"):
        super().__init__(parent)
        self.config_path = config_path
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        group = QGroupBox("Detector activo (inferencia / resultados)")
        form = QFormLayout()

        self.backend_combo = QComboBox()
        for _, label in self.BACKEND_ITEMS:
            self.backend_combo.addItem(label)
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        form.addRow("Backend:", self.backend_combo)

        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setMinimumWidth(280)
        form.addRow("Checkpoint:", self.checkpoint_combo)

        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("📁 Buscar .pth")
        self.browse_btn.clicked.connect(self._browse_checkpoint)
        self.load_btn = QPushButton("✅ Cargar detector")
        self.load_btn.clicked.connect(self.load_detector)
        btn_row.addWidget(self.browse_btn)
        btn_row.addWidget(self.load_btn)
        form.addRow("", btn_row)

        self.status_label = QLabel("No cargado")
        self.status_label.setWordWrap(True)
        form.addRow("Estado:", self.status_label)

        group.setLayout(form)
        layout.addWidget(group)
        self.setLayout(layout)

        self._refresh_checkpoints()
        self.restore_from_config()

    def set_config_path(self, path: str):
        self.config_path = path
        self.restore_from_config()

    def _backend_key(self) -> str:
        idx = self.backend_combo.currentIndex()
        return self.BACKEND_ITEMS[idx][0]

    def _on_backend_changed(self):
        is_vit_dense = self._backend_key() == "vit_dense"
        self.checkpoint_combo.setEnabled(is_vit_dense)
        self.browse_btn.setEnabled(is_vit_dense)

    def _refresh_checkpoints(self):
        self.checkpoint_combo.clear()
        self.checkpoint_combo.addItem("(ninguno)", None)
        for info in list_trained_checkpoints():
            self.checkpoint_combo.addItem(info.label, info.path)

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar checkpoint detector", "runs", "PyTorch (*.pth)"
        )
        if path:
            self.checkpoint_combo.addItem(Path(path).name, path)
            self.checkpoint_combo.setCurrentIndex(self.checkpoint_combo.count() - 1)

    def _resolve_config_path(self) -> Path:
        p = Path(self.config_path)
        if not p.is_absolute():
            root = Path(__file__).resolve().parent.parent
            if (root / p).exists():
                p = root / p
        return p

    def restore_from_config(self):
        p = self._resolve_config_path()
        if not p.exists():
            return
        with open(p, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        active = get_active_detector_config(config)
        backend = active.get("backend", "u2net")
        for i, (key, _) in enumerate(self.BACKEND_ITEMS):
            if key == backend:
                self.backend_combo.setCurrentIndex(i)
                break
        ckpt = active.get("checkpoint")
        if ckpt:
            found = False
            for i in range(self.checkpoint_combo.count()):
                if self.checkpoint_combo.itemData(i) == ckpt:
                    self.checkpoint_combo.setCurrentIndex(i)
                    found = True
                    break
            if not found:
                self.checkpoint_combo.addItem(Path(ckpt).name, ckpt)
                self.checkpoint_combo.setCurrentIndex(self.checkpoint_combo.count() - 1)
        label = active.get("label") or backend
        loaded = active.get("loaded_at") or "—"
        self.status_label.setText(f"Activo: {label} (cargado: {loaded})")

    def load_detector(self, *, silent: bool = False) -> bool:
        backend = self._backend_key()
        ckpt = self.checkpoint_combo.currentData()
        if backend == "vit_dense" and not ckpt:
            if not silent:
                QMessageBox.warning(self, "Checkpoint requerido", "Seleccione un checkpoint para ViT-denso.")
            return False
        if backend == "vit_dense" and not Path(str(ckpt)).exists():
            if not silent:
                QMessageBox.warning(self, "Error", f"Checkpoint no encontrado:\n{ckpt}")
            return False

        p = self._resolve_config_path()
        if not p.exists():
            if not silent:
                QMessageBox.warning(self, "Error", "config.yaml no encontrado")
            return False

        with open(p, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        payload = build_active_detector_payload(backend, ckpt if backend == "vit_dense" else None)
        config.setdefault("grain_detection", {})["active_detector"] = payload
        config["grain_detection"]["localization_detector"] = backend
        config, _ = sync_detector_resolution_config(config)

        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        self.status_label.setText(f"Activo: {payload['label']} (cargado: {payload['loaded_at']})")
        self.detectorLoaded.emit(payload)
        if not silent:
            QMessageBox.information(self, "Detector cargado", f"Detector activo: {payload['label']}")
        return True

    def get_status_text(self) -> str:
        return self.status_label.text()
