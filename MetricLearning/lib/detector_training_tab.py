"""
DetectionTrainingTab — entrenamiento ViT-dense con opciones y logging en tiempo real.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QLabel, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QMessageBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from lib.active_detector_widget import ActiveDetectorWidget
from lib.gui_log_handler import attach_gui_logging, detach_gui_logging
from lib.styles import Styles


class _DetectorWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    log_line = pyqtSignal(str, str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._log_handler = None

    def run(self):
        def on_log(msg, level):
            self.log_line.emit(msg, level)

        self._log_handler = attach_gui_logging(on_log, level=logging.DEBUG)
        root = logging.getLogger()
        root.addHandler(self._log_handler)
        try:
            self.finished.emit(self.func(*self.args, **self.kwargs))
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n\n{traceback.format_exc()}")
        finally:
            root.removeHandler(self._log_handler)
            detach_gui_logging(self._log_handler)


class DetectionTrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self._run_dir = None
        self._estimate_done = False
        self._init_ui()

    def _init_ui(self):
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        cfg_group = QGroupBox("Configuración")
        cfg_form = QFormLayout()
        row = QHBoxLayout()
        self.config_path = QLineEdit("config.yaml")
        browse = QPushButton("📁")
        browse.setFixedWidth(40)
        browse.clicked.connect(self._browse_config)
        row.addWidget(self.config_path)
        row.addWidget(browse)
        cfg_form.addRow("Config:", row)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        cfg_form.addRow("Pipeline:", self.summary_label)
        cfg_group.setLayout(cfg_form)
        layout.addWidget(cfg_group)

        opt_group = QGroupBox("Opciones de entrenamiento detector")
        opt_form = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 200)
        self.epochs_spin.setValue(10)
        self.epochs_spin.setToolTip("Épocas de entrenamiento (10 recomendado para prueba inicial)")
        opt_form.addRow("Épocas:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(8)
        self.batch_spin.setToolTip("Batch size (subir si hay VRAM libre)")
        opt_form.addRow("Batch size:", self.batch_spin)

        self.input_spin = QSpinBox()
        self.input_spin.setRange(224, 896)
        self.input_spin.setSingleStep(14)
        self.input_spin.setValue(504)
        self.input_spin.setToolTip("Resolución cuadrada; múltiplo de 14 (504 = 36×14)")
        opt_form.addRow("Input size (px):", self.input_spin)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 16)
        self.workers_spin.setValue(4)
        opt_form.addRow("Workers:", self.workers_spin)

        self.max_train_spin = QSpinBox()
        self.max_train_spin.setRange(0, 50000)
        self.max_train_spin.setValue(2000)
        self.max_train_spin.setSpecialValueText("Todas")
        self.max_train_spin.setToolTip("0 = dataset completo. 2000 ≈ prueba rápida (~2–4h)")
        opt_form.addRow("Máx. imgs train:", self.max_train_spin)

        self.max_val_spin = QSpinBox()
        self.max_val_spin.setRange(0, 10000)
        self.max_val_spin.setValue(400)
        self.max_val_spin.setSpecialValueText("Todas")
        opt_form.addRow("Máx. imgs val:", self.max_val_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1e-2)
        self.lr_spin.setValue(0.0001)
        self.lr_spin.setSingleStep(0.00005)
        opt_form.addRow("Learning rate:", self.lr_spin)

        self.unfrozen_spin = QSpinBox()
        self.unfrozen_spin.setRange(0, 12)
        self.unfrozen_spin.setValue(2)
        opt_form.addRow("Bloques unfrozen:", self.unfrozen_spin)

        self.val_fast_check = QCheckBox("Validación rápida (SOD en todo val; instancia en submuestra)")
        self.val_fast_check.setChecked(True)
        opt_form.addRow("", self.val_fast_check)

        self.val_f1_spin = QSpinBox()
        self.val_f1_spin.setRange(0, 2000)
        self.val_f1_spin.setValue(100)
        self.val_f1_spin.setToolTip("Imágenes val para det_p/r/f1 por época")
        opt_form.addRow("Muestras instancia/época:", self.val_f1_spin)

        self.loss_combo = QComboBox()
        self.loss_combo.addItem("BCE + Dice (bce_dice)", "bce_dice")
        self.loss_combo.addItem("U²-Net structure (u2net_structure)", "u2net_structure")
        opt_form.addRow("Loss:", self.loss_combo)

        self.metric_combo = QComboBox()
        self.metric_combo.addItem("det_f1 (instancia, IEEE)", "det_f1")
        self.metric_combo.addItem("Fm (SOD, U²-Net)", "Fm")
        self.metric_combo.addItem("mask_iou (rápido)", "mask_iou")
        opt_form.addRow("Métrica checkpoint:", self.metric_combo)

        opt_group.setLayout(opt_form)
        layout.addWidget(opt_group)

        self.detector_widget = ActiveDetectorWidget(config_path=self.config_path.text())
        det_note = QLabel(
            "Detector usado en <b>inferencia</b> y <b>evaluación E2E</b> (no en el entrenamiento DML, "
            "que siempre usa .seg). Elija U²-Net o su ViT-denso entrenado y pulse Guardar."
        )
        det_note.setWordWrap(True)
        det_note.setStyleSheet("color: #555; font-size: 11px; margin-bottom: 4px;")
        layout.addWidget(det_note)
        layout.addWidget(self.detector_widget)

        resume_row = QHBoxLayout()
        self.resume_check = QCheckBox("Reanudar")
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("runs/.../detector/last_detector.pth")
        ck_browse = QPushButton("📁")
        ck_browse.setFixedWidth(40)
        ck_browse.clicked.connect(self._browse_checkpoint)
        resume_row.addWidget(self.resume_check)
        resume_row.addWidget(self.checkpoint_path, 1)
        resume_row.addWidget(ck_browse)
        layout.addLayout(resume_row)

        controls = QHBoxLayout()
        self.save_btn = QPushButton("💾 Guardar")
        self.save_btn.setStyleSheet(Styles.BUTTON_SAVE)
        self.save_btn.setToolTip("Guardar opciones de esta pestaña en config.yaml")
        self.save_btn.clicked.connect(self.save_form)
        self.estimate_btn = QPushButton("⏱️ Estimar")
        self.estimate_btn.clicked.connect(self.estimate_time)
        self.train_btn = QPushButton("🚀 Entrenar detector")
        self.train_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.eval_btn = QPushButton("📊 Evaluar test")
        self.eval_btn.clicked.connect(self.evaluate_detector)
        self.eval_btn.setEnabled(False)
        self.u2net_eval_btn = QPushButton("📊 Evaluar U²-Net baseline")
        self.u2net_eval_btn.clicked.connect(self.evaluate_u2net_baseline)
        self.stop_btn = QPushButton("⏹ Detener")
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.save_btn)
        controls.addWidget(self.estimate_btn)
        controls.addWidget(self.train_btn)
        controls.addWidget(self.eval_btn)
        controls.addWidget(self.u2net_eval_btn)
        controls.addWidget(self.stop_btn)
        layout.addLayout(controls)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Listo — configure opciones y pulse Estimar")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setMinimumHeight(180)
        self.output_area.setStyleSheet(
            "QTextEdit { background:#1e1e1e; color:#d4d4d4; font-family: Consolas; font-size: 12px; }"
        )
        layout.addWidget(self.output_area)

        self.epochs_spin.valueChanged.connect(self._refresh_summary)
        self.batch_spin.valueChanged.connect(self._refresh_summary)
        self.max_train_spin.valueChanged.connect(self._refresh_summary)
        self.input_spin.valueChanged.connect(self._refresh_summary)
        self.loss_combo.currentIndexChanged.connect(self._refresh_summary)
        self.metric_combo.currentIndexChanged.connect(self._refresh_summary)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def sync_config_path(self, path: str):
        if path:
            self.config_path.setText(path)
            self.detector_widget.set_config_path(path)
            self._load_options_from_config()
            self._refresh_summary()

    def _browse_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Config", "", "YAML (*.yaml *.yml)")
        if path:
            self.config_path.setText(path)
            self.detector_widget.set_config_path(path)
            self._load_options_from_config()

    def _browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Checkpoint detector", "", "PyTorch (*.pth)")
        if path:
            self.checkpoint_path.setText(path)

    def _load_options_from_config(self):
        from lib.gui_tab_prefs import apply_detector_tab_to_widgets, load_config

        try:
            p = self._resolve_config_path()
            if not p.exists():
                return
            apply_detector_tab_to_widgets(self, load_config(p))
            self._refresh_summary()
        except Exception:
            pass

    def save_form(self):
        from lib.gui_tab_prefs import (
            _deep_update,
            detector_tab_updates_from_widgets,
            load_config,
            save_config_full,
        )
        from src.utils.detector_resolution_config import sync_detector_resolution_config

        try:
            p = self._resolve_config_path()
            if not p.exists():
                QMessageBox.warning(self, "Error", f"No existe config: {p}")
                return
            cfg = load_config(p)
            _deep_update(cfg, detector_tab_updates_from_widgets(self))
            cfg, _ = sync_detector_resolution_config(cfg)
            save_config_full(p, cfg)
            self.detector_widget.restore_from_config()
            self._refresh_summary()
            self.status_label.setText("Opciones guardadas en config.yaml")
            self._append_log(f"Guardado: {p}", "INFO")
            if self.parent_window and hasattr(self.parent_window, "log_widget"):
                self.parent_window.log_widget.append_log(
                    "Pestaña Detección guardada en config.yaml", "SUCCESS"
                )
            QMessageBox.information(self, "Guardado", "Opciones de detección guardadas en config.yaml")
            if self.parent_window and hasattr(self.parent_window, "training_tab"):
                self.parent_window.training_tab.refresh_detector_for_results_summary()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar:\n{exc}")

    def _resolve_config_path(self) -> Path:
        p = Path(self.config_path.text())
        if not p.is_absolute():
            root = Path(__file__).resolve().parent.parent
            if (root / p).exists():
                p = root / p
        return p

    def _refresh_summary(self):
        import yaml
        from src.utils.detector_resolution_config import resolve_detector_input_size
        try:
            p = self._resolve_config_path()
            with open(p, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            det = cfg.get("detection_training", {})
            size = self.input_spin.value()
            self.summary_label.setText(
                f"ViT-denso | input={size}px | batch={self.batch_spin.value()} | "
                f"epochs={self.epochs_spin.value()} | loss={self.loss_combo.currentData()} | "
                f"metric={self.metric_combo.currentData()} | "
                f"train≤{self.max_train_spin.value() or 'all'}"
            )
        except Exception:
            self.summary_label.setText("(config no disponible)")

    def _append_log(self, msg: str, level: str = "INFO"):
        colors = {
            "DEBUG": "#808080",
            "INFO": "#4ec9b0",
            "WARNING": "#dcdcaa",
            "ERROR": "#f48771",
            "CRITICAL": "#ff6b6b",
        }
        c = colors.get(level, "#d4d4d4")
        self.output_area.append(f'<span style="color:{c}">{msg}</span>')
        if self.parent_window and hasattr(self.parent_window, "log_widget"):
            self.parent_window.log_widget.append_log(msg, level)

    def _warn_unsaved(self) -> bool:
        main = self.parent_window
        if main and hasattr(main, "config_tab") and main.config_tab.is_dirty():
            r = QMessageBox.warning(
                self, "Config sin guardar",
                "Hay cambios sin guardar. ¿Continuar con el YAML en disco + opciones de esta pestaña?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            return r == QMessageBox.Yes
        return True

    def _build_args(self):
        return argparse.Namespace(
            config=str(self._resolve_config_path()),
            resume=self.checkpoint_path.text() if self.resume_check.isChecked() else None,
            gpu=None,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            val_batch_size=self.batch_spin.value(),
            input_size=self.input_spin.value(),
            num_workers=self.workers_spin.value(),
            max_train_images=self.max_train_spin.value(),
            max_val_images=self.max_val_spin.value(),
            val_fast=self.val_fast_check.isChecked(),
            val_det_f1_max_images=self.val_f1_spin.value(),
            num_unfrozen_blocks=self.unfrozen_spin.value(),
            learning_rate=self.lr_spin.value(),
            loss_type=self.loss_combo.currentData(),
            validation_metric=self.metric_combo.currentData(),
        )

    def _start_worker(self, func, *args, **kwargs):
        self.worker = _DetectorWorker(func, *args, **kwargs)
        self.worker.log_line.connect(self._append_log)
        self.worker.finished.connect(lambda r: self._on_worker_finished(r, func))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_worker_finished(self, result, func):
        from scripts.train_detector import run_detector_training
        from scripts.evaluate_detector import evaluate_detector as eval_fn
        from scripts.evaluate_u2net_baseline import evaluate_u2net_baseline as u2_fn
        if func == run_detector_training:
            if isinstance(result, dict) and result.get("status") == "completed":
                self._on_train_done(result)
            else:
                self._on_estimate_done(result)
        elif func == eval_fn:
            self._on_eval_done(result)
            self.eval_btn.setEnabled(True)
        elif func == u2_fn:
            self._on_u2net_eval_done(result)
            self.u2net_eval_btn.setEnabled(True)

    def estimate_time(self):
        if not self._warn_unsaved():
            return
        if self.max_train_spin.value() == 0:
            r = QMessageBox.warning(
                self,
                "Dataset completo",
                "Máx. imgs train = Todas (11k+ imágenes). La estimación puede superar varios días.\n\n"
                "¿Continuar? (Recomendado: 2000 para prueba inicial)",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if r != QMessageBox.Yes:
                return
        self.output_area.clear()
        self._append_log("Iniciando estimación del detector...", "INFO")
        self.estimate_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Estimando...")
        from scripts.train_detector import run_detector_training
        self._start_worker(run_detector_training, self._build_args(), estimate_only=True)

    def _on_estimate_done(self, result):
        self.estimate_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._estimate_done = True
        self._run_dir = result.get("run_dir")
        total = result.get("total_sec", 0)
        train_ep = result.get("train_epoch_sec", 0)
        val_ep = result.get("val_epoch_sec", 0)
        self._append_log("——— Resumen estimación ———", "INFO")
        self._append_log(
            f"Imágenes: train={result.get('train_images')} val={result.get('val_images')}", "INFO"
        )
        self._append_log(
            f"Batches/época: train={result.get('train_batches')} val={result.get('val_batches')}", "INFO"
        )
        self._append_log(
            f"Throughput: {result.get('train_sec_per_batch', 0):.2f}s/batch train | "
            f"{result.get('val_sec_per_batch', 0):.2f}s/batch val", "INFO"
        )
        self._append_log(
            f"Por época: train {train_ep/60:.1f} min + val {val_ep/60:.1f} min", "INFO"
        )
        self._append_log(
            f"TOTAL ({result.get('epochs')} épocas): {total/60:.1f} min ({total/3600:.1f} h)", "INFO"
        )
        self._append_log(f"Run: {result.get('run_dir')}", "DEBUG")
        if total > 86400:
            self._append_log(
                "WARNING: >24h estimado. Reduzca épocas o max_train_images.", "WARNING"
            )
        self.status_label.setText("Estimación completa — puede entrenar")

    def start_training(self):
        if not self._estimate_done and not self._run_dir:
            QMessageBox.information(self, "Estimar primero", "Ejecute 'Estimar' antes de entrenar.")
            return
        if not self._warn_unsaved():
            return
        self._append_log("Iniciando entrenamiento detector...", "INFO")
        self.train_btn.setEnabled(False)
        self.estimate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        from scripts.train_detector import run_detector_training
        self._start_worker(
            run_detector_training,
            self._build_args(),
            estimate_only=False,
            run_dir=self._run_dir,
        )

    def _on_train_done(self, result):
        self.train_btn.setEnabled(True)
        self.estimate_btn.setEnabled(True)
        self.eval_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Entrenamiento completado")
        self._append_log(
            f"Entrenamiento OK — best_{result.get('validation_metric')}="
            f"{result.get('best_metric', 0):.4f}", "INFO"
        )
        det_dir = result.get("detector_dir", "")
        viz_dir = result.get("visualization_dir", "")
        curves_dir = str(Path(det_dir) / "curves") if det_dir else ""
        eval_dir = str(Path(det_dir) / "evaluation") if det_dir else ""
        if curves_dir:
            self._append_log(f"Curvas: {curves_dir}", "INFO")
            self._embed_curve_images(curves_dir)
        if eval_dir:
            self._append_log(f"Evaluación + vis_grid: {eval_dir}", "INFO")
        if viz_dir:
            self._append_log(f"Copia visualización (layout DML): {viz_dir}", "INFO")
        ckpt = str(Path(det_dir) / "best_detector.pth") if det_dir else ""
        if ckpt and Path(ckpt).exists():
            self.detector_widget._refresh_checkpoints()
            self.detector_widget.backend_combo.setCurrentIndex(1)
            for i in range(self.detector_widget.checkpoint_combo.count()):
                if self.detector_widget.checkpoint_combo.itemData(i) == ckpt:
                    self.detector_widget.checkpoint_combo.setCurrentIndex(i)
                    break
            self.detector_widget.load_detector(silent=True)
            reply = QMessageBox.question(
                self,
                "Guardar detector activo",
                "¿Guardar este ViT-denso como detector activo?\n\n"
                "Debe hacerlo <b>antes</b> de entrenar el clasificador para que los "
                "resultados E2E usen este detector junto con su modelo.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                self.save_form()

    def evaluate_detector(self):
        ckpt = self.detector_widget.checkpoint_combo.currentData()
        if not ckpt and self._run_dir:
            p = Path(self._run_dir) / "detector" / "best_detector.pth"
            ckpt = str(p) if p.exists() else None
        if not ckpt or not Path(str(ckpt)).exists():
            QMessageBox.warning(self, "Error", "No hay checkpoint para evaluar.")
            return
        from scripts.evaluate_detector import evaluate_detector as eval_fn
        self.eval_btn.setEnabled(False)
        self._append_log(f"Evaluando checkpoint: {ckpt}", "INFO")
        self.worker = _DetectorWorker(eval_fn, str(self._resolve_config_path()), str(ckpt))
        self.worker.log_line.connect(self._append_log)
        self.worker.finished.connect(self._on_eval_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _embed_curve_images(self, curves_dir: str):
        """Muestra PNGs de curvas en la consola (como pre-análisis DML)."""
        titles = {
            "loss_curves.png": "Loss train/val",
            "sod_metrics.png": "Métricas SOD (Fm, mask_iou, sod_mae)",
            "instance_metrics.png": "Métricas instancia (P/R/F1, grains_mae)",
        }
        for fname, title in titles.items():
            p = Path(curves_dir) / fname
            if p.exists():
                abs_path = p.resolve().as_posix()
                self.output_area.append(
                    f'<br><b>{title}</b><br>'
                    f'<img src="file:///{abs_path}" width="750"><br>'
                )

    def _format_eval_report(self, report: dict, title: str) -> None:
        self._append_log(f"——— {title} ———", "INFO")
        self._append_log(
            f"SOD: Fm={report.get('Fm', 0):.4f} sod_mae={report.get('sod_mae', 0):.4f} "
            f"mask_iou={report.get('mask_iou', 0):.4f}",
            "INFO",
        )
        self._append_log(
            f"Instancia: det_p={report.get('det_precision', 0):.4f} "
            f"det_r={report.get('det_recall', 0):.4f} det_f1={report.get('det_f1', 0):.4f} "
            f"grains_mae={report.get('grains_mae', 0):.4f}",
            "INFO",
        )

    def evaluate_u2net_baseline(self):
        from scripts.evaluate_u2net_baseline import evaluate_u2net_baseline as u2_fn
        out_dir = None
        if self._run_dir:
            out_dir = str(Path(self._run_dir) / "detector" / "evaluation" / "u2net_baseline")
        self.u2net_eval_btn.setEnabled(False)
        self._append_log("Evaluando U²-Net baseline (métricas IEEE A+B)...", "INFO")
        self.worker = _DetectorWorker(
            u2_fn, str(self._resolve_config_path()), out_dir
        )
        self.worker.log_line.connect(self._append_log)
        self.worker.finished.connect(lambda r: self._on_worker_finished(r, u2_fn))
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_u2net_eval_done(self, report):
        self._format_eval_report(report, "U²-Net baseline")

    def _on_eval_done(self, report):
        self.eval_btn.setEnabled(True)
        self._format_eval_report(report, "Eval test ViT-denso")

    def _on_error(self, msg):
        self.estimate_btn.setEnabled(True)
        self.train_btn.setEnabled(self._estimate_done)
        self.eval_btn.setEnabled(True)
        self.u2net_eval_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self._append_log(msg, "ERROR")
        self.status_label.setText("Error")
