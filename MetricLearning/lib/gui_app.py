"""
MetricLearning - Aplicación GUI con PyQt5
Interfaz gráfica profesional siguiendo estándares IEEE para usabilidad.

Arquitectura: Model-View-Controller (MVC)
- Model: Lógica de negocio (src/)
- View: Interfaz gráfica (PyQt5)
- Controller: Coordinación entre Model y View
"""

import logging
import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog,
    QComboBox, QSpinBox, QProgressBar, QMessageBox, QGroupBox,
    QFormLayout, QCheckBox, QListWidget, QSplitter, QStatusBar,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QFont, QTextCursor, QPalette, QColor, QPixmap, QImage
import datetime
import traceback

from src.utils.logging_utils import setup_root_logging
from lib.config_editor import ConfigEditorTab
from lib.dataset_manager_gui import DatasetManagerTab
from lib.inference_tab import InferenceTab
from lib.contour_analysis_tab import ContourAnalysisTab
from lib.detector_training_tab import DetectionTrainingTab
from lib.styles import Styles, Colors, apply_global_styles


class WorkerThread(QThread):
    """
    Thread worker para operaciones largas sin bloquear la GUI.
    Sigue patrón de diseño Observer para comunicación asíncrona.
    """
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
    
    def run(self):
        """Ejecuta función en thread separado"""
        try:
            self.status.emit("Iniciando operación...")
            self.result = self.func(*self.args, **self.kwargs)
            self.finished.emit(self.result)
        except KeyboardInterrupt:
            self.error.emit("Operación cancelada por el usuario.")
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{str(e)}\n\nTraceback:\n{tb}")


class LogWidget(QTextEdit):
    """
    Widget personalizado para mostrar logs en tiempo real.
    Implementa patrón Singleton para logging centralizado.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(120)
        self.setStyleSheet(Styles.LOG_WIDGET)
    
    def append_log(self, message, level="INFO"):
        """Agrega mensaje de log con color según nivel y timestamp"""
        colors = {
            "DEBUG": "#808080",
            "INFO": Colors.SUCCESS,
            "WARNING": Colors.WARNING,
            "ERROR": Colors.ERROR,
            "SUCCESS": Colors.SUCCESS
        }
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = colors.get(level, Colors.TEXT_LIGHT)
        formatted = f'<span style="color: {color};">[{timestamp}] [{level}] {message}</span>'
        self.append(formatted)
        
        # Auto-scroll al final
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)


class TrainingTab(QWidget):
    """
    Tab para entrenamiento del modelo.
    Implementa formulario con validación y feedback visual.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self._run_dir = None
        self._estimate_done = False
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # === Configuración ===
        config_group = QGroupBox("Configuración de Entrenamiento")
        config_layout = QFormLayout()
        
        # Config file
        config_file_layout = QHBoxLayout()
        self.config_path = QLineEdit("config.yaml")
        self.config_path.setPlaceholderText("Ruta al archivo de configuración")
        config_browse = QPushButton("📁 Buscar")
        config_browse.clicked.connect(self.browse_config)
        config_file_layout.addWidget(self.config_path)
        config_file_layout.addWidget(config_browse)
        config_layout.addRow("Archivo Config:", config_file_layout)

        self.ablation_summary_label = QLabel("")
        self.ablation_summary_label.setToolTip(
            "El informe post-entrenamiento (métricas, espectral, full-image) "
            "siempre se genera tras 1 entrenamiento.\n\n"
            "El estudio de ablación F-01…F-07 es OPCIONAL: re-entrena variantes "
            "para una tabla comparativa académica (no sustituye al informe)."
        )
        self.ablation_summary_label.setWordWrap(True)
        self.ablation_summary_label.setStyleSheet(
            "color: #0e639c; font-size: 11px; padding: 4px; background: #e8f4f8; border-radius: 4px;"
        )
        config_layout.addRow("Estudio ablación:", self.ablation_summary_label)

        self.crops_info_label = QLabel(
            "<b>Flujo:</b> 1) Entrenar detector en <b>Detección</b> → "
            "2) Seleccionar detector y <b>Guardar</b> → "
            "3) Entrenar clasificador aquí (crops siempre desde <b>.seg</b>) → "
            "4) Resultados E2E = clasificador + detector elegido."
        )
        self.crops_info_label.setWordWrap(True)
        self.crops_info_label.setStyleSheet(
            "color: #555; font-size: 11px; padding: 6px; background: #f5f5f5; border-radius: 4px;"
        )
        config_layout.addRow("Flujo:", self.crops_info_label)

        self.detector_results_label = QLabel("Detector E2E: (cargando…)")
        self.detector_results_label.setWordWrap(True)
        self.detector_results_label.setStyleSheet(
            "color: #0e639c; font-size: 11px; padding: 6px; background: #e8f4f8; border-radius: 4px;"
        )
        config_layout.addRow("Resultados E2E:", self.detector_results_label)
        
        # Checkpoint (opcional)
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("Opcional: checkpoint para reanudar")
        checkpoint_browse = QPushButton("📁 Buscar")
        checkpoint_browse.clicked.connect(self.browse_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_path)
        checkpoint_layout.addWidget(checkpoint_browse)
        config_layout.addRow("Checkpoint:", checkpoint_layout)
        
        # Resume checkbox
        self.resume_check = QCheckBox("Reanudar entrenamiento")
        config_layout.addRow("", self.resume_check)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # === Controles ===
        controls_layout = QHBoxLayout()

        self.save_btn = QPushButton("💾 Guardar")
        self.save_btn.setStyleSheet(Styles.BUTTON_SAVE)
        self.save_btn.setToolTip("Guardar opciones de esta pestaña en config.yaml")
        self.save_btn.clicked.connect(self.save_form)

        self.estimate_btn = QPushButton("⏱️ Estimar Tiempo")
        self.estimate_btn.clicked.connect(self.estimate_time)
        
        self.train_btn = QPushButton("🚀 Iniciar Entrenamiento")
        self.train_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.train_btn.clicked.connect(self.start_training)
        self.train_btn.setEnabled(False)
        self.train_btn.setToolTip("Primero debes ejecutar 'Estimar Tiempo'")
        
        self.stop_btn = QPushButton("⏹ Detener")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        
        controls_layout.addWidget(self.save_btn)
        controls_layout.addWidget(self.estimate_btn)
        controls_layout.addWidget(self.train_btn)
        controls_layout.addWidget(self.stop_btn)
        layout.addLayout(controls_layout)
        
        # === Progress ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # === Status ===
        self.status_label = QLabel("Listo para entrenar")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # === Terminal de Entrenamiento ===
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
                padding: 10px;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
            }
        """)
        self.output_area.setPlaceholderText("Presiona '⏱️ Estimar Tiempo' para generar el análisis pre-entrenamiento...")
        layout.addWidget(self.output_area, 1)
        
        self.setLayout(layout)
    
    def browse_config(self):
        """Buscar archivo de configuración"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Config", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path.setText(file_path)
    
    def sync_config_path(self, path: str):
        """Sincroniza la ruta de config desde la pestaña Configuración."""
        if path:
            self.config_path.setText(path)
            self.refresh_ablation_summary(path)
            self._load_tab_prefs(path)
            self.refresh_detector_for_results_summary(path)

    def refresh_detector_for_results_summary(self, config_path: str = None):
        """Muestra qué detector se usará en resultados full-image del clasificador."""
        from lib.gui_tab_prefs import load_config, resolve_config_path
        from src.grain_detection.detector_registry import get_active_detector_config, validate_active_detector

        path = config_path or self.config_path.text()
        style_ok = (
            "color: #0e639c; font-size: 11px; padding: 6px; background: #e8f4f8; border-radius: 4px;"
        )
        style_warn = (
            "color: #d9480f; font-size: 11px; padding: 6px; background: #fff4e6; border-radius: 4px;"
        )
        try:
            p = resolve_config_path(path)
            if not p.exists():
                self.detector_results_label.setText("Detector E2E: config no encontrada")
                self.detector_results_label.setStyleSheet(style_warn)
                return
            cfg = load_config(p)
            errors = validate_active_detector(cfg)
            active = get_active_detector_config(cfg)
            label = active.get("label") or active.get("backend", "?")
            if errors:
                self.detector_results_label.setText(
                    "Detector E2E: <b>no configurado</b> — vaya a Detección, seleccione "
                    f"su detector y pulse Guardar.<br>{errors[0]}"
                )
                self.detector_results_label.setStyleSheet(style_warn)
            else:
                saved = active.get("loaded_at")
                saved_note = f" (guardado {saved})" if saved else ""
                self.detector_results_label.setText(
                    f"Detector E2E: <b>{label}</b>{saved_note} — "
                    "se usará en full-image inference al terminar el entrenamiento."
                )
                self.detector_results_label.setStyleSheet(style_ok)
        except Exception:
            self.detector_results_label.setText("Detector E2E: error leyendo config")
            self.detector_results_label.setStyleSheet(style_warn)

    def _load_tab_prefs(self, config_path: str = None):
        from lib.gui_tab_prefs import apply_dml_tab_to_widgets, load_config, resolve_config_path

        path = config_path or self.config_path.text()
        try:
            p = resolve_config_path(path)
            if p.exists():
                apply_dml_tab_to_widgets(self, load_config(p))
                self.refresh_detector_for_results_summary(str(p))
        except Exception:
            pass

    def _ensure_dml_uses_seg(self):
        """Entrenamiento DML siempre con .seg; no altera el detector activo de inferencia."""
        from lib.gui_tab_prefs import load_config, resolve_config_path, save_config_full

        p = resolve_config_path(self.config_path.text())
        if not p.exists():
            return
        cfg = load_config(p)
        cfg.setdefault("data", {})["crop_source"] = "seg"
        save_config_full(p, cfg, backup=False)

    def save_form(self):
        from lib.gui_tab_prefs import (
            _deep_update,
            dml_tab_updates_from_widgets,
            load_config,
            resolve_config_path,
            save_config_full,
        )

        try:
            p = resolve_config_path(self.config_path.text())
            if not p.exists():
                QMessageBox.warning(self, "Error", f"No existe config: {p}")
                return
            cfg = load_config(p)
            _deep_update(cfg, dml_tab_updates_from_widgets(self))
            cfg.setdefault("data", {})["crop_source"] = "seg"
            save_config_full(p, cfg)
            self.status_label.setText("Opciones guardadas en config.yaml")
            if self.parent_window and hasattr(self.parent_window, "log_widget"):
                self.parent_window.log_widget.append_log(
                    "Pestaña Entrenamiento DML guardada en config.yaml", "SUCCESS"
                )
            QMessageBox.information(self, "Guardado", "Opciones de entrenamiento guardadas en config.yaml")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"No se pudo guardar:\n{exc}")

    def refresh_ablation_summary(self, config_path: str = None):
        """Lee evaluation.ablation del YAML y actualiza el resumen en la pestaña Entrenamiento."""
        import yaml
        from src.utils.ablation_matrix import format_ablation_startup_summary

        path = config_path or self.config_path.text()
        summary = "Ablaciones: (config no encontrada)"
        try:
            p = Path(path)
            if not p.is_absolute():
                root = Path(__file__).resolve().parent.parent
                if (root / p).exists():
                    p = root / p
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                from src.utils.ablation_matrix import ensure_ablation_defaults
                ensure_ablation_defaults(cfg)
                summary = format_ablation_startup_summary(cfg)
        except Exception:
            summary = "Ablaciones: error leyendo config"

        self.ablation_summary_label.setText(summary)
        style_on = "color: #0e639c; font-size: 11px; padding: 4px; background: #e8f4f8; border-radius: 4px;"
        style_warn = "color: #d9480f; font-size: 11px; padding: 4px; background: #fff4e6; border-radius: 4px;"
        style_off = "color: #666; font-size: 11px; padding: 4px; background: #f5f5f5; border-radius: 4px;"
        if "re-entrenamiento" in summary:
            self.ablation_summary_label.setStyleSheet(style_on)
        elif "seleccione" in summary:
            self.ablation_summary_label.setStyleSheet(style_warn)
        else:
            self.ablation_summary_label.setStyleSheet(style_off)
    
    def _warn_if_config_unsaved(self) -> bool:
        """
        Advierte si la pestaña Config tiene cambios sin guardar.
        Returns True si se puede continuar, False si el usuario cancela.
        """
        main = self.parent_window
        if main is None or not hasattr(main, "config_tab"):
            return True
        if not main.config_tab.is_dirty():
            return True
        reply = QMessageBox.warning(
            self,
            "Configuración sin guardar",
            "La pestaña Configuración tiene cambios que no se han guardado en disco.\n\n"
            "El entrenamiento usará el archivo YAML en disco, no los valores editados en pantalla.\n\n"
            "¿Desea continuar de todos modos?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes
    
    def browse_checkpoint(self):
        """Buscar checkpoint"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Checkpoint", "", "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.checkpoint_path.setText(file_path)
    
    def _build_args(self):
        """Construye argparse.Namespace desde los campos de la UI"""
        import argparse
        config_path = self.config_path.text()
        checkpoint_path = self.checkpoint_path.text()
        return argparse.Namespace(
            config=config_path,
            resume=checkpoint_path if (self.resume_check.isChecked() and checkpoint_path) else None,
            gpu=None
        )
    
    def _validate_detector_for_results(self) -> bool:
        """Exige detector activo guardado antes de estimar/entrenar el clasificador."""
        from lib.gui_tab_prefs import load_config, resolve_config_path
        from src.grain_detection.detector_registry import get_active_detector_config, validate_active_detector

        p = resolve_config_path(self.config_path.text())
        if not p.exists():
            return True
        cfg = load_config(p)
        errors = validate_active_detector(cfg)
        if errors:
            QMessageBox.warning(
                self,
                "Detector requerido",
                "Antes de entrenar el clasificador debe elegir el detector para los resultados E2E:\n\n"
                "1. Pestaña <b>Detección</b> → entrenar ViT-denso (o U²-Net)\n"
                "2. Seleccionar checkpoint → <b>Guardar</b>\n"
                "3. Volver aquí y estimar/entrenar\n\n"
                + "\n".join(f"• {e}" for e in errors),
            )
            return False
        active = get_active_detector_config(cfg)
        self.parent_window.log_widget.append_log(
            f"Resultados E2E usarán detector: {active.get('label', active.get('backend'))}",
            "INFO",
        )
        return True

    def _validate_inputs(self):
        """Valida inputs y retorna True si son válidos"""
        if not self._warn_if_config_unsaved():
            return False
        config_path = self.config_path.text()
        if not config_path or not Path(config_path).exists():
            QMessageBox.warning(self, "Error", "Archivo de configuración no válido")
            return False
        checkpoint_path = self.checkpoint_path.text()
        if checkpoint_path and not Path(checkpoint_path).exists():
            QMessageBox.warning(self, "Error", "Checkpoint no encontrado")
            return False
        if not self._validate_detector_for_results():
            return False
        return True
    
    def estimate_time(self):
        """Estima tiempo de entrenamiento en background y escribe resultados al log"""
        if not self._validate_inputs():
            return
        self._ensure_dml_uses_seg()
        self.estimate_btn.setEnabled(False)
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("⏱️ Estimando tiempo de entrenamiento...")
        self.parent_window.log_widget.append_log("Estimando tiempo de entrenamiento...", "INFO")
        
        from scripts.train import run_training
        args = self._build_args()
        self.estimate_worker = WorkerThread(run_training, args, estimate_only=True)
        self.estimate_worker.finished.connect(self._on_estimate_ready)
        self.estimate_worker.error.connect(self._on_estimate_error)
        self.estimate_worker.start()
    
    def _on_estimate_error(self, error_msg):
        """Callback cuando falla la estimación"""
        self.estimate_btn.setEnabled(True)
        self.train_btn.setEnabled(self._estimate_done)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Error en estimación")
        self.parent_window.log_widget.append_log(f"Error estimando: {error_msg.split(chr(10))[0]}", "ERROR")
    
    def _on_estimate_ready(self, estimate):
        """Callback cuando la estimación termina — muestra análisis pre-entrenamiento en output_area"""
        try:
            if not isinstance(estimate, dict):
                self.status_label.setText("Error en estimacion")
                self.parent_window.log_widget.append_log("Estimacion no retorno datos validos", "ERROR")
                return

            self._run_dir = estimate.get('run_dir')
            self._estimate_done = True
            self.train_btn.setEnabled(True)
            self.train_btn.setToolTip("")

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analysis = estimate.get('analysis', {})
            charts = analysis.get('charts', {})
            run_paths = analysis.get('run_paths', {})

            stats = analysis.get('stats', {})
            train_stats = stats.get('train', {})

            sep = '<span style="color: #569cd6;">' + '=' * 60 + '</span><br>'
            hdr = lambda t: f'<span style="color: #4ec9b0; font-size: 14px; font-weight: bold;">  {t}</span><br>'
            lbl = lambda t: f'<span style="color: #dcdcaa;">{t}</span>'
            val = lambda t: f'<span style="color: #d4d4d4;">{t}</span>'
            dim = lambda t: f'<span style="color: #808080;">{t}</span>'

            html_parts = [
                '<div style="font-family: Consolas, monospace; color: #d4d4d4; background: #1e1e1e; padding: 8px;">',
            ]

            html_parts += [
                sep,
                f'<span style="color: #4ec9b0; font-size: 16px; font-weight: bold;">  ANALISIS PRE-ENTRENAMIENTO</span><br>',
                sep,
                dim(f'  [{ts}]') + '<br>',
            ]
            if run_paths:
                html_parts.append(dim(f'  Run: {run_paths.get("run_dir", "N/A")}') + '<br>')
            html_parts.append('<br>')

            html_parts += [
                hdr('Modelo & Hiperparametros'),
                val(f'     Backbone:      <b>{estimate.get("model_backbone", "N/A")}</b>') + '<br>',
                val(f'     Embedding dim: <b>{estimate.get("embedding_dim", "N/A")}</b>') + '<br>',
                val(f'     Loss:          <b>{estimate.get("loss_type", "N/A")}</b>') + '<br>',
                val(f'     Optimizer:     <b>{estimate.get("optimizer", "N/A")}</b>') + '<br>',
                val(f'     Learning rate: <b>{estimate.get("learning_rate", "N/A")}</b>') + '<br>',
                val(f'     Scheduler:     <b>{estimate.get("scheduler", "N/A")}</b>') + '<br>',
                val(f'     Mixed prec:    <b>{"Si" if estimate.get("use_amp") else "No"}</b>') + '<br>',
                val(f'     Workers:       <b>{estimate.get("num_workers", "N/A")}</b>') + '<br>',
                '<br>',
            ]

            html_parts += [
                hdr('Dataset'),
                val(f'     Train:      <b>{estimate.get("train_samples", "N/A")}</b> granos') + '<br>',
                val(f'     Val:        <b>{estimate.get("val_samples", "N/A")}</b> granos') + '<br>',
                val(f'     Clases:     <b>{estimate.get("num_classes", "N/A")}</b>') + '<br>',
                val(f'     Batch size: <b>{estimate.get("batch_size", "N/A")}</b>') + '<br>',
            ]

            img_sizes = train_stats.get('image_sizes', {})
            if img_sizes:
                html_parts += [
                    '<br>' + lbl('     Tamano de crops (muestreo):') + '<br>',
                    val(
                        f'       Ancho:  min={img_sizes.get("width_min", "?")}  '
                        f'max={img_sizes.get("width_max", "?")}  '
                        f'media={img_sizes.get("width_mean", "?")}'
                    ) + '<br>',
                    val(
                        f'       Alto:   min={img_sizes.get("height_min", "?")}  '
                        f'max={img_sizes.get("height_max", "?")}  '
                        f'media={img_sizes.get("height_mean", "?")}'
                    ) + '<br>',
                ]
                if img_sizes.get('aspect_ratio_mean') is not None:
                    html_parts.append(
                        val(f'       Aspect ratio medio: <b>{img_sizes["aspect_ratio_mean"]}</b>') + '<br>'
                    )
                if img_sizes.get('sampled') is not None:
                    html_parts.append(dim(f'       (muestreadas {img_sizes["sampled"]} muestras)') + '<br>')

            class_dist = train_stats.get('class_distribution', {})
            if class_dist:
                html_parts.append('<br>' + lbl('     Distribucion por clase:') + '<br>')
                max_name_len = max(len(n) for n in class_dist.keys())
                for cls_name, count in class_dist.items():
                    train_n = estimate.get("train_samples") or 0
                    pct = count / train_n * 100 if train_n > 0 else 0
                    bar_len = min(50, int(pct / 2))
                    bar = '#' * bar_len + '.' * (50 - bar_len)
                    html_parts.append(
                        val(f'       {cls_name:<{max_name_len}}  {count:>5}  ({pct:5.1f}%)  ') +
                        f'<span style="color: #4ec9b0;">{bar}</span><br>'
                    )
                if train_stats.get('min_samples_per_class') is not None:
                    html_parts.append(dim(
                        f'       Min: {train_stats["min_samples_per_class"]} | '
                        f'Max: {train_stats["max_samples_per_class"]} | '
                        f'Media: {train_stats["mean_samples_per_class"]} | '
                        f'Std: {train_stats["std_samples_per_class"]}'
                    ) + '<br>')
            html_parts.append('<br>')

            html_parts += [
                hdr('Benchmark de Rendimiento'),
                val(f'     Forward+Loss+Backward+Opt: <b>{estimate.get("avg_train_step_sec", 0):.3f}s</b>/batch') + '<br>',
                val(f'     Data loading:              <b>{estimate.get("avg_data_load_sec", 0):.3f}s</b>/batch') + '<br>',
                val(f'     Overhead factor:           <b>{estimate.get("data_overhead_factor", 1):.2f}x</b>') + '<br>',
                val(f'     Validacion:                <b>{estimate.get("avg_val_batch_sec", 0):.3f}s</b>/batch') + '<br>',
                '<br>',
            ]

            html_parts += [
                hdr('Estimacion de Tiempo'),
                val(
                    f'     Entrenamiento: {estimate.get("train_batches_per_epoch", 0)} batches x '
                    f'{estimate.get("avg_train_batch_sec", 0):.2f}s = '
                    f'<b>{estimate.get("train_epoch_fmt", "N/A")}</b>'
                ) + '<br>',
                val(
                    f'     Validacion:    {estimate.get("val_batches_per_epoch", 0)} batches x '
                    f'{estimate.get("avg_val_batch_sec", 0):.2f}s + recall = '
                    f'<b>{estimate.get("val_epoch_fmt", "N/A")}</b>'
                ) + '<br>',
                val(
                    f'     Ref. embeddings (una vez, post-training):  '
                    f'<b>{estimate.get("ref_embed_overhead_sec", 0):.0f}s</b>'
                ) + '<br>',
                val(f'     Total epoca:   <b>{estimate.get("epoch_total_fmt", "N/A")}</b>') + '<br>',
                '<br>',
                f'<span style="color: #4ec9b0; font-size: 15px; font-weight: bold;">'
                f'  TOTAL ESTIMADO: {estimate.get("epochs", "?")} epocas x '
                f'{estimate.get("epoch_total_fmt", "N/A")} + ref_embed = '
                f'{estimate.get("total_fmt", "N/A")}</span><br>',
                '<br>', sep,
            ]

            abl_note = estimate.get("ablation_note")
            if abl_note:
                html_parts += [
                    '<span style="color: #ffa94d; font-weight: bold;">'
                    f'  ABLACIONES: {abl_note}</span><br>',
                    '<span style="color: #888;">'
                    '  (se ejecutan automaticamente al pulsar Entrenar, tras el run principal)</span><br>',
                    '<br>', sep,
                ]

            chart_titles = {
                'class_distribution': 'Distribucion de Clases',
                'train_val_balance': 'Balance Train vs Validacion',
                'sample_grid': 'Muestras por Clase (Top 10)',
                'augmentation_preview': 'Vista Previa de Aumentacion',
            }

            for key, title in chart_titles.items():
                img_path = charts.get(key)
                if img_path and Path(img_path).exists():
                    abs_path = Path(img_path).resolve().as_posix()
                    html_parts.append(f'<br>{hdr(title)}')
                    html_parts.append(f'<img src="file:///{abs_path}" width="750"><br>')

            html_parts.append('</div>')

            self.output_area.clear()
            self.output_area.setHtml("".join(html_parts))

            cursor = self.output_area.textCursor()
            cursor.movePosition(QTextCursor.Start)
            self.output_area.setTextCursor(cursor)

            total_fmt = estimate.get('total_fmt', 'N/A')
            epochs = estimate.get('epochs', '?')
            self.status_label.setText(
                f"Analisis listo - Estimado: {total_fmt} ({epochs} epocas) - Listo para entrenar"
            )
            self.parent_window.log_widget.append_log(
                f"Analisis pre-entrenamiento completado: {total_fmt} total ({epochs} epocas)", "SUCCESS"
            )
            self.parent_window.log_widget.append_log(
                f"Carpeta de resultados: {self._run_dir}", "INFO"
            )
        except Exception as e:
            tb = traceback.format_exc()
            self._on_estimate_error(f"{e}\n\nTraceback:\n{tb}")
        finally:
            self.estimate_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
    
    def start_training(self):
        """Inicia entrenamiento directamente en thread separado"""
        if not self._validate_inputs():
            return
        self._ensure_dml_uses_seg()
        if not self._estimate_done or not self._run_dir:
            QMessageBox.warning(self, "Estimación requerida",
                "Debes ejecutar 'Estimar Tiempo' antes de iniciar el entrenamiento.\n"
                "Esto crea la carpeta de resultados y el análisis pre-entrenamiento.")
            return
        
        args = self._build_args()
        
        self.train_btn.setEnabled(False)
        self.estimate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("🔄 Entrenando modelo...")
        
        self.parent_window.log_widget.append_log("Iniciando entrenamiento...", "INFO")
        self.parent_window.log_widget.append_log(f"Config: {args.config}", "DEBUG")
        self.parent_window.log_widget.append_log(f"Resume: {args.resume or 'No'}", "DEBUG")
        self.parent_window.log_widget.append_log(f"Run dir: {self._run_dir}", "DEBUG")
        
        from scripts.train import run_training
        self.worker = WorkerThread(run_training, args, run_dir=self._run_dir)
        self.worker.finished.connect(self.training_finished)
        self.worker.error.connect(self.training_error)
        self.worker.start()
    
    def stop_training(self):
        """Detiene entrenamiento"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        self.training_finished(None)
        self.parent_window.log_widget.append_log("Entrenamiento detenido por usuario", "WARNING")
    
    def training_finished(self, result):
        """Callback cuando termina entrenamiento"""
        self._estimate_done = False
        self._run_dir = None
        self.train_btn.setEnabled(False)
        self.train_btn.setToolTip("Primero debes ejecutar 'Estimar Tiempo'")
        self.estimate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("✅ Entrenamiento completado")
        self.parent_window.log_widget.append_log("Entrenamiento finalizado", "SUCCESS")

        abl_msg = ""
        if isinstance(result, dict):
            abl = result.get("ablation") or {}
            if abl.get("enabled") is not False and abl.get("status") not in (None, "skipped"):
                completed = abl.get("completed", 0)
                failed = abl.get("failed", 0)
                skipped = abl.get("skipped", 0)
                csv_path = abl.get("csv_path")
                if abl.get("status") == "failed" and abl.get("error"):
                    self.parent_window.log_widget.append_log(
                        f"Ablaciones: error — {abl['error']}", "WARNING"
                    )
                    abl_msg = f"\n\nAblaciones: error (run principal OK).\n{abl['error']}"
                elif csv_path:
                    self.parent_window.log_widget.append_log(
                        f"Ablaciones: {completed} OK, {failed} fallos, {skipped} omitidas → {csv_path}",
                        "SUCCESS" if failed == 0 else "WARNING",
                    )
                    abl_msg = (
                        f"\n\nAblaciones: {completed} variantes OK"
                        f"{f', {failed} fallos' if failed else ''}"
                        f"{f', {skipped} omitidas' if skipped else ''}.\n"
                        f"CSV: {csv_path}"
                    )
            elif (
                abl.get("enabled")
                and abl.get("status") == "skipped"
                and abl.get("reason") == "no matrices selected"
            ):
                self.parent_window.log_widget.append_log(
                    "Ablaciones habilitadas pero sin matrices seleccionadas", "WARNING"
                )

        QMessageBox.information(
            self, "Éxito", f"Entrenamiento completado exitosamente.{abl_msg}"
        )
    
    def training_error(self, error_msg):
        """Callback cuando hay error"""
        self._estimate_done = False
        self._run_dir = None
        self.train_btn.setEnabled(False)
        self.train_btn.setToolTip("Primero debes ejecutar 'Estimar Tiempo'")
        self.estimate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Error en entrenamiento")
        
        # Log each line of the error for readability
        for line in error_msg.split('\n'):
            line = line.strip()
            if line:
                self.parent_window.log_widget.append_log(line, "ERROR")
        
        # Show only the first line in the dialog (short message)
        short_msg = error_msg.split('\n')[0]
        QMessageBox.critical(self, "Error", f"Error durante entrenamiento:\n{short_msg}\n\nRevisa el Registro de Actividad para más detalles.")


class EvaluationTab(QWidget):
    """
    Tab para evaluación del modelo.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # === Configuración ===
        config_group = QGroupBox("Post-Entrenamiento — regenera val y test (mismo pipeline que train.py)")
        config_layout = QFormLayout()
        
        # Checkpoint
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_path = QLineEdit()
        self.checkpoint_path.setPlaceholderText("Ruta al checkpoint del modelo")
        checkpoint_browse = QPushButton("📁 Buscar")
        checkpoint_browse.clicked.connect(self.browse_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_path)
        checkpoint_layout.addWidget(checkpoint_browse)
        config_layout.addRow("Checkpoint:", checkpoint_layout)
        
        # Output dir (Auto-detected from checkpoint)
        output_layout = QHBoxLayout()
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Se detectará automáticamente desde el checkpoint")
        self.output_path.setReadOnly(True)
        self.output_path.setStyleSheet("background-color: #f0f0f0; color: #555;")
        output_layout.addWidget(self.output_path)
        config_layout.addRow("Salida (Auto):", output_layout)
        
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # === Controles ===
        self.evaluate_btn = QPushButton("📊 REGENERAR RESULTADOS")
        self.evaluate_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.evaluate_btn.clicked.connect(self.run_evaluation)
        layout.addWidget(self.evaluate_btn)
        
        # === Progress ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(Styles.PROGRESS_BAR)
        layout.addWidget(self.progress_bar)
        
        # === Status ===
        self.status_label = QLabel("✅ Listo para evaluar")
        self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def browse_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Checkpoint", "", "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.checkpoint_path.setText(file_path)
            # Auto-detect run directory
            from pathlib import Path
            try:
                run_dir = Path(file_path).parent.parent
                self.output_path.setText(str(run_dir))
            except Exception:
                pass
    
    def run_evaluation(self):
        """Regenera el pipeline post-entrenamiento completo (val + test, mismo que train.py)"""
        checkpoint = self.checkpoint_path.text()
        output_dir = self.output_path.text()
        
        if not checkpoint or not Path(checkpoint).exists():
            QMessageBox.warning(self, "Error", "Checkpoint no valido o no existe")
            return
        
        self.evaluate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Regenerando resultados post-entrenamiento (val + test)...")
        self.status_label.setStyleSheet(Styles.LABEL_INFO if hasattr(Styles, 'LABEL_INFO') else "")
        
        self.parent_window.log_widget.append_log(f"Evaluacion: {checkpoint}", "INFO")
        self.parent_window.log_widget.append_log(f"Salida: {output_dir}", "INFO")
        
        from scripts.post_training import run_evaluation_from_checkpoint
        
        self.worker = WorkerThread(
            run_evaluation_from_checkpoint,
            checkpoint, 'config.yaml', output_dir
        )
        self.worker.finished.connect(self.evaluation_finished)
        self.worker.error.connect(self.evaluation_error)
        self.worker.start()
    
    def evaluation_finished(self, result):
        self.evaluate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Evaluacion completada")
        self.status_label.setStyleSheet(Styles.LABEL_SUCCESS)
        
        output_dir = self.output_path.text()
        slice_acc = knn_acc = 0.0
        if isinstance(result, dict):
            slice_acc = result.get('slice_accuracy', result.get('accuracy', 0))
            knn_acc = result.get('knn_accuracy', slice_acc)
        self.parent_window.log_widget.append_log(
            f"Evaluacion finalizada — Slice accuracy: {slice_acc:.1%} | kNN: {knn_acc:.1%}",
            "SUCCESS",
        )

        QMessageBox.information(
            self, "Evaluacion Completa",
            f"Resultados guardados en: {output_dir}\n\n"
            f"Slice accuracy (principal): {slice_acc:.1%}\n"
            f"kNN accuracy (secundaria): {knn_acc:.1%}\n\n"
            f"Artefactos: val + test, slice_representatives.pt, Grad-CAM, informes espectrales"
        )
    
    def evaluation_error(self, error_msg):
        self.evaluate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Error en evaluación")
        
        for line in error_msg.split('\n'):
            line = line.strip()
            if line:
                self.parent_window.log_widget.append_log(line, "ERROR")
        
        short_msg = error_msg.split('\n')[0]
        QMessageBox.critical(self, "Error", f"Error durante evaluación:\n{short_msg}\n\nRevisa el Registro de Actividad para más detalles.")


class MainWindow(QMainWindow):
    """
    Ventana principal de la aplicación.
    Implementa patrón MVC y mejores prácticas de usabilidad IEEE.
    """
    def __init__(self):
        super().__init__()
        # Root primero: sin esto los módulos del autoetiquetador no escriben en la terminal.
        setup_root_logging()
        self.logger = logging.getLogger("gui_app")
        self.init_ui()
    
    def init_ui(self):
        """Inicializa interfaz de usuario"""
        self.setWindowTitle("MetricLearning - Sistema de Clasificación de Imágenes")
        self.setGeometry(50, 50, 1400, 900)
        self.setMinimumSize(1000, 700)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)
        
        # === Header ===
        self.header = QLabel("MetricLearning")
        self.header.setAlignment(Qt.AlignCenter)
        self.header.setStyleSheet(Styles.HEADER_MAIN)
        self.header.setWordWrap(True)
        main_layout.addWidget(self.header)
        
        # === Splitter: Tabs arriba, Log abajo ===
        splitter = QSplitter(Qt.Vertical)
        
        # === Tabs ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(Styles.TAB_WIDGET)
        
        # Agregar tabs — dataset tab wrapped in scroll area
        self.config_tab = ConfigEditorTab("config.yaml", self)
        self.dataset_tab = DatasetManagerTab(self)
        self.contour_tab = ContourAnalysisTab(self)
        self.training_tab = TrainingTab(self)
        self.detector_training_tab = DetectionTrainingTab(self)
        self.inference_tab = InferenceTab(self)
        self.evaluation_tab = EvaluationTab(self)
        
        # Wrap dataset tab in scroll area (it has too much content)
        dataset_scroll = QScrollArea()
        dataset_scroll.setWidgetResizable(True)
        dataset_scroll.setWidget(self.dataset_tab)
        dataset_scroll.setFrameShape(QScrollArea.NoFrame)
        
        # Wrap contour tab in scroll area
        contour_scroll = QScrollArea()
        contour_scroll.setWidgetResizable(True)
        contour_scroll.setWidget(self.contour_tab)
        contour_scroll.setFrameShape(QScrollArea.NoFrame)
        
        self.tabs.addTab(self.config_tab, "⚙️ Configuración")
        self.tabs.addTab(dataset_scroll, "📊 Gestor de Datos")
        self.tabs.addTab(contour_scroll, "🔬 Análisis de Contornos")
        self.tabs.addTab(self.training_tab, "🚀 Entrenamiento")
        self.tabs.addTab(self.detector_training_tab, "🎯 Detección")
        self.tabs.addTab(self.inference_tab, "🔍 Inferencia")
        self.tabs.addTab(self.evaluation_tab, "� Evaluación")
        
        self.tabs.currentChanged.connect(self._on_tab_changed)
        
        # Sync Config → Training al guardar y al arranque
        self.config_tab.configSaved.connect(self._on_config_saved)
        self.training_tab.sync_config_path(self.config_tab.get_config_path_str())
        self.detector_training_tab.sync_config_path(self.config_tab.get_config_path_str())
        self.detector_training_tab.detector_widget.detectorLoaded.connect(self._on_detector_loaded)
        
        splitter.addWidget(self.tabs)
        
        # === Log Widget ===
        log_container = QWidget()
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)
        
        log_header_layout = QHBoxLayout()
        log_label = QLabel("Registro de Actividad")
        log_label.setStyleSheet("font-weight: bold; color: #0e639c; font-size: 10pt; padding: 2px;")
        log_header_layout.addWidget(log_label)
        log_header_layout.addStretch()
        
        clear_btn = QPushButton("🗑️ Limpiar Logs")
        clear_btn.setFixedWidth(120)
        clear_btn.clicked.connect(self.log_widget_clear)
        log_header_layout.addWidget(clear_btn)
        log_layout.addLayout(log_header_layout)
        
        self.log_widget = LogWidget()
        log_layout.addWidget(self.log_widget)
        
        log_container.setLayout(log_layout)
        splitter.addWidget(log_container)
        
        # Proportions: 75% tabs, 25% log
        splitter.setSizes([650, 250])
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        central_widget.setLayout(main_layout)
        
        # === Status Bar ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo")
        
        # === Menu Bar ===
        self.create_menu_bar()
        
        # Log inicial
        self.log_widget.append_log("Aplicación iniciada correctamente", "SUCCESS")
        self.logger.info("GUI iniciada")
        self._run_startup_validation()
    
    def _run_startup_validation(self):
        """Valida config.yaml al arranque y muestra estado del pipeline en status bar."""
        from src.utils.config_validator import validate_config_file

        path = self.config_tab.get_config_path_str()
        result = validate_config_file(path)
        summary = result.get("pipeline_summary", "")

        if result["valid"]:
            gpu_ok = not any("CUDA" in w for w in result["warnings"])
            gpu_label = "GPU OK" if gpu_ok else "GPU N/A"
            self._pipeline_status = f"Pipeline: {summary} | {gpu_label}"
            self.status_bar.showMessage(self._pipeline_status)
        else:
            self._pipeline_status = ""
            self.status_bar.showMessage(f"Config inválida: {result['errors'][0]}")
            for err in result["errors"]:
                self.log_widget.append_log(err, "ERROR")

        for warn in result["warnings"]:
            self.log_widget.append_log(warn, "WARNING")

        if result["valid"]:
            self.log_widget.append_log(f"Validación arranque OK — {summary}", "INFO")

        self.training_tab.refresh_ablation_summary(path)
        self.detector_training_tab._refresh_summary()
    
    def log_widget_clear(self):
        """Limpia el widget de logs"""
        self.log_widget.clear()
        self.log_widget.append_log("Logs limpiados", "INFO")
    
    def _on_config_saved(self, config_path: str):
        """Propaga la ruta YAML guardada a pestañas dependientes."""
        self.training_tab.sync_config_path(config_path)
        self.detector_training_tab.sync_config_path(config_path)
        self._run_startup_validation()
        self.training_tab.refresh_ablation_summary(config_path)
        self.log_widget.append_log(
            f"Config sincronizada: {config_path}", "INFO"
        )

    def _on_detector_loaded(self, payload: dict):
        label = payload.get("label", payload.get("backend", "?"))
        self.log_widget.append_log(f"Detector activo: {label}", "SUCCESS")
        if hasattr(self, "training_tab"):
            self.training_tab.refresh_detector_for_results_summary()
        if hasattr(self, "_pipeline_status"):
            self.status_bar.showMessage(f"Detector: {label} | {self._pipeline_status}")
    
    def _on_tab_changed(self, index):
        """Callback cuando se cambia de tab — actualiza header y registra"""
        previous = getattr(self, "_previous_tab_index", 0)
        self._previous_tab_index = index
        
        # Aviso al ir de Configuración → Entrenamiento con cambios sin guardar
        if previous == 0 and index == 3 and self.config_tab.is_dirty():
            QMessageBox.information(
                self,
                "Cambios sin guardar",
                "La pestaña Configuración tiene cambios pendientes.\n\n"
                "Guarde con 💾 Guardar antes de entrenar, o el YAML en disco "
                "no reflejará las ediciones actuales.",
            )
        
        tab_titles = {
            0: "Configuracion del Sistema",
            1: "Gestion de Datasets",
            2: "Analisis de Contornos",
            3: "Entrenamiento DML",
            4: "Entrenamiento Deteccion",
            5: "Inferencia y Prediccion",
            6: "Evaluacion y Resultados",
        }
        tab_names = [
            "Configuracion", "Gestor de Datos", "Contornos",
            "Entrenamiento DML", "Deteccion", "Inferencia", "Evaluacion",
        ]
        if 0 <= index < len(tab_names):
            self.header.setText(f"MetricLearning  |  {tab_titles.get(index, '')}")
            pipeline = getattr(self, "_pipeline_status", "")
            if pipeline:
                self.status_bar.showMessage(f"{tab_names[index]} | {pipeline}")
            else:
                self.status_bar.showMessage(f"Tab: {tab_names[index]}")
            self.log_widget.append_log(f"Navegacion -> {tab_names[index]}", "DEBUG")
    
    def create_menu_bar(self):
        """Crea barra de menú"""
        menubar = self.menuBar()
        
        # Archivo
        file_menu = menubar.addMenu("&Archivo")
        
        exit_action = file_menu.addAction("&Salir")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        # Ayuda
        help_menu = menubar.addMenu("&Ayuda")
        
        about_action = help_menu.addAction("&Acerca de")
        about_action.triggered.connect(self.show_about)
        
        docs_action = help_menu.addAction("&Documentación")
        docs_action.triggered.connect(self.show_docs)
    
    def show_about(self):
        """Muestra diálogo Acerca de"""
        QMessageBox.about(
            self,
            "Acerca de MetricLearning",
            """
            <h2>MetricLearning v2.0</h2>
            <p>Sistema de clasificación de granos con metric learning y slicing dimensional.</p>
            <p><b>Características:</b></p>
            <ul>
                <li>Backbones ViT/CNN (DINOv2, ConvNeXt, etc.) + AnalogyNet</li>
                <li>Sliced MS Loss (expansión espectral por subespacios)</li>
                <li>Clasificación slice-aware + kNN de referencia</li>
                <li>Post-entrenamiento: val + test, Grad-CAM, informe espectral</li>
            </ul>
            <p><b>Desarrollado con:</b> PyTorch, PyQt5</p>
            """
        )
    
    def show_docs(self):
        """Muestra documentación"""
        QMessageBox.information(
            self,
            "Documentación",
            """
            <h3>Documentación disponible en:</h3>
            <p>📁 <b>docs/2026-06-09_INFORME_MODIFICACIONES_PIPELINE_MULTI_SLICE.md</b> — informe unificado pipeline slice-aware</p>
            <p>📁 <b>README.md</b></p>
            """
        )
    
    def closeEvent(self, event):
        """Maneja cierre de aplicación"""
        reply = QMessageBox.question(
            self,
            "Confirmar Salida",
            "¿Está seguro que desea salir?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.logger.info("Aplicación cerrada por usuario")
            event.accept()
        else:
            event.ignore()


def main():
    """Función principal"""
    app = QApplication(sys.argv)
    
    # Aplicar estilos globales centralizados
    apply_global_styles(app)
    
    # Crear y mostrar ventana principal
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
