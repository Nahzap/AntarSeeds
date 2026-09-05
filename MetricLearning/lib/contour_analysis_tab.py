"""
ContourAnalysisTab — Tab GUI para análisis de contornos y segmentación de granos.

Posición: Tab index 2 (después de Gestor de Datos, antes de Entrenamiento).

Secciones:
1. Configuración de segmentación (parámetros U²-Net + botones de acción)
2. Explorador de contornos (imagen con contornos + crop del grano seleccionado)
3. Estadísticas (conteos, distribución por clase)
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QPushButton,
    QComboBox, QSlider, QSpinBox, QProgressBar, QFormLayout, QSplitter,
    QSizePolicy, QMessageBox, QScrollArea, QFrame, QListWidget, QListWidgetItem,
    QCheckBox, QDoubleSpinBox, QShortcut,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QKeySequence

from lib.styles import Styles, Colors
from src.grain_detection.detection_profiles import PROFILES
from src.grain_detection.annotation_paths import (
    DEFAULT_ANNOTATION_ROOT,
    AnnotationIdentityError,
    find_existing_seg_path,
    identity_for,
    identity_params_for,
    migrate_legacy_annotations,
    resolved_image_path,
    seg_matches_image,
    seg_path_for as _seg_path_for_impl,
)

# Perfiles ofrecidos en la GUI. "seed" es el de AntarSeeds (semillas grandes,
# alargadas); el resto proviene del dominio original de polen.
PROFILE_ORDER = ["seed", "balanced", "sensitive", "strict"]
DEFAULT_PROFILE = "seed"

# Encuadre nominal de la Basler (2590x1942 px). Acota los spinbox de área para
# que ningún tope quede por debajo de una semilla real.
FRAME_W, FRAME_H = 2590, 1942

# Anotaciones centralizadas: un único destino para los .seg.
ANNOTATION_ROOT = DEFAULT_ANNOTATION_ROOT
SPLIT_DIRS = ["data/processed/train", "data/processed/val", "data/processed/test"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def seg_path_for(img_path: Path) -> Path:
    """Ruta .seg absoluta: <clase>/<split>__<SAMPLE>__<stem>.seg."""
    return _seg_path_for_impl(img_path, ANNOTATION_ROOT)


def class_images(class_dir: Path) -> List[Path]:
    """Imágenes de una carpeta de clase, ordenadas."""
    class_dir = Path(class_dir)
    if not class_dir.exists():
        return []
    return sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
MAX_FRAME_AREA = FRAME_W * FRAME_H

# Configurar logger con handler de consola
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class ClickableImageLabel(QLabel):
    """QLabel that emits click coordinates mapped to original image space.

    Left-click  → ``left_clicked(x_orig, y_orig)``
    Right-click → ``right_clicked(x_orig, y_orig)``

    The label stores the original image dimensions and the displayed pixmap
    so it can map widget-space coordinates back to the original image.
    """
    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal(int, int)
    left_double_clicked = pyqtSignal(int, int)
    right_double_clicked = pyqtSignal(int, int)
    left_dragged = pyqtSignal(int, int)
    right_dragged = pyqtSignal(int, int)
    left_released = pyqtSignal(int, int)
    right_released = pyqtSignal(int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orig_w = 0
        self._orig_h = 0
        self.setMouseTracking(True)

    def set_original_size(self, w: int, h: int):
        self._orig_w = w
        self._orig_h = h

    # Signals for mouse wheel navigation
    wheel_up = pyqtSignal()
    wheel_down = pyqtSignal()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        # Navegación de imágenes SOLO con Ctrl + wheel.
        # Wheel sin Ctrl queda disponible para el comportamiento normal del widget.
        if event.modifiers() & Qt.ControlModifier:
            if delta > 0:
                self.wheel_up.emit()
            elif delta < 0:
                self.wheel_down.emit()
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):
        mapped = self._map_event_to_orig(event.pos().x(), event.pos().y())
        if mapped is None:
            return super().mousePressEvent(event)
        ox, oy = mapped

        if event.button() == Qt.LeftButton:
            self.left_clicked.emit(ox, oy)
        elif event.button() == Qt.RightButton:
            self.right_clicked.emit(ox, oy)

    def mouseDoubleClickEvent(self, event):
        mapped = self._map_event_to_orig(event.pos().x(), event.pos().y())
        if mapped is None:
            return super().mouseDoubleClickEvent(event)
        ox, oy = mapped
        if event.button() == Qt.LeftButton:
            self.left_double_clicked.emit(ox, oy)
        elif event.button() == Qt.RightButton:
            self.right_double_clicked.emit(ox, oy)

    def mouseMoveEvent(self, event):
        mapped = self._map_event_to_orig(event.pos().x(), event.pos().y())
        if mapped is not None:
            ox, oy = mapped
            btns = event.buttons()
            if btns & Qt.LeftButton:
                self.left_dragged.emit(ox, oy)
            if btns & Qt.RightButton:
                self.right_dragged.emit(ox, oy)
        return super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        mapped = self._map_event_to_orig(event.pos().x(), event.pos().y())
        if mapped is not None:
            ox, oy = mapped
            if event.button() == Qt.LeftButton:
                self.left_released.emit(ox, oy)
            elif event.button() == Qt.RightButton:
                self.right_released.emit(ox, oy)
        return super().mouseReleaseEvent(event)

    def _map_event_to_orig(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        pm = self.pixmap()
        if pm is None or pm.isNull() or self._orig_w == 0:
            return None

        # Widget may be larger than pixmap — compute offset
        lw, lh = self.width(), self.height()
        pw, ph = pm.width(), pm.height()
        off_x = (lw - pw) // 2 if lw > pw else 0
        off_y = (lh - ph) // 2 if lh > ph else 0

        rx = x - off_x
        ry = y - off_y
        if rx < 0 or ry < 0 or rx >= pw or ry >= ph:
            return None

        # Map to original image coords
        ox = int(rx / pw * self._orig_w)
        oy = int(ry / ph * self._orig_h)
        return (ox, oy)


class ManualContourWorker(QThread):
    """Resuelve una semilla de click con el buscador ROI (fuera del hilo de UI)."""
    finished = pyqtSignal(list)   # list of grain dicts (may be empty)
    error = pyqtSignal(str)
    _detector_cache_lock = threading.Lock()
    _detector_cache = {}

    def __init__(self, image_bgr, seeds, config):
        super().__init__()
        self.image_bgr = image_bgr
        self.seeds = [(int(x), int(y)) for x, y in seeds]
        self.click_x, self.click_y = self.seeds[0] if self.seeds else (0, 0)
        self.config = config

    def run(self):
        try:
            from src.grain_detection.seeded_contour import (
                SeededParams,
                resolve_seeded_object,
            )

            # Mismos parámetros que PreviewWorker / Segmentar: sin rama paralela.
            cfg = self.config or {}
            params = SeededParams.from_config(cfg)

            detector = self._get_or_create_detector(cfg)
            logger.info(
                "[Click] CLAHE+U2-Net trazo %d pt(s) (%d,%d) radio=%d",
                len(self.seeds), self.click_x, self.click_y, params.crop_radius,
            )
            grain = resolve_seeded_object(
                self.image_bgr,
                (self.click_x, self.click_y),
                params,
                detector._crop_saliency_provider(),
                seeds=self.seeds,
            )
            self.finished.emit([grain] if grain is not None else [])
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    @staticmethod
    def detector_cache_key(config) -> Tuple[str, int, str]:
        """
        Identidad del modelo, no de los filtros.

        Los filtros viven en SeededParams; si la clave incluyera los sliders,
        cada movimiento recargaría los pesos de U²-Net en GPU.
        """
        cfg = config or {}
        return (
            str(cfg.get("model_type", "u2netp")),
            int(cfg.get("input_size", 320)),
            str(cfg.get("device") or "auto"),
        )

    @classmethod
    def _get_or_create_detector(cls, config):
        from src.grain_detection.grain_detector import PollenGrainDetector

        cfg = config or {}
        key = cls.detector_cache_key(cfg)
        with cls._detector_cache_lock:
            detector = cls._detector_cache.get(key)
            if detector is None:
                detector = PollenGrainDetector(config=dict(cfg))
                cls._detector_cache[key] = detector
        return detector


# Colores para contornos de granos (cíclicos)
GRAIN_COLORS = [
    (66, 133, 244),    # Azul
    (52, 168, 83),     # Verde
    (234, 67, 53),     # Rojo
    (251, 188, 4),     # Amarillo
    (171, 71, 188),    # Púrpura
    (0, 172, 193),     # Cyan
    (255, 112, 67),    # Naranja
    (63, 81, 181),     # Índigo
]


def numpy_to_qpixmap(image: np.ndarray, max_width: int = 800, max_height: int = 600) -> QPixmap:
    """Convierte imagen numpy (BGR o RGB) a QPixmap escalado."""
    if image is None or image.size == 0:
        return QPixmap()

    if len(image.shape) == 2:
        h, w = image.shape
        qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = image.shape
        if ch == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            qimg = QImage(image.data, w, h, ch * w, QImage.Format_RGBA8888)

    pixmap = QPixmap.fromImage(qimg)

    if pixmap.width() > max_width or pixmap.height() > max_height:
        pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    return pixmap


class PreviewWorker(QThread):
    """
    Corre el buscador ROI sobre una imagen completa sin escribir nada.

    Es el lote aplicado a un solo encuadre: mismas semillas automáticas y mismos
    filtros, para calibrar los controles antes de lanzar miles de imágenes.
    """
    finished = pyqtSignal(dict)   # {"grains": [...], "suggestion": {...} | None}
    error = pyqtSignal(str)

    def __init__(self, image_bgr, config):
        super().__init__()
        self.image_bgr = image_bgr
        self.config = config

    def run(self):
        try:
            from src.grain_detection.seeded_contour import (
                SeededParams,
                measure_bodies,
                propose_objects,
                suggest_area_bounds,
            )

            cfg = self.config or {}
            params = SeededParams.from_config(cfg)
            detector = ManualContourWorker._get_or_create_detector(cfg)
            saliency = detector.saliency(self.image_bgr)

            grains = propose_objects(
                saliency,
                self.image_bgr,
                params,
                saliency_provider=detector._crop_saliency_provider(),
            )
            # Se mide siempre: si los topes de área descartan la especie, esto
            # es lo único que dice de qué tamaño son realmente las semillas.
            areas = measure_bodies(saliency, params)
            self.finished.emit(
                {"grains": grains, "suggestion": suggest_area_bounds(areas)}
            )
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class AnnotationWorker(QThread):
    """
    Genera segmentaciones sin bloquear la GUI.

    Un solo worker para cualquier alcance: la lista de splits es el parámetro.
    Un split y los tres comparten anotador, acumulación y forma de las stats.
    """
    progress = pyqtSignal(int, int, str)   # current, total, message
    finished = pyqtSignal(dict)             # stats acumuladas
    error = pyqtSignal(str)

    def __init__(self, splits, overwrite, config, class_filter=None):
        super().__init__()
        self.splits = [str(s) for s in splits]
        self.overwrite = overwrite
        self.config = config
        self.class_filter = class_filter

    @staticmethod
    def empty_stats() -> Dict:
        return {
            "total_images": 0, "annotated": 0, "skipped": 0,
            "errors": 0, "total_grains": 0, "elapsed_ms": 0.0,
            "cancelled": False, "grains_per_class": {}, "per_split": {},
            "annotation_root": ANNOTATION_ROOT,
        }

    @staticmethod
    def accumulate(combined: Dict, split_name: str, stats: Dict) -> Dict:
        """Suma las stats de un split. Pura: es el núcleo verificable del worker."""
        combined["per_split"][split_name] = stats
        for key in ("total_images", "annotated", "skipped", "errors", "total_grains"):
            combined[key] += stats.get(key, 0)
        combined["elapsed_ms"] += float(stats.get("elapsed_ms", 0.0))
        combined["cancelled"] = combined["cancelled"] or bool(stats.get("cancelled"))
        for cls, cnt in (stats.get("grains_per_class") or {}).items():
            combined["grains_per_class"][cls] = combined["grains_per_class"].get(cls, 0) + cnt
        combined["grains_per_image"] = (
            combined["total_grains"] / combined["annotated"] if combined["annotated"] else 0.0
        )
        return combined

    def run(self):
        try:
            from src.grain_detection.annotator import SegmentationAnnotator
            annotator = SegmentationAnnotator(config=self.config)

            combined = self.empty_stats()
            multi = len(self.splits) > 1

            for split_dir in self.splits:
                if self.isInterruptionRequested():
                    combined["cancelled"] = True
                    break
                if not Path(split_dir).exists():
                    continue
                split_name = Path(split_dir).name
                prefix = f"[{split_name}] " if multi else ""

                def on_progress(current, total, msg, _p=prefix):
                    self.progress.emit(current, total, f"{_p}{msg}")

                stats = annotator.annotate_directory(
                    split_dir,
                    overwrite=self.overwrite,
                    progress_callback=on_progress,
                    class_filter=self.class_filter,
                    annotation_root=ANNOTATION_ROOT,
                    cancel_check=self.isInterruptionRequested,
                )
                self.accumulate(combined, split_name, stats)
                if combined["cancelled"]:
                    break

            self.finished.emit(combined)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class ContourAnalysisTab(QWidget):
    """
    Tab de análisis de contornos y segmentación de granos.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self._worker = None
        self._preview_worker = None
        self._preview_grains = []  # propuestas de 'Probar aquí' (no guardadas)
        self._area_suggestion = None  # topes de área medidos en la imagen abierta
        # Calibración por demostración: envolvente de cuerpos aceptados por clase.
        self._class_envelopes = {}
        self._learning_class = None

        # Estado del explorador
        self._current_class = None
        self._current_images = []
        self._current_image_idx = 0
        self._current_grains = []
        self._selected_grain_idx = -1
        self._current_image_bgr = None
        self._seg_dirty = False  # True si se editó el .seg actual sin guardar
        self._current_seg_path = None  # Ruta del .seg de la imagen cargada
        self._class_to_image_indices = {}  # navegación rápida por clase
        
        # Historial de cambios para UNDO (stack de estados de grains)
        self._grains_history = []
        self._active_grain_idx = -1  # Grano activo para fusión incremental

        # ROI manual (puntos ilimitados include/exclude)
        self._roi_include_polygons = []
        self._roi_exclude_polygons = []
        self._roi_current_include = []
        self._roi_current_exclude = []
        self._manual_roi_seed_kind = None  # include | exclude
        self._manual_roi_seed_point = None  # (x, y)
        self._roi_include_mask = None  # acumulación unificada en tiempo real
        self._roi_exclude_mask = None  # acumulación unificada en tiempo real
        self._roi_block_mask = None  # anclas de exclusión (no expansión)
        self._roi_block_points = []  # puntos rojos ancla
        self._roi_last_include_seed = None  # última semilla de inclusión (x, y)
        self._roi_suggest_points = []  # sugerencias de expansión (puntos azules)
        self._seed_retry_depth = 0
        self._pending_ai_seeds = []  # cola FIFO [(points, kind), ...]
        self._last_include_drag_pt = None
        self._last_exclude_drag_pt = None
        self._drag_step_px = 9
        self._preserve_union_mode = True  # no descartar propuestas previas implícitamente
        self._roi_seed_trace = []  # historial visual permanente de semillas usadas
        self._active_seed_trace_idx = -1

        self._init_ui()
        self.setFocusPolicy(Qt.StrongFocus)
        self._setup_shortcuts()

        # Auto-cargar .seg existentes al iniciar (deferred para no bloquear)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(500, self._refresh_explorer)
        from src.utils.logging_utils import set_autolabel_verbosity
        set_autolabel_verbosity(True)
        logger.info("[ContourAnalysisTab] Inicializado — logs objeto/fondo en terminal")

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            # Enter cierra polígono activo; si no hay, aplica ROI.
            if self._roi_current_include or self._roi_current_exclude:
                self._close_active_manual_polygon()
            else:
                self._apply_manual_roi_to_grain()
            return
        if key == Qt.Key_C:
            self._close_active_manual_polygon()
            return
        if key == Qt.Key_A:
            self._apply_manual_roi_to_grain()
            return
        if key == Qt.Key_Escape:
            self._cancel_manual_roi_session()
            return

        if key == Qt.Key_Left:
            self._on_prev_image()
        elif key == Qt.Key_Right:
            self._on_next_image()
        else:
            super().keyPressEvent(event)

    def _setup_shortcuts(self):
        """Atajos rápidos de ROI manual (fase 2)."""
        self._shortcut_close_roi = QShortcut(QKeySequence("C"), self)
        self._shortcut_close_roi.activated.connect(self._close_active_manual_polygon)

        self._shortcut_apply_roi = QShortcut(QKeySequence("A"), self)
        self._shortcut_apply_roi.activated.connect(self._apply_manual_roi_to_grain)

        self._shortcut_cancel_roi = QShortcut(QKeySequence("Esc"), self)
        self._shortcut_cancel_roi.activated.connect(self._cancel_manual_roi_session)

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Header
        header = QLabel("Análisis de Contornos")
        header.setStyleSheet(Styles.HEADER_SECTION)
        layout.addWidget(header)

        # Sin splitter vertical: el tab vive dentro de un QScrollArea, así que un
        # splitter nunca se acota y empuja las estadísticas fuera de la vista.
        # Los parámetros se pliegan cuando estorban.
        layout.addWidget(self._build_config_section())
        layout.addWidget(self._build_explorer_section(), 1)
        layout.addWidget(self._build_stats_section())

        self.setLayout(layout)
        # Los controles arrancan mostrando el preset activo, no valores de polen.
        self._on_profile_changed(self.profile_combo.currentText())
        self._update_bank_info()

    # =========================================================================
    # Sección 1: Configuración de Segmentación
    # =========================================================================

    def _build_config_section(self) -> QWidget:
        group = QGroupBox("Configuración de Segmentación")
        main_layout = QVBoxLayout()

        # Fila 1: Dataset selector
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(["data/processed/train", "data/processed/val", "data/processed/test"])
        self.dataset_combo.setMinimumWidth(250)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        row1.addWidget(self.dataset_combo)

        row1.addWidget(QLabel("Modelo:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["u2netp (rápido, 4.7MB)", "u2net (preciso, 176MB)"])
        self.model_combo.setCurrentIndex(0)
        self.model_combo.setToolTip("u2netp recomendado. u2net requiere descargar 176MB adicionales.")
        row1.addWidget(self.model_combo)

        row1.addWidget(QLabel("Perfil:"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(PROFILE_ORDER)
        self.profile_combo.setCurrentText(DEFAULT_PROFILE)
        self.profile_combo.setToolTip(
            "Carga un preset de filtros morfológicos en los controles de abajo.\n"
            "'seed' es el preset de semillas AntarSeeds; los demás vienen de polen.\n"
            "Tras cargarlo puedes editar cada filtro libremente."
        )
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        row1.addWidget(self.profile_combo)
        row1.addStretch()

        # Plegar los parámetros libera altura para el visor y las estadísticas.
        self.btn_toggle_params = QPushButton("▾ Parámetros")
        self.btn_toggle_params.setCheckable(True)
        self.btn_toggle_params.setChecked(True)
        self.btn_toggle_params.setMaximumWidth(130)
        self.btn_toggle_params.setToolTip(
            "Oculta o muestra los controles de detección.\n"
            "Plegado, el visor y las estadísticas ganan altura."
        )
        self.btn_toggle_params.toggled.connect(self._on_toggle_params)
        row1.addWidget(self.btn_toggle_params)
        main_layout.addLayout(row1)

        # Fila 2: Parámetros — 4 columnas, dentro de un contenedor plegable
        self.params_container = QWidget()
        params_layout = QHBoxLayout()
        params_layout.setContentsMargins(0, 0, 0, 0)

        # Columna 1: Saliency
        col1 = QFormLayout()
        self.threshold_slider = self._make_slider(5, 80, 30, "Threshold")
        col1.addRow("Threshold:", self.threshold_slider)

        self.adaptive_k_slider = self._make_slider(5, 80, 30, "Adaptive k")
        col1.addRow("Adaptive k:", self.adaptive_k_slider)

        self.split_touching_cb = QCheckBox("Separar contacto")
        self.split_touching_cb.setChecked(True)
        self.split_touching_cb.setToolTip(
            "Dos semillas que se tocan forman un solo componente y salían fusionadas.\n"
            "Activado: una semilla por cuerpo (máximos de distancia) + corte por watershed."
        )
        self.split_touching_cb.toggled.connect(self._on_filter_value_changed)
        col1.addRow("Contacto:", self.split_touching_cb)

        self.seed_core_slider = self._make_slider(20, 80, 45, "Núcleo")
        self.seed_core_slider.setToolTip(
            "Prominencia mínima del núcleo de cada cuerpo.\n"
            "Más alto: solo separa cuerpos bien marcados (menos cortes).\n"
            "Más bajo: separa más, con riesgo de partir una semilla en dos."
        )
        col1.addRow("Núcleo:", self.seed_core_slider)

        self.waist_slider = self._make_slider(20, 99, 80, "Cintura")
        self.waist_slider.setToolTip(
            "Un corte solo es real si el istmo es más delgado que los cuerpos.\n"
            "Más bajo: exige una cintura muy marcada (menos cortes rectos falsos).\n"
            "Más alto: acepta cortes con poca cintura."
        )
        col1.addRow("Cintura:", self.waist_slider)

        self.drop_border_cb = QCheckBox("Descartar cuerpos en el borde")
        self.drop_border_cb.setChecked(False)
        self.drop_border_cb.setToolTip(
            "Descarta semillas cortadas por el borde del encuadre.\n"
            "Con el solape de placa (~93 %) esa semilla aparece completa en el FOV\n"
            "vecino, así que activarlo mejora la calidad del banco sin perder datos."
        )
        self.drop_border_cb.toggled.connect(self._on_filter_value_changed)
        col1.addRow("Borde:", self.drop_border_cb)

        params_layout.addLayout(col1)

        # Columna 2: Área y forma
        col2 = QFormLayout()
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(10, MAX_FRAME_AREA)
        self.min_area_spin.setValue(200)
        self.min_area_spin.setSuffix(" px²")
        self.min_area_spin.setSingleStep(1000)
        self.min_area_spin.setToolTip("Área mínima de un objeto (en pixeles²)")
        col2.addRow("Min área:", self.min_area_spin)

        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1000, MAX_FRAME_AREA)
        self.max_area_spin.setValue(200000)
        self.max_area_spin.setSuffix(" px²")
        self.max_area_spin.setSingleStep(10000)
        self.max_area_spin.setToolTip(
            "Área máxima absoluta. Se combina con 'Max área frac': gana el más restrictivo."
        )
        col2.addRow("Max área:", self.max_area_spin)

        self.circularity_slider = self._make_slider(0, 80, 30, "Circularidad")
        self.circularity_slider.setToolTip("Mín circularidad (0=cualquier forma, 0.80=casi circular)")
        col2.addRow("Min circ.:", self.circularity_slider)

        params_layout.addLayout(col2)

        # Columna 3: Filtros de encuadre (antes fijados en código por el perfil)
        col3 = QFormLayout()
        self.max_area_frac_spin = QDoubleSpinBox()
        self.max_area_frac_spin.setRange(0.01, 1.00)
        self.max_area_frac_spin.setSingleStep(0.05)
        self.max_area_frac_spin.setDecimals(2)
        self.max_area_frac_spin.setValue(0.50)
        self.max_area_frac_spin.setToolTip(
            "Área máxima como fracción del encuadre completo.\n"
            "1.00 = sin límite relativo."
        )
        col3.addRow("Max área frac:", self.max_area_frac_spin)

        self.max_side_frac_spin = QDoubleSpinBox()
        self.max_side_frac_spin.setRange(0.05, 1.00)
        self.max_side_frac_spin.setSingleStep(0.05)
        self.max_side_frac_spin.setDecimals(2)
        self.max_side_frac_spin.setValue(0.85)
        self.max_side_frac_spin.setToolTip(
            "Lado mayor del bbox como fracción del lado menor de la imagen.\n"
            "Con polen valía 0.40 y descartaba semillas grandes."
        )
        col3.addRow("Max lado frac:", self.max_side_frac_spin)

        self.max_aspect_spin = QDoubleSpinBox()
        self.max_aspect_spin.setRange(1.0, 20.0)
        self.max_aspect_spin.setSingleStep(0.5)
        self.max_aspect_spin.setDecimals(1)
        self.max_aspect_spin.setValue(8.0)
        self.max_aspect_spin.setToolTip(
            "Relación de aspecto máxima (largo/ancho).\n"
            "Con polen valía 2.8 y descartaba semillas alargadas."
        )
        col3.addRow("Max aspecto:", self.max_aspect_spin)

        params_layout.addLayout(col3)

        # Columna 4: Morfología del buscador ROI + crop DML (no mezclar)
        col4 = QFormLayout()
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 15)
        self.kernel_spin.setValue(3)
        self.kernel_spin.setSingleStep(2)
        self.kernel_spin.setToolTip("Tamaño kernel morfológico (impar) del buscador ROI.")
        col4.addRow("Kernel morf.:", self.kernel_spin)

        self.crop_radius_spin = QSpinBox()
        self.crop_radius_spin.setRange(80, 2000)
        self.crop_radius_spin.setValue(300)
        self.crop_radius_spin.setSingleStep(20)
        self.crop_radius_spin.setSuffix(" px")
        self.crop_radius_spin.setToolTip(
            "Recorte de U²-Net alrededor de la semilla (no es el tamaño del grano).\n"
            "Si el cuerpo queda truncado, el algoritmo amplía el recorte.\n"
            "300–500 px da más detalle que el encuadre entero a 320 px."
        )
        col4.addRow("Radio semilla:", self.crop_radius_spin)

        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 80)
        self.padding_spin.setValue(20)
        self.padding_spin.setSuffix(" %")
        self.padding_spin.setToolTip(
            "Padding del recorte DML/clasificador. No define qué es un objeto."
        )
        col4.addRow("DML padding:", self.padding_spin)

        self.crop_size_spin = QSpinBox()
        self.crop_size_spin.setRange(64, 512)
        self.crop_size_spin.setValue(256)
        self.crop_size_spin.setSingleStep(64)
        self.crop_size_spin.setSuffix(" px")
        self.crop_size_spin.setToolTip(
            "Tamaño del crop cuadrado para DML. No define qué es un objeto."
        )
        col4.addRow("DML crop:", self.crop_size_spin)

        params_layout.addLayout(col4)
        self.params_container.setLayout(params_layout)
        main_layout.addWidget(self.params_container)

        # Lectura de los topes efectivos: lo que se ve es lo que se aplica.
        self.effective_params_label = QLabel("")
        self.effective_params_label.setStyleSheet("color: #888; font-size: 11px;")
        self.effective_params_label.setWordWrap(True)
        main_layout.addWidget(self.effective_params_label)
        for w in self._filter_widgets():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self._on_filter_value_changed)
            else:
                w._slider.valueChanged.connect(self._on_filter_value_changed)

        # Fila 3: Botones de acción
        buttons_layout = QHBoxLayout()

        self.btn_load = QPushButton("Cargar Existentes")
        self.btn_load.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.btn_load.setToolTip("Carga los .seg existentes sin ejecutar el modelo")
        self.btn_load.clicked.connect(self._refresh_explorer)
        buttons_layout.addWidget(self.btn_load)

        # Una sola acción de segmentación con dos modificadores explícitos, en vez
        # de tres botones que solo diferían en alcance y overwrite.
        buttons_layout.addWidget(QLabel("Alcance:"))
        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["Split actual", "Train + Val + Test"])
        self.scope_combo.setToolTip(
            "Split actual: el dataset elegido arriba.\n"
            "Train + Val + Test: los tres splits.\n"
            "En ambos casos solo se segmentan las clases marcadas con tick."
        )
        buttons_layout.addWidget(self.scope_combo)

        self.overwrite_cb = QCheckBox("Sobrescribir")
        self.overwrite_cb.setToolTip(
            "Sin marcar: solo anota imágenes que aún no tienen .seg.\n"
            "Marcado: regenera los .seg existentes (se crea backup .seg.bak)."
        )
        buttons_layout.addWidget(self.overwrite_cb)

        self.verbose_log_cb = QCheckBox("Detalle en terminal")
        self.verbose_log_cb.setChecked(True)
        self.verbose_log_cb.setToolTip(
            "Objeto/fondo, umbral y motivo de cada click o semilla.\n"
            "También en logs/antarseeds_<fecha>.log"
        )
        self.verbose_log_cb.toggled.connect(self._on_verbose_log_toggled)
        buttons_layout.addWidget(self.verbose_log_cb)

        self.btn_segment = QPushButton("Segmentar")
        self.btn_segment.setStyleSheet(
            "QPushButton { background: #e67e22; color: white; padding: 6px 14px; "
            "border-radius: 4px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background: #f39c12; }"
            "QPushButton:disabled { background: #7f8c8d; }"
        )
        self.btn_segment.clicked.connect(self._on_segment)
        buttons_layout.addWidget(self.btn_segment)

        self.btn_cancel_segment = QPushButton("Cancelar")
        self.btn_cancel_segment.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; padding: 6px 14px; "
            "border-radius: 4px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background: #e74c3c; }"
            "QPushButton:disabled { background: #7f8c8d; }"
        )
        self.btn_cancel_segment.setToolTip(
            "Detiene el lote entre imágenes. Los .seg ya escritos se conservan."
        )
        self.btn_cancel_segment.setEnabled(False)
        self.btn_cancel_segment.clicked.connect(self._on_cancel_segment)
        buttons_layout.addWidget(self.btn_cancel_segment)

        self.btn_validate = QPushButton("Validar")
        self.btn_validate.setStyleSheet(Styles.BUTTON_SECONDARY)
        self.btn_validate.clicked.connect(self._on_validate)
        buttons_layout.addWidget(self.btn_validate)

        self.btn_save_registry = QPushButton("Guardar Registros")
        self.btn_save_registry.setStyleSheet(
            "QPushButton { background: #8e44ad; color: white; padding: 6px 14px; "
            "border-radius: 4px; font-weight: bold; font-size: 12px; }"
            "QPushButton:hover { background: #9b59b6; }"
        )
        self.btn_save_registry.setToolTip(
            "Guarda un registro completo de todas las segmentaciones actuales.\n"
            "Permite restaurar el estado sin volver a escanear ni ejecutar U\u00b2-Net."
        )
        self.btn_save_registry.clicked.connect(self._on_save_registry)
        buttons_layout.addWidget(self.btn_save_registry)

        main_layout.addLayout(buttons_layout)

        # Fila 4: banco de datos — el conjunto de .seg que alimenta la fase siguiente.
        bank_layout = QHBoxLayout()
        bank_label = QLabel("Banco de datos (.seg):")
        bank_label.setStyleSheet("font-weight: bold;")
        bank_layout.addWidget(bank_label)

        self.btn_clear_selected_seg = QPushButton("Limpiar clases marcadas")
        self.btn_clear_selected_seg.setStyleSheet(
            "QPushButton { background: #d35400; color: white; padding: 5px 12px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #e67e22; }"
            "QPushButton:disabled { background: #7f8c8d; }"
        )
        self.btn_clear_selected_seg.setToolTip(
            "Borra los .seg de las clases con tick.\n"
            "Úsalo para re-segmentar solo una especie desde cero."
        )
        self.btn_clear_selected_seg.clicked.connect(self._on_clear_selected_annotations)
        bank_layout.addWidget(self.btn_clear_selected_seg)

        self.btn_clear_all_seg = QPushButton("Limpiar TODO")
        self.btn_clear_all_seg.setStyleSheet(
            "QPushButton { background: #922b21; color: white; padding: 5px 12px; "
            "border-radius: 4px; font-weight: bold; }"
            "QPushButton:hover { background: #c0392b; }"
            "QPushButton:disabled { background: #7f8c8d; }"
        )
        self.btn_clear_all_seg.setToolTip(
            "Borra TODOS los .seg del banco, de todas las clases.\n"
            "Deja el proyecto listo para etiquetar desde cero."
        )
        self.btn_clear_all_seg.clicked.connect(self._on_clear_all_annotations)
        bank_layout.addWidget(self.btn_clear_all_seg)

        self.bank_info_label = QLabel("")
        self.bank_info_label.setStyleSheet("color: #888; font-size: 11px;")
        bank_layout.addWidget(self.bank_info_label)

        bank_layout.addStretch()
        main_layout.addLayout(bank_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(Styles.LABEL_INFO)
        main_layout.addWidget(self.status_label)

        group.setLayout(main_layout)
        return group

    def _on_toggle_params(self, expanded: bool):
        self.params_container.setVisible(expanded)
        self.btn_toggle_params.setText("▾ Parámetros" if expanded else "▸ Parámetros")

    def _on_learning_toggled(self, activo: bool):
        if activo:
            self._learning_class = None
            self._adopt_class_envelope()
            logger.info("[Calibración] activada: el etiquetado manual ajusta los topes")
        else:
            self.learning_info_label.setText(
                "Aprendizaje desactivado: los topes solo cambian a mano."
            )
            logger.info("[Calibración] desactivada")

    def _on_verbose_log_toggled(self, verbose: bool):
        """Cambia el detalle que el autoetiquetador escribe en la terminal."""
        from src.utils.logging_utils import set_autolabel_verbosity

        set_autolabel_verbosity(verbose)
        logger.info(
            "Detalle de terminal: %s", "por semilla (DEBUG)" if verbose else "por imagen (INFO)"
        )

    def _make_slider(self, min_val, max_val, default, name) -> QWidget:
        """Crea un widget slider + label de valor."""
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setMinimumWidth(120)

        value_label = QLabel(f"{default / 100:.2f}")
        value_label.setMinimumWidth(40)
        slider.valueChanged.connect(lambda v: value_label.setText(f"{v / 100:.2f}"))

        layout.addWidget(slider)
        layout.addWidget(value_label)
        container.setLayout(layout)

        container._slider = slider
        container._label = value_label
        return container

    # =========================================================================
    # Sección 2: Explorador de Contornos
    # =========================================================================

    def _build_explorer_section(self) -> QWidget:
        group = QGroupBox("Explorador de Contornos")
        main_layout = QVBoxLayout()

        # ===== Layout 3 paneles: Clases (izq) | Imagen (centro) | Grano (der) =====
        view_splitter = QSplitter(Qt.Horizontal)

        # ─── Panel izquierdo: Selector de clases (checkable) ───
        class_panel = QWidget()
        class_layout = QVBoxLayout()
        class_layout.setContentsMargins(2, 2, 2, 2)
        class_layout.setSpacing(4)

        class_header = QLabel("Clases")
        class_header.setStyleSheet("font-weight: bold; font-size: 12px;")
        class_layout.addWidget(class_header)

        # Botones seleccionar todo / ninguno
        class_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("Todas")
        self.btn_select_all.setMaximumHeight(24)
        self.btn_select_all.clicked.connect(self._on_select_all_classes)
        class_btn_layout.addWidget(self.btn_select_all)

        self.btn_select_none = QPushButton("Ninguna")
        self.btn_select_none.setMaximumHeight(24)
        self.btn_select_none.clicked.connect(self._on_select_no_classes)
        class_btn_layout.addWidget(self.btn_select_none)

        self.btn_apply_classes = QPushButton("Aplicar")
        self.btn_apply_classes.setMaximumHeight(24)
        self.btn_apply_classes.setStyleSheet(
            "QPushButton { background: #2980b9; color: white; font-weight: bold; border-radius: 3px; }"
            "QPushButton:hover { background: #3498db; }"
        )
        self.btn_apply_classes.setToolTip("Filtra el explorador por las clases seleccionadas")
        self.btn_apply_classes.clicked.connect(self._on_apply_class_filter)
        class_btn_layout.addWidget(self.btn_apply_classes)
        class_layout.addLayout(class_btn_layout)

        # Lista checkable de clases
        self.class_list = QListWidget()
        self.class_list.setMinimumWidth(160)
        self.class_list.setMaximumWidth(220)
        self.class_list.setStyleSheet(
            f"QListWidget {{ background: {Colors.BG_WIDGET}; border: 1px solid {Colors.BORDER_LIGHT}; }}"
            f"QListWidget::item {{ padding: 3px; }}"
        )
        class_layout.addWidget(self.class_list)

        # Contador de selección
        self.class_count_label = QLabel("0 / 0 clases")
        self.class_count_label.setStyleSheet("color: #888; font-size: 11px;")
        class_layout.addWidget(self.class_count_label)

        class_panel.setLayout(class_layout)
        view_splitter.addWidget(class_panel)

        # ─── Panel central: Imagen + navegación ───
        center_panel = QWidget()
        center_layout = QVBoxLayout()
        center_layout.setContentsMargins(2, 2, 2, 2)

        # Barra de navegación de imágenes
        nav_layout = QHBoxLayout()

        nav_layout.addWidget(QLabel("Imagen:"))
        self.image_combo = QComboBox()
        self.image_combo.setMinimumWidth(220)
        self.image_combo.currentIndexChanged.connect(self._on_image_changed)
        nav_layout.addWidget(self.image_combo)

        self.btn_prev = QPushButton("◀")
        self.btn_prev.setMaximumWidth(32)
        self.btn_prev.clicked.connect(self._on_prev_image)
        nav_layout.addWidget(self.btn_prev)

        self.nav_label = QLabel("0 / 0")
        self.nav_label.setMinimumWidth(70)
        self.nav_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.nav_label)

        self.btn_next = QPushButton("▶")
        self.btn_next.setMaximumWidth(32)
        self.btn_next.clicked.connect(self._on_next_image)
        nav_layout.addWidget(self.btn_next)

        # Navegación rápida por clase + número de imagen en clase
        nav_layout.addWidget(QLabel("Clase:"))
        self.quick_class_combo = QComboBox()
        self.quick_class_combo.setMinimumWidth(140)
        self.quick_class_combo.currentTextChanged.connect(self._on_quick_class_changed)
        nav_layout.addWidget(self.quick_class_combo)

        nav_layout.addWidget(QLabel("Img#:"))
        self.quick_image_spin = QSpinBox()
        self.quick_image_spin.setRange(1, 1)
        self.quick_image_spin.setMaximumWidth(70)
        self.quick_image_spin.valueChanged.connect(self._on_quick_image_number_changed)
        nav_layout.addWidget(self.quick_image_spin)

        self.quick_image_total_label = QLabel("/1")
        self.quick_image_total_label.setMinimumWidth(36)
        nav_layout.addWidget(self.quick_image_total_label)

        nav_layout.addStretch()

        # La prueba vive junto a la imagen: es una acción sobre el encuadre abierto.
        self.btn_preview = QPushButton("Probar aquí")
        self.btn_preview.setStyleSheet(
            "QPushButton { background: #16a085; color: white; padding: 4px 12px; "
            "border-radius: 3px; font-weight: bold; }"
            "QPushButton:hover { background: #1abc9c; }"
            "QPushButton:disabled { background: #7f8c8d; }"
        )
        self.btn_preview.setToolTip(
            "Corre el autoetiquetador en esta imagen con los valores actuales.\n"
            "Dibuja las propuestas en BLANCO y no escribe .seg.\n"
            "Es exactamente lo que hará 'Segmentar' en el lote."
        )
        self.btn_preview.clicked.connect(self._on_preview_current_image)
        nav_layout.addWidget(self.btn_preview)

        self.btn_apply_area = QPushButton("Ajustar área")
        self.btn_apply_area.setStyleSheet(
            "QPushButton { background: #2980b9; color: white; padding: 4px 10px; "
            "border-radius: 3px; font-weight: bold; }"
            "QPushButton:hover { background: #3498db; }"
            "QPushButton:disabled { background: #7f8c8d; }"
        )
        self.btn_apply_area.setToolTip("Pulsa 'Probar aquí' para medir esta imagen.")
        self.btn_apply_area.setEnabled(False)
        self.btn_apply_area.clicked.connect(self._on_apply_area_suggestion)
        nav_layout.addWidget(self.btn_apply_area)

        self.btn_clear_preview = QPushButton("Quitar prueba")
        self.btn_clear_preview.setMaximumWidth(110)
        self.btn_clear_preview.setToolTip("Borra los contornos blancos de la prueba.")
        self.btn_clear_preview.setEnabled(False)
        self.btn_clear_preview.clicked.connect(self._on_clear_preview_clicked)
        nav_layout.addWidget(self.btn_clear_preview)

        center_layout.addLayout(nav_layout)

        # Resultado de la prueba, pegado al visor.
        self.preview_result_label = QLabel(
            "Prueba: sin ejecutar. 'Probar aquí' muestra en blanco lo que "
            "detectaría el lote con los valores actuales."
        )
        self.preview_result_label.setStyleSheet("color: #888; font-size: 11px;")
        self.preview_result_label.setWordWrap(True)
        center_layout.addWidget(self.preview_result_label)

        # Imagen con contornos (clickable for manual contour interaction)
        self.image_label = ClickableImageLabel("Selecciona clases y pulsa Aplicar")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 260)
        self.image_label.setStyleSheet(f"border: 1px solid {Colors.BORDER_LIGHT}; background: {Colors.BG_WIDGET};")
        self.image_label.left_clicked.connect(self._on_image_left_click)
        self.image_label.right_clicked.connect(self._on_image_right_click)
        self.image_label.left_double_clicked.connect(self._on_image_left_double_click)
        self.image_label.right_double_clicked.connect(self._on_image_right_double_click)
        self.image_label.left_dragged.connect(self._on_image_left_drag)
        self.image_label.right_dragged.connect(self._on_image_right_drag)
        self.image_label.left_released.connect(self._on_image_left_release)
        self.image_label.right_released.connect(self._on_image_right_release)
        self.image_label.wheel_up.connect(self._on_prev_image)
        self.image_label.wheel_down.connect(self._on_next_image)
        center_layout.addWidget(self.image_label)

        # Info de granos
        self.grain_info_label = QLabel("")
        self.grain_info_label.setStyleSheet(Styles.LABEL_INFO)
        self.grain_info_label.setWordWrap(True)
        center_layout.addWidget(self.grain_info_label)

        center_panel.setLayout(center_layout)
        view_splitter.addWidget(center_panel)

        # Vista derecha: Crop del grano seleccionado + metadata
        right_container = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(2, 2, 2, 2)

        right_layout.addWidget(QLabel("Grano seleccionado:"))

        self.crop_label = QLabel("Click en un grano")
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setMinimumSize(220, 220)
        self.crop_label.setMaximumSize(300, 300)
        self.crop_label.setStyleSheet(f"border: 1px solid {Colors.BORDER_LIGHT}; background: {Colors.BG_WIDGET};")
        right_layout.addWidget(self.crop_label)

        # Metadata del grano
        self.grain_meta_label = QLabel("")
        self.grain_meta_label.setStyleSheet(Styles.LABEL_INFO)
        self.grain_meta_label.setWordWrap(True)
        right_layout.addWidget(self.grain_meta_label)

        # Selector de grano
        grain_nav = QHBoxLayout()
        self.btn_prev_grain = QPushButton("< Grano")
        self.btn_prev_grain.clicked.connect(self._on_prev_grain)
        grain_nav.addWidget(self.btn_prev_grain)

        self.grain_nav_label = QLabel("—")
        self.grain_nav_label.setAlignment(Qt.AlignCenter)
        grain_nav.addWidget(self.grain_nav_label)

        self.btn_next_grain = QPushButton("Grano >")
        self.btn_next_grain.clicked.connect(self._on_next_grain)
        grain_nav.addWidget(self.btn_next_grain)
        right_layout.addLayout(grain_nav)

        # --- Botones de edición ---
        edit_layout = QHBoxLayout()

        self.btn_delete_grain = QPushButton("Eliminar grano")
        self.btn_delete_grain.setStyleSheet(
            f"QPushButton {{ background: #c0392b; color: white; padding: 4px 10px; "
            f"border-radius: 3px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #e74c3c; }}"
        )
        self.btn_delete_grain.setToolTip("Elimina el grano seleccionado de la anotación")
        self.btn_delete_grain.clicked.connect(self._on_delete_grain)
        edit_layout.addWidget(self.btn_delete_grain)

        self.btn_save_seg = QPushButton("Guardar .seg")
        self.btn_save_seg.setStyleSheet(
            f"QPushButton {{ background: #27ae60; color: white; padding: 4px 10px; "
            f"border-radius: 3px; font-weight: bold; }}"
            f"QPushButton:hover {{ background: #2ecc71; }}"
            f"QPushButton:disabled {{ background: #7f8c8d; }}"
        )
        self.btn_save_seg.setToolTip("Guarda los cambios en el archivo .seg")
        self.btn_save_seg.setEnabled(False)
        self.btn_save_seg.clicked.connect(self._on_save_seg)
        edit_layout.addWidget(self.btn_save_seg)

        self.edit_status_label = QLabel("")
        self.edit_status_label.setStyleSheet("color: #888; font-size: 11px;")
        edit_layout.addWidget(self.edit_status_label)

        edit_layout.addStretch()
        right_layout.addLayout(edit_layout)

        # --- Edición ROI Manual (fase 1 + fase 2) ---
        manual_group = QGroupBox("ROI Manual")
        manual_layout = QVBoxLayout()
        manual_layout.setContentsMargins(6, 6, 6, 6)
        manual_layout.setSpacing(4)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Objetos:"))
        mode_info = QLabel("ROI (semilla U2-Net + exclusiones)")
        mode_info.setStyleSheet("color: #555; font-weight: bold;")
        mode_info.setToolTip(
            "Un solo método: click = semilla del mismo algoritmo que el lote débil.\n"
            "Derecha = ancla/exclusión. No hay un segundo detector paralelo."
        )
        mode_layout.addWidget(mode_info)
        manual_layout.addLayout(mode_layout)

        self.roi_snap_edges_cb = QCheckBox("Snap Canny (solo polígono)")
        self.roi_snap_edges_cb.setChecked(False)
        self.roi_snap_edges_cb.setToolTip(
            "Solo al dibujar polígono a mano. El click U²-Net no se mueve:\n"
            "Canny al marco era el que 'tomaba el borde' del encuadre."
        )
        manual_layout.addWidget(self.roi_snap_edges_cb)

        self.roi_ai_assist_cb = QCheckBox("Asistencia AI por click (U2NET)")
        self.roi_ai_assist_cb.setChecked(True)
        self.roi_ai_assist_cb.setToolTip(
            "Click = semilla U2-Net. El click no usa topes de área ni forma:\n"
            "puedes marcar el grano más grande que veas. Threshold y k sí aplican.\n"
            "Al aceptar (A), esos topes se ajustan para el lote automático."
        )
        manual_layout.addWidget(self.roi_ai_assist_cb)

        self.learn_from_manual_cb = QCheckBox("Aprender de mi etiquetado")
        self.learn_from_manual_cb.setChecked(True)
        self.learn_from_manual_cb.setToolTip(
            "Cada cuerpo que aceptas ajusta los topes del lote (área, circularidad,\n"
            "aspecto, lado, radio) para esa clase. El click de muestra no está\n"
            "limitado: así se descubren semillas más grandes que lo ya visto."
        )
        self.learn_from_manual_cb.toggled.connect(self._on_learning_toggled)
        manual_layout.addWidget(self.learn_from_manual_cb)

        self.learning_info_label = QLabel("")
        self.learning_info_label.setStyleSheet("color: #16a085; font-size: 11px;")
        self.learning_info_label.setWordWrap(True)
        manual_layout.addWidget(self.learning_info_label)

        self.roi_multi_component_cb = QCheckBox("Crear múltiples instancias desde ROI")
        self.roi_multi_component_cb.setChecked(False)
        self.roi_multi_component_cb.setToolTip(
            "Activado: une trazos y puede crear varias islas.\n"
            "Desactivado (recomendado): cada trazo U2-Net es UN ROI; "
            "Aplicar crea una instancia nueva (no sustituye otra por solape)."
        )
        manual_layout.addWidget(self.roi_multi_component_cb)

        close_row = QHBoxLayout()
        self.btn_close_include = QPushButton("Cerrar inclusión")
        self.btn_close_include.clicked.connect(
            lambda: self._close_manual_polygon("include")
        )
        close_row.addWidget(self.btn_close_include)

        self.btn_close_exclude = QPushButton("Cerrar exclusión")
        self.btn_close_exclude.clicked.connect(
            lambda: self._close_manual_polygon("exclude")
        )
        close_row.addWidget(self.btn_close_exclude)
        manual_layout.addLayout(close_row)

        action_row = QHBoxLayout()
        self.btn_apply_roi = QPushButton("Aplicar ROI")
        self.btn_apply_roi.clicked.connect(self._apply_manual_roi_to_grain)
        action_row.addWidget(self.btn_apply_roi)

        self.btn_clear_roi = QPushButton("Limpiar ROI")
        self.btn_clear_roi.clicked.connect(self._clear_manual_roi_points)
        action_row.addWidget(self.btn_clear_roi)

        self.btn_cancel_roi = QPushButton("Cancelar")
        self.btn_cancel_roi.clicked.connect(self._cancel_manual_roi_session)
        action_row.addWidget(self.btn_cancel_roi)
        manual_layout.addLayout(action_row)

        self.roi_hint_label = QLabel(
            "Camino único: CLAHE + U²-Net. Mantén izquierdo: puntos del trazo; "
            "al soltar se cierra el área de mayor puntaje.\n"
            "Derecha = exclusión. CLAHE solo asegura el IoU del mapa.\n"
            "Atajos: C cerrar, A aplicar, Esc cancelar"
        )
        self.roi_hint_label.setStyleSheet("color: #888; font-size: 11px;")
        self.roi_hint_label.setWordWrap(True)
        manual_layout.addWidget(self.roi_hint_label)

        manual_group.setLayout(manual_layout)
        right_layout.addWidget(manual_group)

        right_layout.addStretch()
        right_container.setLayout(right_layout)

        # El panel de grano + ROI es alto; con scroll propio deja de imponer su
        # altura al tab entero (era lo que empujaba las estadísticas fuera de vista).
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QScrollArea.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setWidget(right_container)
        right_scroll.setMinimumWidth(260)
        view_splitter.addWidget(right_scroll)

        view_splitter.setSizes([180, 500, 300])
        view_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(view_splitter)

        group.setLayout(main_layout)
        return group

    # =========================================================================
    # Sección 3: Estadísticas
    # =========================================================================

    def _build_stats_section(self) -> QWidget:
        """
        Barra compacta de dos líneas.

        Antes tenía `setMaximumHeight(80)` con texto envuelto: dentro del
        QScrollArea el contenido se recortaba y no se leía nada.
        """
        group = QGroupBox("Estadísticas")
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)

        self.stats_label = QLabel("Sin datos — genera segmentaciones primero.")
        self.stats_label.setStyleSheet(Styles.LABEL_INFO)
        self.stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.stats_label)

        self.stats_class_label = QLabel("")
        self.stats_class_label.setStyleSheet("color: #888; font-size: 11px;")
        self.stats_class_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.stats_class_label)

        group.setLayout(layout)
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        return group

    def _set_stats(self, totals: str, per_class: Optional[Dict] = None):
        """Único punto de escritura de la barra de estadísticas."""
        self.stats_label.setText(totals)
        per_class = per_class or {}
        if per_class:
            detalle = "  |  ".join(f"{k}: {v}" for k, v in sorted(per_class.items()))
            self.stats_class_label.setText(f"Granos por clase — {detalle}")
            self.stats_class_label.setToolTip(detalle.replace("  |  ", "\n"))
        else:
            self.stats_class_label.setText("")
            self.stats_class_label.setToolTip("")

    # =========================================================================
    # Acciones
    # =========================================================================

    def _get_model_type(self) -> str:
        """Retorna el model_type seleccionado."""
        return "u2netp" if "u2netp" in self.model_combo.currentText() else "u2net"

    def _check_weights_exist(self) -> bool:
        """Verifica que los pesos del modelo seleccionado existen."""
        from pathlib import Path
        model_type = self._get_model_type()
        weights_dir = Path(__file__).resolve().parent.parent / "models" / "weights"
        weights_path = weights_dir / f"{model_type}.pth"
        if not weights_path.exists():
            QMessageBox.critical(
                self, "Pesos no encontrados",
                f"No se encontraron pesos para {model_type} en:\n{weights_path}\n\n"
                f"Opciones:\n"
                f"1. Selecciona 'u2netp' (recomendado, 4.7MB)\n"
                f"2. Ejecuta: python setup_u2net.py --model {model_type}\n"
                f"3. Copia {model_type}.pth desde XYZ_Ctrl_L206_GUI"
            )
            return False
        return True

    def _filter_widgets(self) -> List[QWidget]:
        """Controles que definen qué contornos se aceptan como objeto válido."""
        return [
            self.threshold_slider,
            self.adaptive_k_slider,
            self.min_area_spin,
            self.max_area_spin,
            self.circularity_slider,
            self.max_area_frac_spin,
            self.max_side_frac_spin,
            self.max_aspect_spin,
            self.kernel_spin,
            self.crop_radius_spin,
            self.seed_core_slider,
            self.waist_slider,
        ]

    def _on_profile_changed(self, profile_name: str):
        """Carga un preset en los controles. No aplica nada por sí mismo."""
        preset = PROFILES.get(profile_name)
        if not preset:
            return

        self.adaptive_k_slider._slider.setValue(int(round(preset["adaptive_k"] * 100)))
        self.min_area_spin.setValue(int(min(preset["min_area"], MAX_FRAME_AREA)))
        self.max_area_spin.setValue(int(min(preset["max_area"], MAX_FRAME_AREA)))
        self.circularity_slider._slider.setValue(int(round(preset["min_circularity"] * 100)))
        # Sin max_area_frac en el preset equivale a no acotar por fracción.
        self.max_area_frac_spin.setValue(float(preset.get("max_area_frac", 1.0)))
        self.max_side_frac_spin.setValue(float(preset["max_bbox_side_frac"]))
        self.max_aspect_spin.setValue(float(preset["max_aspect_ratio"]))
        self.kernel_spin.setValue(max(1, int(preset["morph_kernel_size"])))
        if "crop_radius" in preset:
            self.crop_radius_spin.setValue(int(preset["crop_radius"]))

        self._update_effective_params_label()
        logger.info(f"[Contornos] Preset '{profile_name}' cargado en los controles")

    def _on_filter_value_changed(self):
        """Al mover un filtro, la prueba en pantalla deja de ser válida."""
        self._update_effective_params_label()
        if self._clear_preview():
            self._redraw_contours()

    def _update_effective_params_label(self):
        """Muestra los topes que realmente se aplicarán al encuadre actual."""
        if not hasattr(self, "effective_params_label"):
            return

        if self._current_image_bgr is not None:
            h, w = self._current_image_bgr.shape[:2]
            origen = f"imagen actual {w}x{h}"
        else:
            h, w = FRAME_H, FRAME_W
            origen = f"encuadre nominal {w}x{h}"

        area_cap = min(self.max_area_spin.value(), int(self.max_area_frac_spin.value() * h * w))
        side_cap = int(self.max_side_frac_spin.value() * min(h, w))
        self.effective_params_label.setText(
            f"Efectivo ({origen}) — semilla: threshold "
            f"{self.threshold_slider._slider.value() / 100.0:.2f} + k "
            f"{self.adaptive_k_slider._slider.value() / 100.0:.2f}, "
            f"radio {self.crop_radius_spin.value()} px, "
            f"contacto {'separado' if self.split_touching_cb.isChecked() else 'fusionado'}"
            f" (núcleo {self.seed_core_slider._slider.value() / 100.0:.2f}, "
            f"cintura {self.waist_slider._slider.value() / 100.0:.2f})"
            f"{', sin cuerpos de borde' if self.drop_border_cb.isChecked() else ''} | "
            f"cuerpo válido: área {self.min_area_spin.value():,} – {area_cap:,} px², "
            f"lado máx {side_cap:,} px, aspecto máx {self.max_aspect_spin.value():.1f}, "
            f"circ. mín {self.circularity_slider._slider.value() / 100.0:.2f}"
        )

    def _get_config(self) -> Dict:
        """
        Recopila parámetros de la UI como dict de configuración.

        Envía TODOS los filtros de forma explícita: resolve_grain_detection_config
        usa setdefault, así que ningún preset puede sobrescribir la UI en silencio.
        """
        return {
            "model_type": self._get_model_type(),
            "detection_profile": self.profile_combo.currentText(),
            "saliency_threshold": self.threshold_slider._slider.value() / 100.0,
            "adaptive_k": self.adaptive_k_slider._slider.value() / 100.0,
            "min_area": self.min_area_spin.value(),
            "max_area": self.max_area_spin.value(),
            "max_area_frac": self.max_area_frac_spin.value(),
            "min_circularity": self.circularity_slider._slider.value() / 100.0,
            "max_bbox_side_frac": self.max_side_frac_spin.value(),
            "max_aspect_ratio": self.max_aspect_spin.value(),
            "morph_kernel_size": self.kernel_spin.value(),
            "crop_radius": self.crop_radius_spin.value(),
            "split_touching": self.split_touching_cb.isChecked(),
            "seed_core_frac": self.seed_core_slider._slider.value() / 100.0,
            "waist_frac": self.waist_slider._slider.value() / 100.0,
            "drop_border_objects": self.drop_border_cb.isChecked(),
            "object_finder": "roi_seed",
            "crop_padding": self.padding_spin.value() / 100.0,
            "crop_size": self.crop_size_spin.value(),
        }

    def _on_dataset_changed(self, text):
        """Cuando cambia el dataset seleccionado, auto-cargar .seg existentes."""
        self._refresh_explorer()

    def _set_segmentation_busy(self, busy: bool):
        """Único punto que habilita/inhabilita la acción de segmentar."""
        self.btn_segment.setEnabled(not busy)
        self.scope_combo.setEnabled(not busy)
        self.overwrite_cb.setEnabled(not busy)
        self.btn_preview.setEnabled(not busy)
        self.btn_cancel_segment.setEnabled(busy)
        self._set_bank_buttons_enabled(not busy)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setValue(0)

    def _on_preview_current_image(self):
        """Prueba los valores actuales en la imagen abierta, sin escribir .seg."""
        if self._current_image_bgr is None:
            QMessageBox.information(
                self, "Sin imagen",
                "Abre una imagen en el explorador para probar la configuración."
            )
            return
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        if not self._check_weights_exist():
            return

        self.btn_preview.setEnabled(False)
        self.preview_result_label.setText("Probando en esta imagen...")
        self._preview_worker = PreviewWorker(self._current_image_bgr, self._get_config())
        self._preview_worker.finished.connect(self._on_preview_result)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_result(self, payload):
        self.btn_preview.setEnabled(True)
        if isinstance(payload, dict):
            self._preview_grains = payload.get("grains") or []
            self._area_suggestion = payload.get("suggestion")
        else:
            self._preview_grains = payload or []
            self._area_suggestion = None

        self.btn_clear_preview.setEnabled(bool(self._preview_grains))
        n = len(self._preview_grains)
        n_seg = len(self._current_grains)

        # Medición del encuadre: dice el tamaño real de esta especie aunque los
        # topes de área la estén descartando por completo.
        sug = self._area_suggestion
        medido = ""
        if sug:
            medido = (
                f"  |  medido aquí: {sug['measured']} cuerpo(s) de "
                f"{sug['smallest']:,} a {sug['largest']:,} px²"
            )

        min_area = self.min_area_spin.value()
        if n == 0 and sug and sug["largest"] < min_area:
            diagnostico = (
                f"Min área ({min_area:,} px²) es mayor que el cuerpo más grande "
                f"de esta imagen ({sug['largest']:,} px²): se descarta todo. "
                f"Pulsa 'Ajustar área'."
            )
        elif n == 0:
            diagnostico = (
                "Nada detectado: baja Threshold o Adaptive k (más semillas), "
                "o baja Min área."
            )
        elif n < n_seg:
            diagnostico = (
                "Menos que el .seg guardado: baja Threshold / Min área / Min circ., "
                "o sube Radio semilla si los cuerpos salen cortados."
            )
        elif n > n_seg and n_seg > 0:
            diagnostico = (
                "Más que el .seg guardado: sube Min área o Min circ. si aparece fondo."
            )
        else:
            diagnostico = "Coincide con el .seg guardado."

        self._refresh_area_suggestion_button()
        self.preview_result_label.setText(
            f"Prueba (blanco): {n} cuerpo{'s' if n != 1 else ''}  |  "
            f".seg guardado: {n_seg}{medido}  —  {diagnostico}"
        )
        self._redraw_contours()

    # =========================================================================
    # Calibración por demostración: lo que etiquetas a mano fija el automático
    # =========================================================================

    def _current_class_name(self) -> Optional[str]:
        """Clase de la imagen abierta; los topes se aprenden por clase."""
        if not self._current_images:
            return None
        idx = min(max(self._current_image_idx, 0), len(self._current_images) - 1)
        return Path(self._current_images[idx]).parent.name

    def _learn_from_grains(self, grains, reason: str = "etiquetado"):
        """
        Adapta los parámetros a los cuerpos que el usuario acaba de aceptar.

        Es el puente entre el método manual y el automático: el click define qué
        es un cuerpo válido y esos mismos topes son los que usará el lote.
        """
        if not getattr(self, "learn_from_manual_cb", None) or not self.learn_from_manual_cb.isChecked():
            return
        if self._current_image_bgr is None or not grains:
            return

        from src.grain_detection.param_learning import (
            blend_with_current,
            contours_from_grains,
            describe_envelope,
            envelope_from_contours,
            merge_envelopes,
            params_from_envelope,
        )

        h, w = self._current_image_bgr.shape[:2]
        nuevo = envelope_from_contours(contours_from_grains(grains), h, w)
        if nuevo is None:
            return

        clase = self._current_class_name() or "—"
        acumulado = merge_envelopes(self._class_envelopes.get(clase), nuevo)
        self._class_envelopes[clase] = acumulado

        aplicado = blend_with_current(
            params_from_envelope(acumulado, MAX_FRAME_AREA), self._get_config()
        )
        if aplicado is None:
            return

        cambios = self._apply_learned_params(aplicado)
        logger.info(
            "[Calibración] %s en %s — %s", reason, clase, describe_envelope(acumulado)
        )
        if cambios:
            logger.info("[Calibración] %s -> %s", clase, ", ".join(cambios))
            self.status_label.setText(
                f"Parámetros adaptados a {clase} desde tu etiquetado: "
                + ", ".join(cambios)
            )
        self._update_learning_label(clase, acumulado)
        self._persist_class_envelopes()

    def _apply_learned_params(self, valores: Dict, shrink_radius: bool = False) -> List[str]:
        """Escribe los topes aprendidos en los controles. Devuelve qué cambió."""
        cambios: List[str] = []

        def _spin(widget, key, label, fmt="{:,}"):
            nuevo = valores[key]
            if isinstance(widget, QDoubleSpinBox):
                if abs(widget.value() - nuevo) < 1e-6:
                    return
            elif int(widget.value()) == int(nuevo):
                return
            widget.setValue(nuevo)
            cambios.append(f"{label} {fmt.format(nuevo)}")

        def _slider(widget, key, label):
            nuevo = int(round(valores[key] * 100))
            if widget._slider.value() == nuevo:
                return
            widget._slider.setValue(nuevo)
            cambios.append(f"{label} {nuevo / 100.0:.2f}")

        # Max antes que Min: el spinbox recorta si el mínimo supera al máximo.
        _spin(self.max_area_spin, "max_area", "Max área", "{:,} px²")
        _spin(self.min_area_spin, "min_area", "Min área", "{:,} px²")
        _slider(self.circularity_slider, "min_circularity", "Min circ.")
        _spin(self.max_aspect_spin, "max_aspect_ratio", "Max aspecto", "{:.1f}")
        _spin(self.max_side_frac_spin, "max_bbox_side_frac", "Max lado frac", "{:.2f}")

        return cambios

    def _update_learning_label(self, clase: Optional[str] = None, envelope=None):
        """Muestra de qué está aprendiendo el buscador."""
        if not hasattr(self, "learning_info_label"):
            return
        from src.grain_detection.param_learning import describe_envelope

        clase = clase or self._current_class_name()
        envelope = envelope if envelope is not None else self._class_envelopes.get(clase)
        if not envelope:
            self.learning_info_label.setText(
                "Aún sin aprender de esta clase: el click no tiene tope de área. "
                "Marca un cuerpo (también uno grande) y el lote se ajusta."
            )
            return
        self.learning_info_label.setText(
            f"Lote ({clase}): {describe_envelope(envelope)}. "
            "El click no usa estos topes."
        )

    def _adopt_class_envelope(self):
        """Al cambiar de clase, recupera lo aprendido de esa clase."""
        if not getattr(self, "learn_from_manual_cb", None) or not self.learn_from_manual_cb.isChecked():
            return
        clase = self._current_class_name()
        if clase is None or clase == self._learning_class:
            self._update_learning_label(clase)
            return

        self._learning_class = clase
        envelope = self._class_envelopes.get(clase)
        if envelope:
            from src.grain_detection.param_learning import (
                blend_with_current,
                params_from_envelope,
            )

            learned = params_from_envelope(envelope, MAX_FRAME_AREA)
            aplicado = blend_with_current(learned, self._get_config())
            if aplicado:
                cambios = self._apply_learned_params(aplicado)
                if cambios:
                    logger.info(
                        "[Calibración] %s: topes recuperados -> %s",
                        clase, ", ".join(cambios),
                    )
        self._update_learning_label(clase, envelope)

    def _persist_class_envelopes(self):
        from src.grain_detection.param_learning import save_envelopes

        try:
            save_envelopes(ANNOTATION_ROOT, self._class_envelopes)
        except OSError as exc:
            logger.warning("[Calibración] no se pudo guardar envolventes: %s", exc)

    def _load_class_envelopes(self):
        from src.grain_detection.param_learning import load_envelopes, merge_envelopes

        loaded = load_envelopes(ANNOTATION_ROOT)
        if not loaded:
            return
        for clase, env in loaded.items():
            self._class_envelopes[clase] = merge_envelopes(
                self._class_envelopes.get(clase), env
            )
        logger.info(
            "[Calibración] %d clase(s) recuperada(s) de %s",
            len(loaded), ANNOTATION_ROOT,
        )

    def _refresh_area_suggestion_button(self):
        """Habilita 'Ajustar área' solo si la medición cambia los topes actuales."""
        if not hasattr(self, "btn_apply_area"):
            return
        sug = self._area_suggestion
        if not sug:
            self.btn_apply_area.setEnabled(False)
            self.btn_apply_area.setToolTip("Pulsa 'Probar aquí' para medir esta imagen.")
            return

        distinto = (
            sug["min_area"] != self.min_area_spin.value()
            or sug["max_area"] != self.max_area_spin.value()
        )
        self.btn_apply_area.setEnabled(distinto)
        self.btn_apply_area.setToolTip(
            f"Aplica Min área {sug['min_area']:,} y Max área {sug['max_area']:,} px²,\n"
            f"medidos sobre los {sug['measured']} cuerpo(s) de esta imagen "
            f"({sug['smallest']:,}–{sug['largest']:,} px²)."
        )

    def _on_apply_area_suggestion(self):
        """Lleva los topes de área al tamaño real de la especie abierta."""
        sug = self._area_suggestion
        if not sug:
            return

        antes = (self.min_area_spin.value(), self.max_area_spin.value())
        # Max primero: si el mínimo nuevo supera el máximo viejo, el spinbox lo recorta.
        self.max_area_spin.setValue(min(sug["max_area"], MAX_FRAME_AREA))
        self.min_area_spin.setValue(min(sug["min_area"], MAX_FRAME_AREA))

        msg = (
            f"Área ajustada a esta especie: Min {self.min_area_spin.value():,} y "
            f"Max {self.max_area_spin.value():,} px² "
            f"(antes {antes[0]:,} / {antes[1]:,}). Vuelve a pulsar 'Probar aquí'."
        )
        self.preview_result_label.setText(msg)
        self.status_label.setText(msg)
        logger.info(f"[Calibración] {msg}")
        self._refresh_area_suggestion_button()

    def _on_preview_error(self, error_msg: str):
        self.btn_preview.setEnabled(True)
        self._preview_grains = []
        self._area_suggestion = None
        self._refresh_area_suggestion_button()
        self.btn_clear_preview.setEnabled(False)
        self.preview_result_label.setText(f"Error en la prueba: {error_msg[:120]}")
        logger.error(f"[Preview] {error_msg}")

    def _on_clear_preview_clicked(self):
        if self._clear_preview():
            self._redraw_contours()

    def _clear_preview(self) -> bool:
        """La previsualización caduca al cambiar de imagen o de parámetros."""
        if hasattr(self, "btn_clear_preview"):
            self.btn_clear_preview.setEnabled(False)
        if not self._preview_grains:
            return False
        self._preview_grains = []
        if hasattr(self, "preview_result_label"):
            self.preview_result_label.setText(
                "Prueba caducada: los valores cambiaron. Pulsa 'Probar aquí' otra vez."
            )
        return True

    # =========================================================================
    # Banco de datos: los .seg que alimentan la fase de entrenamiento
    # =========================================================================

    def _on_clear_selected_annotations(self):
        """Limpia el banco solo para las clases con tick."""
        marcadas = self._get_checked_classes()
        if not marcadas:
            QMessageBox.warning(
                self, "Sin clases",
                "Marca con tick las clases cuyo .seg quieres borrar."
            )
            return
        self._clear_annotations(marcadas)

    def _on_clear_all_annotations(self):
        """Limpia el banco completo para empezar de cero."""
        self._clear_annotations(None)

    def _clear_annotations(self, class_filter: Optional[List[str]]):
        """
        Único punto de borrado del banco de .seg.

        Ambos botones entran aquí; solo cambia el alcance. Además de los archivos
        se invalida el registro por split (su fast path devolvería conteos viejos)
        y se descarta el estado en memoria de la imagen abierta.
        """
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(
                self, "Ocupado",
                "Hay una segmentación en curso. Cancélala antes de limpiar el banco."
            )
            return

        from src.grain_detection.annotator import count_annotations, delete_annotations

        counts = count_annotations(ANNOTATION_ROOT, class_filter)
        total = sum(counts.values())
        if total == 0:
            alcance = "las clases marcadas" if class_filter else "el banco"
            QMessageBox.information(
                self, "Nada que limpiar", f"No hay .seg en {alcance}."
            )
            return

        if class_filter:
            titulo = "Limpiar clases marcadas"
            detalle = "\n".join(f"  {k}: {v} .seg" for k, v in sorted(counts.items()))
            cuerpo = (
                f"Se borrarán {total} .seg de {len(counts)} clase(s):\n\n{detalle}\n\n"
                f"Las demás clases no se tocan."
            )
        else:
            titulo = "Limpiar TODO el banco"
            cuerpo = (
                f"Se borrarán los {total} .seg de TODAS las clases "
                f"({len(counts)} con datos).\n\n"
                f"El proyecto quedará sin etiquetas y habrá que segmentar de cero."
            )

        reply = QMessageBox.question(
            self, titulo,
            f"{cuerpo}\n\nTambién se borran los backups .seg.bak y el registro.\n"
            f"Esta acción no se puede deshacer. ¿Continuar?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        result = delete_annotations(ANNOTATION_ROOT, class_filter)

        # El registro cachea conteos: sin invalidarlo el explorador mentiría.
        from src.grain_detection.contour_registry import ContourRegistry
        for split in SPLIT_DIRS:
            if Path(split).exists():
                ContourRegistry.delete(split)

        # Estado en memoria de la imagen abierta.
        self._current_grains = []
        self._selected_grain_idx = -1
        self._active_grain_idx = -1
        self._clear_history()
        self._reset_manual_roi_session()
        self._clear_preview()
        self._mark_clean()

        resumen = (
            f"Banco limpiado: {result['seg_removed']} .seg y "
            f"{result['bak_removed']} backups borrados"
            + (f", {result['errors']} errores" if result["errors"] else "")
        )
        self.status_label.setText(resumen)
        self._log_to_parent(resumen, "WARNING" if result["errors"] else "SUCCESS")
        logger.info(f"[Banco] {resumen}")

        self._refresh_explorer()
        self._on_apply_class_filter()
        self._update_bank_info()

    def _update_bank_info(self):
        """Estado del banco: lo que verá la fase siguiente."""
        if not hasattr(self, "bank_info_label"):
            return
        from src.grain_detection.annotator import count_annotations

        counts = count_annotations(ANNOTATION_ROOT)
        total = sum(counts.values())
        if total == 0:
            self.bank_info_label.setText(f"vacío — {ANNOTATION_ROOT}/")
        else:
            self.bank_info_label.setText(
                f"{total} .seg en {len(counts)} clase(s) — {ANNOTATION_ROOT}/"
            )
        self.bank_info_label.setToolTip(
            "\n".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "sin .seg"
        )

    def _set_bank_buttons_enabled(self, enabled: bool):
        self.btn_clear_selected_seg.setEnabled(enabled)
        self.btn_clear_all_seg.setEnabled(enabled)

    def _on_cancel_segment(self):
        """Pide el corte al worker; termina la imagen en curso y conserva lo escrito."""
        if self._worker is None or not self._worker.isRunning():
            return
        self._worker.requestInterruption()
        self.btn_cancel_segment.setEnabled(False)
        self.status_label.setText("Cancelando... se detendrá al terminar la imagen en curso.")

    def _on_segment(self):
        """Despacha la segmentación según alcance y overwrite elegidos."""
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Ocupado", "Ya hay una segmentación en progreso.")
            return
        if not self._check_weights_exist():
            return

        overwrite = self.overwrite_cb.isChecked()
        all_splits = self.scope_combo.currentIndex() == 1

        # El tick manda en cualquier alcance: nunca se anotan clases no marcadas.
        class_filter = self._get_checked_classes()
        if not class_filter:
            QMessageBox.warning(
                self, "Sin clases",
                "Marca con tick al menos una clase.\n"
                "La segmentación solo procesa las clases marcadas."
            )
            return

        alcance = "train + val + test" if all_splits else f"'{self.dataset_combo.currentText()}'"
        clases = ", ".join(class_filter)
        if overwrite:
            reply = QMessageBox.question(
                self, "Confirmar sobrescritura",
                f"¿Regenerar las segmentaciones de {alcance}?\n\n"
                f"Clases ({len(class_filter)}): {clases}\n\n"
                f"Los .seg existentes de esas clases serán sobrescritos "
                f"(se guarda backup .seg.bak).",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        self._run_annotation(overwrite, class_filter, all_splits=all_splits)

    def _run_annotation(self, overwrite: bool, class_filter: List[str], all_splits: bool = False):
        """Lanza el único worker de anotación con el alcance y las clases marcadas."""
        if all_splits:
            if not Path(SPLIT_DIRS[0]).exists():
                QMessageBox.warning(
                    self, "Error",
                    f"No se encontró {SPLIT_DIRS[0]}.\n"
                    "Primero prepara el dataset en 'Gestor de Datos'."
                )
                return
            splits = SPLIT_DIRS
        else:
            root_dir = self.dataset_combo.currentText()
            if not Path(root_dir).exists():
                QMessageBox.warning(self, "Error", f"Directorio no encontrado: {root_dir}")
                return
            splits = [root_dir]

        n_cls = len(class_filter)
        destino = "train + val + test" if all_splits else Path(splits[0]).name
        self.status_label.setText(
            f"Segmentando {destino}: {n_cls} clase{'s' if n_cls != 1 else ''} marcada"
            f"{'s' if n_cls != 1 else ''}..."
        )

        self._set_segmentation_busy(True)
        self._worker = AnnotationWorker(
            splits, overwrite, self._get_config(), class_filter=class_filter
        )
        self._worker.progress.connect(self._on_annotation_progress)
        self._worker.finished.connect(self._on_annotation_finished)
        self._worker.error.connect(self._on_annotation_error)
        self._worker.start()

    def _on_annotation_progress(self, current, total, msg):
        pct = int(current / max(1, total) * 100)
        self.progress_bar.setValue(pct)
        self.status_label.setText(f"[{current}/{total}] {msg}")

    def _on_annotation_finished(self, stats):
        try:
            self._set_segmentation_busy(False)
            self.status_label.setText(self._format_annotation_summary(stats))

            self._update_stats(stats)
            self._update_bank_info()

            # Auto-guardar registro ANTES de refresh (para que fast path lo encuentre)
            self._save_registry_silent(annotation_stats=stats)

            # Refresh con expand_checked=True para mostrar clases recién anotadas
            self._refresh_explorer(expand_checked=True)
            
            # Recargar visor para mostrar las imágenes con .seg recién creados
            self._on_apply_class_filter()
            logger.info("[Explorer] Visor actualizado con .seg recién generados")

            cancelled = bool(stats.get("cancelled"))
            self._log_to_parent(
                f"Segmentación {'CANCELADA' if cancelled else 'completada'} — .seg en "
                f"{stats.get('annotation_root', ANNOTATION_ROOT)}/\n"
                f"Imágenes: {stats.get('total_images', 0)} | "
                f"anotadas: {stats.get('annotated', 0)} | "
                f"omitidas: {stats.get('skipped', 0)} | "
                f"errores: {stats.get('errors', 0)}\n"
                f"Granos: {stats.get('total_grains', 0)} "
                f"({stats.get('grains_per_image', 0):.1f} por imagen)",
                "WARNING" if cancelled else "SUCCESS",
            )
        except Exception as e:
            import traceback
            logger.error(f"[Annotation] Error en callback finished: {e}\n{traceback.format_exc()}")
            self.status_label.setText(f"Error post-anotación: {e}")

    @staticmethod
    def _format_annotation_summary(stats: Dict) -> str:
        """Resumen por split cuando hubo varios; total cuando fue uno."""
        estado = "CANCELADO" if stats.get("cancelled") else "Completado"
        per_split = stats.get("per_split") or {}
        if len(per_split) > 1:
            detalle = " | ".join(
                f"{name}: {s.get('total_grains', 0)} granos "
                f"({s.get('annotated', 0)} nuevos, {s.get('skipped', 0)} existentes)"
                for name, s in per_split.items()
            )
            return f"{estado} — {detalle}"
        return (
            f"{estado}: {stats.get('annotated', 0)} anotados, "
            f"{stats.get('skipped', 0)} omitidos, "
            f"{stats.get('total_grains', 0)} granos totales "
            f"({stats.get('elapsed_ms', 0):.0f}ms)"
        )

    def _on_annotation_error(self, error_msg):
        self._set_segmentation_busy(False)
        self.status_label.setText(f"Error: {error_msg[:100]}")
        QMessageBox.critical(self, "Error", f"Error en segmentación:\n{error_msg[:500]}")

    def _on_save_registry(self):
        """Botón manual: guarda el registro y confirma con diálogo."""
        self._save_registry(interactive=True)

    def _save_registry_silent(self, annotation_stats=None):
        """Auto-guardado tras anotar: mismo guardado, sin diálogos."""
        self._save_registry(annotation_stats=annotation_stats, interactive=False)

    def _save_registry(self, annotation_stats=None, interactive: bool = False):
        """Único punto de guardado del registro de contornos."""
        root_dir = self.dataset_combo.currentText()
        if not Path(root_dir).exists():
            if interactive:
                QMessageBox.warning(self, "Error", f"Directorio no encontrado: {root_dir}")
            return

        try:
            from src.grain_detection.contour_registry import ContourRegistry
            reg_path = ContourRegistry.save(
                root_dir,
                detection_params=self._get_config(),
                annotation_stats=annotation_stats,
                annotation_root=ANNOTATION_ROOT,
            )
        except Exception as e:
            if interactive:
                import traceback
                QMessageBox.critical(self, "Error", f"Error guardando registro:\n{e}")
                logger.error(f"[Registry] Error guardando: {e}\n{traceback.format_exc()}")
            else:
                logger.warning(f"[Registry] Error en auto-save: {e}")
            return

        logger.info(f"[Registry] Guardado: {reg_path}")
        if not interactive:
            self._log_to_parent(f"Registro auto-guardado: {reg_path.name}", "DEBUG")
            return

        summary = ContourRegistry.load(root_dir)["summary"]
        self.status_label.setText(
            f"Registros guardados: {summary['images_with_seg']} imgs con .seg, "
            f"{summary['total_grains']} granos — {reg_path.name}"
        )
        self._log_to_parent(
            f"Registro guardado: {reg_path} "
            f"({summary['images_with_seg']} imgs, {summary['total_grains']} granos, "
            f"{summary['num_classes']} clases)",
            "SUCCESS",
        )
        QMessageBox.information(
            self, "Registros Guardados",
            f"Registro guardado exitosamente en:\n{reg_path}\n\n"
            f"Imágenes: {summary['total_images']}\n"
            f"Con .seg: {summary['images_with_seg']}\n"
            f"Granos totales: {summary['total_grains']}\n"
            f"Clases: {summary['num_classes']}\n\n"
            f"Los .seg viven en {ANNOTATION_ROOT}/. 'Cargar Existentes' restaura desde aquí."
        )

    def _log_to_parent(self, message: str, level: str = "INFO"):
        """Log en el panel de la ventana principal si existe."""
        if self.parent_window and hasattr(self.parent_window, "log_widget"):
            self.parent_window.log_widget.append_log(message, level)

    def _on_validate(self):
        """Valida integridad de pares imagen/.seg (no requiere modelo U²-Net)."""
        root_dir = self.dataset_combo.currentText()
        if not Path(root_dir).exists():
            QMessageBox.warning(self, "Error", f"Directorio no encontrado: {root_dir}")
            return

        try:
            from src.grain_detection.annotator import SegmentationAnnotator
            annotator = SegmentationAnnotator()  # Lazy: no carga modelo
            
            result = annotator.validate_annotations(
                root_dir, annotation_root=ANNOTATION_ROOT
            )

            self._log_to_parent(
                f"Validación completada: {result['valid']}/{result['total_images']} con .seg válido"
            )

            msg = (
                f"Total imágenes: {result['total_images']}\n"
                f"Con .seg válido: {result['valid']}\n"
                f"Sin .seg: {len(result['missing_seg'])}\n"
                f"Corruptos: {len(result['corrupt_seg'])}\n"
                f"Huérfanos: {len(result['orphan_seg'])}\n\n"
                f"Buscando en: {ANNOTATION_ROOT}/"
            )
            QMessageBox.information(self, "Validación", msg)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error validando:\n{e}")

    # =========================================================================
    # Explorador
    # =========================================================================

    def _refresh_explorer(self, expand_checked=False):
        """Refresca la lista de clases y el explorador con el dataset actual.

        Args:
            expand_checked: If True, auto-check all classes that have .seg files
                (in addition to previously checked classes). Used after annotation
                to ensure newly annotated classes become visible.

        Fast path: Si existe contour_registry.json, carga conteos desde ahí
        (instantáneo, sin escanear archivos). Fallback: escaneo de directorio.
        """
        root_dir = self.dataset_combo.currentText()
        root = Path(root_dir)
        if not root.exists():
            self.status_label.setText(f"Directorio no encontrado: {root_dir}")
            return

        # Una sola vez por sesión: renombra .seg legacy ambiguos entre placas.
        if not getattr(self, "_legacy_seg_migrated", False):
            try:
                mig = migrate_legacy_annotations(ANNOTATION_ROOT, SPLIT_DIRS)
                self._legacy_seg_migrated = True
                if mig.get("renamed"):
                    msg = (
                        f"Migración .seg: {mig['renamed']} renombrados con placa. "
                        f"Si un stem existía en varias placas, se asignó a train "
                        f"(val/test de ese stem deben re-etiquetarse)."
                    )
                    logger.warning("[Explorer] %s", msg)
                    if self.parent_window and hasattr(self.parent_window, "log_widget"):
                        self.parent_window.log_widget.append_log(msg, "WARNING")
            except Exception as e:
                self._legacy_seg_migrated = True
                logger.warning("[Explorer] Migración legacy falló: %s", e)

        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])

        # Intentar fast path desde registro
        seg_counts = {}
        img_counts = {}
        registry_loaded = False

        try:
            from src.grain_detection.contour_registry import ContourRegistry
            registry = ContourRegistry.load(root_dir)
            if registry and not ContourRegistry.is_stale(root_dir, ANNOTATION_ROOT):
                # Fast path: conteos desde el registro
                for entry in registry["images"]:
                    cls = entry["class"]
                    img_counts[cls] = img_counts.get(cls, 0) + 1
                    if entry.get("seg_path"):
                        seg_counts[cls] = seg_counts.get(cls, 0) + 1

                # Restaurar parámetros de detección del registro
                self._restore_params_from_registry(registry.get("detection_params", {}))

                summary = registry["summary"]
                self.status_label.setText(
                    f"Registro cargado: {summary['images_with_seg']} imgs con .seg, "
                    f"{summary['total_grains']} granos "
                    f"({registry['timestamp'][:19]})"
                )
                registry_loaded = True
                logger.info("[Explorer] Cargado desde contour_registry.json (fast path)")
        except Exception as e:
            logger.debug(f"[Explorer] Registry fast-path falló: {e}")

        if not registry_loaded:
            # Fallback: escaneo directo de directorio
            logger.info(f"[Explorer] Escaneando directorio (fallback) - .seg en {ANNOTATION_ROOT}")

            for cls in classes:
                imgs = class_images(root / cls)
                segs = [f for f in imgs if find_existing_seg_path(f, ANNOTATION_ROOT)]
                img_counts[cls] = len(imgs)
                seg_counts[cls] = len(segs)
                logger.debug(f"[Explorer] {cls}: {len(imgs)} imgs, {len(segs)} con .seg")

        # Guardar clases previamente checkeadas
        prev_checked = self._get_checked_classes()

        # Poblar lista checkable
        self.class_list.clear()
        for cls in classes:
            n_seg = seg_counts.get(cls, 0)
            n_img = img_counts.get(cls, 0)
            item = QListWidgetItem(f"{cls}  ({n_seg}/{n_img})")
            item.setData(Qt.UserRole, cls)  # nombre limpio
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            # Mantener check previo o activar por default si tiene .seg
            if expand_checked:
                # After annotation: check if previously checked OR has .seg data
                item.setCheckState(Qt.Checked if (cls in prev_checked or n_seg > 0) else Qt.Unchecked)
            elif prev_checked:
                item.setCheckState(Qt.Checked if cls in prev_checked else Qt.Unchecked)
            else:
                item.setCheckState(Qt.Checked if n_seg > 0 else Qt.Unchecked)
            self.class_list.addItem(item)

        self._update_class_count_label()
        self._load_class_envelopes()
        self._learning_class = None
        self._on_apply_class_filter()
        self._update_stats_from_directory(root_dir)
        self._update_bank_info()

    def _restore_params_from_registry(self, params: Dict):
        """Restaura los parámetros de la GUI desde los guardados en el registro."""
        if not params:
            return
        try:
            # El perfil se aplica primero: los valores guardados deben ganar sobre el preset.
            if params.get("detection_profile") in PROFILES:
                self.profile_combo.setCurrentText(params["detection_profile"])
            if "saliency_threshold" in params:
                self.threshold_slider._slider.setValue(int(params["saliency_threshold"] * 100))
            if "adaptive_k" in params:
                self.adaptive_k_slider._slider.setValue(int(params["adaptive_k"] * 100))
            if "min_area" in params:
                self.min_area_spin.setValue(int(params["min_area"]))
            if "max_area" in params:
                self.max_area_spin.setValue(int(min(params["max_area"], MAX_FRAME_AREA)))
            if params.get("max_area_frac") is not None:
                self.max_area_frac_spin.setValue(float(params["max_area_frac"]))
            if "max_bbox_side_frac" in params:
                self.max_side_frac_spin.setValue(float(params["max_bbox_side_frac"]))
            if "max_aspect_ratio" in params:
                self.max_aspect_spin.setValue(float(params["max_aspect_ratio"]))
            if "min_circularity" in params:
                self.circularity_slider._slider.setValue(int(params["min_circularity"] * 100))
            if "morph_kernel_size" in params:
                self.kernel_spin.setValue(params["morph_kernel_size"])
            if "split_touching" in params:
                self.split_touching_cb.setChecked(bool(params["split_touching"]))
            if params.get("seed_core_frac") is not None:
                self.seed_core_slider._slider.setValue(
                    int(round(float(params["seed_core_frac"]) * 100))
                )
            if params.get("waist_frac") is not None:
                self.waist_slider._slider.setValue(
                    int(round(float(params["waist_frac"]) * 100))
                )
            if "drop_border_objects" in params:
                self.drop_border_cb.setChecked(bool(params["drop_border_objects"]))
            if "crop_padding" in params:
                self.padding_spin.setValue(int(params["crop_padding"] * 100))
            if "crop_size" in params:
                self.crop_size_spin.setValue(params["crop_size"])
            if "model_type" in params:
                idx = 0 if params["model_type"] == "u2netp" else 1
                self.model_combo.setCurrentIndex(idx)
            self._update_effective_params_label()
            logger.info("[Explorer] Parámetros restaurados desde registro")
        except Exception as e:
            logger.warning(f"[Explorer] Error restaurando parámetros: {e}")

    def _get_checked_classes(self) -> List[str]:
        """Retorna lista de nombres de clases checkeadas."""
        checked = []
        for i in range(self.class_list.count()):
            item = self.class_list.item(i)
            if item.checkState() == Qt.Checked:
                checked.append(item.data(Qt.UserRole))
        return checked

    def _update_class_count_label(self):
        """Actualiza el contador de clases seleccionadas."""
        total = self.class_list.count()
        checked = len(self._get_checked_classes())
        self.class_count_label.setText(f"{checked} / {total} clases")

    def _on_select_all_classes(self):
        """Checkea todas las clases."""
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(Qt.Checked)
        self._update_class_count_label()

    def _on_select_no_classes(self):
        """Descheckea todas las clases."""
        for i in range(self.class_list.count()):
            self.class_list.item(i).setCheckState(Qt.Unchecked)
        self._update_class_count_label()

    def _on_apply_class_filter(self):
        """Aplica el filtro de clases: recarga imágenes de las clases seleccionadas.

        Shows ALL images from selected classes, prioritizing those with .seg files first.
        Images without .seg are marked as "[sin .seg]" for easy identification.
        """
        selected_classes = self._get_checked_classes()
        self._update_class_count_label()

        root_dir = self.dataset_combo.currentText()
        root = Path(root_dir)

        # Recopilar TODAS las imágenes de las clases seleccionadas
        images_with_seg = []
        images_without_seg = []

        for cls in selected_classes:
            for f in class_images(root / cls):
                if find_existing_seg_path(f, ANNOTATION_ROOT):
                    images_with_seg.append(f)
                else:
                    images_without_seg.append(f)


        # Combinar: primero las que tienen .seg, luego las que no
        images = images_with_seg + images_without_seg
        
        logger.info(f"[Explorer] Encontradas {len(images)} imágenes totales "
                   f"({len(images_with_seg)} con .seg, {len(images_without_seg)} sin .seg) "
                   f"en {len(selected_classes)} clases")

        self._current_images = images
        self._current_image_idx = 0
        self._class_to_image_indices = {}
        for idx, img in enumerate(images):
            cls = img.parent.name
            self._class_to_image_indices.setdefault(cls, []).append(idx)

        # Poblar combo: etiqueta con split|placa|clase para no confundir stems.
        self.image_combo.blockSignals(True)
        self.image_combo.clear()
        for img in images_with_seg:
            try:
                split, plate, stem = identity_for(img)
                label = f"[{split}|{plate}|{img.parent.name}] {stem}"
            except AnnotationIdentityError:
                label = f"[{img.parent.name}] {img.stem} [SIN-ID]"
            self.image_combo.addItem(label, str(img))
        for img in images_without_seg:
            try:
                split, plate, stem = identity_for(img)
                label = f"[{split}|{plate}|{img.parent.name}] {stem} [sin .seg]"
            except AnnotationIdentityError:
                label = f"[{img.parent.name}] {img.stem} [sin .seg]"
            self.image_combo.addItem(label, str(img))
        self.image_combo.blockSignals(False)
        self._refresh_quick_class_controls()

        self._current_seg_path = None
        self._seg_dirty = False

        if images:
            self.image_combo.blockSignals(True)
            self.image_combo.setCurrentIndex(0)
            self.image_combo.blockSignals(False)
            self._current_image_idx = 0
            self._load_current_image()
        else:
            n_cls = len(selected_classes)
            self.image_label.setText(
                f"Sin imágenes en {n_cls} clase{'s' if n_cls != 1 else ''} seleccionada{'s' if n_cls != 1 else ''}"
                if n_cls > 0 else "Selecciona al menos una clase"
            )
            self.grain_info_label.setText("")
            self.nav_label.setText("0 / 0")
            self.quick_image_spin.blockSignals(True)
            self.quick_image_spin.setRange(1, 1)
            self.quick_image_spin.setValue(1)
            self.quick_image_spin.blockSignals(False)
            self.quick_image_total_label.setText("/0")

    def _on_image_changed(self, new_idx: int):
        """Único punto de cambio de imagen (combo / prev / next / quick).

        El índice viejo (`_current_image_idx`) sigue apuntando a los granos en
        memoria hasta confirmar dirty. Solo entonces se actualiza y se carga.
        """
        if new_idx < 0 or new_idx >= len(self._current_images):
            return
        if new_idx == self._current_image_idx and self._current_image_bgr is not None:
            return

        if self._seg_dirty:
            old_name = Path(self._current_images[self._current_image_idx]).name
            reply = QMessageBox.question(
                self,
                "Cambios sin guardar",
                f"Hay cambios sin guardar en:\n{old_name}\n\n"
                "¿Guardar antes de cambiar?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._on_save_seg()
                if self._seg_dirty:
                    # Guardado cancelado o fallido: no cambiar de imagen.
                    self.image_combo.blockSignals(True)
                    self.image_combo.setCurrentIndex(self._current_image_idx)
                    self.image_combo.blockSignals(False)
                    return
            elif reply == QMessageBox.Cancel:
                self.image_combo.blockSignals(True)
                self.image_combo.setCurrentIndex(self._current_image_idx)
                self.image_combo.blockSignals(False)
                return
            else:
                self._seg_dirty = False
                self.btn_save_seg.setEnabled(False)
                self.edit_status_label.setText("")

        self._current_image_idx = new_idx
        self._load_current_image()

    def _on_prev_image(self):
        if self._current_image_idx > 0:
            self.image_combo.setCurrentIndex(self._current_image_idx - 1)

    def _on_next_image(self):
        if self._current_image_idx < len(self._current_images) - 1:
            self.image_combo.setCurrentIndex(self._current_image_idx + 1)

    def _refresh_quick_class_controls(self):
        """Refresca combos de navegación rápida clase/índice."""
        self.quick_class_combo.blockSignals(True)
        current_cls = None
        if self._current_images and 0 <= self._current_image_idx < len(self._current_images):
            current_cls = self._current_images[self._current_image_idx].parent.name

        self.quick_class_combo.clear()
        for cls in sorted(self._class_to_image_indices.keys()):
            self.quick_class_combo.addItem(cls)

        if current_cls and current_cls in self._class_to_image_indices:
            idx = self.quick_class_combo.findText(current_cls)
            if idx >= 0:
                self.quick_class_combo.setCurrentIndex(idx)
        self.quick_class_combo.blockSignals(False)
        self._update_quick_image_controls()

    def _update_quick_image_controls(self):
        """Ajusta spinner de índice según clase activa en quick nav."""
        cls = self.quick_class_combo.currentText()
        indices = self._class_to_image_indices.get(cls, [])
        n = len(indices)
        self.quick_image_spin.blockSignals(True)
        if n <= 0:
            self.quick_image_spin.setRange(1, 1)
            self.quick_image_spin.setValue(1)
            self.quick_image_total_label.setText("/0")
            self.quick_image_spin.blockSignals(False)
            return

        self.quick_image_spin.setRange(1, n)
        if self._current_images and 0 <= self._current_image_idx < len(self._current_images):
            if self._current_image_idx in indices:
                pos = indices.index(self._current_image_idx) + 1
            else:
                pos = 1
        else:
            pos = 1
        self.quick_image_spin.setValue(pos)
        self.quick_image_total_label.setText(f"/{n}")
        self.quick_image_spin.blockSignals(False)

    def _on_quick_class_changed(self, class_name: str):
        """Salta a la primera imagen de la clase seleccionada."""
        if not class_name:
            return
        indices = self._class_to_image_indices.get(class_name, [])
        if not indices:
            self._update_quick_image_controls()
            return
        self.image_combo.setCurrentIndex(indices[0])

    def _on_quick_image_number_changed(self, value: int):
        """Salta a la imagen N de la clase seleccionada."""
        cls = self.quick_class_combo.currentText()
        indices = self._class_to_image_indices.get(cls, [])
        if not indices:
            return
        pos = int(np.clip(value, 1, len(indices))) - 1
        target_idx = indices[pos]
        if target_idx == self._current_image_idx:
            return
        self.image_combo.setCurrentIndex(target_idx)

    def _load_current_image(self):
        """Carga `_current_images[_current_image_idx]` y su .seg (sin diálogo dirty)."""
        if not self._current_images or self._current_image_idx >= len(self._current_images):
            return

        img_path = Path(self._current_images[self._current_image_idx])

        self._clear_history()
        self._reset_manual_roi_session()
        self._clear_preview()

        try:
            self._current_seg_path = seg_path_for(img_path)
        except AnnotationIdentityError as e:
            self._current_seg_path = None
            logger.error("[Explorer] %s", e)
            self.image_label.setText(
                f"Sin identidad absoluta (split+SAMPLE): {img_path.name}"
            )
            return

        existing_seg = find_existing_seg_path(img_path, ANNOTATION_ROOT)
        logger.info(
            "[Explorer] idx=%d path=%s seg=%s",
            self._current_image_idx, img_path, existing_seg or self._current_seg_path,
        )

        image = cv2.imread(str(img_path))
        if image is None:
            self.image_label.setText(f"Error leyendo {img_path.name}")
            return

        self._current_image_bgr = image.copy()
        h_img, w_img = image.shape[:2]
        self.image_label.set_original_size(w_img, h_img)
        self._update_effective_params_label()

        self._current_grains = []
        if existing_seg is not None:
            try:
                from src.grain_detection.seg_format import SegFileReader
                if not seg_matches_image(existing_seg, img_path, (w_img, h_img)):
                    logger.warning(
                        "[Explorer] .seg %s no corresponde a %s — no se dibuja",
                        existing_seg.name, img_path,
                    )
                    if self.parent_window and hasattr(self.parent_window, "log_widget"):
                        self.parent_window.log_widget.append_log(
                            f".seg omitido (identidad distinta): {existing_seg.name}",
                            "WARNING",
                        )
                else:
                    result = SegFileReader.read(str(existing_seg))
                    self._current_grains = result.get("grains", [])
                    self._current_seg_path = seg_path_for(img_path)
            except Exception as e:
                self._current_grains = []
                logger.warning(f"Error leyendo {existing_seg.name}: {e}")

        annotated = image.copy()
        for i, grain in enumerate(self._current_grains):
            color = GRAIN_COLORS[i % len(GRAIN_COLORS)]
            contour = grain.get("contour")
            if contour is not None and len(contour) >= 3:
                pts = contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(annotated, [pts], -1, color, 2)

            bx, by, bw, bh = grain["bbox"]
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 1)
            cv2.putText(annotated, f"#{i}", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        pixmap = numpy_to_qpixmap(annotated, max_width=700, max_height=500)
        self.image_label.setPixmap(pixmap)

        n = len(self._current_grains)
        active_info = f" [Grano #{self._active_grain_idx} en edición]" if self._active_grain_idx >= 0 else ""
        self.grain_info_label.setText(
            f"{n} grano{'s' if n != 1 else ''}{active_info} — {self._interaction_hint()}"
        )

        self.nav_label.setText(f"{self._current_image_idx + 1} / {len(self._current_images)}")
        self._sync_quick_controls_to_current_image()

        self._adopt_class_envelope()
        clase = self._current_class_name()
        if clase and clase not in self._class_envelopes and self._current_grains:
            self._learn_from_grains(self._current_grains, reason=".seg existente")

        if self._current_grains:
            self._selected_grain_idx = 0
            self._show_selected_grain()
        else:
            self._selected_grain_idx = -1
            self.crop_label.setText("Sin granos")
            self.grain_meta_label.setText("")
            self.grain_nav_label.setText("—")

        self._seg_dirty = False
        self.btn_save_seg.setEnabled(False)
        self.edit_status_label.setText("")

    def _sync_quick_controls_to_current_image(self):
        """Sincroniza quick class/spin al índice de imagen actual."""
        if not self._current_images or self._current_image_idx >= len(self._current_images):
            return
        cls = self._current_images[self._current_image_idx].parent.name
        idx_cls = self.quick_class_combo.findText(cls)
        if idx_cls >= 0 and self.quick_class_combo.currentIndex() != idx_cls:
            self.quick_class_combo.blockSignals(True)
            self.quick_class_combo.setCurrentIndex(idx_cls)
            self.quick_class_combo.blockSignals(False)
        self._update_quick_image_controls()

    def _on_prev_grain(self):
        if self._current_grains and self._selected_grain_idx > 0:
            self._selected_grain_idx -= 1
            self._show_selected_grain()

    def _on_next_grain(self):
        if self._current_grains and self._selected_grain_idx < len(self._current_grains) - 1:
            self._selected_grain_idx += 1
            self._show_selected_grain()

    def _show_selected_grain(self):
        """Muestra el crop del grano seleccionado."""
        if (self._current_image_bgr is None or
                self._selected_grain_idx < 0 or
                self._selected_grain_idx >= len(self._current_grains)):
            return

        grain = self._current_grains[self._selected_grain_idx]
        image = self._current_image_bgr

        # Extraer crop
        h_img, w_img = image.shape[:2]
        bx, by, bw, bh = grain["bbox"]
        padding = self.padding_spin.value() / 100.0

        pad_x = int(bw * padding)
        pad_y = int(bh * padding)
        side = max(bw + 2 * pad_x, bh + 2 * pad_y)

        cx = bx + bw // 2
        cy = by + bh // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w_img, x1 + side)
        y2 = min(h_img, y1 + side)

        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crop = cv2.resize(crop, (256, 256))
        else:
            crop = np.zeros((256, 256, 3), dtype=np.uint8)

        pixmap = numpy_to_qpixmap(crop, max_width=280, max_height=280)
        self.crop_label.setPixmap(pixmap)

        # Metadata
        sal = grain.get("saliency", 0)
        contour = grain.get("contour")
        n_pts = len(contour) if contour is not None else 0
        area_approx = bw * bh

        self.grain_meta_label.setText(
            f"Grano #{self._selected_grain_idx}\n"
            f"BBox: ({bx}, {by}, {bw}, {bh})\n"
            f"Saliency: {sal:.3f}\n"
            f"Área bbox: {area_approx} px²\n"
            f"Contorno: {n_pts} puntos"
        )

        total = len(self._current_grains)
        self.grain_nav_label.setText(f"{self._selected_grain_idx + 1} / {total}")

    # =========================================================================
    # Edición de segmentaciones
    # =========================================================================

    def _mark_dirty(self):
        """Marca el .seg actual como modificado."""
        self._seg_dirty = True
        self.btn_save_seg.setEnabled(True)
        self.edit_status_label.setText("● Cambios sin guardar")
        self.edit_status_label.setStyleSheet("color: #e67e22; font-size: 11px; font-weight: bold;")

    def _mark_clean(self):
        """Marca el .seg actual como guardado."""
        self._seg_dirty = False
        self.btn_save_seg.setEnabled(False)
        self.edit_status_label.setText("✓ Guardado")
        self.edit_status_label.setStyleSheet("color: #27ae60; font-size: 11px;")

    def _is_ai_roi_assist_enabled(self) -> bool:
        return hasattr(self, "roi_ai_assist_cb") and self.roi_ai_assist_cb.isChecked()

    def _interaction_hint(self) -> str:
        """Único texto de ayuda del visor: antes había dos versiones divergentes."""
        if self._is_ai_roi_assist_enabled():
            return (
                "Izq+arrastre: trazo U2-Net (cierra al soltar) | Der: exclusión | "
                "C cerrar | A aplicar | Esc cancelar | Ctrl+Wheel navegar"
            )
        return (
            "Izq: puntos de inclusión | Der: puntos de exclusión | "
            "doble click: cerrar polígono | A aplicar | Esc cancelar | Ctrl+Wheel navegar"
        )

    def _clear_manual_roi_points(self):
        """Botón 'Limpiar ROI'."""
        self._reset_manual_roi_session()
        self._redraw_contours()
        self.grain_info_label.setText("ROI manual limpio.")

    def _cancel_manual_roi_session(self):
        """Esc / botón 'Cancelar'."""
        self._reset_manual_roi_session()
        self._redraw_contours()
        self.grain_info_label.setText("Edición ROI cancelada.")

    def _reset_manual_roi_session(self):
        """
        Estado ROI a cero. Único reset: limpiar y cancelar dejaban rastros
        distintos (semillas pendientes y traza vieja sobrevivían a 'Limpiar').
        """
        self._roi_include_polygons = []
        self._roi_exclude_polygons = []
        self._roi_current_include = []
        self._roi_current_exclude = []
        self._manual_roi_seed_kind = None
        self._manual_roi_seed_point = None
        self._roi_include_mask = None
        self._roi_exclude_mask = None
        self._roi_block_mask = None
        self._roi_block_points = []
        self._roi_last_include_seed = None
        self._roi_suggest_points = []
        self._seed_retry_depth = 0
        self._pending_ai_seeds = []
        self._last_include_drag_pt = None
        self._last_exclude_drag_pt = None
        self._roi_seed_trace = []
        self._active_seed_trace_idx = -1

    def _close_manual_polygon(self, kind: str):
        if kind == "include":
            if len(self._roi_current_include) >= 3:
                self._roi_include_polygons.append(self._roi_current_include.copy())
            self._roi_current_include = []
        elif kind == "exclude":
            if len(self._roi_current_exclude) >= 3:
                self._roi_exclude_polygons.append(self._roi_current_exclude.copy())
            self._roi_current_exclude = []
        self._redraw_contours()

    def _close_active_manual_polygon(self):
        if len(self._roi_current_include) >= 3:
            self._close_manual_polygon("include")
            self.grain_info_label.setText("Polígono de inclusión cerrado.")
            return
        if len(self._roi_current_exclude) >= 3:
            self._close_manual_polygon("exclude")
            self.grain_info_label.setText("Polígono de exclusión cerrado.")
            return
        self.grain_info_label.setText("No hay polígono activo con al menos 3 puntos.")

    def _maybe_snap_to_edge(self, x: int, y: int, radius: int = 12) -> Tuple[int, int]:
        """Ajusta el punto al borde local más cercano (opcional)."""
        if self._current_image_bgr is None or not self.roi_snap_edges_cb.isChecked():
            return (x, y)

        img = self._current_image_bgr
        h, w = img.shape[:2]
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius + 1)
        y2 = min(h, y + radius + 1)
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return (x, y)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            return (x, y)

        abs_xs = xs + x1
        abs_ys = ys + y1
        d2 = (abs_xs - x) ** 2 + (abs_ys - y) ** 2
        k = int(np.argmin(d2))
        return (int(abs_xs[k]), int(abs_ys[k]))

    def _ensure_roi_masks(self):
        """Inicializa máscaras acumuladas de ROI con tamaño de imagen actual."""
        if self._current_image_bgr is None:
            return
        h, w = self._current_image_bgr.shape[:2]
        if self._roi_include_mask is None or self._roi_include_mask.shape != (h, w):
            self._roi_include_mask = np.zeros((h, w), dtype=np.uint8)
        if self._roi_exclude_mask is None or self._roi_exclude_mask.shape != (h, w):
            self._roi_exclude_mask = np.zeros((h, w), dtype=np.uint8)
        if self._roi_block_mask is None or self._roi_block_mask.shape != (h, w):
            self._roi_block_mask = np.zeros((h, w), dtype=np.uint8)

    def _add_exclusion_anchor(self, x: int, y: int, radius: int = 4):
        """
        Agrega ancla roja de exclusión:
        - bloquea expansión futura de inclusión en ese punto;
        - elimina localmente ese punto del área verde ya acumulada.
        """
        self._ensure_roi_masks()
        if self._roi_block_mask is None:
            return
        cx, cy = int(x), int(y)
        # Frente de puntos: cada ancla agrega disco + posible conexión al punto previo.
        cv2.circle(self._roi_block_mask, (cx, cy), int(radius), 255, -1)
        if self._roi_block_points:
            px, py = self._roi_block_points[-1]
            dist = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)
            if dist <= 120.0:
                thickness = max(2, int(radius * 2))
                cv2.line(
                    self._roi_block_mask,
                    (int(px), int(py)),
                    (cx, cy),
                    255,
                    thickness=thickness,
                )
        self._roi_block_points.append((cx, cy))
        if self._roi_include_mask is not None:
            inhibit = self._get_anchor_inhibit_mask()
            self._roi_include_mask[inhibit > 0] = 0
            self._harmonize_include_mask(seed=self._roi_last_include_seed)

    def _get_anchor_inhibit_mask(self, dilation_px: int = 6) -> np.ndarray:
        """Construye máscara de inhibición a partir de anclas rojas."""
        self._ensure_roi_masks()
        if self._roi_block_mask is None:
            return None
        if dilation_px <= 0:
            return self._roi_block_mask.copy()
        k = 2 * int(dilation_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(self._roi_block_mask, kernel, iterations=1)

    def _prune_disconnected_include_regions(self, seed: Optional[Tuple[int, int]] = None):
        """
        Elimina regiones verdes inconexas, conservando solo el componente principal.
        Prioridad:
        1) componente que contiene la semilla indicada;
        2) componente que contiene la última semilla de inclusión;
        3) componente de mayor área.
        """
        if self._roi_include_mask is None:
            return
        mask = self._roi_include_mask
        if np.count_nonzero(mask) == 0:
            return

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num_labels <= 2:
            return

        h, w = mask.shape[:2]

        def _label_at(pt):
            if pt is None:
                return 0
            x, y = int(pt[0]), int(pt[1])
            if x < 0 or y < 0 or x >= w or y >= h:
                return 0
            return int(labels[y, x])

        keep_label = _label_at(seed)
        if keep_label <= 0:
            keep_label = _label_at(self._roi_last_include_seed)
        if keep_label <= 0:
            comps = [(lbl, int(stats[lbl, cv2.CC_STAT_AREA])) for lbl in range(1, num_labels)]
            comps = [c for c in comps if c[1] > 0]
            if not comps:
                return
            keep_label = max(comps, key=lambda c: c[1])[0]

        pruned = np.zeros_like(mask, dtype=np.uint8)
        pruned[labels == keep_label] = 255
        self._roi_include_mask = pruned

    def _harmonize_include_mask(self, seed: Optional[Tuple[int, int]] = None):
        """
        Uniformiza la máscara verde: huecos internos e islas.
        El borde lo fija U²-Net; no se dilata aquí.
        """
        if self._roi_include_mask is None:
            return
        if np.count_nonzero(self._roi_include_mask) == 0:
            return

        from src.grain_detection.seeded_contour import fill_holes

        before = self._roi_include_mask.copy()
        self._roi_include_mask = fill_holes(self._roi_include_mask)
        filled = int(np.count_nonzero(self._roi_include_mask))
        if filled >= self._roi_include_mask.size:
            logger.info(
                "[ROI] fill_holes = encuadre (%d px) — no es un objeto, dejo la máscara anterior",
                filled,
            )
            self._roi_include_mask = before
            return

        # Reaplicar barrera de anclas tras armonizar
        inhibit = self._get_anchor_inhibit_mask()
        if inhibit is not None:
            self._roi_include_mask[inhibit > 0] = 0

        # Modo unión persistente: no descartar componentes previos automáticamente.
        if not self._preserve_union_mode:
            self._prune_disconnected_include_regions(seed=seed)

    def _apply_ai_exclusion_contour(self, contour: np.ndarray, seed: Optional[Tuple[int, int]] = None):
        """
        Aplica exclusión asistida por U2NET de forma controlada:
        - toma la región propuesta por U2NET;
        - selecciona el componente conectado más relevante para la semilla;
        - restringe la sustracción al entorno de la máscara verde (evita recortes lejanos).
        """
        self._ensure_roi_masks()
        if contour is None or len(contour) < 3 or self._roi_include_mask is None:
            return

        h, w = self._roi_include_mask.shape[:2]
        cand = np.zeros((h, w), dtype=np.uint8)
        pts = contour.reshape(-1, 1, 2).astype(np.int32)
        cv2.drawContours(cand, [pts], -1, 255, -1)

        # Elegir componente conectado guiado por semilla (o más cercano).
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cand, 8)
        chosen = None
        if num_labels > 1:
            if seed is not None:
                sx = int(np.clip(seed[0], 0, w - 1))
                sy = int(np.clip(seed[1], 0, h - 1))
                lbl = int(labels[sy, sx])
                if lbl > 0 and stats[lbl, cv2.CC_STAT_AREA] >= 3:
                    chosen = (labels == lbl).astype(np.uint8) * 255
            if chosen is None:
                # fallback: componente más cercana a semilla o mayor área
                if seed is not None:
                    sx, sy = seed
                    best_lbl = -1
                    best_d2 = float("inf")
                    for lbl in range(1, num_labels):
                        area = int(stats[lbl, cv2.CC_STAT_AREA])
                        if area < 3:
                            continue
                        cx, cy = centroids[lbl]
                        d2 = (cx - sx) ** 2 + (cy - sy) ** 2
                        if d2 < best_d2:
                            best_d2 = d2
                            best_lbl = lbl
                    if best_lbl > 0:
                        chosen = (labels == best_lbl).astype(np.uint8) * 255
                if chosen is None:
                    areas = [(lbl, int(stats[lbl, cv2.CC_STAT_AREA])) for lbl in range(1, num_labels)]
                    areas = [x for x in areas if x[1] >= 3]
                    if areas:
                        best_lbl = max(areas, key=lambda x: x[1])[0]
                        chosen = (labels == best_lbl).astype(np.uint8) * 255
        else:
            chosen = cand

        if chosen is None:
            return

        # Exclusión LOCAL alrededor de la semilla para evitar "comerse" toda la máscara.
        include_local = cv2.bitwise_and(chosen, self._roi_include_mask)
        if np.count_nonzero(include_local) == 0:
            # Si no intersecta con verde, al menos bloquear su región para siguientes expansiones.
            self._roi_block_mask = cv2.bitwise_or(self._roi_block_mask, chosen)
            return

        x, y, bw, bh = cv2.boundingRect(chosen)
        local_radius = int(np.clip(0.22 * max(bw, bh), 10, 42))
        if seed is not None:
            sx = int(np.clip(seed[0], 0, w - 1))
            sy = int(np.clip(seed[1], 0, h - 1))
        else:
            sx = int(np.clip(x + bw // 2, 0, w - 1))
            sy = int(np.clip(y + bh // 2, 0, h - 1))

        local_circle = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(local_circle, (sx, sy), local_radius, 255, -1)
        remove_mask = cv2.bitwise_and(include_local, local_circle)

        if np.count_nonzero(remove_mask) == 0:
            # fallback local: punto más cercano dentro de include_local
            ys, xs = np.where(include_local > 0)
            if len(xs) == 0:
                return
            d2 = (xs - sx) ** 2 + (ys - sy) ** 2
            k = int(np.argmin(d2))
            px, py = int(xs[k]), int(ys[k])
            local_circle = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(local_circle, (px, py), max(8, local_radius // 2), 255, -1)
            remove_mask = cv2.bitwise_and(include_local, local_circle)

        if np.count_nonzero(remove_mask) == 0:
            return

        # Límite de seguridad: no retirar más de una fracción por click.
        include_area = int(np.count_nonzero(self._roi_include_mask))
        max_remove = max(40, int(0.18 * include_area))
        rem_area = int(np.count_nonzero(remove_mask))
        if rem_area > max_remove and seed is not None:
            ys, xs = np.where(remove_mask > 0)
            d2 = (xs - sx) ** 2 + (ys - sy) ** 2
            order = np.argsort(d2)
            keep_idx = order[:max_remove]
            limited = np.zeros((h, w), dtype=np.uint8)
            limited[ys[keep_idx], xs[keep_idx]] = 255
            remove_mask = limited

        # Quitar realmente de la máscara verde y bloquear re-expansión local.
        self._roi_include_mask[remove_mask > 0] = 0
        self._roi_block_mask = cv2.bitwise_or(self._roi_block_mask, remove_mask)
        self._harmonize_include_mask(seed=self._roi_last_include_seed)

    def _accumulate_roi_seed_contour(self, kind: str, contour: np.ndarray, seed: Optional[Tuple[int, int]] = None):
        """
        Acumula la región de una semilla en máscaras unificadas (unión progresiva).
        Esto evita que queden submáscaras separadas sin integración.
        """
        self._ensure_roi_masks()
        if contour is None or len(contour) < 3:
            return

        poly = contour.reshape(-1, 2).astype(np.int32)
        fill = np.zeros_like(self._roi_include_mask, dtype=np.uint8)
        cv2.drawContours(fill, [poly.reshape(-1, 1, 2)], -1, 255, -1)

        if kind == "include":
            inhibit = self._get_anchor_inhibit_mask()
            if inhibit is not None:
                fill = cv2.bitwise_and(fill, cv2.bitwise_not(inhibit))
            before = self._roi_include_mask.copy()
            one_at_a_time = (
                hasattr(self, "roi_multi_component_cb")
                and not self.roi_multi_component_cb.isChecked()
            )
            if one_at_a_time:
                self._roi_include_mask = fill
            else:
                self._roi_include_mask = cv2.bitwise_or(self._roi_include_mask, fill)
            self._roi_last_include_seed = seed
            self._harmonize_include_mask(seed=seed)
            filled = int(np.count_nonzero(self._roi_include_mask))
            if filled >= self._roi_include_mask.size:
                logger.info(
                    "[ROI] máscara = encuadre (%d px) — no es un objeto, descarto este trazo",
                    filled,
                )
                self._roi_include_mask = before
                return
            # En modo AI, evitar duplicar geometría en listas de polígonos
            # (la máscara acumulada es la fuente de verdad).
            if not self._is_ai_roi_assist_enabled():
                self._roi_include_polygons.append([(int(p[0]), int(p[1])) for p in poly])
        else:
            self._roi_exclude_mask = cv2.bitwise_or(self._roi_exclude_mask, fill)
            if not self._is_ai_roi_assist_enabled():
                self._roi_exclude_polygons.append([(int(p[0]), int(p[1])) for p in poly])

    def _request_ai_roi_seed(self, x_orig: int, y_orig: int, kind: str):
        self._request_ai_roi_stroke([(int(x_orig), int(y_orig))], kind)

    def _request_ai_roi_stroke(self, points, kind: str):
        """Un trazo → una inferencia U²-Net. Cierra el área de mayor puntaje."""
        if self._current_image_bgr is None:
            return
        pts = [(int(x), int(y)) for x, y in points]
        if not pts:
            return
        trace_entry = {
            "x": pts[0][0],
            "y": pts[0][1],
            "kind": kind,
            "status": "queued",
            "n": len(pts),
        }
        self._roi_seed_trace.append(trace_entry)
        trace_idx = len(self._roi_seed_trace) - 1

        if hasattr(self, "_manual_worker") and self._manual_worker is not None and self._manual_worker.isRunning():
            item = (pts, kind)
            if self._pending_ai_seeds:
                last_pts, last_kind = self._pending_ai_seeds[-1]
                if last_kind == kind:
                    self._pending_ai_seeds[-1] = item
                elif len(self._pending_ai_seeds) >= 2:
                    self._pending_ai_seeds = [self._pending_ai_seeds[-1], item]
                else:
                    self._pending_ai_seeds.append(item)
            else:
                self._pending_ai_seeds.append(item)
            self.grain_info_label.setText(
                f"U2-Net ocupado. Trazo en cola ({len(self._pending_ai_seeds)} pendiente/s)."
            )
            return
        if not self._check_weights_exist():
            self._roi_seed_trace[trace_idx]["status"] = "failed"
            return

        sx, sy = pts[0]
        self._manual_roi_seed_kind = kind
        self._manual_roi_seed_point = (sx, sy)
        self._active_seed_trace_idx = trace_idx
        self._roi_seed_trace[trace_idx]["status"] = "running"

        config = self._get_config()
        self.grain_info_label.setText(
            f"U2-Net cerrando trazo de {len(pts)} punto(s) "
            f"({'inclusión' if kind == 'include' else 'exclusión'})..."
        )
        self._manual_worker = ManualContourWorker(
            self._current_image_bgr, pts, config
        )
        self._manual_worker.finished.connect(self._on_manual_roi_seed_result)
        self._manual_worker.error.connect(self._on_manual_roi_seed_error)
        self._manual_worker.start()

    def _drain_pending_ai_seed(self):
        if not self._pending_ai_seeds:
            return
        pts, kind = self._pending_ai_seeds.pop(0)
        self._request_ai_roi_stroke(pts, kind)

    def _is_far_enough_from_last(self, last_pt: Optional[Tuple[int, int]], x: int, y: int) -> bool:
        if last_pt is None:
            return True
        dx = int(x) - int(last_pt[0])
        dy = int(y) - int(last_pt[1])
        return (dx * dx + dy * dy) >= (self._drag_step_px * self._drag_step_px)

    def _suggest_expansion_points(
        self, seed: Optional[Tuple[int, int]], max_points: int = 6
    ) -> List[Tuple[int, int]]:
        """
        Sugiere puntos de expansión alrededor del borde del área verde actual.
        Prioriza vecindad de la semilla fallida y cobertura del frente de crecimiento.
        """
        if self._roi_include_mask is None or np.count_nonzero(self._roi_include_mask) == 0:
            return []

        mask = self._roi_include_mask
        h, w = mask.shape[:2]
        ring = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
            iterations=1,
        )
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))
        ys, xs = np.where(ring > 0)
        if len(xs) == 0:
            return []

        if seed is not None:
            sx = int(np.clip(seed[0], 0, w - 1))
            sy = int(np.clip(seed[1], 0, h - 1))
        else:
            sx, sy = w // 2, h // 2

        d2 = (xs - sx) ** 2 + (ys - sy) ** 2
        order = np.argsort(d2)
        candidates = [(int(xs[i]), int(ys[i])) for i in order[: min(len(order), 2000)]]

        picked = []
        min_sep2 = 18 * 18
        for p in candidates:
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= min_sep2 for q in picked):
                picked.append(p)
            if len(picked) >= max_points:
                break
        return picked

    def _on_manual_roi_seed_result(self, grain_dicts):
        """Callback de semilla AI: agrega contorno propuesto a include o exclude."""
        kind = self._manual_roi_seed_kind
        seed = self._manual_roi_seed_point
        self._manual_roi_seed_kind = None
        self._manual_roi_seed_point = None
        trace_idx = self._active_seed_trace_idx
        self._active_seed_trace_idx = -1

        if kind not in ("include", "exclude"):
            if 0 <= trace_idx < len(self._roi_seed_trace):
                self._roi_seed_trace[trace_idx]["status"] = "failed"
            self.grain_info_label.setText("Semilla AI sin contexto de edición.")
            return

        contour = None
        if grain_dicts:
            contour = grain_dicts[0].get("contour")

        if contour is None or len(contour) < 3:
            if 0 <= trace_idx < len(self._roi_seed_trace):
                self._roi_seed_trace[trace_idx]["status"] = "failed"
            self._roi_current_include = []
            if kind == "include" and self._is_ai_roi_assist_enabled():
                self._roi_suggest_points = self._suggest_expansion_points(seed, max_points=6)
                self._redraw_contours()
            self.grain_info_label.setText(
                "U2-Net no cerró el trazo: la semilla no está sobre el objeto. "
                "Arrastra por la textura (arriba-izquierda) y suelta."
            )
            self._drain_pending_ai_seed()
            return

        if kind == "exclude" and self._is_ai_roi_assist_enabled():
            self._apply_ai_exclusion_contour(contour, seed=seed)
        else:
            self._accumulate_roi_seed_contour(kind, contour, seed=seed)
            if kind == "include":
                self._seed_retry_depth = 0
                self._roi_suggest_points = []
                self._roi_current_include = []
        if 0 <= trace_idx < len(self._roi_seed_trace):
            self._roi_seed_trace[trace_idx]["status"] = "ok"

        self._redraw_contours()
        incl_stat = (
            np.count_nonzero(self._roi_include_mask) if self._roi_include_mask is not None else len(self._roi_include_polygons)
        )
        excl_stat = (
            len(self._roi_block_points) if self._is_ai_roi_assist_enabled()
            else (np.count_nonzero(self._roi_exclude_mask) if self._roi_exclude_mask is not None else len(self._roi_exclude_polygons))
        )
        trace = ""
        if grain_dicts and grain_dicts[0]:
            trace = grain_dicts[0].get("trace") or ""
        extra = f" | {trace}" if trace else ""
        self.grain_info_label.setText(
            f"Región AI {'incluida' if kind == 'include' else 'excluida'} desde semilla {seed}. "
            f"Incl: {incl_stat} | Excl/anclas: {excl_stat}{extra}"
        )
        if trace:
            logger.info("[Click] %s", trace)
        self._drain_pending_ai_seed()

    def _on_manual_roi_seed_error(self, error_msg: str):
        self._manual_roi_seed_kind = None
        self._manual_roi_seed_point = None
        trace_idx = self._active_seed_trace_idx
        self._active_seed_trace_idx = -1
        if 0 <= trace_idx < len(self._roi_seed_trace):
            self._roi_seed_trace[trace_idx]["status"] = "failed"
        self.grain_info_label.setText(f"Error U2-Net semilla ROI: {error_msg[:80]}")
        logger.error(f"[ManualROI] Error semilla AI: {error_msg}")
        self._drain_pending_ai_seed()

    def _build_manual_roi_mask(self) -> Optional[np.ndarray]:
        from src.grain_detection.manual_roi import rasterize_polygons

        if self._current_image_bgr is None:
            return None
        h, w = self._current_image_bgr.shape[:2]

        # Polígonos manuales (solo cuando NO se usa asistencia AI por semilla)
        if self._is_ai_roi_assist_enabled():
            include_mask_poly = np.zeros((h, w), dtype=np.uint8)
            exclude_mask_poly = np.zeros((h, w), dtype=np.uint8)
        else:
            include_polys = self._roi_include_polygons.copy()
            exclude_polys = self._roi_exclude_polygons.copy()
            # Cerrar implícitamente polígonos en curso al aplicar si tienen >=3 puntos
            if len(self._roi_current_include) >= 3:
                include_polys.append(self._roi_current_include.copy())
            if len(self._roi_current_exclude) >= 3:
                exclude_polys.append(self._roi_current_exclude.copy())
            include_mask_poly = rasterize_polygons((h, w), include_polys)
            exclude_mask_poly = rasterize_polygons((h, w), exclude_polys)

        include_mask = include_mask_poly
        exclude_mask = exclude_mask_poly

        # Integrar también acumulación unificada de semillas AI
        if self._roi_include_mask is not None and self._roi_include_mask.shape == (h, w):
            include_mask = cv2.bitwise_or(include_mask, self._roi_include_mask)
        if self._roi_exclude_mask is not None and self._roi_exclude_mask.shape == (h, w):
            exclude_mask = cv2.bitwise_or(exclude_mask, self._roi_exclude_mask)
        if self._roi_block_mask is not None and self._roi_block_mask.shape == (h, w):
            exclude_mask = cv2.bitwise_or(exclude_mask, self._roi_block_mask)

        if np.count_nonzero(include_mask) == 0:
            return None
        keep_mask = cv2.bitwise_and(include_mask, cv2.bitwise_not(exclude_mask))
        return keep_mask

    def _draw_manual_roi_overlay(self, base_image: np.ndarray) -> np.ndarray:
        """Dibuja overlay semitransparente de ROI manual sobre la imagen."""
        out = base_image.copy()
        overlay = out.copy()
        ai_mode = self._is_ai_roi_assist_enabled()

        def _draw_poly(polys, color_fill, color_line):
            for poly in polys:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.fillPoly(overlay, [pts], color_fill)
                    cv2.polylines(out, [pts], isClosed=True, color=color_line, thickness=2)

        # En modo manual puro, mostrar polígonos explícitos.
        # En modo AI, la fuente de verdad son las máscaras acumuladas.
        if not ai_mode:
            _draw_poly(self._roi_include_polygons, (0, 140, 0), (0, 255, 0))
            for poly in self._roi_exclude_polygons:
                if len(poly) >= 3:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(out, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        # Máscara verde acumulada unificada
        if self._roi_include_mask is not None:
            overlay[self._roi_include_mask > 0] = (0, 140, 0)
        # Zona bloqueada/excluida en rojo suave para depuración visual
        if self._roi_block_mask is not None:
            overlay[self._roi_block_mask > 0] = (0, 0, 120)

        cv2.addWeighted(overlay, 0.25, out, 0.75, 0.0, out)

        # Dibujar anclas rojas explícitas
        for px, py in self._roi_block_points:
            cv2.circle(out, (int(px), int(py)), 4, (0, 0, 255), 2)

        # Sugerencias de expansión (puntos azules)
        for sx, sy in self._roi_suggest_points:
            cv2.circle(out, (int(sx), int(sy)), 4, (255, 120, 0), 2)

        # Historial permanente de semillas propuestas
        for s in self._roi_seed_trace:
            sx = int(s.get("x", 0))
            sy = int(s.get("y", 0))
            status = s.get("status", "queued")
            kind = s.get("kind", "include")
            if status == "ok":
                color = (0, 220, 0) if kind == "include" else (0, 0, 220)
            elif status == "failed":
                color = (0, 165, 255)  # naranja
            elif status == "running":
                color = (255, 255, 0)  # amarillo
            else:
                color = (220, 220, 220)  # queued
            cv2.circle(out, (sx, sy), 2, color, -1)

        def _draw_open(points, color):
            if not points:
                return
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=False, color=color, thickness=2)
            for p in points:
                cv2.circle(out, (int(p[0]), int(p[1])), 3, color, -1)

        _draw_open(self._roi_current_include, (0, 255, 0))
        _draw_open(self._roi_current_exclude, (0, 0, 255))
        return out

    def _apply_manual_roi_to_grain(self):
        """Aplica ROI manual include/exclude al grano activo o crea instancias nuevas."""
        if self._current_image_bgr is None:
            return

        mask = self._build_manual_roi_mask()
        if mask is None:
            self.grain_info_label.setText("ROI manual vacío: falta polígono de inclusión.")
            return

        from src.grain_detection.manual_roi import extract_contours_from_mask, contour_bbox
        from src.grain_detection.seeded_contour import covers_full_frame

        contours = extract_contours_from_mask(mask, min_area=3.0)
        if not contours:
            self.grain_info_label.setText("ROI resultante vacío tras aplicar exclusión.")
            return

        multi = self.roi_multi_component_cb.isChecked()
        contour_list = contours if multi else contours[:1]
        img_h, img_w = self._current_image_bgr.shape[:2]
        target_idx = self._find_best_overlapping_grain(mask)
        base_sal = (
            float(self._current_grains[target_idx].get("saliency", 0.5))
            if target_idx >= 0
            else 0.5
        )

        created = []
        for c in contour_list:
            if covers_full_frame(c, img_h, img_w):
                logger.info("[ROI] contorno = encuadre — no es un objeto")
                continue
            bx, by, bw, bh = contour_bbox(c)
            created.append(
                {
                    "index": 0,
                    "class_index": 0,
                    "bbox": (bx, by, bw, bh),
                    "saliency": base_sal,
                    "contour": c.astype(np.int32),
                }
            )

        if not created:
            self.grain_info_label.setText("ROI descartado: el encuadre no es un objeto.")
            return

        self._push_history()

        replace_selected = (
            not multi
            and self._selected_grain_idx >= 0
            and self._selected_grain_idx < len(self._current_grains)
            and target_idx == self._selected_grain_idx
        )
        if replace_selected:
            self._current_grains[self._selected_grain_idx] = created[0]
            self._active_grain_idx = self._selected_grain_idx
        else:
            self._current_grains.extend(created)
            self._selected_grain_idx = len(self._current_grains) - len(created)
            self._active_grain_idx = self._selected_grain_idx

        for i, g in enumerate(self._current_grains):
            g["index"] = i

        self._mark_dirty()
        self._learn_from_grains(created, reason="ROI aplicado")
        self._reset_manual_roi_session()
        self._redraw_contours()
        self._show_selected_grain()
        self.grain_info_label.setText(
            f"ROI aplicado. {len(created)} instancia{'s' if len(created) != 1 else ''} "
            f"actualizada{'s' if len(created) != 1 else ''}."
        )

    def _find_best_overlapping_grain(self, roi_mask: np.ndarray, min_overlap: float = 0.25) -> int:
        """
        Retorna el índice del grano con mayor solapamiento con roi_mask.
        Si no hay solapamiento suficiente, retorna -1 para forzar creación de nuevo objeto.
        """
        if roi_mask is None or not self._current_grains:
            return -1

        best_idx = -1
        best_score = 0.0
        h, w = roi_mask.shape[:2]

        for idx, grain in enumerate(self._current_grains):
            contour = grain.get("contour")
            if contour is None or len(contour) < 3:
                continue

            gmask = np.zeros((h, w), dtype=np.uint8)
            pts = contour.reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(gmask, [pts], -1, 255, -1)

            inter = cv2.bitwise_and(roi_mask, gmask)
            inter_area = float(np.count_nonzero(inter))
            grain_area = float(np.count_nonzero(gmask))
            if grain_area <= 0:
                continue

            # fracción del grano cubierta por ROI nueva
            overlap = inter_area / grain_area
            if overlap > best_score:
                best_score = overlap
                best_idx = idx

        return best_idx if best_score >= min_overlap else -1

    def _delete_grain_near_point(self, x_orig: int, y_orig: int) -> bool:
        """
        Elimina el grano cercano al punto (si corresponde). Retorna True si eliminó.
        """
        if self._current_image_bgr is None or not self._current_grains:
            return False

        best_idx = -1
        best_dist = float("inf")
        for i, grain in enumerate(self._current_grains):
            bx, by, bw, bh = grain["bbox"]
            cx = bx + bw // 2
            cy = by + bh // 2
            dist = ((x_orig - cx) ** 2 + (y_orig - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx < 0:
            return False

        grain = self._current_grains[best_idx]
        bx, by, bw, bh = grain["bbox"]
        max_dist = max(bw, bh) * 1.5
        if best_dist > max_dist:
            return False

        self._push_history()
        removed = self._current_grains.pop(best_idx)
        logger.info(f"[ManualContour] Eliminado grano #{best_idx} por click derecho, bbox={removed['bbox']}")

        for i, g in enumerate(self._current_grains):
            g["index"] = i

        self._mark_dirty()
        self._redraw_contours()

        if len(self._current_grains) == 0:
            self._selected_grain_idx = -1
            self._active_grain_idx = -1
            self.crop_label.setText("Sin granos")
            self.grain_meta_label.setText("")
            self.grain_nav_label.setText("—")
        else:
            self._selected_grain_idx = min(best_idx, len(self._current_grains) - 1)
            if self._active_grain_idx == best_idx:
                self._active_grain_idx = -1
            self._show_selected_grain()

        self.grain_info_label.setText(
            f"Grano #{best_idx} eliminado. {len(self._current_grains)} restantes."
        )
        return True

    def _on_delete_grain(self):
        """Elimina el grano seleccionado de la anotación actual."""
        if not self._current_grains or self._selected_grain_idx < 0:
            QMessageBox.information(self, "Info", "No hay grano seleccionado para eliminar.")
            return

        idx = self._selected_grain_idx
        n = len(self._current_grains)

        reply = QMessageBox.question(
            self, "Confirmar eliminación",
            f"¿Eliminar grano #{idx} de {n}?\n"
            f"BBox: {self._current_grains[idx]['bbox']}\n"
            f"Saliency: {self._current_grains[idx].get('saliency', 0):.3f}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Eliminar el grano
        removed = self._current_grains.pop(idx)
        logger.info(f"[Edit] Eliminado grano #{idx}, bbox={removed['bbox']}")

        # Re-indexar granos restantes
        for i, g in enumerate(self._current_grains):
            g["index"] = i

        self._mark_dirty()

        # Ajustar selección
        if len(self._current_grains) == 0:
            self._selected_grain_idx = -1
            self.crop_label.setText("Sin granos")
            self.grain_meta_label.setText("")
            self.grain_nav_label.setText("—")
        else:
            self._selected_grain_idx = min(idx, len(self._current_grains) - 1)

        # Re-dibujar imagen con contornos actualizados
        self._redraw_contours()

        if self._selected_grain_idx >= 0:
            self._show_selected_grain()

    def _on_save_seg(self):
        """Guarda el .seg actual con los granos editados."""
        if self._current_seg_path is None:
            QMessageBox.warning(self, "Error", "No hay .seg cargado para guardar.")
            return

        # Safety: check if we'd lose data by saving fewer grains
        existing_count = 0
        if self._current_seg_path and Path(self._current_seg_path).exists():
            try:
                from src.grain_detection.seg_format import SegFileReader
                existing_count = SegFileReader.count_grains(str(self._current_seg_path))
            except Exception:
                pass

        new_count = len(self._current_grains) if self._current_grains else 0

        if new_count == 0 and existing_count > 0:
            reply = QMessageBox.warning(
                self, "Advertencia: Pérdida de datos",
                f"El .seg actual tiene {existing_count} grano(s).\n"
                f"Vas a guardar 0 granos — esto BORRARÁ las anotaciones existentes.\n\n"
                f"Se creará un backup automático (.seg.bak).\n"
                f"¿Continuar?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        elif new_count < existing_count:
            reply = QMessageBox.question(
                self, "Confirmar reducción",
                f"El .seg actual tiene {existing_count} grano(s), "
                f"pero guardarás {new_count}.\n"
                f"Se creará un backup automático (.seg.bak).\n"
                f"¿Continuar?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                return

        try:
            from src.grain_detection.seg_format import SegFileWriter

            # Siempre la imagen de `_current_image_idx` (aún la vieja si venimos
            # del diálogo dirty en `_on_image_changed`).
            img_path = Path(self._current_images[self._current_image_idx])
            h, w = self._current_image_bgr.shape[:2] if self._current_image_bgr is not None else (0, 0)
            self._current_seg_path = seg_path_for(img_path)
            params = {"editado_manualmente": "true"}
            params.update(identity_params_for(img_path))

            SegFileWriter.write(
                str(self._current_seg_path),
                resolved_image_path(img_path),
                (h, w),
                self._current_grains,
                params,
            )

            self._mark_clean()
            logger.info(
                "[Edit] Guardado %s <- %s (%d granos)",
                self._current_seg_path.name, img_path, len(self._current_grains),
            )

            if self.parent_window and hasattr(self.parent_window, 'log_widget'):
                self.parent_window.log_widget.append_log(
                    f"Seg guardado: {self._current_seg_path.name} ({len(self._current_grains)} granos)",
                    "SUCCESS"
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando .seg:\n{e}")
            logger.error(f"[Edit] Error guardando {self._current_seg_path}: {e}")

    # =========================================================================
    # Historial de cambios (UNDO)
    # =========================================================================
    
    def _push_history(self):
        """Guarda el estado actual de granos en el historial para UNDO."""
        import copy
        # Deep copy para evitar referencias compartidas
        state_snapshot = copy.deepcopy(self._current_grains)
        self._grains_history.append(state_snapshot)
        
        # Limitar historial a últimos 20 estados para no consumir memoria
        if len(self._grains_history) > 20:
            self._grains_history.pop(0)
        
        logger.debug(f"[History] Estado guardado. Historial: {len(self._grains_history)} estados")
    
    def _pop_history(self):
        """Restaura el último estado de granos del historial."""
        if not self._grains_history:
            return self._current_grains
        
        import copy
        restored_state = self._grains_history.pop()
        logger.debug(f"[History] Estado restaurado. Historial: {len(self._grains_history)} estados")
        return copy.deepcopy(restored_state)
    
    def _clear_history(self):
        """Limpia el historial de cambios (usado al cambiar de imagen)."""
        self._grains_history = []
        self._active_grain_idx = -1
        logger.debug("[History] Historial limpiado")

    # =========================================================================
    # Manual contour interaction (click on image)
    # =========================================================================

    def _on_image_left_click(self, x_orig, y_orig):
        """Click izquierdo: semilla del buscador ROI, o punto de polígono manual."""
        if self._current_image_bgr is None:
            return

        if self._is_ai_roi_assist_enabled():
            self._seed_retry_depth = 0
            self._roi_suggest_points = []
            self._last_include_drag_pt = (int(x_orig), int(y_orig))
            self._roi_current_include = [(int(x_orig), int(y_orig))]
            self._redraw_contours()
            self.grain_info_label.setText(
                "Trazando inclusión. Arrastra y suelta: U²-Net cierra el área de mayor puntaje."
            )
        else:
            sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
            self._last_include_drag_pt = (int(sx), int(sy))
            self._roi_current_include.append((sx, sy))
            self._redraw_contours()
            self.grain_info_label.setText(
                f"Inclusión manual: {len(self._roi_current_include)} punto(s) en polígono activo."
            )

    def _on_image_left_double_click(self, x_orig, y_orig):
        """Doble click izquierdo: cierra polígono de inclusión."""
        self._close_manual_polygon("include")
        self.grain_info_label.setText("Polígono de inclusión cerrado.")

    def _on_image_right_double_click(self, x_orig, y_orig):
        """Doble click derecho: cierra polígono de exclusión."""
        self._close_manual_polygon("exclude")
        self.grain_info_label.setText("Polígono de exclusión cerrado.")

    def _on_image_right_click(self, x_orig, y_orig):
        """
        Click derecho, en orden de prioridad:

        1. Sin ROI en curso y con un grano cerca → eliminar ese grano.
        2. Sin ROI en curso y sin grano cerca → deshacer la última operación.
        3. Con ROI en curso → ancla de exclusión (o punto de polígono manual).
        """
        if self._current_image_bgr is None:
            return

        if not self._has_include_work():
            if self._delete_grain_near_point(x_orig, y_orig):
                self.grain_info_label.setText(
                    f"{self.grain_info_label.text()} Click der en vacío para UNDO."
                )
                return
            if self._undo_last_operation():
                return

        sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
        self._last_exclude_drag_pt = (int(sx), int(sy))

        if self._is_ai_roi_assist_enabled():
            self._add_exclusion_anchor(sx, sy, radius=4)
            self._redraw_contours()
            self.grain_info_label.setText(
                f"Barrera roja extendida en ({sx}, {sy}). Bloquea expansión del ROI verde."
            )
        else:
            self._roi_current_exclude.append((sx, sy))
            self._redraw_contours()
            self.grain_info_label.setText(
                f"Exclusión manual: {len(self._roi_current_exclude)} punto(s) en polígono activo."
            )

    def _has_include_work(self) -> bool:
        """Hay una inclusión en construcción (máscara acumulada o polígono)."""
        if self._roi_include_mask is not None and np.count_nonzero(self._roi_include_mask) > 0:
            return True
        return bool(self._roi_include_polygons or self._roi_current_include)

    def _undo_last_operation(self) -> bool:
        """Restaura el estado anterior de granos. False si no hay historial."""
        if not self._grains_history:
            self.grain_info_label.setText(
                "No hay contorno cerca del click ni operaciones para deshacer."
            )
            return False

        self._current_grains = self._pop_history()
        for i, g in enumerate(self._current_grains):
            g["index"] = i

        self._mark_dirty()
        self._redraw_contours()

        if self._selected_grain_idx >= len(self._current_grains):
            self._selected_grain_idx = len(self._current_grains) - 1
        if self._active_grain_idx >= len(self._current_grains):
            self._active_grain_idx = -1

        if self._selected_grain_idx >= 0:
            self._show_selected_grain()
        else:
            self.crop_label.setText("Sin granos")
            self.grain_meta_label.setText("")
            self.grain_nav_label.setText("—")

        n_history = len(self._grains_history)
        self.grain_info_label.setText(
            f"Última operación deshecha. {n_history} "
            f"operacion{'es' if n_history != 1 else ''} en historial."
        )
        logger.info(f"[ManualContour] UNDO ejecutado. Historial: {n_history} estados")
        return True

    def _on_image_left_drag(self, x_orig, y_orig):
        """Arrastre izquierdo: sembrado continuo para inclusión."""
        if self._current_image_bgr is None:
            return
        if not self._is_far_enough_from_last(self._last_include_drag_pt, x_orig, y_orig):
            return

        if self._is_ai_roi_assist_enabled():
            self._last_include_drag_pt = (int(x_orig), int(y_orig))
            self._roi_current_include.append((int(x_orig), int(y_orig)))
            self._redraw_contours()
            self.grain_info_label.setText(
                f"Trazando inclusión: {len(self._roi_current_include)} puntos. Suelta para cerrar."
            )
        else:
            sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
            self._last_include_drag_pt = (int(sx), int(sy))
            self._roi_current_include.append((sx, sy))
            self._redraw_contours()

    def _on_image_right_drag(self, x_orig, y_orig):
        """Arrastre derecho: barrera roja continua o exclusión manual continua."""
        if self._current_image_bgr is None:
            return
        if not self._is_far_enough_from_last(self._last_exclude_drag_pt, x_orig, y_orig):
            return

        sx, sy = self._maybe_snap_to_edge(x_orig, y_orig)
        self._last_exclude_drag_pt = (int(sx), int(sy))
        if self._is_ai_roi_assist_enabled():
            self._add_exclusion_anchor(sx, sy, radius=4)
        else:
            self._roi_current_exclude.append((sx, sy))
        self._redraw_contours()

    def _on_image_left_release(self, x_orig, y_orig):
        if (
            self._is_ai_roi_assist_enabled()
            and self._current_image_bgr is not None
            and self._roi_current_include
        ):
            if self._is_far_enough_from_last(self._last_include_drag_pt, x_orig, y_orig):
                self._roi_current_include.append((int(x_orig), int(y_orig)))
            self._request_ai_roi_stroke(list(self._roi_current_include), "include")
        self._last_include_drag_pt = None

    def _on_image_right_release(self, x_orig, y_orig):
        self._last_exclude_drag_pt = None

    def _redraw_contours(self):
        """Re-dibuja los contornos sobre la imagen actual sin recargar."""
        if self._current_image_bgr is None:
            return

        annotated = self._current_image_bgr.copy()
        for i, grain in enumerate(self._current_grains):
            color = GRAIN_COLORS[i % len(GRAIN_COLORS)]
            contour = grain.get("contour")
            if contour is not None and len(contour) >= 3:
                pts = contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(annotated, [pts], -1, color, 2)

            bx, by, bw, bh = grain["bbox"]
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 1)
            cv2.putText(annotated, f"#{i}", (bx, by - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        annotated = self._draw_preview_overlay(annotated)
        annotated = self._draw_manual_roi_overlay(annotated)

        pixmap = numpy_to_qpixmap(annotated, max_width=700, max_height=500)
        self.image_label.setPixmap(pixmap)

        n = len(self._current_grains)
        edit_tag = " ● Editado" if self._seg_dirty else ""
        preview_tag = (
            f" | prueba: {len(self._preview_grains)} cuerpos (blanco)"
            if self._preview_grains else ""
        )
        self.grain_info_label.setText(
            f"{n} grano{'s' if n != 1 else ''}{edit_tag}{preview_tag} — {self._interaction_hint()}"
        )

    def _draw_preview_overlay(self, base_image: np.ndarray) -> np.ndarray:
        """Propuestas de 'Probar aquí' en blanco: no están guardadas en el .seg."""
        if not self._preview_grains:
            return base_image
        out = base_image
        for grain in self._preview_grains:
            contour = grain.get("contour")
            if contour is None or len(contour) < 3:
                continue
            pts = contour.reshape(-1, 1, 2).astype(np.int32)
            cv2.drawContours(out, [pts], -1, (255, 255, 255), 2)
        return out

    # =========================================================================
    # Estadísticas
    # =========================================================================

    def _update_stats(self, stats: Dict):
        """Actualiza la sección de estadísticas con el resultado de la anotación."""
        self._set_stats(
            f"Imágenes: {stats.get('total_images', 0)}  |  "
            f"Granos: {stats.get('total_grains', 0)}  |  "
            f"Promedio/img: {stats.get('grains_per_image', 0):.1f}",
            stats.get("grains_per_class"),
        )

    def _update_stats_from_directory(self, root_dir: str):
        """Calcula estadísticas rápidas escaneando .seg existentes."""
        root = Path(root_dir)
        if not root.exists():
            return

        total_images = 0
        total_seg = 0
        total_grains = 0
        grains_per_class = {}

        from src.grain_detection.seg_format import SegFileReader

        for class_dir in sorted(root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            class_grains = 0

            for f in class_images(class_dir):
                total_images += 1
                seg = find_existing_seg_path(f, ANNOTATION_ROOT)
                if seg is not None:
                    total_seg += 1
                    try:
                        class_grains += SegFileReader.count_grains(str(seg))
                    except Exception:
                        pass

            total_grains += class_grains
            if class_grains > 0:
                grains_per_class[class_name] = class_grains

        if total_images == 0:
            self._set_stats("Sin datos — genera segmentaciones primero.")
            return

        avg = total_grains / total_seg if total_seg > 0 else 0.0
        self._set_stats(
            f"Imágenes: {total_images}  |  Con .seg: {total_seg}  |  "
            f"Granos: {total_grains}  |  Promedio/img: {avg:.1f}",
            grains_per_class,
        )
