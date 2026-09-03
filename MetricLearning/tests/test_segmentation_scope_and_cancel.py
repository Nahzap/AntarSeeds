"""
Alcance y cancelación del lote de pre-segmentación.

Cubren tres fallos de uso reales: no se podía cortar un lote iniciado, el
alcance 'Train + Val + Test' ignoraba el tick de clases, y no había forma de
evaluar la configuración antes de lanzar miles de imágenes.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.grain_detection.annotator import SegmentationAnnotator
from src.grain_detection.detected_grain import DetectedGrain


class _StubDetector:
    """Detector determinista: un grano por imagen, sin cargar U²-Net."""

    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return None, [
            DetectedGrain(
                index=0,
                bbox=(10, 10, 30, 30),
                area=900.0,
                saliency_prob=0.9,
                centroid=(25, 25),
                contour=np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.int32),
            )
        ]

    def get_parameters(self):
        return {"object_finder": "roi_seed"}


@pytest.fixture
def dataset(tmp_path):
    for cls in ("C_Quitensis", "D_Antartica"):
        cls_dir = tmp_path / "train" / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            img = np.full((60, 60, 3), 30, np.uint8)
            cv2.imwrite(str(cls_dir / f"{cls}_{i:03d}.png"), img)
    return tmp_path


# ---------------------------------------------------------------------------
# Cancelación
# ---------------------------------------------------------------------------

def test_cancel_stops_batch_and_keeps_written_seg(dataset, tmp_path):
    detector = _StubDetector()
    annotator = SegmentationAnnotator(detector=detector)
    annot_root = tmp_path / "annotations"

    # Corta después de la segunda imagen.
    state = {"seen": 0}

    def cancel_check():
        state["seen"] += 1
        return state["seen"] > 2

    stats = annotator.annotate_directory(
        str(dataset / "train"),
        annotation_root=str(annot_root),
        cancel_check=cancel_check,
    )

    assert stats["cancelled"] is True
    assert stats["annotated"] == 2
    assert detector.calls == 2
    assert len(list(annot_root.rglob("*.seg"))) == 2


def test_no_cancel_check_processes_everything(dataset, tmp_path):
    annotator = SegmentationAnnotator(detector=_StubDetector())
    stats = annotator.annotate_directory(
        str(dataset / "train"), annotation_root=str(tmp_path / "a")
    )
    assert stats["cancelled"] is False
    assert stats["annotated"] == 6


def test_cancel_before_first_image_writes_nothing(dataset, tmp_path):
    annot_root = tmp_path / "annotations"
    annotator = SegmentationAnnotator(detector=_StubDetector())
    stats = annotator.annotate_directory(
        str(dataset / "train"),
        annotation_root=str(annot_root),
        cancel_check=lambda: True,
    )
    assert stats["cancelled"] is True
    assert stats["annotated"] == 0
    assert list(annot_root.rglob("*.seg")) == []


# ---------------------------------------------------------------------------
# Alcance: el tick manda
# ---------------------------------------------------------------------------

def test_class_filter_limits_annotation_to_ticked_classes(dataset, tmp_path):
    annot_root = tmp_path / "annotations"
    annotator = SegmentationAnnotator(detector=_StubDetector())
    stats = annotator.annotate_directory(
        str(dataset / "train"),
        annotation_root=str(annot_root),
        class_filter=["C_Quitensis"],
    )

    assert stats["annotated"] == 3
    assert (annot_root / "C_Quitensis").exists()
    assert not (annot_root / "D_Antartica").exists()


def test_worker_passes_ticked_classes_to_every_split(tmp_path):
    """El alcance multi-split no puede perder el filtro de clases."""
    from unittest.mock import MagicMock, patch

    from lib.contour_analysis_tab import AnnotationWorker

    splits = []
    for name in ("train", "val", "test"):
        d = tmp_path / name
        d.mkdir()
        splits.append(str(d))

    with patch("src.grain_detection.annotator.SegmentationAnnotator") as MockAnnotator:
        mock = MagicMock()
        mock.annotate_directory.return_value = {
            "total_images": 1, "annotated": 1, "skipped": 0, "errors": 0,
            "total_grains": 2, "elapsed_ms": 1.0, "cancelled": False,
            "grains_per_class": {"C_Quitensis": 2},
        }
        MockAnnotator.return_value = mock

        worker = AnnotationWorker(
            splits, overwrite=False, config={}, class_filter=["C_Quitensis"]
        )
        worker.run()

        assert mock.annotate_directory.call_count == 3
        for call in mock.annotate_directory.call_args_list:
            assert call[1]["class_filter"] == ["C_Quitensis"]
            assert callable(call[1]["cancel_check"])


def test_worker_marks_combined_stats_as_cancelled(tmp_path):
    from lib.contour_analysis_tab import AnnotationWorker

    combined = AnnotationWorker.empty_stats()
    AnnotationWorker.accumulate(combined, "train", {
        "total_images": 5, "annotated": 2, "skipped": 0, "errors": 0,
        "total_grains": 4, "elapsed_ms": 2.0, "cancelled": True,
        "grains_per_class": {"C_Quitensis": 4},
    })
    assert combined["cancelled"] is True
    assert "CANCELADO" in _summary(combined)


def _summary(stats):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5.QtWidgets")
    from lib.contour_analysis_tab import ContourAnalysisTab

    return ContourAnalysisTab._format_annotation_summary(stats)


# ---------------------------------------------------------------------------
# Panel de configuración: la prueba usa exactamente lo que usará el lote
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def contour_tab():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    from lib.contour_analysis_tab import ContourAnalysisTab

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    tab = ContourAnalysisTab()
    yield tab
    tab.deleteLater()
    app.processEvents()


def test_cancel_button_only_enabled_while_busy(contour_tab):
    contour_tab._set_segmentation_busy(True)
    assert contour_tab.btn_cancel_segment.isEnabled()
    assert not contour_tab.btn_segment.isEnabled()
    assert not contour_tab.btn_preview.isEnabled()

    contour_tab._set_segmentation_busy(False)
    assert not contour_tab.btn_cancel_segment.isEnabled()
    assert contour_tab.btn_segment.isEnabled()
    assert contour_tab.btn_preview.isEnabled()


def test_threshold_reaches_effective_label(contour_tab):
    """Threshold quedaba fuera de los filtros que refrescan el panel."""
    assert contour_tab.threshold_slider in contour_tab._filter_widgets()
    contour_tab.threshold_slider._slider.setValue(52)
    assert "0.52" in contour_tab.effective_params_label.text()


def test_preview_expires_when_a_filter_moves(contour_tab):
    contour_tab._preview_grains = [{"contour": np.zeros((4, 2), np.int32)}]
    contour_tab.min_area_spin.setValue(contour_tab.min_area_spin.value() + 1000)
    assert contour_tab._preview_grains == []


def test_preview_button_sits_with_the_image(contour_tab):
    """La prueba es una acción sobre el encuadre abierto, no del panel de lote."""
    assert contour_tab.btn_preview.text() == "Probar aquí"
    assert hasattr(contour_tab, "preview_result_label")
    assert hasattr(contour_tab, "btn_clear_preview")


def test_preview_result_suggests_what_to_adjust(contour_tab):
    contour_tab._current_grains = []
    contour_tab._on_preview_result([])
    texto = contour_tab.preview_result_label.text()
    assert "0 cuerpo" in texto
    assert "Threshold" in texto or "Min área" in texto


# ---------------------------------------------------------------------------
# Layout: las estadísticas deben poder leerse
# ---------------------------------------------------------------------------

def test_stats_bar_is_not_height_clipped(contour_tab):
    """Tenía setMaximumHeight(80) con texto envuelto: se recortaba."""
    stats_group = contour_tab.layout().itemAt(3).widget()
    assert stats_group.maximumHeight() > 200, "la barra no puede tener tope de altura"
    assert stats_group.minimumSizeHint().height() <= stats_group.sizeHint().height()


def test_stats_shows_totals_and_per_class_on_separate_labels(contour_tab):
    contour_tab._set_stats(
        "Imágenes: 1442  |  Con .seg: 1442  |  Granos: 1649",
        {"C_Quitensis": 512, "D_Antartica": 128},
    )
    assert "1442" in contour_tab.stats_label.text()
    assert "C_Quitensis: 512" in contour_tab.stats_class_label.text()
    assert "D_Antartica" in contour_tab.stats_class_label.toolTip()


def test_stats_clears_per_class_line_when_empty(contour_tab):
    contour_tab._set_stats("Sin datos")
    assert contour_tab.stats_class_label.text() == ""


def test_no_vertical_splitter_inside_scroll_area(contour_tab):
    """Un QSplitter vertical dentro del QScrollArea nunca se acota: 'no baja más'."""
    from PyQt5.QtWidgets import QSplitter

    top_level = [
        contour_tab.layout().itemAt(i).widget()
        for i in range(contour_tab.layout().count())
    ]
    assert not any(isinstance(w, QSplitter) for w in top_level if w is not None)


def test_folding_params_frees_height(contour_tab):
    contour_tab.show()
    config_group = contour_tab.layout().itemAt(1).widget()

    contour_tab.btn_toggle_params.setChecked(True)
    alto_desplegado = config_group.minimumSizeHint().height()
    contour_tab.btn_toggle_params.setChecked(False)
    alto_plegado = config_group.minimumSizeHint().height()

    assert alto_plegado < alto_desplegado
    assert not contour_tab.params_container.isVisible()
    assert contour_tab.btn_toggle_params.text().startswith("▸")
    contour_tab.btn_toggle_params.setChecked(True)
