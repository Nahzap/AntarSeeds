"""
Tests del puente GUI -> detector del autoetiquetador weak.

El fallo que cubren: la pestaña de contornos enviaba solo 8 claves de config, así
que resolve_grain_detection_config rellenaba los filtros de encuadre con el preset
'balanced' (polen: aspecto 2.8, lado 0.40). El perfil 'seed' era inalcanzable desde
la interfaz y las semillas grandes o alargadas se descartaban en silencio.
"""

import os

import numpy as np
import pytest

from src.grain_detection.detection_profiles import (
    FILTER_KEYS,
    PROFILES,
    missing_filter_keys,
    resolve_grain_detection_config,
)
from src.grain_detection.grain_detector import PollenGrainDetector
from src.grain_detection.saliency_grain_extraction import extract_grains_from_saliency

# Encuadre real de la Basler.
FRAME_H, FRAME_W = 1942, 2590


def _elongated_seed_saliency(length=900, width=150):
    """Saliency con una sola semilla alargada centrada (tipo J_Bufonius)."""
    sal = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
    cy, cx = FRAME_H // 2, FRAME_W // 2
    y0, y1 = cy - width // 2, cy + width // 2
    x0, x1 = cx - length // 2, cx + length // 2
    sal[y0:y1, x0:x1] = 0.95
    return sal


def _blank_frame():
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Contrato de configuración
# ---------------------------------------------------------------------------

def test_partial_config_lets_preset_override_filters():
    """Omitir un filtro cede el control al preset: el bug de raíz."""
    partial = {"min_area": 30000, "max_area": 500000, "min_circularity": 0.30}
    assert "max_aspect_ratio" in missing_filter_keys(partial)

    resolved = resolve_grain_detection_config(partial)
    assert resolved["max_aspect_ratio"] == PROFILES["balanced"]["max_aspect_ratio"]
    assert resolved["max_bbox_side_frac"] == PROFILES["balanced"]["max_bbox_side_frac"]


def test_complete_config_survives_resolution_unchanged():
    """Con todos los filtros declarados, ningún preset puede pisarlos."""
    explicit = {
        "detection_profile": "balanced",
        "adaptive_k": 0.20,
        "min_area": 30000,
        "max_area": 500000,
        "max_area_frac": 0.50,
        "min_circularity": 0.12,
        "max_bbox_side_frac": 0.85,
        "max_aspect_ratio": 8.0,
        "morph_kernel_size": 3,
    }
    assert missing_filter_keys(explicit) == ()

    resolved = resolve_grain_detection_config(explicit)
    for key in FILTER_KEYS:
        assert resolved[key] == explicit[key], f"{key} fue sobrescrito por el preset"


# ---------------------------------------------------------------------------
# Efecto real sobre la detección
# ---------------------------------------------------------------------------

def test_balanced_rejects_elongated_seed():
    """Los límites de polen descartan una semilla alargada válida."""
    kwargs = resolve_grain_detection_config({"detection_profile": "balanced"})
    grains = extract_grains_from_saliency(
        _elongated_seed_saliency(),
        _blank_frame(),
        min_area=1000,
        max_area=1000000,
        min_circularity=0.10,
        max_bbox_side_frac=kwargs["max_bbox_side_frac"],
        max_aspect_ratio=kwargs["max_aspect_ratio"],
    )
    assert grains == []


def test_seed_profile_accepts_elongated_seed():
    """El perfil 'seed' sí acepta la misma semilla."""
    kwargs = resolve_grain_detection_config({"detection_profile": "seed"})
    grains = extract_grains_from_saliency(
        _elongated_seed_saliency(),
        _blank_frame(),
        min_area=kwargs["min_area"],
        max_area=kwargs["max_area"],
        min_circularity=kwargs["min_circularity"],
        max_bbox_side_frac=kwargs["max_bbox_side_frac"],
        max_aspect_ratio=kwargs["max_aspect_ratio"],
    )
    assert len(grains) == 1
    _, _, w, h = grains[0].bbox
    assert max(w, h) / min(w, h) > PROFILES["balanced"]["max_aspect_ratio"]


# ---------------------------------------------------------------------------
# max_area vs max_area_frac: ambos vinculantes
# ---------------------------------------------------------------------------

def test_area_caps_combine_most_restrictive_wins():
    sal = np.zeros((500, 500), dtype=np.float32)
    sal[100:300, 100:300] = 0.9  # 40000 px

    common = dict(
        min_circularity=0.1,
        max_bbox_side_frac=0.9,
        max_aspect_ratio=8.0,
        min_area=100,
    )
    # frac permisiva (0.5*250000=125000), max_area absoluto restrictivo
    assert extract_grains_from_saliency(
        sal, np.zeros((500, 500, 3), np.uint8),
        max_area=10000, max_area_frac=0.5, **common
    ) == []
    # max_area permisivo, frac restrictiva (0.01*250000=2500)
    assert extract_grains_from_saliency(
        sal, np.zeros((500, 500, 3), np.uint8),
        max_area=1000000, max_area_frac=0.01, **common
    ) == []
    # ambos permisivos
    assert len(extract_grains_from_saliency(
        sal, np.zeros((500, 500, 3), np.uint8),
        max_area=1000000, max_area_frac=0.5, **common
    )) == 1


# ---------------------------------------------------------------------------
# Trazabilidad: el .seg debe registrar los filtros aplicados
# ---------------------------------------------------------------------------

def test_detector_reports_every_applied_filter():
    """get_parameters() alimenta la cabecera del .seg; no puede omitir filtros."""
    detector = PollenGrainDetector(config={"detection_profile": "seed"})
    params = detector.get_parameters()
    for key in FILTER_KEYS:
        assert key in params, f"{key} no queda registrado en el .seg"
    assert params["detection_profile"] == "seed"


def test_detector_honours_explicit_filters_over_profile():
    detector = PollenGrainDetector(
        config={"detection_profile": "balanced", "max_aspect_ratio": 9.5}
    )
    assert detector.max_aspect_ratio == 9.5


# ---------------------------------------------------------------------------
# La GUI real: lo que se ve en los controles es lo que se ejecuta
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


def test_gui_config_declares_every_filter(contour_tab):
    assert missing_filter_keys(contour_tab._get_config()) == ()


def test_gui_defaults_to_seed_profile(contour_tab):
    cfg = contour_tab._get_config()
    assert cfg["detection_profile"] == "seed"
    assert cfg["max_aspect_ratio"] == PROFILES["seed"]["max_aspect_ratio"]
    assert cfg["max_bbox_side_frac"] == PROFILES["seed"]["max_bbox_side_frac"]


def test_gui_profile_switch_reaches_detector(contour_tab):
    contour_tab.profile_combo.setCurrentText("strict")
    cfg = contour_tab._get_config()
    assert cfg["max_aspect_ratio"] == PROFILES["strict"]["max_aspect_ratio"]

    contour_tab.profile_combo.setCurrentText("seed")
    contour_tab.max_aspect_spin.setValue(11.0)
    resolved = resolve_grain_detection_config(contour_tab._get_config())
    assert resolved["max_aspect_ratio"] == 11.0


def test_gui_area_spinboxes_cover_full_frame(contour_tab):
    """Un objeto no puede quedar fuera de rango por un tope de widget."""
    assert contour_tab.max_area_spin.maximum() >= FRAME_H * FRAME_W
    assert contour_tab.min_area_spin.maximum() >= FRAME_H * FRAME_W


def test_gui_object_finder_is_roi_seed(contour_tab):
    """ROI es el único buscador: click y lote leen las mismas claves."""
    cfg = contour_tab._get_config()
    assert cfg["object_finder"] == "roi_seed"
    assert cfg["crop_radius"] == contour_tab.crop_radius_spin.value()


def test_gui_has_no_edit_mode_switch(contour_tab):
    """Sin estado de modo: no puede existir un segundo camino de detección."""
    assert not hasattr(contour_tab, "edit_mode_combo")
    assert not hasattr(contour_tab, "_edit_mode")
    assert not hasattr(contour_tab, "_on_edit_mode_changed")
    assert not hasattr(contour_tab, "_on_manual_contour_result")


def test_gui_crop_radius_reaches_seeded_params(contour_tab):
    contour_tab.crop_radius_spin.setValue(420)
    contour_tab.threshold_slider._slider.setValue(45)
    cfg = contour_tab._get_config()
    from src.grain_detection.seeded_contour import SeededParams

    params = SeededParams.from_config(cfg)
    assert params.crop_radius == 420
    assert abs(params.saliency_threshold - 0.45) < 1e-9
    assert abs(params.min_circularity - contour_tab.circularity_slider._slider.value() / 100.0) < 1e-9

