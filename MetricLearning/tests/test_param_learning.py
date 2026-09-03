"""
Calibración por demostración: el etiquetado manual fija los topes.

El fallo que cubren: los topes de área son propios de cada especie. Calibrados
para `C_Quitensis` (~290.000 px² por grano), `min_area=80000` descartaba todos
los cuerpos de `J_Bufonius` (~60.000 px²) y el lote devolvía 0 semillas.
"""

import cv2
import numpy as np
import pytest

from src.grain_detection.param_learning import (
    blend_with_current,
    contours_from_grains,
    describe_envelope,
    envelope_from_contours,
    measure_contour,
    merge_envelopes,
    params_from_envelope,
)
from src.grain_detection.seeded_contour import (
    SeededParams,
    passes_validity_filters,
    peak_seeds,
    propose_objects,
)

FRAME_H, FRAME_W = 1942, 2590
FRAME_AREA = FRAME_H * FRAME_W


def _circle_contour(cx, cy, r, n=64):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1).astype(np.int32)


def _rect_contour(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)


# ---------------------------------------------------------------------------
# Medición de un cuerpo
# ---------------------------------------------------------------------------

def test_measure_circle_is_round_and_square():
    m = measure_contour(_circle_contour(500, 500, 100), FRAME_H, FRAME_W)
    assert m["circularity"] > 0.9
    assert m["aspect_ratio"] == pytest.approx(1.0, abs=0.1)
    assert m["area"] == pytest.approx(np.pi * 100 ** 2, rel=0.05)


def test_measure_elongated_reports_aspect():
    m = measure_contour(_rect_contour(0, 0, 400, 100), FRAME_H, FRAME_W)
    assert m["aspect_ratio"] == pytest.approx(4.0, rel=0.05)
    assert m["side_px"] == pytest.approx(400, rel=0.05)


def test_measure_rejects_degenerate():
    assert measure_contour(None, FRAME_H, FRAME_W) is None
    assert measure_contour(np.array([[1, 1], [2, 2]]), FRAME_H, FRAME_W) is None


# ---------------------------------------------------------------------------
# Envolvente y acumulación
# ---------------------------------------------------------------------------

def test_envelope_brackets_every_body():
    contours = [_circle_contour(400, 400, 60), _circle_contour(900, 900, 140)]
    env = envelope_from_contours(contours, FRAME_H, FRAME_W)
    assert env["n"] == 2
    assert env["min_area"] < env["max_area"]
    assert env["max_side_px"] == pytest.approx(280, rel=0.05)


def test_envelope_of_nothing_is_none():
    assert envelope_from_contours([], FRAME_H, FRAME_W) is None


def test_merge_accumulates_across_images():
    a = envelope_from_contours([_circle_contour(400, 400, 60)], FRAME_H, FRAME_W)
    b = envelope_from_contours([_circle_contour(900, 900, 140)], FRAME_H, FRAME_W)
    merged = merge_envelopes(a, b)

    assert merged["n"] == 2
    assert merged["min_area"] == pytest.approx(a["min_area"])
    assert merged["max_area"] == pytest.approx(b["max_area"])


def test_merge_with_none_keeps_the_other():
    a = envelope_from_contours([_circle_contour(400, 400, 60)], FRAME_H, FRAME_W)
    assert merge_envelopes(None, a)["n"] == 1
    assert merge_envelopes(a, None)["n"] == 1
    assert merge_envelopes(None, None) is None


# ---------------------------------------------------------------------------
# Topes derivados: deben aceptar lo etiquetado
# ---------------------------------------------------------------------------

def test_learned_params_accept_the_labelled_body():
    contour = _circle_contour(1200, 900, 130)
    env = envelope_from_contours([contour], FRAME_H, FRAME_W)
    learned = params_from_envelope(env, FRAME_AREA)

    params = SeededParams.from_config({**learned, "max_area_frac": 1.0})
    assert passes_validity_filters(contour, FRAME_H, FRAME_W, params)


def test_learned_params_accept_elongated_body():
    contour = _rect_contour(200, 200, 900, 380)
    env = envelope_from_contours([contour], FRAME_H, FRAME_W)
    learned = params_from_envelope(env, FRAME_AREA)

    params = SeededParams.from_config({**learned, "max_area_frac": 1.0})
    assert passes_validity_filters(contour, FRAME_H, FRAME_W, params)


def test_learned_min_area_is_below_the_smallest_body():
    env = envelope_from_contours(
        [_circle_contour(400, 400, 60), _circle_contour(900, 900, 140)],
        FRAME_H, FRAME_W,
    )
    learned = params_from_envelope(env, FRAME_AREA)
    assert learned["min_area"] < env["min_area"]
    assert learned["max_area"] > env["max_area"]


def test_learning_does_not_set_crop_radius():
    """El radio es de U²-Net, no del tamaño del grano. 2072 px era el marco."""
    env = envelope_from_contours([_circle_contour(900, 900, 200)], FRAME_H, FRAME_W)
    learned = params_from_envelope(env, FRAME_AREA)
    assert "crop_radius" not in learned


def test_learned_params_are_clamped_to_the_frame():
    env = envelope_from_contours(
        [_rect_contour(0, 0, FRAME_W - 1, FRAME_H - 1)], FRAME_H, FRAME_W
    )
    learned = params_from_envelope(env, FRAME_AREA)
    assert learned["max_area"] <= FRAME_AREA
    assert learned["max_bbox_side_frac"] <= 1.0
    assert learned["max_aspect_ratio"] <= 20.0


def test_params_from_no_envelope_is_none():
    assert params_from_envelope(None, FRAME_AREA) is None


# ---------------------------------------------------------------------------
# El caso real: topes de otra especie dejaban el lote en cero
# ---------------------------------------------------------------------------

def _small_seed_scene():
    """Dos cuerpos de ~60.000 px², como J_Bufonius."""
    sal = np.zeros((900, 900), np.float32)
    cv2.ellipse(sal, (250, 300), (150, 120), 0, 0, 360, 1.0, -1)
    cv2.ellipse(sal, (650, 600), (145, 115), 0, 0, 360, 1.0, -1)
    return sal, np.zeros((900, 900, 3), np.uint8)


BIG_SPECIES_CFG = dict(
    min_area=80000, max_area=800000, max_area_frac=0.5, min_circularity=0.12,
    max_bbox_side_frac=0.85, max_aspect_ratio=8.0, morph_kernel_size=3,
    saliency_threshold=0.3, adaptive_k=0.2, crop_radius=300, peak_max=16,
)


def test_params_of_another_species_find_nothing():
    """Reproduce el síntoma reportado: 0 semillas, 0 cuerpos."""
    sal, img = _small_seed_scene()
    params = SeededParams(**BIG_SPECIES_CFG)
    assert peak_seeds(sal, params) == []
    assert propose_objects(sal, img, params) == []


def test_learning_from_one_manual_body_unblocks_the_class():
    """Un solo cuerpo etiquetado a mano basta para recalibrar la clase."""
    sal, img = _small_seed_scene()

    # El usuario marca a mano el primer cuerpo.
    manual = _circle_contour(250, 300, 130)
    env = envelope_from_contours([manual], 900, 900)
    learned = blend_with_current(
        params_from_envelope(env, 900 * 900), BIG_SPECIES_CFG
    )

    params = SeededParams.from_config({**BIG_SPECIES_CFG, **learned, "max_area_frac": 1.0})
    assert peak_seeds(sal, params), "sigue sin generar semillas tras aprender"
    assert len(propose_objects(sal, img, params)) >= 1


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def test_blend_rescales_area_to_the_species():
    """Es el arreglo del síntoma: min_area baja al tamaño real de la clase."""
    learned = {"min_area": 21988, "max_area": 122157, "min_circularity": 0.75,
               "max_aspect_ratio": 1.5, "max_bbox_side_frac": 0.36, "crop_radius": 208}
    mezclado = blend_with_current(learned, BIG_SPECIES_CFG)

    assert mezclado["min_area"] == 21988
    assert mezclado["max_area"] == 122157


def test_blend_never_tightens_shape_limits():
    """Un círculo perfecto no puede subir la circularidad mínima a 0.75."""
    learned = {"min_area": 21988, "max_area": 122157, "min_circularity": 0.75,
               "max_aspect_ratio": 1.5, "max_bbox_side_frac": 0.36, "crop_radius": 100}
    mezclado = blend_with_current(learned, BIG_SPECIES_CFG)

    assert mezclado["min_circularity"] == BIG_SPECIES_CFG["min_circularity"]
    assert mezclado["max_aspect_ratio"] == BIG_SPECIES_CFG["max_aspect_ratio"]
    assert mezclado["max_bbox_side_frac"] == BIG_SPECIES_CFG["max_bbox_side_frac"]
    assert "crop_radius" not in mezclado


def test_blend_relaxes_when_the_body_needs_it():
    """Una semilla alargada sí debe ampliar el aspecto."""
    learned = {"min_area": 5000, "max_area": 90000, "min_circularity": 0.05,
               "max_aspect_ratio": 11.0, "max_bbox_side_frac": 0.93}
    mezclado = blend_with_current(learned, BIG_SPECIES_CFG)

    assert mezclado["max_aspect_ratio"] == 11.0
    assert mezclado["max_bbox_side_frac"] == 0.93
    assert mezclado["min_circularity"] == 0.05


def test_blend_without_current_uses_learned():
    learned = {"min_area": 10, "max_area": 20, "min_circularity": 0.5,
               "max_aspect_ratio": 2.0, "max_bbox_side_frac": 0.5}
    assert blend_with_current(learned, {}) == learned
    assert blend_with_current(None, BIG_SPECIES_CFG) is None


def test_blended_params_still_accept_the_labelled_body():
    """La mezcla no puede romper la garantía: lo etiquetado sigue siendo válido."""
    contour = _rect_contour(200, 200, 900, 380)
    env = envelope_from_contours([contour], FRAME_H, FRAME_W)
    mezclado = blend_with_current(
        params_from_envelope(env, FRAME_AREA), BIG_SPECIES_CFG
    )
    params = SeededParams.from_config({**BIG_SPECIES_CFG, **mezclado, "max_area_frac": 1.0})
    assert passes_validity_filters(contour, FRAME_H, FRAME_W, params)


def test_contours_from_grains_handles_seg_shapes():
    grains = [
        {"contour": _circle_contour(100, 100, 30)},
        {"contour": _circle_contour(200, 200, 30).reshape(-1, 1, 2)},
        {"contour": None},
        {},
    ]
    out = contours_from_grains(grains)
    assert len(out) == 2
    assert all(c.ndim == 2 and c.shape[1] == 2 for c in out)


def test_envelopes_roundtrip_on_disk(tmp_path):
    from src.grain_detection.param_learning import load_envelopes, save_envelopes

    env = envelope_from_contours([_circle_contour(400, 400, 90)], FRAME_H, FRAME_W)
    save_envelopes(tmp_path, {"J_Bufonius": env})
    loaded = load_envelopes(tmp_path)
    assert "J_Bufonius" in loaded
    assert loaded["J_Bufonius"]["n"] == env["n"]
    assert loaded["J_Bufonius"]["min_area"] == pytest.approx(env["min_area"])


def test_load_envelopes_missing_file_is_empty(tmp_path):
    from src.grain_detection.param_learning import load_envelopes

    assert load_envelopes(tmp_path) == {}


def test_load_envelopes_ignores_corrupt_file(tmp_path):
    from src.grain_detection.param_learning import ENVELOPES_FILENAME, load_envelopes

    (tmp_path / ENVELOPES_FILENAME).write_text("{no json", encoding="utf-8")
    assert load_envelopes(tmp_path) == {}


def test_describe_envelope_is_readable():
    env = envelope_from_contours([_circle_contour(400, 400, 90)], FRAME_H, FRAME_W)
    texto = describe_envelope(env)
    assert "1 cuerpo" in texto and "px²" in texto
    assert describe_envelope(None) == "sin cuerpos medidos"


# ---------------------------------------------------------------------------
# Integración con la interfaz: el click mueve los controles
# ---------------------------------------------------------------------------

@pytest.fixture
def tab(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    import lib.contour_analysis_tab as contour_tab

    monkeypatch.setattr(contour_tab, "ANNOTATION_ROOT", str(tmp_path))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    widget = contour_tab.ContourAnalysisTab()
    widget._current_image_bgr = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    widget._current_images = [Path("data/processed/train/J_Bufonius/a.png")]
    widget._current_image_idx = 0
    yield widget
    widget.deleteLater()
    app.processEvents()


def _grain(cx, cy, rx, ry):
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pts = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], axis=1)
    return {"contour": pts.astype(np.int32)}


def test_registry_does_not_restore_crop_radius(tab):
    """990 px en el registro hacía 4 inferencias casi a marco y U²-Net perdía el grano."""
    tab.crop_radius_spin.setValue(300)
    tab._restore_params_from_registry({"crop_radius": 990, "min_area": 145269})
    assert tab.crop_radius_spin.value() == 300


def test_learning_is_on_by_default(tab):
    assert tab.learn_from_manual_cb.isChecked()


def test_manual_body_lowers_min_area(tab):
    tab.min_area_spin.setValue(80000)
    tab._learn_from_grains([_grain(400, 400, 130, 120)], reason="test")
    assert tab.min_area_spin.value() < 80000


def test_learning_accumulates_across_clicks(tab):
    tab._learn_from_grains([_grain(400, 400, 130, 120)], reason="test")
    tras_uno = tab.min_area_spin.value()
    tab._learn_from_grains([_grain(900, 700, 95, 88)], reason="test")
    assert tab.min_area_spin.value() < tras_uno


def test_learning_keeps_shape_limits_of_the_preset(tab):
    circ_antes = tab.circularity_slider._slider.value()
    asp_antes = tab.max_aspect_spin.value()
    tab._learn_from_grains([_grain(400, 400, 130, 128)], reason="test")
    assert tab.circularity_slider._slider.value() <= circ_antes
    assert tab.max_aspect_spin.value() >= asp_antes


def test_each_class_keeps_its_own_scale(tab):
    from pathlib import Path

    tab._learn_from_grains([_grain(400, 400, 130, 120)], reason="test")
    pequeno = tab.min_area_spin.value()

    tab._current_images = [Path("data/processed/train/C_Quitensis/b.png")]
    tab._learn_from_grains([_grain(1200, 900, 310, 290)], reason="test")
    grande = tab.min_area_spin.value()
    assert grande > pequeno

    radio_antes = tab.crop_radius_spin.value()
    tab._current_images = [Path("data/processed/train/J_Bufonius/c.png")]
    tab._learning_class = None
    tab._adopt_class_envelope()
    assert tab.min_area_spin.value() == pequeno
    assert tab.crop_radius_spin.value() == radio_antes


def test_disabled_learning_leaves_controls_alone(tab):
    tab.learn_from_manual_cb.setChecked(False)
    tab.min_area_spin.setValue(80000)
    tab._learn_from_grains([_grain(400, 400, 130, 120)], reason="test")
    assert tab.min_area_spin.value() == 80000


def test_learned_config_reaches_the_batch_finder(tab):
    """El puente: los topes aprendidos son los que usará 'Segmentar'."""
    tab._learn_from_grains([_grain(400, 400, 130, 120)], reason="test")
    params = SeededParams.from_config(tab._get_config())
    assert params.min_area == tab.min_area_spin.value()
    assert params.max_area == tab.max_area_spin.value()


def test_manual_click_ignores_learned_area_caps(tab):
    """Si el click heredara max_area, un grano más grande no se podría enseñar."""
    cfg = tab._get_config()
    demo = SeededParams.from_config(cfg).for_demonstration(FRAME_H, FRAME_W)
    assert demo.max_area >= FRAME_H * FRAME_W
    assert demo.max_area > cfg["max_area"]
    assert demo.min_area < cfg["min_area"]
    assert demo.max_area_frac == 1.0


def test_learning_persists_per_class(tab, tmp_path):
    from src.grain_detection.param_learning import load_envelopes

    tab._learn_from_grains([_grain(400, 400, 130, 120)], reason="test")
    loaded = load_envelopes(tmp_path)
    assert "J_Bufonius" in loaded
    assert loaded["J_Bufonius"]["n"] >= 1


def test_existing_seg_bootstraps_only_once(tab):
    """Navegar un .seg no recalibra en cada imagen: solo arranca la clase."""
    tab._current_grains = [_grain(400, 400, 130, 120)]
    tab.min_area_spin.setValue(80000)
    tab._learn_from_grains(tab._current_grains, reason=".seg existente")
    assert "J_Bufonius" in tab._class_envelopes
    # El gancho de carga solo llama a aprender si la clase aún no tiene envolvente.
    assert tab._current_class_name() in tab._class_envelopes
