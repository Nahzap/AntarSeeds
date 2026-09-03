"""
Efectos de borde: ningún contorno puede tener un lado recto artificial.

Dos causas producían la misma línea recta sobre una semilla:

1. Watershed cortando un cuerpo real cuyo máximo de distancia se desdobla
   (semilla alargada o con una muesca). El corte atravesaba la semilla entera.
2. Contorno truncado por el recorte y aceptado como resultado. El lado recto
   era el canto del recorte, no el perfil del objeto.
"""

import cv2
import numpy as np
import pytest

from src.grain_detection.seeded_contour import (
    SeededParams,
    crop_saliency_provider,
    distance_cores,
    merge_shallow_splits,
    peak_seeds,
    propose_objects,
    resolve_seeded_object,
    touches_image_border,
    touches_interior_crop_border,
    watershed_regions,
)

BASE = dict(
    min_area=2000, max_area=400000, max_area_frac=0.8, min_circularity=0.05,
    max_bbox_side_frac=0.95, max_aspect_ratio=12.0, morph_kernel_size=3,
    saliency_threshold=0.3, adaptive_k=0.1, crop_radius=160, peak_max=16,
)


def _params(**kwargs):
    return SeededParams(**{**BASE, **kwargs})


def _mask(saliency):
    return (saliency >= 0.5).astype(np.uint8) * 255


def _elongated():
    """Una sola semilla alargada: el corte por la mitad sería un error."""
    sal = np.zeros((500, 500), np.float32)
    cv2.ellipse(sal, (250, 250), (150, 62), 0, 0, 360, 1.0, -1)
    return sal


def _touching():
    """Dos semillas tangentes: aquí el corte SÍ corresponde."""
    sal = np.zeros((500, 500), np.float32)
    cv2.circle(sal, (200, 250), 70, 1.0, -1)
    cv2.circle(sal, (320, 250), 70, 1.0, -1)
    return sal


def _n_labels(markers):
    return len({int(v) for v in np.unique(markers) if v > 1})


# ---------------------------------------------------------------------------
# Cintura: no partir un cuerpo sin istmo real
# ---------------------------------------------------------------------------

def test_elongated_body_is_not_split():
    """El caso del borde recto atravesando una semilla."""
    assert watershed_regions(_mask(_elongated()), _params()) is None


def test_elongated_body_yields_one_seed():
    seeds = peak_seeds(_elongated(), _params(split_touching=True))
    assert len(seeds) == 1


def test_elongated_body_yields_one_core():
    cores = distance_cores(_mask(_elongated()), _params())
    assert len(cores) == 1


def test_elongated_body_stays_whole_end_to_end():
    sal = _elongated()
    granos = propose_objects(sal, np.zeros((500, 500, 3), np.uint8), _params())
    assert len(granos) == 1
    _x, _y, bw, _bh = granos[0]["bbox"]
    assert bw > 250, f"ancho {bw}: la semilla salió partida"


def test_touching_bodies_still_split():
    """La cintura no puede desactivar la separación legítima."""
    markers = watershed_regions(_mask(_touching()), _params())
    assert markers is not None and _n_labels(markers) == 2


def test_waist_too_permissive_merges_everything():
    """waist_frac alto acepta cortes sin cintura: el fallo que se corrigió."""
    binary = _mask(_elongated())
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    from src.grain_detection.seeded_contour import distance_peaks

    sure_fg = distance_peaks(binary, _params())
    n, markers = cv2.connectedComponents(sure_fg)
    if n <= 2:
        pytest.skip("la escena no genera picos múltiples")

    markers = markers.astype(np.int32) + 1
    markers[cv2.subtract(binary, sure_fg) > 0] = 0
    cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)

    laxo = merge_shallow_splits(markers, binary, dist, _params(waist_frac=0.99))
    estricto = merge_shallow_splits(markers, binary, dist, _params(waist_frac=0.50))
    assert _n_labels(estricto) < _n_labels(laxo) or _n_labels(estricto) == 1


# ---------------------------------------------------------------------------
# Truncado por el recorte
# ---------------------------------------------------------------------------

def test_interior_crop_side_is_an_artifact():
    """Lado del recorte que no es borde de la imagen: recta inventada."""
    contour = np.array([[0, 20], [60, 20], [60, 80], [0, 80]], dtype=np.int32)
    box = (500, 500, 600, 600)  # recorte interior en una imagen 2000x2000
    assert touches_interior_crop_border(contour, box, 2000, 2000) is True


def test_crop_side_on_the_image_edge_is_legitimate():
    """El mismo lado, pero coincidiendo con el borde del encuadre: es real."""
    contour = np.array([[0, 20], [60, 20], [60, 80], [0, 80]], dtype=np.int32)
    box = (0, 0, 100, 100)  # recorte pegado a la esquina de una imagen 100x100
    assert touches_interior_crop_border(contour, box, 100, 100) is False


def test_contour_inside_the_crop_has_no_artifact():
    contour = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.int32)
    box = (500, 500, 600, 600)
    assert touches_interior_crop_border(contour, box, 2000, 2000) is False


def test_truncated_contour_is_never_returned():
    """
    Radio de recorte minúsculo frente al objeto: antes se devolvía el contorno
    cortado (lado recto). Ahora el recorte crece hasta el encuadre y el cuerpo
    sale completo.
    """
    sal = _elongated()
    img = np.zeros((500, 500, 3), np.uint8)
    params = _params(crop_radius=60)

    grano = resolve_seeded_object(img, (250, 250), params, crop_saliency_provider(sal))
    assert grano is not None, "el objeto legítimo no debe perderse"
    _x, _y, bw, _bh = grano["bbox"]
    assert bw > 250, f"ancho {bw}: sigue truncado por el recorte"


def test_seed_far_from_any_body_is_rejected():
    sal = np.zeros((400, 400), np.float32)
    img = np.zeros((400, 400, 3), np.uint8)
    assert resolve_seeded_object(img, (200, 200), _params(), crop_saliency_provider(sal)) is None


# ---------------------------------------------------------------------------
# Cuerpos cortados por el encuadre
# ---------------------------------------------------------------------------

def _body_on_edge():
    """Semilla partida por el borde superior del encuadre."""
    sal = np.zeros((400, 400), np.float32)
    cv2.circle(sal, (200, 10), 80, 1.0, -1)
    return sal


def test_touches_image_border_detects_partial_body():
    contour = np.array([[100, 0], [300, 0], [300, 90], [100, 90]], dtype=np.int32)
    assert touches_image_border(contour, 400, 400) is True


def test_interior_body_does_not_touch_image_border():
    contour = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.int32)
    assert touches_image_border(contour, 400, 400) is False


def test_border_objects_kept_by_default():
    sal, img = _body_on_edge(), np.zeros((400, 400, 3), np.uint8)
    granos = propose_objects(sal, img, _params(min_area=1000))
    assert len(granos) == 1


def test_border_objects_dropped_when_asked():
    sal, img = _body_on_edge(), np.zeros((400, 400, 3), np.uint8)
    granos = propose_objects(
        sal, img, _params(min_area=1000, drop_border_objects=True)
    )
    assert granos == []


def test_interior_body_survives_the_border_policy():
    sal = np.zeros((400, 400), np.float32)
    cv2.circle(sal, (200, 200), 70, 1.0, -1)
    granos = propose_objects(
        sal, np.zeros((400, 400, 3), np.uint8),
        _params(min_area=1000, drop_border_objects=True),
    )
    assert len(granos) == 1


# ---------------------------------------------------------------------------
# Transporte de los ajustes
# ---------------------------------------------------------------------------

def test_params_carry_edge_settings():
    p = SeededParams.from_config({"waist_frac": 0.55, "drop_border_objects": True})
    assert p.waist_frac == pytest.approx(0.55)
    assert p.drop_border_objects is True


def test_detector_records_edge_settings_in_seg():
    from src.grain_detection.grain_detector import PollenGrainDetector

    det = PollenGrainDetector(
        config={"detection_profile": "seed", "waist_frac": 0.6, "drop_border_objects": True}
    )
    params = det.get_parameters()
    assert params["waist_frac"] == pytest.approx(0.6)
    assert params["drop_border_objects"] is True
