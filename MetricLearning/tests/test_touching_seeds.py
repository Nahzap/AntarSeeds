"""
Semillas en contacto: el autoetiquetador debe separarlas.

El fallo que cubren: `peak_seeds` tomaba UN centroide por componente conectado.
Dos semillas que se tocan forman un solo componente, así que la semilla caía en
la unión y salía un único cuerpo fusionado. Un humano hace click en el centro de
cada semilla; el sembrador automático debe hacer lo mismo.
"""

import cv2
import numpy as np
import pytest

from src.grain_detection.seeded_contour import (
    SeededParams,
    crop_saliency_provider,
    distance_cores,
    distance_peaks,
    peak_seeds,
    propose_objects,
    region_containing,
    resolve_seeded_object,
    watershed_regions,
)

BASE = dict(
    min_area=2000, max_area=200000, max_area_frac=0.6, min_circularity=0.05,
    max_bbox_side_frac=0.95, max_aspect_ratio=8.0, morph_kernel_size=3,
    saliency_threshold=0.3, adaptive_k=0.1, crop_radius=140, peak_max=16,
)

# Dos círculos r=70 con centros a 120 px: se solapan 20 px.
TOUCHING_CENTERS = ((200, 250), (320, 250))


def _params(**kwargs):
    return SeededParams(**{**BASE, **kwargs})


def _touching_saliency():
    sal = np.zeros((500, 500), np.float32)
    for cx, cy in TOUCHING_CENTERS:
        cv2.circle(sal, (cx, cy), 70, 1.0, -1)
    return sal


def _separate_saliency():
    sal = np.zeros((500, 500), np.float32)
    cv2.circle(sal, (130, 130), 60, 1.0, -1)
    cv2.circle(sal, (370, 370), 60, 1.0, -1)
    return sal


def _mask(saliency):
    return (saliency >= 0.5).astype(np.uint8) * 255


def _blank():
    return np.zeros((500, 500, 3), np.uint8)


# ---------------------------------------------------------------------------
# Núcleos de distancia: un pico por cuerpo
# ---------------------------------------------------------------------------

def test_distance_peaks_finds_one_core_per_touching_body():
    peaks = distance_peaks(_mask(_touching_saliency()), _params())
    n, _lbl, _st, _cent = cv2.connectedComponentsWithStats(peaks, 8)
    assert n - 1 == 2, "el cuello no se cortó: siguen siendo un solo núcleo"


def test_distance_peaks_keeps_single_body_single():
    sal = np.zeros((400, 400), np.float32)
    cv2.circle(sal, (200, 200), 80, 1.0, -1)
    peaks = distance_peaks(_mask(sal), _params())
    n, _lbl, _st, _cent = cv2.connectedComponentsWithStats(peaks, 8)
    assert n - 1 == 1, "una semilla sola no debe partirse"


def test_distance_cores_sit_near_real_centers():
    cores = distance_cores(_mask(_touching_saliency()), _params())
    assert len(cores) == 2
    for _area, cx, cy in cores:
        d = min(
            (cx - rx) ** 2 + (cy - ry) ** 2 for rx, ry in TOUCHING_CENTERS
        ) ** 0.5
        assert d < 25, f"núcleo en ({cx},{cy}) lejos de todo centro real"


def test_distance_peaks_on_empty_mask_is_empty():
    empty = np.zeros((50, 50), np.uint8)
    assert np.count_nonzero(distance_peaks(empty, _params())) == 0
    assert distance_cores(empty, _params()) == []


# ---------------------------------------------------------------------------
# Sembrado automático
# ---------------------------------------------------------------------------

def test_seeds_land_on_each_body_not_on_the_junction():
    sal = _touching_saliency()
    seeds = peak_seeds(sal, _params(split_touching=True))
    assert len(seeds) == 2
    for sx, sy in seeds:
        assert abs(sx - 260) > 30, "la semilla cayó en la unión de los dos cuerpos"


def test_without_split_the_seed_falls_on_the_junction():
    """Comportamiento anterior, conservado como referencia del fallo."""
    seeds = peak_seeds(_touching_saliency(), _params(split_touching=False))
    assert len(seeds) == 1
    assert abs(seeds[0][0] - 260) < 15


def test_separate_bodies_unaffected_by_the_switch():
    sal = _separate_saliency()
    con = peak_seeds(sal, _params(split_touching=True))
    sin = peak_seeds(sal, _params(split_touching=False))
    assert len(con) == len(sin) == 2


# ---------------------------------------------------------------------------
# Watershed: el contorno no invade la vecina
# ---------------------------------------------------------------------------

def test_watershed_splits_touching_mask():
    markers = watershed_regions(_mask(_touching_saliency()), _params())
    assert markers is not None
    assert len({int(v) for v in np.unique(markers) if v > 1}) == 2


def test_watershed_returns_none_for_single_body():
    sal = np.zeros((400, 400), np.float32)
    cv2.circle(sal, (200, 200), 80, 1.0, -1)
    assert watershed_regions(_mask(sal), _params()) is None


def test_region_containing_picks_the_seed_side():
    binary = _mask(_touching_saliency())
    markers = watershed_regions(binary, _params())
    left = region_containing(markers, binary, 200, 250)
    right = region_containing(markers, binary, 320, 250)

    assert left is not None and right is not None
    assert np.count_nonzero(cv2.bitwise_and(left, right)) == 0, "las regiones se solapan"
    assert left[250, 200] > 0 and left[250, 320] == 0
    assert right[250, 320] > 0 and right[250, 200] == 0


# ---------------------------------------------------------------------------
# Efecto extremo a extremo: click y lote
# ---------------------------------------------------------------------------

def test_batch_yields_one_body_per_touching_seed():
    sal, img = _touching_saliency(), _blank()
    fusionado = propose_objects(sal, img, _params(split_touching=False))
    separado = propose_objects(sal, img, _params(split_touching=True))

    assert len(fusionado) == 1
    assert len(separado) == 2


def test_split_bodies_are_each_about_one_seed_wide():
    sal, img = _touching_saliency(), _blank()
    granos = propose_objects(sal, img, _params(split_touching=True))
    assert len(granos) == 2
    for g in granos:
        _x, _y, bw, _bh = g["bbox"]
        # Un cuerpo solo mide ~140 px; el fusionado medía ~260 px.
        assert bw < 190, f"ancho {bw}: el cuerpo sigue abarcando a la vecina"


def test_click_on_one_of_two_touching_seeds_returns_only_that_one():
    """Es el caso que el usuario resuelve a mano: el click debe respetar el corte."""
    sal, img = _touching_saliency(), _blank()
    params = _params(split_touching=True)
    provider = crop_saliency_provider(sal)

    izq = resolve_seeded_object(img, (200, 250), params, provider)
    der = resolve_seeded_object(img, (320, 250), params, provider)

    assert izq is not None and der is not None
    assert izq["bbox"] != der["bbox"]
    for grano in (izq, der):
        _x, _y, bw, _bh = grano["bbox"]
        assert bw < 190


@pytest.mark.parametrize("frac", [0.25, 0.45, 0.65])
def test_separation_holds_across_core_fractions(frac):
    sal, img = _touching_saliency(), _blank()
    granos = propose_objects(sal, img, _params(split_touching=True, seed_core_frac=frac))
    assert len(granos) == 2


# ---------------------------------------------------------------------------
# La GUI y el .seg deben transportar los dos ajustes nuevos
# ---------------------------------------------------------------------------

def test_params_from_config_carry_split_keys():
    p = SeededParams.from_config({"split_touching": False, "seed_core_frac": 0.7})
    assert p.split_touching is False
    assert p.seed_core_frac == pytest.approx(0.7)


def test_defaults_separate_touching_seeds():
    assert SeededParams().split_touching is True
    assert SeededParams.from_config({}).split_touching is True


def test_detector_records_split_settings_in_seg_params():
    from src.grain_detection.grain_detector import PollenGrainDetector

    det = PollenGrainDetector(
        config={"detection_profile": "seed", "split_touching": True, "seed_core_frac": 0.5}
    )
    params = det.get_parameters()
    assert params["split_touching"] is True
    assert params["seed_core_frac"] == pytest.approx(0.5)
