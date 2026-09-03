"""El buscador ROI: click y lote deben compartir el mismo algoritmo."""

import cv2
import numpy as np

from src.grain_detection.salient_detector import SalientObjectDetector
from src.grain_detection.seeded_contour import (
    SeededParams,
    compare_object_background,
    contour_touches_border,
    covers_full_frame,
    crop_saliency_provider,
    fill_holes,
    is_valid_seeded_contour,
    passes_validity_filters,
    peak_seeds,
    propose_objects,
    resolve_seeded_object,
    u2net_seeded_contour,
)


def _disk(h, w, cy, cx, r, value=0.95):
    sal = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]
    sal[(x - cx) ** 2 + (y - cy) ** 2 <= r ** 2] = value
    return sal


def test_seeded_contour_from_center_recovers_blob():
    sal = _disk(240, 240, 120, 120, 40)
    cnt = u2net_seeded_contour(sal, 120, 120, SeededParams(min_area=200, morph_kernel_size=3))
    assert cnt is not None and len(cnt) >= 3
    xs, ys = cnt[:, 0], cnt[:, 1]
    assert xs.min() < 120 < xs.max()
    assert ys.min() < 120 < ys.max()
    _x, _y, bw, bh = cv2.boundingRect(cnt.reshape(-1, 1, 2))
    assert 70 <= bw <= 96
    assert 70 <= bh <= 96


def test_validity_filters_reject_tiny_and_keep_blob():
    params = SeededParams(min_area=5000, max_area=80000, min_circularity=0.10, max_aspect_ratio=8.0)
    tiny = np.array([[10, 10], [12, 10], [12, 12], [10, 12]], dtype=np.int32)
    blob = np.array([[80, 80], [160, 80], [160, 160], [80, 160]], dtype=np.int32)
    assert not passes_validity_filters(tiny, 240, 240, params)
    assert passes_validity_filters(blob, 240, 240, params)


def test_peak_seeds_finds_two_blobs():
    sal = _disk(400, 400, 100, 100, 35) + _disk(400, 400, 300, 300, 35)
    sal = np.clip(sal, 0, 1)
    params = SeededParams(min_area=400, saliency_threshold=0.3, adaptive_k=0.1, peak_max=8)
    seeds = peak_seeds(sal, params)
    assert len(seeds) == 2


def test_propose_objects_two_peaks_same_as_click_algorithm():
    sal = _disk(400, 400, 110, 110, 40) + _disk(400, 400, 290, 290, 40)
    sal = np.clip(sal, 0, 1)
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    params = SeededParams(
        min_area=400,
        max_area=80000,
        max_area_frac=0.5,
        min_circularity=0.10,
        max_bbox_side_frac=0.85,
        max_aspect_ratio=8.0,
        morph_kernel_size=3,
        saliency_threshold=0.25,
        adaptive_k=0.15,
        crop_radius=90,
        peak_max=8,
    )
    grains = propose_objects(sal, image, params)
    assert len(grains) == 2
    for g in grains:
        assert "contour" in g and "bbox" in g
        assert passes_validity_filters(g["contour"], 400, 400, params)


def _two_blob_scene():
    sal = np.clip(_disk(400, 400, 110, 110, 40) + _disk(400, 400, 290, 290, 40), 0, 1)
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    params = SeededParams(
        min_area=400,
        max_area=80000,
        max_area_frac=0.5,
        min_circularity=0.10,
        max_bbox_side_frac=0.85,
        max_aspect_ratio=8.0,
        morph_kernel_size=3,
        saliency_threshold=0.25,
        adaptive_k=0.15,
        crop_radius=90,
        peak_max=8,
    )
    return sal, image, params


def test_click_and_batch_share_one_algorithm():
    """El click en un pico debe dar el mismo cuerpo que el lote en ese pico."""
    sal, image, params = _two_blob_scene()
    provider = crop_saliency_provider(sal)

    batch = propose_objects(sal, image, params)
    seeds = peak_seeds(sal, params)
    assert len(seeds) == len(batch) == 2

    for seed in seeds:
        clicked = resolve_seeded_object(image, seed, params, provider)
        assert clicked is not None
        match = min(batch, key=lambda g: abs(g["bbox"][0] - clicked["bbox"][0]))
        assert match["bbox"] == clicked["bbox"]


def test_resolve_seeded_object_clips_seed_outside_frame():
    sal, image, params = _two_blob_scene()
    provider = crop_saliency_provider(sal)
    assert resolve_seeded_object(image, (-50, -50), params, provider) is None


def test_resolve_seeded_object_maps_to_global_coords():
    """El contorno vuelve en coordenadas de la imagen, no del recorte."""
    sal, image, params = _two_blob_scene()
    grain = resolve_seeded_object(image, (290, 290), params, crop_saliency_provider(sal))
    assert grain is not None
    bx, by, bw, bh = grain["bbox"]
    assert bx > 200 and by > 200
    assert bx <= 290 <= bx + bw
    assert 0.0 < grain["saliency"] <= 1.0


def test_resolve_runs_at_most_two_inferences():
    sal, image, params = _two_blob_scene()
    boxes = []

    def provider(img, box):
        boxes.append(box)
        x1, y1, x2, y2 = box
        return sal[y1:y2, x1:x2]

    assert resolve_seeded_object(image, (110, 110), params, provider) is not None
    assert len(boxes) <= 2


def test_enlarge_that_loses_the_object_keeps_first_crop():
    """Ampliar al marco no debe tirar el recorte que ya separó (T_officinale)."""
    sal = np.zeros((400, 400), np.float32)
    sal[80:320, 180:220] = 0.95
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    boxes = []

    def provider(_img, box):
        boxes.append(box)
        x1, y1, x2, y2 = box
        if (x2 - x1) >= 400 and (y2 - y1) >= 400:
            return np.zeros((400, 400), np.float32)
        return sal[y1:y2, x1:x2]

    grain = resolve_seeded_object(
        image, (200, 200), SeededParams(crop_radius=80, min_area=50), provider
    )
    assert grain is not None
    assert len(boxes) <= 2
    bx, by, bw, bh = grain["bbox"]
    assert bx <= 200 <= bx + bw
    assert by <= 200 <= by + bh


def test_saliency_provider_is_used_per_crop():
    """El provider recibe la caja del recorte: así el lote puede correr la red."""
    sal, image, params = _two_blob_scene()
    boxes = []

    def provider(img, box):
        boxes.append(box)
        x1, y1, x2, y2 = box
        return sal[y1:y2, x1:x2]

    resolve_seeded_object(image, (110, 110), params, provider)
    assert boxes
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 110 <= x2 and y1 <= 110 <= y2


def test_contour_touches_border_detects_truncation():
    full = np.array([[0, 0], [99, 0], [99, 99], [0, 99]], dtype=np.int32)
    inner = np.array([[20, 20], [70, 20], [70, 70], [20, 70]], dtype=np.int32)
    assert contour_touches_border(full, 100, 100) is True
    assert contour_touches_border(inner, 100, 100) is False


def test_demonstration_accepts_body_that_batch_rejects_for_size():
    """El click no puede heredar max_area: si no, nunca se enseña un grano más grande."""
    sal = _disk(900, 900, 450, 450, 220)
    image = np.zeros((900, 900, 3), dtype=np.uint8)
    batch = SeededParams(
        min_area=80000,
        max_area=20000,
        max_area_frac=0.05,
        min_circularity=0.50,
        max_bbox_side_frac=0.20,
        max_aspect_ratio=1.2,
        crop_radius=120,
        saliency_threshold=0.25,
        adaptive_k=0.15,
    )
    provider = crop_saliency_provider(sal)
    assert resolve_seeded_object(image, (450, 450), batch, provider) is None

    demo = batch.for_demonstration(900, 900)
    grain = resolve_seeded_object(image, (450, 450), demo, provider)
    assert grain is not None
    area = abs(__import__("cv2").contourArea(grain["contour"].reshape(-1, 1, 2)))
    assert area > 20000


def test_for_demonstration_opens_size_caps_and_keeps_saliency():
    p = SeededParams(min_area=80000, max_area=50000, max_area_frac=0.1,
                     min_circularity=0.4, crop_radius=120,
                     saliency_threshold=0.41, adaptive_k=0.33)
    d = p.for_demonstration(1942, 2590)
    assert d.min_area <= 50
    assert d.max_area >= 1942 * 2590
    assert d.max_area_frac == 1.0
    assert d.max_bbox_side_frac == 1.0
    assert d.crop_radius == 120
    assert d.max_crop_fill == p.max_crop_fill
    assert d.split_touching is False
    assert d.demonstration is True
    assert abs(d.saliency_threshold - 0.41) < 1e-9
    assert abs(d.adaptive_k - 0.33) < 1e-9


def test_fill_holes_at_origin_does_not_become_the_frame():
    """Incl: 5 Mpx: el objeto tocaba (0,0) y fill_holes invertía el fondo."""
    mask = np.zeros((80, 100), np.uint8)
    mask[0:40, 0:50] = 255
    filled = fill_holes(mask)
    assert int(np.count_nonzero(filled)) == int(np.count_nonzero(mask))
    assert int(np.count_nonzero(filled)) < filled.size


def test_covers_full_frame_is_the_image():
    frame = np.array([[0, 0], [99, 0], [99, 79], [0, 79]], dtype=np.int32)
    body = np.array([[5, 5], [40, 5], [40, 30], [5, 30]], dtype=np.int32)
    assert covers_full_frame(frame, 80, 100)
    assert not covers_full_frame(body, 80, 100)


def test_frame_rectangle_is_not_an_object():
    """Incl: 5 Mpx era el encuadre, no el grano."""
    frame = np.array([[0, 0], [399, 0], [399, 399], [0, 399]], dtype=np.int32)
    body = np.array([[40, 60], [360, 60], [360, 280], [40, 280]], dtype=np.int32)
    assert not is_valid_seeded_contour(frame, 400, 400, 200, 200, max_ratio=0.70)
    assert is_valid_seeded_contour(body, 400, 400, 200, 200, max_ratio=0.70)


def test_clahe_raises_luminance_contrast():
    img = np.full((120, 160, 3), 140, np.uint8)
    img[20:100, 30:90] = 155
    before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].std()
    after = cv2.cvtColor(SalientObjectDetector.enhance(img), cv2.COLOR_BGR2LAB)[:, :, 0].std()
    assert after > before


def test_compare_finds_gap_on_a_disk():
    sal = _disk(200, 200, 100, 100, 35)
    cmp = compare_object_background(sal, (100, 100))
    assert cmp.separated
    assert cmp.object_sal > cmp.background_sal
    assert cmp.background_sal <= cmp.threshold <= cmp.object_sal


def test_compare_uniform_map_is_not_separated():
    sal = np.full((180, 180), 0.4, np.float32)
    cmp = compare_object_background(sal, (90, 90))
    assert not cmp.separated


def test_seeded_contour_rejects_when_no_separation():
    sal = np.full((180, 180), 0.4, np.float32)
    assert u2net_seeded_contour(sal, 90, 90, SeededParams()) is None


def test_resolver_and_extractor_share_one_threshold():
    sal = _disk(240, 240, 120, 120, 40)
    cmp = compare_object_background(sal, (120, 120))
    a = u2net_seeded_contour(sal, 120, 120, SeededParams(), separation=cmp)
    b = u2net_seeded_contour(sal, 120, 120, SeededParams())
    assert a is not None and b is not None
    assert np.array_equal(a, b)


def test_interior_high_saliency_closes_local_crop():
    """Textura interior (sal=1 en el recorte): no es fallo, es el objeto."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    def provider(_img, box):
        x1, y1, x2, y2 = box
        return np.ones((y2 - y1, x2 - x1), np.float32)

    params = SeededParams(crop_radius=80).for_demonstration(400, 400)
    grain = resolve_seeded_object(image, (200, 200), params, provider)
    assert grain is not None
    bx, by, bw, bh = grain["bbox"]
    assert bx <= 200 <= bx + bw
    assert by <= 200 <= by + bh


def test_stroke_uses_one_crop_for_all_points():
    sal = _disk(400, 400, 200, 200, 80)
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    boxes = []

    def provider(_img, box):
        boxes.append(box)
        x1, y1, x2, y2 = box
        return sal[y1:y2, x1:x2]

    stroke = [(180, 200), (200, 200), (220, 205)]
    grain = resolve_seeded_object(
        image, stroke[0], SeededParams(crop_radius=90, min_area=50),
        provider, seeds=stroke,
    )
    assert grain is not None
    assert len(boxes) <= 2
    x1, y1, x2, y2 = boxes[0]
    assert x1 <= 180 and x2 >= 220


def test_full_frame_saliency_is_rejected():
    """U²-Net marcando todo el recorte no debe devolver el marco."""
    sal = np.ones((400, 400), np.float32)
    cnt = u2net_seeded_contour(sal, 200, 200, SeededParams(max_crop_fill=0.70))
    assert cnt is None


def test_resolver_rejects_full_frame_as_object():
    image = np.zeros((80, 100, 3), dtype=np.uint8)

    def provider(_img, box):
        x1, y1, x2, y2 = box
        return np.ones((y2 - y1, x2 - x1), np.float32)

    params = SeededParams(crop_radius=200).for_demonstration(80, 100)
    assert resolve_seeded_object(image, (50, 40), params, provider) is None


def test_seeded_params_from_gui_config_keys():
    cfg = {
        "min_area": 1234,
        "max_area": 99999,
        "max_area_frac": 0.4,
        "min_circularity": 0.21,
        "max_bbox_side_frac": 0.7,
        "max_aspect_ratio": 6.5,
        "morph_kernel_size": 5,
        "saliency_threshold": 0.41,
        "adaptive_k": 0.33,
        "crop_radius": 420,
    }
    p = SeededParams.from_config(cfg)
    assert p.min_area == 1234
    assert p.crop_radius == 420
    assert abs(p.adaptive_k - 0.33) < 1e-9
    assert abs(p.saliency_threshold - 0.41) < 1e-9
