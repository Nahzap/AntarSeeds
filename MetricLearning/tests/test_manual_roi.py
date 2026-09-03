import numpy as np

from src.grain_detection.manual_roi import (
    build_manual_roi_mask,
    contour_bbox,
    extract_contours_from_mask,
)


def test_build_manual_roi_mask_excludes_region():
    include = [[(10, 10), (90, 10), (90, 90), (10, 90)]]
    exclude = [[(40, 40), (60, 40), (60, 60), (40, 60)]]
    mask = build_manual_roi_mask((100, 100), include, exclude)

    # Punto dentro de inclusión pero fuera de exclusión
    assert mask[20, 20] == 255
    # Punto dentro de exclusión debe quedar fuera
    assert mask[50, 50] == 0


def test_extract_contours_from_mask_multi_component():
    mask = np.zeros((120, 120), dtype=np.uint8)
    mask[10:30, 10:30] = 255
    mask[60:90, 70:100] = 255

    contours = extract_contours_from_mask(mask, min_area=5.0)
    assert len(contours) == 2

    # Contorno mayor primero por área descendente
    b0 = contour_bbox(contours[0])
    b1 = contour_bbox(contours[1])
    area0 = b0[2] * b0[3]
    area1 = b1[2] * b1[3]
    assert area0 >= area1


def test_unbounded_points_polygon():
    # Polígono con muchos puntos (simula edición sin límite práctico)
    points = [(x, 20 + (x % 5)) for x in range(5, 95)]
    # Cerrar forma de manera simple
    points.extend([(95, 95), (5, 95)])

    mask = build_manual_roi_mask((120, 120), [points], [])
    contours = extract_contours_from_mask(mask, min_area=5.0)
    assert len(contours) >= 1
