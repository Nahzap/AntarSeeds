"""Tests for morphological grain extraction filters."""

import numpy as np

from src.grain_detection.saliency_grain_extraction import extract_grains_from_saliency


def _blank_image(h=504, w=504):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_rejects_oversized_bbox():
    sal = np.zeros((504, 504), dtype=np.float32)
    # Blob que ocupa >35% del lado
    sal[10:250, 10:250] = 0.9
    grains = extract_grains_from_saliency(
        sal, _blank_image(),
        min_area=100,
        max_area=200000,
        min_circularity=0.1,
        max_bbox_side_frac=0.35,
    )
    assert len(grains) == 0


def test_keeps_small_round_grain():
    sal = np.zeros((504, 504), dtype=np.float32)
    cy, cx, r = 252, 252, 40
    y, x = np.ogrid[:504, :504]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
    sal[mask] = 0.95
    grains = extract_grains_from_saliency(
        sal, _blank_image(),
        min_area=500,
        max_area=80000,
        min_circularity=0.4,
        max_bbox_side_frac=0.35,
    )
    assert len(grains) == 1
