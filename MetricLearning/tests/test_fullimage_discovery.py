"""Full-image inference discovery mode tests (model-based detector)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.grain_detection.detected_grain import DetectedGrain
from src.grain_detection.saliency_grain_extraction import bbox_iou, nms_merge_grains


class TestNmsMerge:
    def test_merge_overlapping_boxes(self):
        g1 = DetectedGrain(
            index=0, bbox=(10, 10, 40, 40), area=100, saliency_prob=0.9,
            centroid=(30, 30), contour=None,
        )
        g2 = DetectedGrain(
            index=1, bbox=(12, 12, 38, 38), area=90, saliency_prob=0.7,
            centroid=(31, 31), contour=None,
        )
        g3 = DetectedGrain(
            index=2, bbox=(200, 200, 30, 30), area=80, saliency_prob=0.6,
            centroid=(215, 215), contour=None,
        )
        merged = nms_merge_grains([g1, g2, g3], iou_threshold=0.3)
        assert len(merged) == 2
        assert merged[0].saliency_prob == 0.9


class TestBboxIou:
    def test_iou_identical(self):
        assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_iou_disjoint(self):
        assert bbox_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
