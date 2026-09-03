"""
Shared grain crop / mask utilities — training (SegmentedGrainDataset) and E2E inference.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.grain_detection.detected_grain import DetectedGrain


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-6)


def _grain_crop_bbox(grain: Union[Dict, DetectedGrain]) -> Tuple[int, int, int, int]:
    if isinstance(grain, DetectedGrain):
        return grain.crop_bbox()
    tight = grain.get("classification_bbox")
    if tight is not None:
        return tuple(tight)
    return tuple(grain["bbox"])


def extract_grain_crop(
    image: np.ndarray,
    grain: Union[Dict, DetectedGrain],
    *,
    crop_padding: float = 0.2,
    border_value: Tuple[int, int, int] = (128, 128, 128),
) -> Optional[np.ndarray]:
    """Square crop centered on grain bbox with proportional padding (matches training)."""
    h_img, w_img = image.shape[:2]
    x, y, w, h = _grain_crop_bbox(grain)

    pad_x = int(w * crop_padding)
    pad_y = int(h * crop_padding)
    side = max(w + 2 * pad_x, h + 2 * pad_y)

    cx = x + w // 2
    cy = y + h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    valid_x1 = max(0, x1)
    valid_y1 = max(0, y1)
    valid_x2 = min(w_img, x2)
    valid_y2 = min(h_img, y2)

    crop = image[valid_y1:valid_y2, valid_x1:valid_x2]
    if crop.size == 0:
        return None

    if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
        top = max(0, -y1)
        bottom = max(0, y2 - h_img)
        left = max(0, -x1)
        right = max(0, x2 - w_img)
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=border_value)

    return crop


def generate_grain_mask(
    grain: Union[Dict, DetectedGrain],
    image_shape: Tuple[int, int],
    *,
    crop_padding: float = 0.2,
) -> np.ndarray:
    """Binary mask for the crop region from grain contour (matches SegmentedGrainDataset)."""
    h_img, w_img = image_shape
    x, y, w, h = _grain_crop_bbox(grain)
    if isinstance(grain, DetectedGrain):
        contour = grain.contour
    else:
        contour = grain.get("contour")

    if contour is None or len(contour) < 3:
        side = max(w, h)
        return np.full((side, side), 255, dtype=np.uint8)

    pad_x = int(w * crop_padding)
    pad_y = int(h * crop_padding)
    side = max(w + 2 * pad_x, h + 2 * pad_y)

    cx = x + w // 2
    cy = y + h // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
    pts = np.asarray(contour).reshape(-1, 1, 2).astype(np.int32)
    cv2.drawContours(mask_full, [pts], -1, 255, -1)

    valid_x1 = max(0, x1)
    valid_y1 = max(0, y1)
    valid_x2 = min(w_img, x2)
    valid_y2 = min(h_img, y2)
    mask_crop = mask_full[valid_y1:valid_y2, valid_x1:valid_x2]

    if x1 < 0 or y1 < 0 or x2 > w_img or y2 > h_img:
        top = max(0, -y1)
        bottom = max(0, y2 - h_img)
        left = max(0, -x1)
        right = max(0, x2 - w_img)
        mask_crop = cv2.copyMakeBorder(mask_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    return mask_crop


def apply_mask_bg(crop: np.ndarray, mask_crop: np.ndarray, mask_bg_mode: str) -> np.ndarray:
    """Apply background fill outside grain mask (BGR uint8)."""
    bg_pixels = mask_crop == 0
    if not np.any(bg_pixels):
        return crop

    crop = crop.copy()
    if mask_bg_mode == "imagenet_neutral":
        from src.utils.dataset_norm import imagenet_neutral_bgr_uint8

        b, g, r = imagenet_neutral_bgr_uint8()
        crop[bg_pixels] = (b, g, r)
    elif mask_bg_mode == "random":
        bg_color = np.random.randint(0, 256, size=(1, 1, 3), dtype=np.uint8)
        crop[bg_pixels] = bg_color[0, 0]
    elif mask_bg_mode == "gray":
        crop[bg_pixels] = 128
    elif mask_bg_mode == "black":
        crop[bg_pixels] = 0
    return crop


def prepare_grain_crop_and_mask(
    image: np.ndarray,
    grain: Union[Dict, DetectedGrain],
    *,
    crop_padding: float = 0.2,
    use_mask: bool = True,
    mask_bg_mode: str = "imagenet_neutral",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract crop + optional masked crop using the same rules as training."""
    crop = extract_grain_crop(image, grain, crop_padding=crop_padding)
    if crop is None:
        return None, None

    if not use_mask:
        return crop, None

    contour = grain.contour if isinstance(grain, DetectedGrain) else grain.get("contour")
    if contour is None or len(contour) < 3:
        mask_crop = np.full(crop.shape[:2], 255, dtype=np.uint8)
        return crop, mask_crop

    mask_crop = generate_grain_mask(grain, image.shape[:2], crop_padding=crop_padding)
    crop = apply_mask_bg(crop, mask_crop, mask_bg_mode)
    return crop, mask_crop


def nms_detected_grains(
    grains: List[DetectedGrain],
    *,
    iou_threshold: float = 0.5,
) -> List[DetectedGrain]:
    """Suppress overlapping detections, keeping highest saliency."""
    if len(grains) <= 1:
        return grains

    ordered = sorted(grains, key=lambda g: float(g.saliency_prob), reverse=True)
    kept: List[DetectedGrain] = []
    for grain in ordered:
        if all(bbox_iou(grain.bbox, k.bbox) < iou_threshold for k in kept):
            kept.append(grain)

    for i, grain in enumerate(kept):
        grain.index = i
    return kept
