"""
Config-driven detection quality gates for full-image inference.

All thresholds come from grain_detection in config.yaml — no hardcoded policy.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.grain_detection.detected_grain import DetectedGrain


def resolve_quality_config(grain_cfg: Optional[dict]) -> Dict:
    """Read optional quality-gate keys from grain_detection config."""
    cfg = grain_cfg or {}
    return {
        "min_visible_bbox_frac": cfg.get("min_visible_bbox_frac"),
        "reject_edge_partial": bool(cfg.get("reject_edge_partial", False)),
        "edge_margin_frac": float(cfg.get("edge_margin_frac", 0.008)),
        "min_edge_bbox_side_frac": float(cfg.get("min_edge_bbox_side_frac", 0.06)),
        "min_centroid_edge_inset_frac": float(cfg.get("min_centroid_edge_inset_frac", 0.22)),
        "split_multi_peak_enabled": bool(cfg.get("split_multi_peak_enabled", False)),
        "split_min_area_frac": float(cfg.get("split_min_area_frac", 0.025)),
        "split_peak_min_saliency": float(cfg.get("split_peak_min_saliency", 0.35)),
        "split_peak_min_area": int(cfg.get("split_peak_min_area", 400)),
        "split_peak_morph_kernel": int(cfg.get("split_peak_morph_kernel", 3)),
        "split_peak_min_side_frac": float(cfg.get("split_peak_min_side_frac", 0.028)),
        "min_saliency_fill_frac": cfg.get("min_saliency_fill_frac"),
        "min_mask_fill_frac": cfg.get("min_mask_fill_frac"),
        "saliency_fill_threshold": float(cfg.get("saliency_fill_threshold", 0.30)),
        "reject_classify_max_area_frac": cfg.get("reject_classify_max_area_frac"),
        "reject_multi_peak_area_frac": cfg.get("reject_multi_peak_area_frac"),
        "min_classify_side_frac": cfg.get("min_classify_side_frac"),
    }


def visible_bbox_fraction(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
) -> float:
    x, y, w, h = bbox
    h_img, w_img = image_shape[:2]
    vis_w = max(0, min(x + w, w_img) - max(x, 0))
    vis_h = max(0, min(y + h, h_img) - max(y, 0))
    return (vis_w * vis_h) / max(w * h, 1)


def _edge_margin_px(image_shape: Tuple[int, int], margin_frac: float) -> int:
    h_img, w_img = image_shape[:2]
    return max(1, int(min(h_img, w_img) * margin_frac))


def touches_image_edge(
    bbox: Tuple[int, int, int, int],
    image_shape: Tuple[int, int],
    margin_frac: float,
) -> bool:
    x, y, w, h = bbox
    h_img, w_img = image_shape[:2]
    m = _edge_margin_px(image_shape, margin_frac)
    return x <= m or y <= m or (x + w) >= (w_img - m) or (y + h) >= (h_img - m)


def is_edge_partial_detection(
    grain: DetectedGrain,
    image_shape: Tuple[int, int],
    cfg: Dict,
) -> bool:
    """Reject small boxes clipped by frame borders (partial grains)."""
    if not cfg.get("reject_edge_partial"):
        return False
    bbox = grain.crop_bbox()
    x, y, w, h = bbox
    if not touches_image_edge(bbox, image_shape, cfg["edge_margin_frac"]):
        return False
    img_side = min(image_shape[:2])
    min_side = cfg["min_edge_bbox_side_frac"] * img_side
    if min(w, h) < min_side:
        return True

    inset = cfg["min_centroid_edge_inset_frac"] * min(w, h)
    cx, cy = grain.centroid
    m = _edge_margin_px(image_shape, cfg["edge_margin_frac"])
    h_img, w_img = image_shape[:2]
    if x <= m and (cx - x) < inset:
        return True
    if y <= m and (cy - y) < inset:
        return True
    if (x + w) >= (w_img - m) and ((x + w) - cx) < inset:
        return True
    if (y + h) >= (h_img - m) and ((y + h) - cy) < inset:
        return True
    return False


def saliency_fill_fraction(
    grain: DetectedGrain,
    saliency: np.ndarray,
    threshold: float,
) -> float:
    x, y, w, h = grain.crop_bbox()
    h_img, w_img = saliency.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = saliency[y1:y2, x1:x2]
    return float((roi >= threshold).mean())


def mask_fill_fraction(
    grain: DetectedGrain,
    image_shape: Tuple[int, int],
    crop_padding: float,
) -> Optional[float]:
    if grain.contour is None or len(grain.contour) < 3:
        return None
    from src.grain_detection.grain_crop_utils import generate_grain_mask

    mask = generate_grain_mask(grain, image_shape, crop_padding=crop_padding)
    if mask.size == 0:
        return 0.0
    return float((mask > 0).mean())


def count_saliency_components(
    grain: DetectedGrain,
    saliency: np.ndarray,
    cfg: Dict,
) -> int:
    """Count separated high-saliency blobs inside a detection bbox."""
    h_img, w_img = saliency.shape[:2]
    x, y, w, h = grain.crop_bbox()
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    roi = saliency[y1:y2, x1:x2]
    if roi.size == 0:
        return 0

    binary = (roi >= cfg["split_peak_min_saliency"]).astype(np.uint8) * 255
    k = max(3, int(cfg["split_peak_morph_kernel"]))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
    min_area = int(cfg["split_peak_min_area"])
    count = 0
    for label_id in range(1, n_labels):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= min_area:
            count += 1
    return count


def rejection_reason(
    grain: DetectedGrain,
    image_shape: Tuple[int, int],
    cfg: Dict,
    *,
    saliency: Optional[np.ndarray] = None,
    crop_padding: float = 0.2,
) -> Optional[str]:
    """Return rejection code if grain fails quality gates, else None."""
    bbox = grain.crop_bbox()
    min_vis = cfg.get("min_visible_bbox_frac")
    if min_vis is not None and visible_bbox_fraction(bbox, image_shape) < float(min_vis):
        return "clipped_bbox"

    if is_edge_partial_detection(grain, image_shape, cfg):
        return "edge_partial"

    min_side_frac = cfg.get("min_classify_side_frac")
    if min_side_frac is not None:
        _, _, w, h = bbox
        if min(w, h) < float(min_side_frac) * min(image_shape[:2]):
            return "too_small"

    multi_peak_frac = cfg.get("reject_multi_peak_area_frac")
    if multi_peak_frac is not None and saliency is not None:
        _, _, w, h = bbox
        h_img, w_img = image_shape[:2]
        area_frac = (w * h) / max(h_img * w_img, 1)
        if area_frac > float(multi_peak_frac):
            if count_saliency_components(grain, saliency, cfg) > 1:
                return "oversized_multi_peak"

    # Legacy key: same as reject_multi_peak_area_frac (multi-peak only, not single large grains)
    max_area_frac = cfg.get("reject_classify_max_area_frac")
    if max_area_frac is not None and saliency is not None:
        _, _, w, h = bbox
        h_img, w_img = image_shape[:2]
        area_frac = (w * h) / max(h_img * w_img, 1)
        if area_frac > float(max_area_frac) and count_saliency_components(grain, saliency, cfg) > 1:
            return "oversized_multi_peak"

    min_sal_fill = cfg.get("min_saliency_fill_frac")
    if min_sal_fill is not None and saliency is not None:
        fill = saliency_fill_fraction(
            grain, saliency, cfg["saliency_fill_threshold"]
        )
        if fill < float(min_sal_fill):
            return "low_saliency_fill"

    min_mask_fill = cfg.get("min_mask_fill_frac")
    if min_mask_fill is not None:
        mfill = mask_fill_fraction(grain, image_shape, crop_padding)
        if mfill is not None and mfill < float(min_mask_fill):
            return "low_mask_fill"

    return None


def filter_detections(
    grains: List[DetectedGrain],
    image_shape: Tuple[int, int],
    cfg: Dict,
    *,
    saliency: Optional[np.ndarray] = None,
    crop_padding: float = 0.2,
) -> Tuple[List[DetectedGrain], List[Tuple[DetectedGrain, str]]]:
    """Keep grains passing quality gates; return rejected with reasons."""
    kept: List[DetectedGrain] = []
    rejected: List[Tuple[DetectedGrain, str]] = []
    for grain in grains:
        reason = rejection_reason(
            grain, image_shape, cfg, saliency=saliency, crop_padding=crop_padding
        )
        if reason:
            rejected.append((grain, reason))
        else:
            kept.append(grain)
    for i, g in enumerate(kept):
        g.index = i
    return kept, rejected


def split_multi_peak_detection(
    grain: DetectedGrain,
    saliency: np.ndarray,
    cfg: Dict,
) -> List[DetectedGrain]:
    """Split one bbox into several when saliency shows multiple separated peaks."""
    if not cfg.get("split_multi_peak_enabled"):
        return [grain]

    h_img, w_img = saliency.shape[:2]
    x, y, w, h = grain.bbox
    area_frac = (w * h) / max(h_img * w_img, 1)
    if area_frac < cfg["split_min_area_frac"]:
        return [grain]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    roi = saliency[y1:y2, x1:x2]
    if roi.size == 0:
        return [grain]

    binary = (roi >= cfg["split_peak_min_saliency"]).astype(np.uint8) * 255
    k = max(3, int(cfg["split_peak_morph_kernel"]))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    parts: List[DetectedGrain] = []
    min_area = int(cfg["split_peak_min_area"])
    min_side_px = int(cfg["split_peak_min_side_frac"] * min(h_img, w_img))
    for label_id in range(1, n_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        px = int(stats[label_id, cv2.CC_STAT_LEFT])
        py = int(stats[label_id, cv2.CC_STAT_TOP])
        pw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        ph = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if pw < min_side_px or ph < min_side_px:
            continue
        comp_mask = labels == label_id
        prob = float(roi[comp_mask].mean()) if np.any(comp_mask) else grain.saliency_prob
        cx = int(centroids[label_id][0]) + x1
        cy = int(centroids[label_id][1]) + y1
        parts.append(
            DetectedGrain(
                index=len(parts),
                bbox=(x1 + px, y1 + py, pw, ph),
                area=float(area),
                saliency_prob=prob,
                centroid=(cx, cy),
                contour=None,
                classification_bbox=(x1 + px, y1 + py, pw, ph),
            )
        )

    return parts if len(parts) > 1 else [grain]


def split_oversized_detections(
    grains: List[DetectedGrain],
    saliency: np.ndarray,
    cfg: Dict,
) -> List[DetectedGrain]:
    """Apply multi-peak split to each detection; re-index."""
    out: List[DetectedGrain] = []
    for grain in grains:
        out.extend(split_multi_peak_detection(grain, saliency, cfg))
    for i, g in enumerate(out):
        g.index = i
    return out
