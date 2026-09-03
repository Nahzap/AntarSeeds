"""
Coarse-to-fine grain detection: global scan -> per-candidate verification.

Stage 1 (coarse): fast single-pass saliency on the full frame.
Stage 2 (discover): sliding-window + saliency peaks for missed regions.
Stage 3 (verify): zoom each candidate; retain coarse hits if saliency confirms.
Classification crops always use the tight bbox (classification_bbox), matching .seg training.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.grain_detection.detected_grain import DetectedGrain
from src.grain_detection.saliency_grain_extraction import bbox_iou, nms_merge_grains

logger = logging.getLogger(__name__)


def crop_with_padding(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding_frac: float,
) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    if image is None or image.size == 0:
        return None, (0, 0)
    x, y, w, h = bbox
    pad_x = max(4, int(w * padding_frac))
    pad_y = max(4, int(h * padding_frac))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None, (0, 0)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def tag_classification_bbox(grain: DetectedGrain) -> DetectedGrain:
    """Preserve tight bbox for classifier crops (training-aligned)."""
    grain.classification_bbox = grain.bbox
    return grain


def reject_oversized_detection(
    grain: DetectedGrain,
    image_shape: Tuple[int, int],
    *,
    max_area_frac: Optional[float] = None,
    max_side_frac: Optional[float] = None,
) -> bool:
    """Drop detections that cover too much of the frame (config-driven artefact filter)."""
    if max_area_frac is None and max_side_frac is None:
        return False
    h_img, w_img = image_shape[:2]
    _, _, w, h = grain.bbox
    img_side = min(h_img, w_img)
    if img_side <= 0:
        return False
    if max_area_frac is not None and w * h > max_area_frac * img_side * img_side:
        return True
    if max_side_frac is not None and max(w, h) > max_side_frac * img_side:
        return True
    return False


def expand_grain_bbox(
    grain: DetectedGrain,
    expand_frac: float,
    image_shape: Tuple[int, int],
) -> DetectedGrain:
    """Optional display-only bbox expansion; classification uses classification_bbox."""
    if expand_frac <= 0:
        return grain
    h_img, w_img = image_shape[:2]
    x, y, w, h = grain.bbox
    pad_x = int(w * expand_frac)
    pad_y = int(h * expand_frac)
    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y)
    nx2 = min(w_img, x + w + pad_x)
    ny2 = min(h_img, y + h + pad_y)
    nw, nh = nx2 - nx, ny2 - ny
    return DetectedGrain(
        index=grain.index,
        bbox=(nx, ny, nw, nh),
        area=float(nw * nh),
        saliency_prob=grain.saliency_prob,
        centroid=(nx + nw // 2, ny + nh // 2),
        contour=grain.contour,
        classification_bbox=grain.classification_bbox or grain.bbox,
    )


def shift_grain(grain: DetectedGrain, offset_x: int, offset_y: int) -> DetectedGrain:
    bx, by, bw, bh = grain.bbox
    cx, cy = grain.centroid
    contour = grain.contour
    if contour is not None:
        contour = contour.copy()
        contour[:, 0] += offset_x
        contour[:, 1] += offset_y
    return DetectedGrain(
        index=grain.index,
        bbox=(bx + offset_x, by + offset_y, bw, bh),
        area=grain.area,
        saliency_prob=grain.saliency_prob,
        centroid=(cx + offset_x, cy + offset_y),
        contour=contour,
        classification_bbox=(
            (
                grain.classification_bbox[0] + offset_x,
                grain.classification_bbox[1] + offset_y,
                grain.classification_bbox[2],
                grain.classification_bbox[3],
            )
            if grain.classification_bbox is not None
            else None
        ),
    )


def is_viable_candidate(
    grain: DetectedGrain,
    *,
    min_area: int = 1500,
    min_side: int = 32,
) -> bool:
    _, _, w, h = grain.bbox
    if w < min_side or h < min_side:
        return False
    return w * h >= min_area


def merge_candidate_lists(
    *lists: List[DetectedGrain],
    iou_threshold: float = 0.30,
    min_area: int = 1500,
    min_side: int = 32,
) -> List[DetectedGrain]:
    merged: List[DetectedGrain] = []
    for items in lists:
        merged.extend(
            g for g in items if is_viable_candidate(g, min_area=min_area, min_side=min_side)
        )
    return nms_merge_grains(merged, iou_threshold=iou_threshold)


def supplement_candidates_from_saliency(
    saliency: np.ndarray,
    image: np.ndarray,
    existing: List[DetectedGrain],
    *,
    extract_fn,
    min_saliency: float,
    min_overlap_iou: float = 0.15,
) -> List[DetectedGrain]:
    grains = extract_fn(saliency, image)
    supplements: List[DetectedGrain] = []
    for grain in grains:
        if grain.saliency_prob < min_saliency:
            continue
        if any(bbox_iou(grain.bbox, base.bbox) >= min_overlap_iou for base in existing):
            continue
        supplements.append(grain)
    return supplements


def discover_peak_candidates(
    saliency: np.ndarray,
    existing: List[DetectedGrain],
    *,
    min_peak: float = 0.38,
    min_area: int = 800,
    min_side: int = 28,
    min_overlap_iou: float = 0.12,
    morph_kernel_size: int = 5,
) -> List[DetectedGrain]:
    """Propose candidates from high-saliency blobs contour extraction may miss."""
    binary = (saliency >= min_peak).astype(np.uint8) * 255
    k = max(3, int(morph_kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    peaks: List[DetectedGrain] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_side or h < min_side:
            continue
        if any(bbox_iou((x, y, w, h), e.bbox) >= min_overlap_iou for e in existing):
            continue
        mask = np.zeros(saliency.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        prob = float(np.mean(saliency[mask > 0]))
        if prob < min_peak:
            continue
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"]) if M["m00"] > 0 else x + w // 2
        cy = int(M["m01"] / M["m00"]) if M["m00"] > 0 else y + h // 2
        peaks.append(
            DetectedGrain(
                index=len(peaks),
                bbox=(x, y, w, h),
                area=float(area),
                saliency_prob=prob,
                centroid=(cx, cy),
                contour=contour.reshape(-1, 2),
            )
        )
    return peaks


def verify_candidate(
    full_image: np.ndarray,
    candidate: DetectedGrain,
    *,
    predict_and_extract_fn,
    crop_padding: float,
    min_saliency: float,
    min_iou: float,
    saliency_fallback_min: float,
) -> Optional[DetectedGrain]:
    crop, (ox, oy) = crop_with_padding(full_image, candidate.bbox, crop_padding)
    if crop is None or min(crop.shape[:2]) < 48:
        return None

    saliency, grains = predict_and_extract_fn(crop)

    bx, by, bw, bh = candidate.bbox
    rx1 = max(0, bx - ox)
    ry1 = max(0, by - oy)
    rx2 = min(crop.shape[1], rx1 + bw)
    ry2 = min(crop.shape[0], ry1 + bh)
    peak = candidate.saliency_prob
    if rx2 > rx1 and ry2 > ry1:
        peak = float(np.max(saliency[ry1:ry2, rx1:rx2]))

    if peak < min_saliency:
        return None

    best: Optional[DetectedGrain] = None
    best_score = -1.0
    for grain in grains:
        shifted = shift_grain(grain, ox, oy)
        iou = bbox_iou(shifted.bbox, candidate.bbox)
        if iou < min_iou or shifted.saliency_prob < min_saliency:
            continue
        score = shifted.saliency_prob * (0.5 + 0.5 * iou)
        if score > best_score:
            best_score = score
            best = shifted

    if best is not None:
        return best

    if peak >= saliency_fallback_min:
        return DetectedGrain(
            index=candidate.index,
            bbox=candidate.bbox,
            area=candidate.area,
            saliency_prob=peak,
            centroid=candidate.centroid,
            contour=candidate.contour,
        )
    return None


def run_coarse_fine_detection(
    image: np.ndarray,
    *,
    coarse_pass_fn,
    sliding_saliency_fn,
    extract_fn,
    extract_fn_sensitive,
    verify_pass_fn,
    needs_sliding_fn,
    supplement_enabled: bool,
    supplement_min_saliency: float,
    supplement_min_peak: float,
    verify_crop_padding: float,
    verify_min_saliency: float,
    verify_min_iou: float,
    coarse_retain_min_saliency: float,
    supplement_k_boost: Optional[float] = None,
    bbox_expand_frac: float = 0.0,
    min_candidate_area: int = 1500,
    supplement_merge_iou: float = 0.25,
    merge_iou: float = 0.28,
    nms_iou: float = 0.32,
    peak_min_area: Optional[int] = None,
    peak_min_side: int = 28,
    peak_morph_kernel_size: int = 5,
    reject_max_area_frac: Optional[float] = None,
    reject_max_side_frac: Optional[float] = None,
) -> Tuple[np.ndarray, List[DetectedGrain], dict]:
    h, w = image.shape[:2]
    img_pixels = h * w
    peak_area = peak_min_area if peak_min_area is not None else max(400, min_candidate_area // 2)
    reject_area = reject_max_area_frac if reject_max_area_frac is not None else None
    reject_side = reject_max_side_frac

    sal_coarse = coarse_pass_fn(image)
    coarse_candidates = extract_fn(sal_coarse, image)

    saliency = sal_coarse
    supplement: List[DetectedGrain] = []
    if supplement_enabled and needs_sliding_fn(h, w):
        sal_sw = sliding_saliency_fn(image)
        saliency = np.maximum(sal_coarse, sal_sw)
        supplement = supplement_candidates_from_saliency(
            sal_sw,
            image,
            coarse_candidates,
            extract_fn=lambda s, img: extract_fn_sensitive(
                s, img, k_override=supplement_k_boost
            ),
            min_saliency=supplement_min_saliency,
        )
        peak_hits = discover_peak_candidates(
            saliency,
            coarse_candidates + supplement,
            min_peak=supplement_min_peak,
            min_area=peak_area,
            min_side=peak_min_side,
            morph_kernel_size=peak_morph_kernel_size,
        )
        supplement = merge_candidate_lists(
            supplement,
            peak_hits,
            iou_threshold=supplement_merge_iou,
            min_area=peak_area,
            min_side=peak_min_side,
        )

    tagged: List[Tuple[DetectedGrain, str]] = [(g, "coarse") for g in coarse_candidates]
    for g in supplement:
        tagged.append((g, "supplement"))

    merged = merge_candidate_lists(
        coarse_candidates,
        supplement,
        iou_threshold=merge_iou,
        min_area=min_candidate_area,
    )
    source_by_bbox = {}
    for g, src in tagged:
        source_by_bbox[g.bbox] = src

    verified: List[DetectedGrain] = []
    rejected = 0
    retained = 0
    for cand in merged:
        src = source_by_bbox.get(cand.bbox, "supplement")
        fallback_min = (
            coarse_retain_min_saliency if src == "coarse" else supplement_min_saliency
        )
        refined = verify_candidate(
            image,
            cand,
            predict_and_extract_fn=verify_pass_fn,
            crop_padding=verify_crop_padding,
            min_saliency=verify_min_saliency,
            min_iou=verify_min_iou,
            saliency_fallback_min=fallback_min,
        )
        if refined is not None:
            verified.append(refined)
        elif cand.saliency_prob >= fallback_min:
            verified.append(cand)
            retained += 1
        else:
            rejected += 1

    verified = nms_merge_grains(verified, iou_threshold=nms_iou)
    final: List[DetectedGrain] = []
    for grain in verified:
        if reject_oversized_detection(
            grain,
            (h, w),
            max_area_frac=reject_area,
            max_side_frac=reject_side,
        ):
            rejected += 1
            continue
        grain = tag_classification_bbox(grain)
        if bbox_expand_frac > 0:
            grain = expand_grain_bbox(grain, bbox_expand_frac, (h, w))
        final.append(grain)
    verified = final

    stats = {
        "coarse_candidates": len(coarse_candidates),
        "supplement_candidates": len(supplement),
        "merged_candidates": len(merged),
        "verified": len(verified),
        "rejected": rejected,
        "retained_fallback": retained,
    }
    logger.info(
        "[CoarseFine] coarse=%d supplement=%d verified=%d retained=%d rejected=%d",
        stats["coarse_candidates"],
        stats["supplement_candidates"],
        stats["verified"],
        stats["retained_fallback"],
        stats["rejected"],
    )
    return saliency, verified, stats
