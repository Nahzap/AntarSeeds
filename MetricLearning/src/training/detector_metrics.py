"""Detection metrics: SOD (Fm, MAE) + instance-level Det-P/R/F1."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from src.grain_detection.saliency_grain_extraction import bbox_iou, extract_grains_from_saliency

_DummyImage = np.zeros((1, 1, 3), dtype=np.uint8)


def mask_iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    pb = pred >= threshold
    gb = gt >= threshold
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)


def sod_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute error between saliency maps in [0, 1] (SOD benchmark standard)."""
    pred = np.clip(pred.astype(np.float64), 0.0, 1.0)
    gt = np.clip(gt.astype(np.float64), 0.0, 1.0)
    return float(np.mean(np.abs(pred - gt)))


def sod_f_measure(
    pred: np.ndarray,
    gt: np.ndarray,
    beta: float = 0.3,
    n_thresholds: int = 51,
) -> float:
    """
    Maximum F-beta over threshold sweep (Fm) — Qin et al. U²-Net SOD protocol.
    beta < 1 weights precision more (default 0.3 in SOD literature).
    """
    pred = np.clip(pred.astype(np.float64), 0.0, 1.0)
    gt = np.clip(gt.astype(np.float64), 0.0, 1.0)
    gt_bin = gt >= 0.5
    gt_sum = gt_bin.sum()
    if gt_sum == 0 and pred.max() < 1e-8:
        return 1.0
    if gt_sum == 0:
        return 0.0

    beta2 = beta * beta
    best = 0.0
    for t in np.linspace(0.0, 1.0, n_thresholds):
        pb = pred >= t
        inter = np.logical_and(pb, gt_bin).sum()
        if inter == 0:
            continue
        precision = inter / max(pb.sum(), 1)
        recall = inter / max(gt_sum, 1)
        denom = beta2 * precision + recall
        if denom > 0:
            f = (1.0 + beta2) * precision * recall / denom
            if f > best:
                best = f
    return float(best)


@torch.inference_mode()
def batch_sod_metrics_torch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    beta: float = 0.3,
    n_thresholds: int = 51,
) -> Dict[str, torch.Tensor]:
    """
    GPU-batched SOD metrics. pred/gt shape (N, 1, H, W) in [0, 1].
    Returns per-image tensors (N,).
    """
    pred = pred.float().clamp(0.0, 1.0)
    gt = gt.float().clamp(0.0, 1.0)
    gt_bin = gt >= 0.5

    inter_union = _batch_binary_iou(pred, gt, threshold=0.5)
    mae = (pred - gt).abs().mean(dim=(1, 2, 3))

    beta2 = beta * beta
    gt_sum = gt_bin.flatten(1).sum(dim=1).clamp(min=1)
    best_f = torch.zeros(pred.shape[0], device=pred.device)
    thresholds = torch.linspace(0.0, 1.0, n_thresholds, device=pred.device)
    pred_flat = pred.flatten(1)
    gt_flat = gt_bin.flatten(1).float()
    for t in thresholds:
        pb = pred_flat >= t
        inter = (pb & gt_bin.flatten(1)).sum(dim=1).float()
        precision = inter / pb.sum(dim=1).clamp(min=1).float()
        recall = inter / gt_sum.float()
        denom = beta2 * precision + recall
        f = torch.where(
            denom > 0,
            (1.0 + beta2) * precision * recall / denom,
            torch.zeros_like(denom),
        )
        best_f = torch.maximum(best_f, f)

    empty_gt = gt_bin.flatten(1).sum(dim=1) == 0
    flat_pred = pred.flatten(1).amax(dim=1) < 1e-8
    best_f = torch.where(empty_gt & flat_pred, torch.ones_like(best_f), best_f)
    best_f = torch.where(empty_gt & ~flat_pred, torch.zeros_like(best_f), best_f)

    return {"mask_iou": inter_union, "Fm": best_f, "sod_mae": mae}


def _batch_binary_iou(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> torch.Tensor:
    pb = pred >= threshold
    gb = gt >= threshold
    inter = (pb & gb).flatten(1).sum(dim=1).float()
    union = (pb | gb).flatten(1).sum(dim=1).float()
    return torch.where(union > 0, inter / union, torch.ones_like(inter))





def _grains_to_bboxes(grains) -> List[Tuple[int, int, int, int]]:

    return [g.bbox for g in grains]





def det_precision_recall_f1(

    pred_bboxes: List[Tuple[int, int, int, int]],

    gt_bboxes: List[Tuple[int, int, int, int]],

    iou_threshold: float = 0.5,

) -> Dict[str, float]:

    if not gt_bboxes and not pred_bboxes:

        return {"det_precision": 1.0, "det_recall": 1.0, "det_f1": 1.0}

    if not gt_bboxes:

        return {"det_precision": 0.0, "det_recall": 1.0, "det_f1": 0.0}

    if not pred_bboxes:

        return {"det_precision": 1.0, "det_recall": 0.0, "det_f1": 0.0}



    matched_gt = set()

    tp = 0

    for pb in pred_bboxes:

        best_iou = 0.0

        best_j = -1

        for j, gb in enumerate(gt_bboxes):

            if j in matched_gt:

                continue

            iou = bbox_iou(pb, gb)

            if iou > best_iou:

                best_iou = iou

                best_j = j

        if best_iou >= iou_threshold and best_j >= 0:

            tp += 1

            matched_gt.add(best_j)



    precision = tp / max(len(pred_bboxes), 1)

    recall = tp / max(len(gt_bboxes), 1)

    if precision + recall == 0:

        f1 = 0.0

    else:

        f1 = 2 * precision * recall / (precision + recall)

    return {

        "det_precision": float(precision),

        "det_recall": float(recall),

        "det_f1": float(f1),

    }





def evaluate_prediction_map(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    image_bgr: np.ndarray | None = None,
    *,
    grain_cfg: dict,
) -> Dict[str, float]:
    """SOD + instance metrics from a saliency probability map."""
    miou = mask_iou(pred_map, gt_map)
    fm = sod_f_measure(pred_map, gt_map)
    smae = sod_mae(pred_map, gt_map)
    from src.grain_detection.saliency_grain_extraction import grain_extraction_kwargs

    dummy = image_bgr if image_bgr is not None else _DummyImage
    pred_kw = grain_extraction_kwargs(grain_cfg, for_gt=False)
    gt_kw = grain_extraction_kwargs(grain_cfg, for_gt=True)
    grains_pred = extract_grains_from_saliency(pred_map, dummy, **pred_kw)
    grains_gt = extract_grains_from_saliency(gt_map, dummy, **gt_kw)

    prf = det_precision_recall_f1(

        _grains_to_bboxes(grains_pred),

        _grains_to_bboxes(grains_gt),

    )

    prf["mask_iou"] = miou

    prf["Fm"] = fm

    prf["sod_mae"] = smae

    prf["grains_mae"] = abs(len(grains_pred) - len(grains_gt))

    return prf





def aggregate_metric_lists(metric_lists: Dict[str, list]) -> Dict[str, float]:

    """Mean per-key over non-empty lists."""

    return {k: float(np.mean(v)) if v else 0.0 for k, v in metric_lists.items()}


