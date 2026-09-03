"""Shared evaluation utilities for ViT-dense and U²-Net detector baselines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from src.data.detection_transforms import letterbox_to_square
from src.grain_detection.saliency_grain_extraction import (
    extract_grains_from_saliency,
    grain_extraction_kwargs,
)
from src.training.detector_metrics import aggregate_metric_lists, evaluate_prediction_map
from src.utils.detector_resolution_config import resolve_detector_input_size

logger = logging.getLogger(__name__)

METRIC_KEYS = (
    "mask_iou", "Fm", "sod_mae",
    "det_precision", "det_recall", "det_f1", "grains_mae",
)

TITLE_BAR_H = 32
LEGEND_H = 36
COLOR_GT = (0, 220, 120)      # verde — contorno GT (.seg)
COLOR_PRED = (60, 60, 255)     # rojo — contorno predicción
COLOR_PRED_FILL = (80, 80, 255)


def _add_title_bar(panel: np.ndarray, title: str) -> np.ndarray:
    bar = np.full((TITLE_BAR_H, panel.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(bar, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def _draw_grain_contours(
    canvas: np.ndarray,
    grains,
    color: tuple,
    *,
    thickness: int = 2,
    fill_alpha: float = 0.0,
) -> None:
    for g in grains:
        if g.contour is None or len(g.contour) < 3:
            continue
        cnt = np.asarray(g.contour, dtype=np.int32).reshape(-1, 1, 2)
        if fill_alpha > 0:
            overlay = canvas.copy()
            cv2.drawContours(overlay, [cnt], -1, color, -1)
            cv2.addWeighted(overlay, fill_alpha, canvas, 1.0 - fill_alpha, 0, canvas)
        cv2.drawContours(canvas, [cnt], -1, color, thickness, cv2.LINE_AA)


def _panel_gt_overlay(image_bgr: np.ndarray, gt_map: np.ndarray, grain_cfg: dict) -> np.ndarray:
    """Original + contornos GT verdes desde máscara .seg."""
    panel = image_bgr.copy()
    gt_kw = grain_extraction_kwargs(grain_cfg, for_gt=True)
    gt_grains = extract_grains_from_saliency(gt_map, image_bgr, **gt_kw)
    _draw_grain_contours(panel, gt_grains, COLOR_GT, thickness=2, fill_alpha=0.15)
    return _add_title_bar(panel, "2 GT (.seg)")


def _panel_pred_heatmap(pred_map: np.ndarray) -> np.ndarray:
    """Mapa de saliency con colormap."""
    pred_u8 = (np.clip(pred_map, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(pred_u8, cv2.COLORMAP_INFERNO)
    return _add_title_bar(heat, "3 Prediccion")


def _panel_compare(
    image_bgr: np.ndarray,
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    grain_cfg: dict,
) -> np.ndarray:
    """Original + contornos GT (verde) y pred (rojo); sin cajas axis-aligned."""
    panel = image_bgr.copy()
    pred_kw = grain_extraction_kwargs(grain_cfg, for_gt=False)
    gt_kw = grain_extraction_kwargs(grain_cfg, for_gt=True)
    pred_grains = extract_grains_from_saliency(pred_map, image_bgr, **pred_kw)
    gt_grains = extract_grains_from_saliency(gt_map, image_bgr, **gt_kw)

    _draw_grain_contours(panel, gt_grains, COLOR_GT, thickness=2)
    _draw_grain_contours(panel, pred_grains, COLOR_PRED, thickness=2, fill_alpha=0.12)

    n_gt, n_pred = len(gt_grains), len(pred_grains)
    subtitle = f"GT={n_gt}  Pred={n_pred}"
    bar_panel = _add_title_bar(panel, f"4 Comparacion ({subtitle})")
    return bar_panel


def _legend_bar(width: int) -> np.ndarray:
    bar = np.full((LEGEND_H, width, 3), 30, dtype=np.uint8)
    x = 12
    cv2.rectangle(bar, (x, 12), (x + 18, 26), COLOR_GT, -1)
    cv2.putText(bar, "GT (.seg)", (x + 24, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    x += 120
    cv2.rectangle(bar, (x, 12), (x + 18, 26), COLOR_PRED, -1)
    cv2.putText(bar, "Pred (filtrada)", (x + 24, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    x += 160
    cv2.putText(
        bar,
        "Filtros: area>=400px2, circularidad>=0.32",
        (x, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (160, 160, 160),
        1,
    )
    return bar


def save_vis_grid(
    image_bgr: np.ndarray,
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    out_path: Path,
    *,
    grain_cfg: dict,
) -> None:
    """Panel 4 columnas: Original | GT | Heatmap | Comparacion con leyenda."""
    h, w = image_bgr.shape[:2]
    p1 = _add_title_bar(image_bgr.copy(), "1 Original")
    p2 = _panel_gt_overlay(image_bgr, gt_map, grain_cfg)
    p3 = _panel_pred_heatmap(pred_map)
    p4 = _panel_compare(image_bgr, pred_map, gt_map, grain_cfg)

    row = np.hstack([p1, p2, p3, p4])
    legend = _legend_bar(row.shape[1])
    panel = np.vstack([row, legend])

    max_w = 2800
    if panel.shape[1] > max_w:
        scale = max_w / panel.shape[1]
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])


def run_detector_evaluation(
    dataset,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    *,
    config: dict,
    output_dir: str | Path,
    checkpoint_label: str = "",
    vis_every_n: int = 50,
) -> dict:
    """Evaluate a detector on a FullImageDetectionDataset."""
    grain_cfg = config.get("grain_detection", {}) or {}
    input_size = resolve_detector_input_size(config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "vis_grid"

    agg = {k: [] for k in METRIC_KEYS}

    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        image = cv2.imread(sample["image_path"])
        if image is None:
            continue
        image_lb, _, _ = letterbox_to_square(image, None, input_size)
        pred_map = predict_fn(image_lb)
        gt_map = dataset[idx]["saliency_gt"][0].numpy()

        prf = evaluate_prediction_map(pred_map, gt_map, image_lb, grain_cfg=grain_cfg)
        for k in METRIC_KEYS:
            agg[k].append(prf[k])

        if vis_every_n > 0 and idx % vis_every_n == 0:
            save_vis_grid(
                image_lb, pred_map, gt_map,
                vis_dir / f"sample_{idx:04d}.jpg",
                grain_cfg=grain_cfg,
            )

    report = aggregate_metric_lists(agg)
    report["num_images"] = len(dataset.samples)
    report["protocol"] = "instance_iou0.5 + sod_fm_mae"
    if checkpoint_label:
        report["checkpoint"] = checkpoint_label
    report["test_dir"] = str(getattr(dataset, "root", ""))

    with open(out / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"Evaluation: Fm={report['Fm']:.4f} sod_mae={report['sod_mae']:.4f} "
        f"det_f1={report['det_f1']:.4f} det_p={report['det_precision']:.4f} "
        f"det_r={report['det_recall']:.4f}"
    )
    return report
