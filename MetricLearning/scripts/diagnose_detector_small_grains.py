"""Diagnose small-grain recall: model vs post-processing filters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cv2
import numpy as np
import torch
import yaml

from src.data.detection_dataset import FullImageDetectionDataset
from src.data.detection_transforms import letterbox_to_square, to_model_tensor
from src.grain_detection.saliency_grain_extraction import (
    bbox_iou,
    extract_grains_from_saliency,
    grain_extraction_kwargs,
)
from src.models.vit_dense_segmentation import ViTDenseSegmentationModel
from src.utils.detector_resolution_config import resolve_detector_input_size


def _load_model(checkpoint: str, device: torch.device) -> ViTDenseSegmentationModel:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ViTDenseSegmentationModel.from_config(ckpt.get("config", {}))
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(device).eval()
    return model


def diagnose(
    config_path: str,
    checkpoint: str,
    *,
    max_images: int = 200,
) -> dict:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    grain_cfg = config.get("grain_detection", {}) or {}
    data_cfg = config["data"]
    input_size = resolve_detector_input_size(config)
    test_dir = data_cfg.get("test_dir") or data_cfg.get("val_dir")
    ds = FullImageDetectionDataset(
        test_dir,
        annotation_root=data_cfg.get("annotation_root"),
        input_size=input_size,
        augment=False,
        max_images=max_images or None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint, device)

    strict_kw = grain_extraction_kwargs(grain_cfg, for_gt=False)
    sensitive_kw = dict(strict_kw)
    sensitive_kw.update(
        {
            "min_area": int(grain_cfg.get("min_area_sensitive", 250)),
            "min_circularity": float(grain_cfg.get("min_circularity_sensitive", 0.25)),
            "adaptive_k": float(grain_cfg.get("adaptive_k_sensitive", 0.18)),
            "morph_kernel_size": int(grain_cfg.get("morph_kernel_size_sensitive", 2)),
        }
    )
    gt_kw = grain_extraction_kwargs(grain_cfg, for_gt=True)

    gt_total = pred_strict = pred_sensitive = missed_strict = 0
    gt_areas: list[float] = []
    missed_areas: list[float] = []

    for idx in range(len(ds)):
        sample = ds.samples[idx]
        image = cv2.imread(sample["image_path"])
        if image is None:
            continue
        image_lb, _, _ = letterbox_to_square(image, None, input_size)
        gt_map = ds[idx]["saliency_gt"][0].numpy()
        with torch.inference_mode():
            logits = model(to_model_tensor(image_lb).unsqueeze(0).to(device))
            pred_map = torch.sigmoid(logits)[0, 0].cpu().numpy()

        gt_grains = extract_grains_from_saliency(gt_map, image_lb, **gt_kw)
        pr_strict = extract_grains_from_saliency(pred_map, image_lb, **strict_kw)
        pr_sensitive = extract_grains_from_saliency(pred_map, image_lb, **sensitive_kw)

        gt_total += len(gt_grains)
        pred_strict += len(pr_strict)
        pred_sensitive += len(pr_sensitive)

        for g in gt_grains:
            gt_areas.append(g.area)
            if not any(bbox_iou(g.bbox, p.bbox) >= 0.3 for p in pr_strict):
                missed_strict += 1
                missed_areas.append(g.area)

    areas = np.asarray(gt_areas, dtype=np.float64)
    missed = np.asarray(missed_areas, dtype=np.float64)
    report = {
        "images": len(ds),
        "gt_grains": gt_total,
        "pred_strict": pred_strict,
        "pred_sensitive": pred_sensitive,
        "recall_strict_count": pred_strict / max(gt_total, 1),
        "recall_sensitive_count": pred_sensitive / max(gt_total, 1),
        "gt_unmatched_strict": missed_strict,
        "gt_unmatched_strict_pct": 100.0 * missed_strict / max(len(areas), 1),
        "min_area_strict": strict_kw["min_area"],
        "min_circularity_strict": strict_kw["min_circularity"],
    }
    if len(areas):
        report["gt_area_px2"] = {
            "p10": float(np.percentile(areas, 10)),
            "p25": float(np.percentile(areas, 25)),
            "median": float(np.median(areas)),
            "p75": float(np.percentile(areas, 75)),
        }
        small = areas < 800
        report["gt_fraction_area_lt_800"] = float(small.mean())
        if len(missed):
            report["missed_area_median"] = float(np.median(missed))
            report["missed_fraction_area_lt_800"] = float((missed < 800).mean())
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="runs/2026-06-13_01-40-07/detector/best_detector.pth",
    )
    parser.add_argument("--max-images", type=int, default=200)
    args = parser.parse_args()
    report = diagnose(args.config, args.checkpoint, max_images=args.max_images)
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
