"""Full E2E analysis on exact images in fullimage_inference gallery."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.post_training import _fullimage_bbox_iou
from scripts.regenerate_fullimage_folder import _parse_manifest, _resolve_image_path
from src.grain_detection.full_image_classifier import FullImageClassifier
from src.grain_detection.seg_format import SegFileReader
from src.utils.config_utils import merge_global_active_detector, resolve_run_config_path


def analyze_gallery(run_dir: str, gallery_dir: str):
    run = Path(run_dir)
    gallery = Path(gallery_dir)
    ckpt = run / "checkpoints" / "best_model.pth"
    cfg_path = resolve_run_config_path(str(ckpt), str(run / "config" / "config.yaml"))
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = merge_global_active_detector(config, str(ROOT / "config.yaml"))

    clf = FullImageClassifier.from_checkpoint(str(ckpt), config_path=cfg_path, k=5)
    class_names = clf.class_names
    annotation_root = Path(config["data"]["annotation_root"])
    manifest = _parse_manifest(gallery, class_names)

    totals = dict(annotated=0, detected=0, matched=0, missed=0, extra_fp=0,
                  matched_cls_ok=0, matched_cls_bad=0, legacy_ok=0, legacy_total=0)
    per_class = {}
    per_image = []
    errors = []

    for cls_name, img_stem in manifest:
        cls_idx = class_names.index(cls_name)
        img_path = _resolve_image_path(cls_name, img_stem, annotation_root)
        if img_path is None:
            errors.append({"class": cls_name, "stem": img_stem, "error": "image_not_found"})
            continue

        seg_path = annotation_root / cls_name / f"{img_stem}.seg"
        annotated = []
        if seg_path.exists():
            annotated = SegFileReader.read(str(seg_path)).get("grains", [])

        results = clf.classify_full_image(str(img_path), return_crops=False)
        grains = results.get("grains", [])

        matched_ann = set()
        matched_ok = matched_bad = legacy_ok = 0
        for grain in grains:
            pred = grain.get("raw_predicted_class", grain["predicted_class"])
            pred_idx = class_names.index(pred) if pred in class_names else -1
            if pred_idx == cls_idx:
                legacy_ok += 1
            best_iou = 0.0
            for ann_i, ann in enumerate(annotated):
                iou = _fullimage_bbox_iou(grain["bbox"], ann["bbox"])
                if iou >= 0.3:
                    matched_ann.add(ann_i)
                best_iou = max(best_iou, iou)
            if best_iou >= 0.3:
                if pred_idx == cls_idx:
                    matched_ok += 1
                else:
                    matched_bad += 1

        n_ann = len(annotated)
        n_det = len(grains)
        n_matched = len(matched_ann)
        n_missed = n_ann - n_matched
        n_extra = sum(
            1 for g in grains
            if not any(_fullimage_bbox_iou(g["bbox"], a["bbox"]) >= 0.3 for a in annotated)
        )

        totals["annotated"] += n_ann
        totals["detected"] += n_det
        totals["matched"] += n_matched
        totals["missed"] += n_missed
        totals["extra_fp"] += n_extra
        totals["matched_cls_ok"] += matched_ok
        totals["matched_cls_bad"] += matched_bad
        totals["legacy_ok"] += legacy_ok
        totals["legacy_total"] += n_det

        rec = {
            "class": cls_name,
            "image": img_path.name,
            "annotated": n_ann,
            "detected": n_det,
            "matched": n_matched,
            "missed": n_missed,
            "extra_fp": n_extra,
            "legacy_acc": legacy_ok / n_det if n_det else None,
            "matched_cls_acc": matched_ok / n_matched if n_matched else None,
            "det_recall": n_matched / n_ann if n_ann else None,
        }
        per_image.append(rec)

        pc = per_class.setdefault(cls_name, {"legacy_ok": 0, "legacy_total": 0, "matched_ok": 0, "matched_total": 0})
        pc["legacy_ok"] += legacy_ok
        pc["legacy_total"] += n_det
        pc["matched_ok"] += matched_ok
        pc["matched_total"] += n_matched

    legacy_acc = totals["legacy_ok"] / totals["legacy_total"] if totals["legacy_total"] else 0
    det_recall = totals["matched"] / totals["annotated"] if totals["annotated"] else 0
    cls_matched = (
        totals["matched_cls_ok"] / (totals["matched_cls_ok"] + totals["matched_cls_bad"])
        if (totals["matched_cls_ok"] + totals["matched_cls_bad"]) else 0
    )

    report = {
        "run_dir": str(run),
        "gallery_dir": str(gallery),
        "active_detector": config.get("grain_detection", {}).get("active_detector"),
        "pipeline": {
            "mask_bg_mode": clf.mask_bg_mode,
            "max_load_side": clf.max_load_side,
            "nms_iou_threshold": clf.nms_iou_threshold,
            "detection_strategy": getattr(clf.detector, "detection_strategy", "single_pass"),
            "sliding_window_enabled": bool(
                getattr(clf.detector, "sliding_window_enabled", False)
            ),
        },
        "n_images": len(manifest),
        "totals": totals,
        "legacy_e2e_accuracy": legacy_acc,
        "detection_recall_on_annotated": det_recall,
        "classification_on_matched_only": cls_matched,
        "per_class": {
            k: {
                "legacy_accuracy": v["legacy_ok"] / v["legacy_total"] if v["legacy_total"] else None,
                "matched_cls_accuracy": v["matched_ok"] / v["matched_total"] if v["matched_total"] else None,
                "grains_detected": v["legacy_total"],
            }
            for k, v in per_class.items()
        },
        "worst_images": sorted(
            [r for r in per_image if (r["legacy_acc"] or 0) < 1.0 or (r["det_recall"] or 1) < 1.0],
            key=lambda r: (r["legacy_acc"] or 0, r["det_recall"] or 0),
        )[:12],
        "missing_images": errors,
    }

    out = gallery / "gallery_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    analyze_gallery(
        "runs/2026-06-13_09-50-06",
        "runs/2026-06-13_09-50-06/images/post_training/val/fullimage_inference",
    )
