"""
Formal decomposition of full-image inference vs oracle crops for a DML run.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml

from scripts.post_training import _fullimage_bbox_iou
from src.grain_detection.seg_format import SegFileReader
from src.grain_detection.full_image_classifier import FullImageClassifier
from src.utils.config_utils import merge_global_active_detector, resolve_run_config_path


IOU_MATCH = 0.3


def _find_image(seg_path: Path, cls_name: str, img_name: str) -> Path | None:
    for candidate in (
        seg_path.parent / img_name,
        Path("data/processed/train") / cls_name / img_name,
        Path("data/processed/val") / cls_name / img_name,
        Path("data/processed/test") / cls_name / img_name,
    ):
        if candidate.exists():
            return candidate
    return None


def _parse_seg_image_name(seg_path: Path) -> str | None:
    with open(seg_path, encoding="utf-8") as f:
        for line in f.readlines()[:10]:
            if line.startswith("# Imagen:"):
                return line.split(":", 1)[1].strip()
    return None


def analyze_run(run_dir: str, prefix: str = "val", n_images_per_class: int = 3, seed: int = 42):
    run_dir = Path(run_dir)
    ckpt = run_dir / "checkpoints" / "best_model.pth"
    config_path = resolve_run_config_path(str(ckpt), str(run_dir / "config" / "config.yaml"))
    config = yaml.safe_load(open(config_path, encoding="utf-8"))
    config = merge_global_active_detector(config, "config.yaml")

    classifier = FullImageClassifier.from_checkpoint(str(ckpt), config_path=config_path, k=5)
    class_names = classifier.class_names
    annotation_root = Path(config["data"]["annotation_root"])

    rng = np.random.RandomState(seed)
    per_image = []
    totals = {
        "annotated": 0,
        "detected": 0,
        "matched": 0,
        "missed": 0,
        "extra_fp": 0,
        "matched_cls_correct": 0,
        "matched_cls_wrong": 0,
        "legacy_correct": 0,
        "legacy_total": 0,
    }

    for cls_idx, cls_name in enumerate(class_names):
        if cls_name not in {p.name for p in annotation_root.iterdir() if p.is_dir()}:
            continue
        cls_dir = annotation_root / cls_name
        seg_files = sorted(cls_dir.glob("*.seg"))
        if not seg_files:
            continue
        sel = [seg_files[i] for i in rng.choice(len(seg_files), min(n_images_per_class, len(seg_files)), replace=False)]

        for seg_path in sel:
            img_name = _parse_seg_image_name(seg_path)
            if not img_name:
                continue
            img_path = _find_image(seg_path, cls_name, img_name)
            if img_path is None:
                continue

            seg_data = SegFileReader.read(str(seg_path))
            annotated = seg_data.get("grains", [])
            results = classifier.classify_full_image(str(img_path), return_crops=False)
            grains = results.get("grains", [])

            matched_ann = set()
            matched_cls_ok = 0
            matched_cls_bad = 0
            legacy_ok = 0

            for grain in grains:
                pred = grain["predicted_class"]
                pred_idx = class_names.index(pred) if pred in class_names else -1
                if pred_idx == cls_idx:
                    legacy_ok += 1

                best_iou = 0.0
                best_i = -1
                for ann_i, ann in enumerate(annotated):
                    iou = _fullimage_bbox_iou(grain["bbox"], ann["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_i = ann_i
                if best_iou >= IOU_MATCH and best_i >= 0:
                    matched_ann.add(best_i)
                    if pred_idx == cls_idx:
                        matched_cls_ok += 1
                    else:
                        matched_cls_bad += 1

            n_ann = len(annotated)
            n_det = len(grains)
            n_matched = len(matched_ann)
            n_missed = n_ann - n_matched
            n_extra = sum(
                1
                for g in grains
                if not any(_fullimage_bbox_iou(g["bbox"], a["bbox"]) >= IOU_MATCH for a in annotated)
            )

            totals["annotated"] += n_ann
            totals["detected"] += n_det
            totals["matched"] += n_matched
            totals["missed"] += n_missed
            totals["extra_fp"] += n_extra
            totals["matched_cls_correct"] += matched_cls_ok
            totals["matched_cls_wrong"] += matched_cls_bad
            totals["legacy_correct"] += legacy_ok
            totals["legacy_total"] += n_det

            per_image.append(
                {
                    "class": cls_name,
                    "image": img_path.name,
                    "annotated": n_ann,
                    "detected": n_det,
                    "matched": n_matched,
                    "missed": n_missed,
                    "extra_fp": n_extra,
                    "legacy_acc": legacy_ok / n_det if n_det else 0.0,
                    "matched_cls_acc": matched_cls_ok / n_matched if n_matched else None,
                    "det_recall": n_matched / n_ann if n_ann else 0.0,
                }
            )

    # Oracle upper bound: classify .seg crops with same classifier (no detector)
    oracle_correct = 0
    oracle_total = 0
    for rec in per_image:
        cls_name = rec["class"]
        cls_idx = class_names.index(cls_name)
        seg_path = None
        for sp in (annotation_root / cls_name).glob("*.seg"):
            if _parse_seg_image_name(sp) == rec["image"]:
                seg_path = sp
                break
        if seg_path is None:
            continue
        img_path = _find_image(seg_path, cls_name, rec["image"])
        if img_path is None:
            continue
        seg_data = SegFileReader.read(str(seg_path))
        full_img = cv2.imread(str(img_path))
        if full_img is None:
            continue
        for ann in seg_data.get("grains", []):
            from src.grain_detection.grain_crop_utils import prepare_grain_crop_and_mask

            grain = {"bbox": ann["bbox"], "contour": ann.get("contour")}
            crop_bgr, mask_crop = prepare_grain_crop_and_mask(
                full_img,
                grain,
                crop_padding=classifier.crop_padding,
                use_mask=classifier.use_mask,
                mask_bg_mode=classifier.mask_bg_mode,
            )
            if crop_bgr is None:
                continue
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            if classifier.transforms:
                mask_uint8 = (mask_crop * 255).astype(np.uint8) if mask_crop is not None else None
                if mask_uint8 is not None and hasattr(classifier.transforms, "__call__"):
                    tensor = classifier.transforms(crop_rgb, mask=mask_uint8)
                    if isinstance(tensor, tuple):
                        tensor = tensor[0]
                else:
                    tensor = classifier.transforms(crop_rgb)
            else:
                tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
            mask = (
                torch.from_numpy(mask_crop.astype(np.float32) / 255.0)
                if mask_crop is not None
                else torch.ones(classifier.crop_size, classifier.crop_size)
            )
            with torch.no_grad():
                emb = classifier.model(
                    tensor.unsqueeze(0).to(classifier.device),
                    mask=mask.unsqueeze(0).to(classifier.device) if classifier.masked_pooling else None,
                )
            if classifier.slice_classifier is not None:
                pred_idx = int(classifier.slice_classifier.predict(emb)[0])
            else:
                pred_idx = int(classifier.knn.predict(emb.cpu().numpy())[0])
            oracle_total += 1
            if pred_idx == cls_idx:
                oracle_correct += 1

    report = {
        "run_dir": str(run_dir),
        "prefix": prefix,
        "active_detector": config.get("grain_detection", {}).get("active_detector"),
        "n_images": len(per_image),
        "totals": totals,
        "legacy_metric": {
            "correct": totals["legacy_correct"],
            "total_detected": totals["legacy_total"],
            "accuracy": totals["legacy_correct"] / totals["legacy_total"] if totals["legacy_total"] else 0,
            "note": "Current post_training metric: pred==image_class for ALL detections",
        },
        "decomposed": {
            "detection_recall_on_annotated": totals["matched"] / totals["annotated"] if totals["annotated"] else 0,
            "classification_on_matched_only": (
                totals["matched_cls_correct"] / (totals["matched_cls_correct"] + totals["matched_cls_wrong"])
                if (totals["matched_cls_correct"] + totals["matched_cls_wrong"])
                else 0
            ),
            "false_positive_detections": totals["extra_fp"],
            "missed_annotations": totals["missed"],
        },
        "oracle_bbox_classification": {
            "correct": oracle_correct,
            "total": oracle_total,
            "accuracy": oracle_correct / oracle_total if oracle_total else 0,
            "note": "Same classifier on .seg bboxes (no detector) — upper bound if localization were perfect",
        },
        "per_class_legacy": {},
        "per_class_matched_cls": {},
        "worst_images": sorted(per_image, key=lambda r: r["legacy_acc"])[:8],
    }

    for cls in class_names:
        imgs = [r for r in per_image if r["class"] == cls]
        if not imgs:
            continue
        lc = sum(int(r["legacy_acc"] * r["detected"]) for r in imgs)
        lt = sum(r["detected"] for r in imgs)
        mc = sum(
            int((r["matched_cls_acc"] or 0) * r["matched"])
            for r in imgs
            if r["matched"]
        )
        mt = sum(r["matched"] for r in imgs)
        report["per_class_legacy"][cls] = lc / lt if lt else 0.0
        report["per_class_matched_cls"][cls] = mc / mt if mt else None

    out = run_dir / "images" / "post_training" / prefix / "fullimage_inference" / "formal_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="runs/2026-06-13_09-50-06")
    p.add_argument("--prefix", default="val")
    args = p.parse_args()
    analyze_run(args.run_dir, args.prefix)
