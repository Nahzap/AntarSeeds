"""
Regenerate full-image inference PNGs for an existing gallery folder.

Uses the same source images as filenames already present in the folder
({class}_{image_stem}.png), so results are comparable run-to-run.

Example:
    python scripts/regenerate_fullimage_folder.py ^
        --run_dir runs/2026-06-12_17-16-54 ^
        --gallery_dir runs/2026-06-12_17-16-54/images/post_training/test/fullimage_inference
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.post_training import (  # noqa: E402
    DEFAULT_FIG_DPI,
    _draw_fullimage_grain_labels,
    _fullimage_bbox_iou,
)
from scripts.train import create_model  # noqa: E402
from src.data.transforms_enhanced import get_val_transforms  # noqa: E402
from src.grain_detection.full_image_classifier import FullImageClassifier  # noqa: E402
from src.grain_detection.seg_format import SegFileReader  # noqa: E402
from src.inference.slice_classifier import load_slice_classifier_from_run  # noqa: E402
from src.utils.inference_utils import overlay_full_image_gradcam  # noqa: E402
from src.utils.visual_explainer import create_explainer  # noqa: E402

logger = logging.getLogger(__name__)


def _parse_manifest(gallery_dir: Path, class_names: list[str]) -> list[tuple[str, str]]:
    """Map existing output PNG names to (class_name, image_stem)."""
    ordered = sorted(class_names, key=len, reverse=True)
    manifest: list[tuple[str, str]] = []
    for png in sorted(gallery_dir.glob("*.png")):
        if png.name == "fullimage_summary.png":
            continue
        stem = png.stem
        matched = None
        for cls in ordered:
            prefix = f"{cls}_"
            if stem.startswith(prefix):
                matched = (cls, stem[len(prefix) :])
                break
        if matched is None:
            raise ValueError(f"Cannot parse class from gallery file: {png.name}")
        manifest.append(matched)
    return manifest


def _resolve_image_path(cls_name: str, img_stem: str, annotation_root: Path) -> Path | None:
    """Resolve source image for a gallery entry (stem from output PNG name)."""
    cls_dir = annotation_root / cls_name
    if not cls_dir.is_dir():
        return None

    # 1) Exact stem from gallery filename (.png / .jpg)
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        direct = cls_dir / f"{img_stem}{ext}"
        if direct.exists():
            return direct
    for root in (
        Path("data/processed/train") / cls_name,
        Path("data/processed/val") / cls_name,
        Path("data/processed/test") / cls_name,
    ):
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            candidate = root / f"{img_stem}{ext}"
            if candidate.exists():
                return candidate

    # 2) Matching .seg for this stem → # Imagen header (same logic as post_training)
    seg_path = cls_dir / f"{img_stem}.seg"
    if seg_path.exists():
        try:
            with open(seg_path, encoding="utf-8") as f:
                for line in f.readlines()[:10]:
                    if line.startswith("# Imagen:"):
                        img_name = line.split(":", 1)[1].strip()
                        for candidate in (
                            cls_dir / img_name,
                            Path("data/processed/train") / cls_name / img_name,
                            Path("data/processed/val") / cls_name / img_name,
                            Path("data/processed/test") / cls_name / img_name,
                        ):
                            if candidate.exists():
                                return candidate
                        break
        except OSError:
            pass

    return None


def regenerate_fullimage_folder(
    run_dir: Path,
    gallery_dir: Path,
    *,
    with_gradcam: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    gallery_dir = Path(gallery_dir)
    gallery_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from src.utils.config_utils import merge_global_active_detector

    config = merge_global_active_detector(config, str(ROOT / "config.yaml"))

    ckpt_path = run_dir / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and config.get("hardware", {}).get("use_gpu", True)
        else "cpu"
    )
    logger.info("Device: %s", device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    config.setdefault("model", {})["pretrained"] = False
    config["model"]["freeze_backbone"] = False
    model = create_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    embed_dim = config.get("model", {}).get("embedding_dim", 128)
    num_slices = config.get("training", {}).get("loss", {}).get("params", {}).get("num_slices", 4)
    slice_classifier = load_slice_classifier_from_run(run_dir, device, embed_dim, num_slices)
    if slice_classifier is None:
        raise RuntimeError(f"slice_representatives.pt missing in {run_dir}")

    class_names = list(slice_classifier.reps.class_names)
    manifest = _parse_manifest(gallery_dir, class_names)
    logger.info("Regenerating %d full-image outputs in %s", len(manifest), gallery_dir)

    ref_path = run_dir / "reference_embeddings.pt"
    ref_data = torch.load(ref_path, map_location="cpu", weights_only=False)
    reference_embeddings = (
        ref_data["embeddings"].cpu().numpy()
        if torch.is_tensor(ref_data["embeddings"])
        else ref_data["embeddings"]
    )
    reference_labels = (
        ref_data["labels"].cpu().numpy()
        if torch.is_tensor(ref_data["labels"])
        else ref_data["labels"]
    )

    val_config = {**config.get("augmentation", {}), **config["data"]}
    val_transform = get_val_transforms(val_config)
    grain_config = config.get("grain_detection", {})
    discovery_mode = bool(grain_config.get("discovery_mode", False))

    full_classifier = FullImageClassifier(
        model=model,
        transforms=val_transform,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        class_names=class_names,
        device=device,
        grain_config=grain_config,
        slice_classifier=slice_classifier,
    )
    logger.info(
        "Localization backend: %s | discovery_mode=%s",
        full_classifier.localization_backend,
        discovery_mode,
    )

    explainer = create_explainer(model, device) if with_gradcam else None
    annotation_root = Path(config["data"].get("annotation_root", "data/annotations"))

    crops_dir = gallery_dir / "crops"
    if crops_dir.exists():
        for old in crops_dir.glob("*.png"):
            old.unlink()
    crops_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}
    total_grains = 0
    total_correct = 0

    for cls_name, img_stem in manifest:
        cls_idx = class_names.index(cls_name)
        img_path = _resolve_image_path(cls_name, img_stem, annotation_root)
        if img_path is None:
            logger.warning("Image not found for %s / %s", cls_name, img_stem)
            continue

        seg_path = annotation_root / cls_name / f"{img_stem}.seg"
        annotated_grains = []
        n_annotated = 0
        if seg_path.exists():
            seg_data = SegFileReader.read(str(seg_path))
            annotated_grains = seg_data.get("grains", [])
            n_annotated = len(annotated_grains)

        results = full_classifier.classify_full_image(
            str(img_path),
            return_crops=True,
            discovery_mode=discovery_mode,
        )
        grains = results.get("grains", [])
        rejected_dets = results.get("rejected_detections", [])
        full_img = cv2.imread(str(img_path))
        if full_img is None:
            logger.warning("Could not read %s", img_path)
            continue

        h_img, w_img = full_img.shape[:2]
        display_img = full_img.copy()
        font_scale = max(0.4, min(h_img, w_img) / 2000)
        thickness = max(1, int(min(h_img, w_img) / 800))

        img_grains_correct = 0
        img_grains_total = len(grains)
        heatmaps_and_bboxes = []
        matched_ann: set[int] = set()

        for g_idx, grain in enumerate(grains):
            eval_cls_name = grain.get("raw_predicted_class", grain["predicted_class"])
            pred_cls_name = grain["predicted_class"]
            pred_cls = class_names.index(eval_cls_name) if eval_cls_name in class_names else -1
            if pred_cls == cls_idx and not grain.get("rejected_by_classifier"):
                img_grains_correct += 1

            for ann_i, ann in enumerate(annotated_grains):
                if _fullimage_bbox_iou(grain["bbox"], ann["bbox"]) >= 0.3:
                    matched_ann.add(ann_i)

            if grain.get("crop") is not None:
                safe_cls = cls_name.replace(" ", "_").replace("/", "_")
                safe_pred = eval_cls_name.replace(" ", "_").replace("/", "_")
                crop_save = crops_dir / f"{safe_cls}_{img_stem}_grain_{g_idx}_{safe_pred}.png"
                cv2.imwrite(str(crop_save), grain["crop"])

            if with_gradcam and explainer is not None and grain.get("tensor") is not None:
                target_cls = pred_cls if pred_cls != -1 else 0
                pred_target = slice_classifier.get_target_for_gradcam(target_cls, device)
                heatmap = explainer.generate_heatmap(
                    grain["tensor"].unsqueeze(0), centroid=pred_target
                )
                crop_bbox = grain.get("classification_bbox") or grain["bbox"]
                heatmaps_and_bboxes.append((heatmap, crop_bbox))

        for rej in rejected_dets:
            bx, by, bw, bh = rej["bbox"]
            cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), (160, 160, 160), 2, cv2.LINE_AA)
            reason = rej.get("rejection_reason", "filtered")
            cv2.putText(
                display_img,
                f"filtrado ({reason})",
                (bx, max(12, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.75,
                (160, 160, 160),
                thickness,
                cv2.LINE_AA,
            )

        if with_gradcam and explainer is not None and heatmaps_and_bboxes:
            full_img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
            gradcam_bgr = cv2.cvtColor(
                overlay_full_image_gradcam(
                    full_image=full_img_rgb,
                    heatmaps_and_bboxes=heatmaps_and_bboxes,
                    crop_padding=full_classifier.crop_padding,
                    alpha=0.45,
                ),
                cv2.COLOR_RGB2BGR,
            )
            display_img = cv2.addWeighted(gradcam_bgr, 0.85, full_img, 0.15, 0)

        for ann_i, ann in enumerate(annotated_grains):
            if ann_i in matched_ann:
                continue
            bx, by, bw, bh = ann["bbox"]
            cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(
                display_img,
                "anotado (no detectado)",
                (bx, max(12, by - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.85,
                (255, 200, 0),
                thickness,
                cv2.LINE_AA,
            )

        _draw_fullimage_grain_labels(
            display_img,
            grains,
            expected_cls_idx=cls_idx,
            class_names=class_names,
            h_img=h_img,
            w_img=w_img,
        )

        n_extra_discovered = sum(
            1
            for grain in grains
            if not any(
                _fullimage_bbox_iou(grain["bbox"], ann["bbox"]) >= 0.3
                for ann in annotated_grains
            )
        )
        n_missed_annotated = n_annotated - len(matched_ann)
        total_grains += img_grains_total
        total_correct += img_grains_correct
        acc = img_grains_correct / img_grains_total if img_grains_total > 0 else 0.0

        title = (
            f"{cls_name} | {img_path.name} | "
            f"Evaluados: {img_grains_total} | Filtrados: {len(rejected_dets)} | "
            f"Anotados .seg: {n_annotated} | "
            f"Correctos: {img_grains_correct}/{img_grains_total} ({acc:.0%}) | "
            f"Extra descubiertos: {n_extra_discovered} | "
            f"Anotados perdidos: {n_missed_annotated}"
        )
        bar_h = max(36, int(h_img * 0.045))
        title_bar = np.zeros((bar_h, w_img, 3), dtype=np.uint8)
        title_bar[:] = (40, 40, 40)
        cv2.putText(
            title_bar,
            title,
            (10, bar_h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.1,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        display_img = np.vstack([title_bar, display_img])

        save_path = gallery_dir / f"{cls_name}_{img_stem}.png"
        cv2.imwrite(str(save_path), display_img)
        logger.info("Saved %s (%d grains, %d/%d correct)", save_path.name, img_grains_total, img_grains_correct, img_grains_total)

        all_results.setdefault(cls_name, []).append(
            {
                "image": img_path.name,
                "grains": img_grains_total,
                "annotated": n_annotated,
                "correct": img_grains_correct,
                "accuracy": acc,
                "discovered_extra": n_extra_discovered,
                "missed_annotated": n_missed_annotated,
                "quality_filtered": len(rejected_dets),
            }
        )

    if all_results:
        overall_acc = total_correct / total_grains if total_grains > 0 else 0.0
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(all_results.keys())
        accs = []
        for name in names:
            c = sum(r["correct"] for r in all_results[name])
            t = sum(r["grains"] for r in all_results[name])
            accs.append(c / t if t > 0 else 0.0)
        colors_bar = ["#2ecc71" if a >= 0.9 else ("#f39c12" if a >= 0.7 else "#e74c3c") for a in accs]
        bars = ax.barh(names, accs, color=colors_bar, edgecolor="#333", linewidth=0.5)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Grain Classification Accuracy", fontsize=12)
        ax.set_title(
            f"Full-Image Inference — {total_correct}/{total_grains} grains ({overall_acc:.1%})",
            fontsize=14,
            fontweight="bold",
        )
        ax.axvline(x=0.9, color="green", linestyle="--", alpha=0.5)
        for bar, a in zip(bars, accs):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{a:.1%}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(gallery_dir / "fullimage_summary.png", dpi=DEFAULT_FIG_DPI, bbox_inches="tight")
        plt.close()
        logger.info(
            "Summary: %d/%d grains correct (%.1f%%)",
            total_correct,
            total_grains,
            overall_acc * 100,
        )

    return all_results


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Regenerate an existing fullimage_inference gallery")
    parser.add_argument("--run_dir", required=True, help="Run directory with best_model.pth")
    parser.add_argument("--gallery_dir", required=True, help="Target fullimage_inference folder")
    parser.add_argument("--no-gradcam", action="store_true", help="Skip Grad-CAM overlays (faster)")
    args = parser.parse_args()
    regenerate_fullimage_folder(
        Path(args.run_dir),
        Path(args.gallery_dir),
        with_gradcam=not args.no_gradcam,
    )


if __name__ == "__main__":
    main()
