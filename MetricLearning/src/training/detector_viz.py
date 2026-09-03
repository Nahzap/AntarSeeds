"""Visualization utilities for detector training (curves + val samples)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def plot_detector_curves(history: List[dict], output_dir: str | Path) -> Dict[str, str]:
    """Plot training/validation metric curves from epoch history."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not history:
        logger.warning("plot_detector_curves: history vacío")
        return {}

    epochs = [int(m.get("epoch", i + 1)) for i, m in enumerate(history)]
    artifacts: Dict[str, str] = {}

    def _series(key: str) -> List[float]:
        return [float(m.get(key, 0.0)) for m in history]

    # Loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, _series("train_loss"), "o-", label="train_loss", color="#3498db")
    ax.plot(epochs, _series("val_loss"), "s-", label="val_loss", color="#e74c3c")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Detector — Loss curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = out / "loss_curves.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    artifacts["loss_curves"] = str(loss_path)

    # SOD metrics (higher Fm / mask_iou better; lower sod_mae better)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, _series("Fm"), "o-", color="#2ecc71")
    axes[0].set_title("Fm (SOD)")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, _series("mask_iou"), "o-", color="#9b59b6")
    axes[1].set_title("mask_iou")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, _series("sod_mae"), "o-", color="#e67e22")
    axes[2].set_title("sod_mae (↓ mejor)")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    sod_path = out / "sod_metrics.png"
    fig.savefig(sod_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    artifacts["sod_metrics"] = str(sod_path)

    # Instance metrics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, key, title in (
        (axes[0, 0], "det_precision", "det_precision"),
        (axes[0, 1], "det_recall", "det_recall"),
        (axes[1, 0], "det_f1", "det_f1"),
        (axes[1, 1], "grains_mae", "grains_mae (↓ mejor)"),
    ):
        ax.plot(epochs, _series(key), "o-")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    inst_path = out / "instance_metrics.png"
    fig.savefig(inst_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    artifacts["instance_metrics"] = str(inst_path)

    logger.info(f"Detector curves saved to {out}")
    return artifacts


def save_val_vis_batch(
    model: torch.nn.Module,
    val_loader,
    *,
    device: torch.device,
    config: dict,
    output_dir: str | Path,
    epoch: int,
    max_samples: int = 6,
) -> int:
    """Save GT|pred|overlay panels for a few val images."""
    import cv2
    from src.data.detection_transforms import letterbox_to_square
    from src.training.detector_eval_utils import save_vis_grid
    from src.utils.detector_resolution_config import resolve_detector_input_size

    out = Path(output_dir) / f"vis_epoch_{epoch:03d}"
    out.mkdir(parents=True, exist_ok=True)
    grain_cfg = config.get("grain_detection", {}) or {}
    input_size = resolve_detector_input_size(config)
    saved = 0

    model.eval()
    with torch.inference_mode():
        for batch in val_loader:
            images = batch["image"].to(device)
            targets = batch["saliency_gt"]
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = targets.numpy()

            for i in range(probs.shape[0]):
                if saved >= max_samples:
                    return saved
                path = batch["image_path"][i]
                image = cv2.imread(path)
                if image is None:
                    continue
                image_lb, _, _ = letterbox_to_square(image, None, input_size)
                save_vis_grid(
                    image_lb,
                    probs[i, 0],
                    targets_np[i, 0],
                    out / f"sample_{saved:02d}.jpg",
                    grain_cfg=grain_cfg,
                )
                saved += 1
            if saved >= max_samples:
                break
    logger.info(f"Saved {saved} val vis samples → {out}")
    return saved


def mirror_detector_artifacts_to_run_images(
    run_dir: str | Path,
    detector_dir: str | Path,
    curve_artifacts: Dict[str, str],
) -> Path:
    """Copy key PNGs to runs/{ts}/images/post_training/detector/ (familiar DML layout)."""
    import shutil

    dest = Path(run_dir) / "images" / "post_training" / "detector"
    dest.mkdir(parents=True, exist_ok=True)
    for src in curve_artifacts.values():
        p = Path(src)
        if p.exists():
            shutil.copy2(p, dest / p.name)

    eval_vis = Path(detector_dir) / "evaluation" / "vis_grid"
    if eval_vis.exists():
        dest_vis = dest / "vis_grid"
        if dest_vis.exists():
            shutil.rmtree(dest_vis)
        shutil.copytree(eval_vis, dest_vis)

    report = Path(detector_dir) / "evaluation" / "evaluation_report.json"
    if report.exists():
        shutil.copy2(report, dest / "evaluation_report.json")

    return dest
