#!/usr/bin/env python3
"""
Compute dataset normalization stats and optional sensor noise calibration (Phase E).

Usage:
    python scripts/compute_dataset_norm.py --config config.yaml
    python scripts/compute_dataset_norm.py --config config.yaml --max-samples 500 --noise
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.grain_dataset import SegmentedGrainDataset
from src.utils.dataset_norm import compute_channel_mean_std, save_norm_stats
from src.utils.resolution_config import resolve_effective_image_size
from src.utils.sensor_noise import estimate_gaussian_noise_std, save_noise_model

logger = logging.getLogger(__name__)


def _collect_crop_rgb_samples(config: dict, max_samples: int) -> list:
    """Sample RGB uint8 crops from train set (same pipeline as training, sin augment)."""
    data_cfg = config.get("data", {})
    grain_cfg = config.get("grain_detection", {})
    train_dir = data_cfg.get("train_dir")
    if not train_dir or not Path(train_dir).exists():
        raise FileNotFoundError(f"data.train_dir no encontrado: {train_dir}")

    image_size = resolve_effective_image_size(config)

    ds = SegmentedGrainDataset(
        root=train_dir,
        transform=None,
        crop_padding=float(grain_cfg.get("crop_padding", 0.2)),
        crop_size=image_size,
        use_mask=bool(grain_cfg.get("use_mask", True)),
        mask_bg_mode=str(grain_cfg.get("train_mask_bg_mode", "imagenet_neutral")),
        return_mask=False,
        annotation_root=data_cfg.get("annotation_root"),
    )

    n = min(max_samples, len(ds))
    if n == 0:
        raise RuntimeError("Train dataset vacío")

    indices = np.linspace(0, len(ds) - 1, n, dtype=int)
    images = []
    for idx in indices:
        tensor, _ = ds[int(idx)]
        arr = (tensor.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        images.append(arr)
    return images


def main():
    parser = argparse.ArgumentParser(description="Compute dataset norm + sensor noise stats")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--noise", action="store_true", help="Also calibrate sensor noise model")
    parser.add_argument(
        "--output",
        default=None,
        help="Override stats output path (default: data.normalize.stats_path from config)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config_path = Path(args.config)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_norm = (config.get("data") or {}).setdefault("normalize", {})
    stats_path = Path(args.output or data_norm.get("stats_path") or "data/stats/train_norm.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Sampling up to {args.max_samples} train crops...")
    images = _collect_crop_rgb_samples(config, args.max_samples)
    mean, std = compute_channel_mean_std(images)
    save_norm_stats(
        stats_path,
        mean,
        std,
        meta={
            "source": str(config.get("data", {}).get("train_dir")),
            "num_samples": len(images),
            "config": str(config_path),
        },
    )
    logger.info(f"mean={mean}")
    logger.info(f"std={std}")

    if args.noise:
        aug = config.setdefault("augmentation", {})
        noise_cfg = aug.setdefault("sensor_noise_model", {})
        noise_path = Path(noise_cfg.get("stats_path") or "data/stats/sensor_noise.json")
        p10, p90 = estimate_gaussian_noise_std(images)
        save_noise_model(
            noise_path,
            (p10, p90),
            meta={"num_samples": len(images)},
        )
        logger.info(f"sensor_noise var_limit=[{p10:.2f}, {p90:.2f}] → {noise_path}")

    print(f"OK: stats → {stats_path}")


if __name__ == "__main__":
    main()
