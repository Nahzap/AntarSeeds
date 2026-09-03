"""
Backfill post-training artifacts for existing runs (no training).

This script re-runs post-training analysis on existing run directories to
generate/refresh evaluation artifacts such as:
  - training_report.md (updated sections)
  - evaluation_metrics_summary.json
  - training_protocol.md
  - images/post_training/classification_report.csv
  - images/post_training/per_class_f1.png
  - images/post_training/confusion_matrix_normalized.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

# Ensure project root is on path when invoked as script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.post_training import (
    run_post_training_analysis,
    create_eval_dataloaders,
    finalize_self_contained_checkpoint,
    compute_and_save_centroids,
    load_slice_classifier_from_run,
)
from src.data.dataloader_utils import dispose_data_engine
from src.models.analogy_net import AnalogyNet
from src.data.dataset import MielDataset
from src.data.transforms_enhanced import get_val_transforms
from src.utils.config_utils import get_backbone_name


logger = logging.getLogger("backfill_post_training")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def create_model_from_config(config: dict) -> AnalogyNet:
    from src.utils.model_factory import create_analogy_net_from_config
    return create_analogy_net_from_config(config)


def create_val_loader_from_config(config: dict):
    data_cfg = config["data"]
    aug_cfg = config.get("augmentation", {})
    transform_cfg = {**aug_cfg, **data_cfg}
    backbone_name = get_backbone_name(config)
    val_transforms = get_val_transforms(transform_cfg, backbone_name=backbone_name)

    grain_cfg = config.get("grain_detection", {})
    if bool(grain_cfg.get("enabled", True)):
        from src.data.grain_dataset import SegmentedGrainDataset
        val_dataset = SegmentedGrainDataset(
            root=data_cfg["val_dir"],
            transform=val_transforms,
            crop_padding=grain_cfg.get("crop_padding", 0.2),
            crop_size=grain_cfg.get("crop_size", 252),
            use_mask=grain_cfg.get("use_mask", True),
            mask_bg_mode="gray",
            min_saliency=0.0,
            return_mask=grain_cfg.get("masked_pooling", False),
            annotation_root=data_cfg.get("annotation_root", "data/annotations"),
        )
    else:
        val_dataset = MielDataset(root=data_cfg["val_dir"], transform=val_transforms)

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=0,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=False,
    )
    return val_loader


def backfill_run(run_dir: Path) -> None:
    config_path = run_dir / "config" / "config.yaml"
    best_ckpt = run_dir / "checkpoints" / "best_model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Missing best checkpoint: {best_ckpt}")

    logger.info("=" * 80)
    logger.info("Backfilling run: %s", run_dir)
    logger.info("=" * 80)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _, val_loader, test_loader, data_engine = create_eval_dataloaders(config)
    model = create_model_from_config(config)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and config.get("hardware", {}).get("use_gpu", True)
        else "cpu"
    )
    logger.info("Using device: %s", device)
    model.to(device)

    logger.info("Loading best checkpoint: %s", best_ckpt)
    ckpt = torch.load(str(best_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    embed_dim = config.get("model", {}).get("embedding_dim", 128)
    loss_params = config.get("training", {}).get("loss", {}).get("params", {})
    num_slices = int(loss_params.get("num_slices", 4))
    slice_classifier = load_slice_classifier_from_run(run_dir, device, embed_dim, num_slices)

    if slice_classifier is None:
        from src.data.data_engine import GrainDataEngine
        import copy
        train_cfg = copy.deepcopy(config)
        train_cfg.setdefault("data", {})["pin_memory"] = False
        train_engine = GrainDataEngine.from_config(train_cfg)
        _, slice_classifier = compute_and_save_centroids(
            model,
            train_engine.train_loader,
            device,
            str(run_dir),
            train_engine.train_loader.dataset.classes,
            num_slices=num_slices,
            loss_state_dict=ckpt.get("loss_state_dict"),
            loss_type=ckpt.get("loss_type"),
        )
        dispose_data_engine(train_engine)

    run_post_training_analysis(
        model=model,
        val_loader=val_loader,
        config=config,
        run_dir=str(run_dir),
        device=device,
        slice_classifier=slice_classifier,
        prefix="val",
    )

    if test_loader is not None:
        run_post_training_analysis(
            model=model,
            val_loader=test_loader,
            config=config,
            run_dir=str(run_dir),
            device=device,
            slice_classifier=slice_classifier,
            prefix="test",
        )

    finalize_self_contained_checkpoint(run_dir)
    dispose_data_engine(data_engine)

    logger.info("Backfill completed: %s", run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill post-training artifacts for existing runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories to backfill, e.g. runs/2026-03-20_ConvNextV2-Tiny_00",
    )
    args = parser.parse_args()

    setup_logging()

    failures = []
    for run in args.runs:
        run_path = Path(run)
        try:
            backfill_run(run_path)
        except Exception as exc:
            logger.exception("Backfill failed for run %s: %s", run_path, exc)
            failures.append(str(run_path))

    if failures:
        logger.error("Backfill finished with failures in %d run(s): %s", len(failures), failures)
        raise SystemExit(1)

    logger.info("All requested runs were backfilled successfully.")


if __name__ == "__main__":
    main()

