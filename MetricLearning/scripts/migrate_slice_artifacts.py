"""
Migrate legacy runs to slice-aware inference artifacts.

For runs trained with sliced_proxy before loss_state_dict persistence:
  1. Try loss_state_dict / class_proxies in checkpoint
  2. Detect optimizer proxy group (values usually unavailable)
  3. Compute slice_representatives from reference_embeddings.pt or train loader

Optionally regenerates post-training with the slice-aware pipeline (val + test by default; use --skip-test for val-only edge cases).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.slice_classifier import (
    SliceRepresentatives,
    compute_slice_representatives_from_reference,
    extract_proxies_from_checkpoint,
    save_slice_representatives,
)
from scripts.post_training import compute_and_save_centroids, run_evaluation_from_checkpoint

logger = logging.getLogger("migrate_slice_artifacts")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _load_run_config(run_dir: Path) -> dict:
    for candidate in (run_dir / "config" / "config.yaml", run_dir / "config.yaml"):
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No config.yaml found under {run_dir}")


def migrate_run(
    run_dir: Path,
    regenerate_post_training: bool = False,
    force: bool = False,
    skip_test: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    slice_path = run_dir / "slice_representatives.pt"
    ckpt_path = run_dir / "checkpoints" / "best_model.pth"
    if slice_path.exists() and not force and not regenerate_post_training:
        logger.info("slice_representatives.pt already exists — skipping (use --force)")
        return slice_path

    config = _load_run_config(run_dir)
    loss_cfg = config.get("training", {}).get("loss", {})
    loss_params = loss_cfg.get("params", {})
    num_slices = int(loss_params.get("num_slices", 4))
    embed_dim = int(config.get("model", {}).get("embedding_dim", 128))
    slice_dim = embed_dim // num_slices

    centroids_path = run_dir / "class_centroids.pt"
    ref_path = run_dir / "reference_embeddings.pt"

    n_classes = None
    class_names = None
    if centroids_path.exists():
        cd = torch.load(str(centroids_path), map_location="cpu", weights_only=False)
        class_names = list(cd["class_names"])
        n_classes = len(class_names)

    reps: SliceRepresentatives | None = None

    if ckpt_path.exists() and n_classes:
        proxies = extract_proxies_from_checkpoint(
            ckpt_path, num_slices=num_slices, num_classes=n_classes, slice_dim=slice_dim
        )
        if proxies is not None:
            reps = SliceRepresentatives(
                representatives=proxies,
                class_names=class_names,
                num_slices=num_slices,
                slice_dim=slice_dim,
                embedding_dim=embed_dim,
                mode="learned_proxies",
                loss_type="sliced_proxy",
            )
            proxy_out = run_dir / "class_proxies.pt"
            torch.save(
                {
                    "proxies": proxies,
                    "class_names": class_names,
                    "num_slices": num_slices,
                    "slice_dim": slice_dim,
                    "embedding_dim": embed_dim,
                    "loss_type": "sliced_proxy",
                },
                proxy_out,
            )
            logger.info("Saved learned proxies: %s", proxy_out)

    if reps is None and ref_path.exists():
        logger.info("Computing slice representatives from reference_embeddings.pt...")
        reps = compute_slice_representatives_from_reference(ref_path, num_slices=num_slices)

    if reps is None:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
        logger.info("Computing slice representatives from training set (slow)...")
        from scripts.train import create_model
        from src.data.data_engine import GrainDataEngine
        import copy

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        train_cfg = copy.deepcopy(config)
        train_cfg.setdefault("data", {})["pin_memory"] = False
        engine = GrainDataEngine.from_config(train_cfg)
        model = create_model(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        _, slice_clf = compute_and_save_centroids(
            model,
            engine.train_loader,
            device,
            str(run_dir),
            engine.train_loader.dataset.classes,
            num_slices=num_slices,
            loss_state_dict=ckpt.get("loss_state_dict"),
            loss_type=ckpt.get("loss_type") or loss_cfg.get("type"),
        )
        from src.data.dataloader_utils import dispose_data_engine
        dispose_data_engine(engine)
        logger.info("Slice artifacts written via compute_and_save_centroids")
        if regenerate_post_training:
            run_evaluation_from_checkpoint(str(ckpt_path), skip_test=skip_test)
        return run_dir / "slice_representatives.pt"

    if reps is not None:
        save_slice_representatives(slice_path, reps)
        logger.info(
            "Saved slice_representatives.pt: shape %s, mode=%s",
            tuple(reps.representatives.shape),
            reps.mode,
        )
    elif not slice_path.exists():
        raise RuntimeError("Could not create slice_representatives.pt")

    if regenerate_post_training and ckpt_path.exists():
        run_evaluation_from_checkpoint(str(ckpt_path), skip_test=skip_test)

    return slice_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy runs to slice-aware artifacts")
    parser.add_argument(
        "--run",
        type=str,
        default="runs/2026-06-09_13-33-01",
        help="Run directory to migrate",
    )
    parser.add_argument(
        "--regenerate-post-training",
        action="store_true",
        help="Regenerate all post-training artifacts after migration",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing slice_representatives.pt",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test post-training during --regenerate-post-training (default: run val + test)",
    )
    args = parser.parse_args()
    setup_logging()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    migrate_run(
        run_dir,
        regenerate_post_training=args.regenerate_post_training,
        force=args.force,
        skip_test=args.skip_test,
    )
    logger.info("Migration complete: %s", run_dir)


if __name__ == "__main__":
    main()
