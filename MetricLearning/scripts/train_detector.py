"""
ViT-dense pollen localization training script.

Usage (GUI invokes ``run_detector_training``):
    python scripts/train_detector.py --config config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import torch
import yaml

from src.data.detector_data_engine import DetectorDataEngine
from src.models.vit_dense_segmentation import ViTDenseSegmentationModel
from src.training.detector_trainer import DetectorTrainer
from src.utils.config_utils import load_config
from src.utils.detector_resolution_config import sync_detector_resolution_config
from src.utils.config_validator import validate_config


def setup_logging(log_path: Path, level: str = "INFO") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_detector")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    sh = logging.StreamHandler()
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _benchmark_batches(model, loader, device, use_amp: bool, n_warmup: int = 2, n_timed: int = 5) -> float:
    """Segundos por batch (warmup + media)."""
    model.eval()
    it = iter(loader)
    with torch.inference_mode():
        for _ in range(n_warmup):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(images)
        t0 = time.perf_counter()
        n = 0
        for _ in range(n_timed):
            try:
                batch = next(it)
            except StopIteration:
                break
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                model(images)
            n += 1
    return (time.perf_counter() - t0) / max(n, 1)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.1f} h"


def apply_detector_gui_overrides(config: dict, args) -> dict:
    """Fusiona overrides de la GUI en detection_training."""
    det = dict(config.get("detection_training") or {})
    for key in (
        "epochs", "batch_size", "val_batch_size", "input_size", "num_workers",
        "max_train_images", "max_val_images", "val_fast", "val_det_f1_max_images",
        "num_unfrozen_blocks", "learning_rate", "validation_metric",
    ):
        val = getattr(args, key, None)
        if val is not None:
            det[key] = val
    loss_type = getattr(args, "loss_type", None)
    if loss_type is not None:
        det.setdefault("loss", {})["type"] = loss_type
    out = dict(config)
    out["detection_training"] = det
    out, _ = sync_detector_resolution_config(out)
    return out


def run_detector_training(
    args,
    estimate_only: bool = False,
    run_dir: str | None = None,
):
    config = load_config(args.config)
    config = apply_detector_gui_overrides(config, args)
    config, _ = sync_detector_resolution_config(config)

    if getattr(args, "gpu", None) is not None:
        config.setdefault("hardware", {})["gpu_ids"] = [args.gpu]

    validation = validate_config(config)
    if not validation["valid"]:
        raise ValueError("Config inválida:\n" + "\n".join(validation["errors"]))

    from scripts.run_setup import create_run_directory, ensure_detector_run_directory

    if run_dir is None:
        run_paths = create_run_directory(config.get("logging", {}).get("tensorboard", {}).get("log_dir", "runs"))
        run_dir = run_paths["run_dir"]
    det_paths = ensure_detector_run_directory(run_dir)
    detector_dir = Path(det_paths["detector_dir"])

    log = setup_logging(detector_dir / "logs" / "training.log", config.get("logging", {}).get("log_level", "INFO"))

    # Snapshot config
    cfg_dst = detector_dir / "config.yaml"
    shutil.copy2(args.config, cfg_dst)

    seed = config.get("reproducibility", {}).get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    engine = DetectorDataEngine.from_config(config)
    train_stats = engine.train_dataset.get_stats()
    val_stats = engine.val_dataset.get_stats()

    pre_dir = Path(det_paths["pre_training_dir"])
    with open(pre_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_stats, "val": val_stats}, f, indent=2)

    log.info("=" * 70)
    log.info("DETECTOR TRAINING — ViT-dense saliency localization")
    log.info("=" * 70)
    log.info(f"Train images: {train_stats['num_images']} | Val: {val_stats['num_images']}")
    log.info(f"Grains/img (train mean): {train_stats.get('grains_per_image_mean', 0):.2f}")
    det_cfg = config.get("detection_training", {})
    log.info(
        f"Config efectiva: epochs={det_cfg.get('epochs')} batch={det_cfg.get('batch_size')} "
        f"input={det_cfg.get('input_size')} workers={det_cfg.get('num_workers')} "
        f"max_train={det_cfg.get('max_train_images', 0) or 'all'} "
        f"max_val={det_cfg.get('max_val_images', 0) or 'all'} "
        f"val_fast={det_cfg.get('val_fast', True)}"
    )
    if validation.get("warnings"):
        for w in validation["warnings"][:5]:
            log.warning(w)

    model = ViTDenseSegmentationModel.from_config(config)
    hw = config.get("hardware", {}) or {}
    device = torch.device("cuda" if torch.cuda.is_available() and hw.get("use_gpu", True) else "cpu")
    log.info(f"Device: {device}")
    model.to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    use_amp = bool(hw.get("use_amp", True)) and device.type == "cuda"
    log.info(f"AMP: {use_amp}")

    n_train_batches = len(engine.train_loader)
    n_val_batches = len(engine.val_loader)
    log.info(f"Batches/época: train={n_train_batches} val={n_val_batches}")
    log.debug("Benchmark train (warmup+5 batches)...")
    train_batch_s = _benchmark_batches(model, engine.train_loader, device, use_amp)
    log.debug("Benchmark val (warmup+5 batches)...")
    val_batch_s = _benchmark_batches(model, engine.val_loader, device, use_amp)
    log.info(f"Throughput: train {train_batch_s:.2f}s/batch | val {val_batch_s:.2f}s/batch")

    epochs = int(det_cfg.get("epochs", 30))
    val_fast = bool(det_cfg.get("val_fast", True))
    val_f1_max = int(det_cfg.get("val_det_f1_max_images", 100) or 0)
    if not val_fast and val_f1_max <= 0:
        val_f1_max = 100
    val_batch_size = int(det_cfg.get("val_batch_size", 0) or det_cfg.get("batch_size", 4))
    val_images = val_stats["num_images"]
    inst_images = min(val_images, val_f1_max) if val_f1_max > 0 else 0
    val_gpu_s = n_val_batches * val_batch_s
    val_morph_s = inst_images * 0.08
    val_epoch_s = val_gpu_s + val_morph_s
    train_epoch_s = n_train_batches * train_batch_s
    est_epoch = train_epoch_s + val_epoch_s
    estimate = {
        "run_dir": run_dir,
        "detector_dir": str(detector_dir),
        "train_images": train_stats["num_images"],
        "val_images": val_stats["num_images"],
        "train_batches": n_train_batches,
        "val_batches": n_val_batches,
        "train_sec_per_batch": train_batch_s,
        "val_sec_per_batch": val_batch_s,
        "train_epoch_sec": train_epoch_s,
        "val_epoch_sec": val_epoch_s,
        "epochs": epochs,
        "sec_per_epoch": est_epoch,
        "total_sec": est_epoch * epochs,
        "val_fast": val_fast,
        "analysis": {
            "stats": {"train": train_stats, "val": val_stats},
            "run_paths": det_paths,
        },
    }

    if estimate_only:
        engine.release_workers()
        log.info(
            f"Estimación: train {_format_duration(train_epoch_s)} + val {_format_duration(val_epoch_s)} "
            f"= {_format_duration(est_epoch)}/época × {epochs} = {_format_duration(estimate['total_sec'])} total"
        )
        if estimate["total_sec"] > 86400:
            log.warning(
                "Entrenamiento >24h estimado. Reduzca epochs, aumente batch_size, "
                "o use max_train_images/max_val_images en opciones de la pestaña."
            )
        return estimate

    resume = getattr(args, "resume", None)
    trainer = DetectorTrainer(
        model=model,
        train_loader=engine.train_loader,
        val_loader=engine.val_loader,
        config=config,
        run_dir=run_dir,
        logger_inst=log,
    )
    summary = trainer.train(resume_path=resume)
    summary["run_dir"] = run_dir
    summary["status"] = "completed"
    summary["detector_dir"] = str(detector_dir)
    log.info(f"Detector training complete. best_{summary['validation_metric']}={summary['best_metric']:.4f}")

    auto_eval = bool(det_cfg.get("auto_evaluate", True))
    best_ckpt = detector_dir / "best_detector.pth"
    if auto_eval and best_ckpt.exists():
        try:
            from scripts.evaluate_detector import evaluate_detector
            from src.training.detector_viz import mirror_detector_artifacts_to_run_images

            eval_dir = detector_dir / "evaluation"
            log.info(f"Post-entrenamiento: evaluando {best_ckpt.name} → {eval_dir}")
            report = evaluate_detector(
                args.config,
                str(best_ckpt),
                str(eval_dir),
            )
            summary["evaluation_report"] = report
            mirrored = mirror_detector_artifacts_to_run_images(
                run_dir,
                detector_dir,
                summary.get("curve_artifacts", {}),
            )
            summary["visualization_dir"] = str(mirrored)
            log.info(
                f"Visualizaciones: {detector_dir / 'curves'} | "
                f"eval vis_grid: {eval_dir / 'vis_grid'} | copia: {mirrored}"
            )
        except Exception as exc:
            log.warning(f"Auto-evaluación post-entrenamiento falló: {exc}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train ViT-dense pollen detector")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    result = run_detector_training(args, estimate_only=args.estimate_only)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
