"""Regenerate detector curves and evaluation visuals from an existing run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import logging

from scripts.evaluate_detector import evaluate_detector
from src.training.detector_viz import mirror_detector_artifacts_to_run_images, plot_detector_curves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def regenerate_detector_visuals(
    run_dir: str,
    config_path: str = "config.yaml",
    *,
    skip_eval: bool = False,
) -> dict:
    run = Path(run_dir)
    detector_dir = run / "detector"
    metrics_json = detector_dir / "metrics.json"
    best_ckpt = detector_dir / "best_detector.pth"

    if not metrics_json.exists():
        raise FileNotFoundError(f"No existe {metrics_json}")
    with open(metrics_json, encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("history", [])

    curves_dir = detector_dir / "curves"
    curve_artifacts = plot_detector_curves(history, curves_dir)
    logger.info(f"Curvas guardadas en {curves_dir}")

    report = None
    if not skip_eval and best_ckpt.exists():
        eval_dir = detector_dir / "evaluation"
        report = evaluate_detector(config_path, str(best_ckpt), str(eval_dir))
        logger.info(f"Evaluación + vis_grid en {eval_dir}")

    mirrored = mirror_detector_artifacts_to_run_images(run, detector_dir, curve_artifacts)
    logger.info(f"Copia en {mirrored}")

    return {
        "curve_artifacts": curve_artifacts,
        "visualization_dir": str(mirrored),
        "evaluation_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerar curvas y vis_grid de un run detector")
    parser.add_argument("--run-dir", required=True, help="runs/YYYY-MM-DD_HH-MM-SS")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    result = regenerate_detector_visuals(args.run_dir, args.config, skip_eval=args.skip_eval)
    print(json.dumps({k: v for k, v in result.items() if k != "evaluation_report"}, indent=2))


if __name__ == "__main__":
    main()
