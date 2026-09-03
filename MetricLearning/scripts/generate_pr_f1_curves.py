"""
Generate PR trajectory and F1 curves by epoch for a run.

Usage:
    python scripts/generate_pr_f1_curves.py --run-dir "runs/2026-03-20_DINOv2 ViT-s_00"
"""

import argparse
import logging
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.post_training import generate_pr_f1_epoch_curves


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(description='Generate PR/F1 epoch curves from training_metrics.csv')
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Path to run directory containing training_metrics.csv'
    )
    parser.add_argument(
        '--output-subdir',
        type=str,
        default='images/post_training',
        help='Subdirectory inside run-dir where images will be saved'
    )
    args = parser.parse_args()

    _setup_logging()
    run_dir = Path(args.run_dir)
    csv_path = run_dir / 'training_metrics.csv'
    output_dir = run_dir / args.output_subdir

    artifacts = generate_pr_f1_epoch_curves(str(csv_path), str(output_dir))
    if artifacts:
        logging.info("PR/F1 epoch curve generation completed.")
        for name, path in artifacts.items():
            logging.info(f"  {name}: {path}")
    else:
        logging.warning("No PR/F1 artifacts were generated. Check CSV availability/columns.")


if __name__ == '__main__':
    main()
