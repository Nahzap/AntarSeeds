#!/usr/bin/env python3
"""
Smoke test: 1-epoch training with configurable pooling + sliced_ms.

Usage:
  python scripts/smoke_test_pooling_slice.py
  python scripts/smoke_test_pooling_slice.py --config test_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("smoke_test")


def main():
    parser = argparse.ArgumentParser(description="Smoke test pooling + slices (1 epoch)")
    parser.add_argument("--config", default="test_config.yaml")
    args = parser.parse_args()

    from src.utils.config_validator import validate_config_file
    from src.utils.config_utils import load_config

    config_path = ROOT / args.config
    validation = validate_config_file(str(config_path), check_data_dirs=True)
    logger.info("Pipeline: %s", validation["pipeline_summary"])
    for w in validation.get("warnings", []):
        logger.warning("Config: %s", w)
    if not validation["valid"]:
        for e in validation["errors"]:
            logger.error("Config: %s", e)
        raise SystemExit(1)

    config = load_config(str(config_path))
    config["training"]["epochs"] = 1

    logger.info("Running unit tests (pooling + slices)...")
    import pytest

    code = pytest.main(
        [
            "-q",
            str(ROOT / "tests" / "test_pooling_module.py"),
            str(ROOT / "tests" / "test_multi_tower.py"),
            str(ROOT / "tests" / "test_slice_utils_and_config_validator.py"),
            str(ROOT / "tests" / "test_slice_classifier.py"),
        ]
    )
    if code != 0:
        raise SystemExit(code)

    logger.info("Running 1-epoch training smoke test...")
    from scripts.train import main as train_main

    sys.argv = ["train.py", "--config", str(config_path)]
    train_main()


if __name__ == "__main__":
    main()
