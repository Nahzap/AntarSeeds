#!/usr/bin/env python3
"""
Orchestrate ablation matrices F-01 … F-07 (Phase F).

Examples:
    # List variants without training
    python scripts/run_ablation_matrix.py --matrix pooling --dry-run

    # Generate configs only
    python scripts/run_ablation_matrix.py --matrix slices --prepare-only

    # Run full training for each variant
    python scripts/run_ablation_matrix.py --matrix augment --run

    # Aggregate existing batch manifest
    python scripts/run_ablation_matrix.py --collect ablations/augment_2026-06-12_120000/manifest.json
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.ablation_matrix import (
    MATRIX_REGISTRY,
    aggregate_manifest_to_csv,
    build_variant_config,
    collect_run_metrics,
    create_batch_dir,
    find_run_dir_by_experiment,
    get_matrix,
    iter_variants,
    list_matrices,
    save_manifest,
    write_results_csv,
)

logger = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_training(config_path: Path, python_exe: str) -> int:
    cmd = [python_exe, str(ROOT / "scripts" / "train.py"), "--config", str(config_path)]
    logger.info("Ejecutando: %s", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation matrices (Phase F)")
    parser.add_argument("--config", default="config.yaml", help="Base config.yaml")
    parser.add_argument(
        "--matrix",
        choices=list_matrices() + ["all"],
        help="Ablation matrix to run",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print variants only")
    parser.add_argument("--prepare-only", action="store_true", help="Write variant configs, no train")
    parser.add_argument("--run", action="store_true", help="Launch train.py per variant")
    parser.add_argument(
        "--collect",
        type=str,
        default=None,
        help="Path to manifest.json to aggregate into CSV",
    )
    parser.add_argument("--output", type=str, default=None, help="CSV output path")
    parser.add_argument("--batch-dir", type=str, default="ablations", help="Output root for batches")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Runs directory for matching")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for quick ablations")
    parser.add_argument("--python", default=sys.executable, help="Python executable for subprocess")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.collect:
        manifest = Path(args.collect)
        out = Path(args.output) if args.output else None
        csv_path = aggregate_manifest_to_csv(manifest, out)
        print(f"OK: {csv_path}")
        return 0

    if not args.matrix:
        parser.error("--matrix es obligatorio (o usa --collect)")

    base_config = _load_config(ROOT / args.config)
    matrices = list_matrices() if args.matrix == "all" else [args.matrix]

    all_rows = []
    for matrix_name in matrices:
        matrix = get_matrix(matrix_name)
        logger.info("=== Matriz %s (%s): %s ===", matrix["id"], matrix_name, matrix["description"])

        if args.dry_run:
            for variant, cfg in iter_variants(matrix_name, base_config, epochs_override=args.epochs):
                print(f"  [{variant['id']}] {variant.get('label')} -> experiment.name={cfg['experiment']['name']}")
            continue

        batch_dir = create_batch_dir(ROOT / args.batch_dir, matrix_name)
        manifest = {
            "matrix": matrix_name,
            "matrix_id": matrix["id"],
            "base_config": str(args.config),
            "batch_dir": str(batch_dir),
            "runs": [],
        }

        for variant, cfg in iter_variants(matrix_name, base_config, epochs_override=args.epochs):
            cfg_path = batch_dir / "configs" / f"{variant['id']}.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
            logger.info("Config: %s", cfg_path)

            entry = {
                "variant_id": variant["id"],
                "variant_label": variant.get("label"),
                "config_path": str(cfg_path),
                "run_dir": None,
            }

            if args.run:
                rc = _run_training(cfg_path, args.python)
                if rc != 0:
                    logger.error("Train falló para %s (exit %s)", variant["id"], rc)
                run_dir = find_run_dir_by_experiment(ROOT / args.runs_dir, variant["id"])
                if run_dir:
                    entry["run_dir"] = str(run_dir)
                    logger.info("Run detectado: %s", run_dir)
                else:
                    logger.warning("No se encontró run para experiment.name=%s", variant["id"])

            manifest["runs"].append(entry)

        manifest_path = batch_dir / "manifest.json"
        save_manifest(manifest_path, manifest)

        rows = []
        for entry in manifest["runs"]:
            row = {"matrix": matrix_name, "variant_id": entry["variant_id"], "variant_label": entry["variant_label"]}
            if entry.get("run_dir"):
                row.update(collect_run_metrics(Path(entry["run_dir"])))
            rows.append(row)

        if rows:
            csv_path = batch_dir / "results.csv"
            write_results_csv(rows, csv_path)
            all_rows.extend(rows)
            logger.info("Resultados: %s", csv_path)

        if args.prepare_only and not args.run:
            logger.info("Manifest: %s (prepare-only)", manifest_path)

    if args.matrix == "all" and all_rows and not args.dry_run:
        combined = ROOT / args.batch_dir / "all_matrices_results.csv"
        write_results_csv(all_rows, combined)
        print(f"OK: {combined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
