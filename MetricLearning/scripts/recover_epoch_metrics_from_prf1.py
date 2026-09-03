"""
Recover epoch-level precision trend when training_metrics.csv was flattened.

This script targets runs where `precision_macro` became constant after a legacy
backfill. It reconstructs per-epoch precision using the same formula currently
used by post-training PR/F1 analysis:

    P = (F1 * R) / (2R - F1)

where:
  - F1 -> `F1_macro`
  - R  -> `recall_at_1`

The result is written back into `precision_macro` and a backup CSV is created.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _reconstruct_precision(df: pd.DataFrame) -> pd.Series:
    f1 = pd.to_numeric(df["F1_macro"], errors="coerce")
    recall = pd.to_numeric(df["recall_at_1"], errors="coerce")
    denom = (2.0 * recall) - f1
    precision = np.where(np.abs(denom) > 1e-12, (f1 * recall) / denom, np.nan)
    precision = pd.to_numeric(pd.Series(precision), errors="coerce")
    valid = precision.notna() & (precision >= 0.0) & (precision <= 1.0)
    return precision.where(valid, np.nan)


def recover_run(run_dir: Path) -> None:
    csv_path = run_dir / "training_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"F1_macro", "recall_at_1", "precision_macro"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    precision_unique = pd.to_numeric(df["precision_macro"], errors="coerce").dropna().nunique()
    f1_unique = pd.to_numeric(df["F1_macro"], errors="coerce").dropna().nunique()
    recall_unique = pd.to_numeric(df["recall_at_1"], errors="coerce").dropna().nunique()

    if precision_unique > 1:
        print(f"SKIP: {run_dir} -> precision_macro already varies ({precision_unique} unique values)")
        return
    if f1_unique <= 1 or recall_unique <= 1:
        raise ValueError(
            f"Cannot reconstruct precision trajectory (F1 unique={f1_unique}, recall unique={recall_unique})"
        )

    reconstructed = _reconstruct_precision(df)
    valid_count = int(reconstructed.notna().sum())
    if valid_count == 0:
        raise ValueError("Reconstruction produced no valid precision values")

    backup_path = csv_path.with_name(f"training_metrics_backup_before_precision_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(backup_path, index=False, encoding="utf-8")

    df["precision_macro"] = reconstructed.fillna(pd.to_numeric(df["precision_macro"], errors="coerce")).round(6)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"OK: {run_dir}")
    print(f"  backup: {backup_path}")
    print(f"  reconstructed precision rows: {valid_count}/{len(df)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover epoch precision trend from F1/Recall.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories")
    args = parser.parse_args()

    failures = []
    for run in args.runs:
        run_dir = Path(run)
        try:
            recover_run(run_dir)
        except Exception as exc:
            print(f"FAIL: {run_dir} -> {exc}")
            failures.append(run)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
