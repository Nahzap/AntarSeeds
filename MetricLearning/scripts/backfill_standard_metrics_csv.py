"""
Backfill standard classification metric columns into historical training_metrics.csv.

For legacy runs, per-epoch standard metrics are unavailable because those runs did not
persist per-epoch predictions/embeddings. This script backfills columns using the
run-level post-training metrics from evaluation_metrics_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


COLUMN_MAP = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": "precision_macro",
    "precision_weighted": "precision_weighted",
    "f1_weighted": "F1_weighted",
}


def _has_variation(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return int(numeric.nunique()) > 1


def backfill_run(run_dir: Path) -> None:
    csv_path = run_dir / "training_metrics.csv"
    summary_path = run_dir / "evaluation_metrics_summary.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    std = summary.get("classification_standard", {})

    df = pd.read_csv(csv_path)

    # Fill columns only when absent or empty.
    # Do NOT overwrite varying per-epoch metrics that may already be valid.
    for src_key, dst_col in COLUMN_MAP.items():
        run_level_val = round(float(std.get(src_key, 0.0)), 6)
        if dst_col not in df.columns:
            df[dst_col] = run_level_val
            continue

        col_numeric = pd.to_numeric(df[dst_col], errors="coerce")
        # If column is entirely empty, fill it.
        if col_numeric.dropna().empty:
            df[dst_col] = run_level_val
            continue

        # If per-epoch variation exists, preserve original values.
        if _has_variation(col_numeric):
            continue

        # Column is constant and non-empty:
        # preserve by default to avoid silently flattening potentially valid data.
        # Legacy callers can still force overwrite externally if needed.
        continue

    # Keep column ordering human-friendly.
    desired_order = [
        "epoch", "train_loss", "val_loss",
        "recall_at_1", "recall_at_5", "recall_at_10",
        "mAP@R", "NMI", "F1_macro",
        "accuracy", "balanced_accuracy", "precision_macro", "precision_weighted", "F1_weighted",
        "learning_rate", "is_best", "best_recall_at_1", "epoch_time_sec",
    ]
    cols = [c for c in desired_order if c in df.columns] + [c for c in df.columns if c not in desired_order]
    df = df[cols]

    # Enforce max 6 decimals in standard metric columns
    for col in COLUMN_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(6)

    df.to_csv(csv_path, index=False, encoding="utf-8")

    # Keep consistency across run-level artifacts
    class_csv = run_dir / "images" / "post_training" / "classification_report.csv"
    if class_csv.exists():
        cdf = pd.read_csv(class_csv)
        for col in ("precision", "recall", "f1"):
            if col in cdf.columns:
                cdf[col] = pd.to_numeric(cdf[col], errors="coerce").round(6)
        cdf.to_csv(class_csv, index=False, encoding="utf-8")

    # Round summary JSON numeric metrics to 6 decimals
    cls = summary.get("classification_standard", {})
    for k, v in list(cls.items()):
        if isinstance(v, (int, float)):
            cls[k] = round(float(v), 6)
    ml = summary.get("metric_learning", {})
    for k, v in list(ml.items()):
        if isinstance(v, (int, float)):
            ml[k] = round(float(v), 6)
    for row in summary.get("per_class", []):
        for key in ("precision", "recall", "f1"):
            if key in row and isinstance(row[key], (int, float)):
                row[key] = round(float(row[key]), 6)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill standard metric columns in training_metrics.csv")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories")
    args = parser.parse_args()

    failed = []
    for run in args.runs:
        run_dir = Path(run)
        try:
            backfill_run(run_dir)
            print(f"OK: {run_dir}")
        except Exception as exc:
            print(f"FAIL: {run_dir} -> {exc}")
            failed.append(run)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

