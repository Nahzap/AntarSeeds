"""
Backfill lightweight standard-metric visualizations for existing runs.

Reads already-generated artifacts:
  - evaluation_metrics_summary.json
  - images/post_training/classification_report.csv

Generates:
  - images/post_training/standard_metrics_overview.png
  - images/post_training/per_class_precision_recall_f1.png

Also appends references to training_report.md if missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_DPI = 1200


def plot_standard_metrics(summary_json: Path, output_path: Path, dpi: int = DEFAULT_DPI) -> None:
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cls = data.get("classification_standard", {})
    ordered_keys = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("precision_macro", "Precision Macro"),
        ("precision_weighted", "Precision Weighted"),
        ("f1_macro", "F1 Macro"),
        ("f1_weighted", "F1 Weighted"),
    ]
    labels = [label for key, label in ordered_keys]
    values = [float(cls.get(key, 0.0)) for key, _ in ordered_keys]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = list(range(len(labels)))
    ax.plot(x, values, marker="o", linewidth=2.5, markersize=7, color="#2b6cb0")
    ax.fill_between(x, values, [0] * len(values), color="#2b6cb0", alpha=0.08)
    # Dynamic y-range reduces excessive whitespace for high-score runs.
    ymin = max(0.0, min(values) - 0.08)
    ymax = min(1.02, max(values) + 0.05)
    if ymax - ymin < 0.12:
        pad = (0.12 - (ymax - ymin)) / 2
        ymin = max(0.0, ymin - pad)
        ymax = min(1.02, ymax + pad)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Standard Classification Metrics (k-NN LOO)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)

    for i, val in enumerate(values):
        ax.text(i, min(val + 0.015, 1.02), f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_per_class_prf1(class_report_csv: Path, output_path: Path, dpi: int = DEFAULT_DPI) -> None:
    df = pd.read_csv(class_report_csv)
    classes = df["class"].tolist()
    plt.style.use("seaborn-v0_8-whitegrid")
    matrix_df = pd.DataFrame(
        [df["precision"].values, df["recall"].values, df["f1"].values],
        index=["Precision", "Recall", "F1"],
        columns=classes,
    )

    fig, ax = plt.subplots(figsize=(max(11, len(classes) * 1.5), 4.8))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".6f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="#e5e7eb",
        cbar_kws={"label": "Score"},
        ax=ax,
    )
    ax.set_title("Per-Class Metrics Matrix (Precision / Recall / F1)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Metric", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def patch_training_report(report_path: Path) -> None:
    if not report_path.exists():
        return
    text = report_path.read_text(encoding="utf-8")
    needed_lines = [
        "- `images/post_training/standard_metrics_overview.png` — Accuracy/Balanced Accuracy/Precision/F1 overview",
        "- `images/post_training/per_class_precision_recall_f1.png` — Precision/Recall/F1 by class",
    ]
    changed = False
    for line in needed_lines:
        if line not in text:
            text += f"\n{line}"
            changed = True
    if changed:
        report_path.write_text(text, encoding="utf-8")


def process_run(run_dir: Path, dpi: int = DEFAULT_DPI) -> None:
    summary_json = run_dir / "evaluation_metrics_summary.json"
    class_csv = run_dir / "images" / "post_training" / "classification_report.csv"
    post_dir = run_dir / "images" / "post_training"
    if not summary_json.exists():
        raise FileNotFoundError(f"Missing {summary_json}")
    if not class_csv.exists():
        raise FileNotFoundError(f"Missing {class_csv}")

    plot_standard_metrics(summary_json, post_dir / "standard_metrics_overview.png", dpi=dpi)
    plot_per_class_prf1(class_csv, post_dir / "per_class_precision_recall_f1.png", dpi=dpi)
    patch_training_report(run_dir / "training_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill standard metric visualizations.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Output DPI for generated figures")
    args = parser.parse_args()

    failed = []
    for run in args.runs:
        run_dir = Path(run)
        try:
            process_run(run_dir, dpi=args.dpi)
            print(f"OK: {run_dir}")
        except Exception as exc:
            print(f"FAIL: {run_dir} -> {exc}")
            failed.append(run)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

