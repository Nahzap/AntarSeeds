from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

matplotlib.use("Agg")


PERCENT_COLUMNS = [
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mAP@R",
    "NMI",
    "F1_macro",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "precision_weighted",
    "F1_weighted",
    "best_recall_at_1",
]


def prettify_backbone_name(raw_name: str) -> str:
    mapping = {
        "convnext_v2_tiny": "ConvNeXt V2 Tiny",
        "resnet50": "ResNet-50",
        "dinov2_vits14": "DINOv2 ViT-S/14",
        "deit_small_patch16_224": "DeiT-S",
    }
    return mapping.get(raw_name, raw_name)


def compute_completion_percent(epoch_series: pd.Series) -> pd.Series:
    epochs = pd.to_numeric(epoch_series, errors="coerce")
    min_epoch = float(epochs.min())
    max_epoch = float(epochs.max())
    if max_epoch <= min_epoch:
        return pd.Series([100.0] * len(epochs), index=epochs.index)
    return ((epochs - min_epoch) / (max_epoch - min_epoch)) * 100.0


def load_run_data(run_dir: Path) -> tuple[pd.DataFrame, dict]:
    metrics_path = run_dir / "training_metrics.csv"
    config_path = run_dir / "config" / "config.yaml"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metrics_df = pd.read_csv(metrics_path)
    if "epoch" not in metrics_df.columns:
        raise ValueError(f"Column 'epoch' not found in {metrics_path}")
    metrics_df = metrics_df.sort_values("epoch").reset_index(drop=True)

    return metrics_df, config


def create_progress_table(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    progress_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for run_dir in run_dirs:
        metrics_df, config = load_run_data(run_dir)

        backbone_raw = str(config.get("model", {}).get("backbone_name", run_dir.name))
        model_name = prettify_backbone_name(backbone_raw)
        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        loss_cfg = train_cfg.get("loss", {})
        hard_mining_cfg = train_cfg.get("hard_mining", {})

        df = metrics_df.copy()
        df["model_name"] = model_name
        df["backbone_name"] = backbone_raw
        df["run_dir"] = str(run_dir)
        df["completion_pct"] = compute_completion_percent(df["epoch"]).round(2)

        for col in PERCENT_COLUMNS:
            if col in df.columns:
                df[f"{col}_pct"] = pd.to_numeric(df[col], errors="coerce") * 100.0

        if "recall_at_1_pct" in df.columns:
            df["score_pct"] = df["recall_at_1_pct"]
        else:
            df["score_pct"] = pd.NA

        output_cols = [
            "model_name",
            "backbone_name",
            "run_dir",
            "epoch",
            "completion_pct",
            "score_pct",
            "recall_at_1_pct",
            "recall_at_5_pct",
            "recall_at_10_pct",
            "mAP@R_pct",
            "NMI_pct",
            "F1_macro_pct",
            "accuracy_pct",
            "balanced_accuracy_pct",
            "F1_weighted_pct",
            "train_loss",
            "val_loss",
            "learning_rate",
            "is_best",
            "epoch_time_sec",
        ]
        output_cols = [c for c in output_cols if c in df.columns]
        progress_frames.append(df[output_cols])

        best_idx = pd.to_numeric(df.get("recall_at_1"), errors="coerce").idxmax()
        best_row = df.loc[best_idx]
        final_row = df.iloc[-1]
        summary_rows.append(
            {
                "model_name": model_name,
                "backbone_name": backbone_raw,
                "run_dir": str(run_dir),
                "experiment_name": config.get("experiment", {}).get("name", ""),
                "description": config.get("experiment", {}).get("description", ""),
                "epochs_configured": train_cfg.get("epochs"),
                "embedding_dim": config.get("model", {}).get("embedding_dim"),
                "image_size": data_cfg.get("image_size"),
                "batch_size": data_cfg.get("batch_size"),
                "samples_per_class": data_cfg.get("samples_per_class"),
                "optimizer": train_cfg.get("optimizer"),
                "scheduler": train_cfg.get("scheduler"),
                "learning_rate": train_cfg.get("learning_rate"),
                "weight_decay": train_cfg.get("weight_decay"),
                "loss_type": loss_cfg.get("type"),
                "loss_margin": loss_cfg.get("params", {}).get("margin"),
                "loss_scale": loss_cfg.get("params", {}).get("scale"),
                "hard_mining_enabled": hard_mining_cfg.get("enabled"),
                "hard_mining_ratio": hard_mining_cfg.get("ratio"),
                "best_epoch": int(best_row["epoch"]),
                "best_recall_at_1_pct": float(best_row.get("recall_at_1", 0.0)) * 100.0,
                "best_mAP@R_pct": float(best_row.get("mAP@R", 0.0)) * 100.0,
                "best_nmi_pct": float(best_row.get("NMI", 0.0)) * 100.0,
                "best_f1_macro_pct": float(best_row.get("F1_macro", 0.0)) * 100.0,
                "final_recall_at_1_pct": float(final_row.get("recall_at_1", 0.0)) * 100.0,
                "final_mAP@R_pct": float(final_row.get("mAP@R", 0.0)) * 100.0,
                "final_nmi_pct": float(final_row.get("NMI", 0.0)) * 100.0,
                "final_f1_macro_pct": float(final_row.get("F1_macro", 0.0)) * 100.0,
            }
        )

    progress_df = pd.concat(progress_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    return progress_df, summary_df


def save_comparison_plot(progress_df: pd.DataFrame, output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    for model_name, model_df in progress_df.groupby("model_name"):
        plot_df = model_df.sort_values("completion_pct")
        ax.plot(
            plot_df["completion_pct"],
            plot_df["score_pct"],
            marker="o",
            linewidth=2.2,
            markersize=4,
            label=model_name,
        )

    ax.set_title("Comparacion de rendimiento por progreso de entrenamiento", fontsize=13, fontweight="bold")
    ax.set_xlabel("Completacion del entrenamiento (%)")
    ax.set_ylabel("Score (Recall@1, %)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(title="Modelo", loc="lower right")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_readme(
    output_dir: Path,
    run_dirs: list[Path],
    progress_csv: Path,
    pivot_csv: Path,
    summary_csv: Path,
    plot_path: Path,
) -> None:
    text = [
        "# Comparacion de 4 modelos",
        "",
        "Este reporte compara los entrenamientos por epoca normalizados a 0-100% de completacion.",
        "",
        "## Runs incluidos",
    ]
    text.extend([f"- `{run_dir}`" for run_dir in run_dirs])
    text.extend(
        [
            "",
            "## Archivos generados",
            f"- `{progress_csv.name}`: tabla larga con metricas por epoca y score principal (Recall@1).",
            f"- `{pivot_csv.name}`: tabla pivot para comparar Recall@1 (%) de los 4 modelos por progreso.",
            f"- `{summary_csv.name}`: configuraciones clave + mejores/finales metricas por modelo.",
            f"- `{plot_path.name}`: grafico unico de Recall@1 (%) vs completacion (%).",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparative CSVs and plot for multiple training runs."
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to compare")
    parser.add_argument(
        "--reports-root",
        default="reports",
        help="Root directory where timestamped report folder will be created",
    )
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.runs]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.reports_root) / f"model_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    progress_df, summary_df = create_progress_table(run_dirs)

    progress_csv = output_dir / "epoch_progress_comparison_long.csv"
    progress_df.to_csv(progress_csv, index=False, float_format="%.6f")

    pivot_df = (
        progress_df.pivot_table(
            index="completion_pct",
            columns="model_name",
            values="score_pct",
            aggfunc="mean",
        )
        .sort_index()
        .reset_index()
    )
    pivot_csv = output_dir / "epoch_progress_comparison_pivot.csv"
    pivot_df.to_csv(pivot_csv, index=False, float_format="%.6f")

    summary_df = summary_df.sort_values("best_recall_at_1_pct", ascending=False).reset_index(drop=True)
    summary_csv = output_dir / "model_configuration_and_summary.csv"
    summary_df.to_csv(summary_csv, index=False, float_format="%.6f")

    plot_path = output_dir / "all_models_progress_comparison.png"
    save_comparison_plot(progress_df, plot_path)

    write_readme(output_dir, run_dirs, progress_csv, pivot_csv, summary_csv, plot_path)

    print(f"Report generated successfully: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
