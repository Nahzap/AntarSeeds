"""
Complete implementation pipeline for paper-facing improvements.

This script executes a full "0 to 100%" completion workflow over selected runs:
  1) Validates required artifacts per run
  2) Backfills standard metrics into training_metrics.csv
  3) Backfills standard metric visualizations
  4) Aggregates 4-backbone comparison artifacts (CSV + Markdown)
  5) Computes speedup/throughput narrative indicators from explicit assumptions
  6) Writes a timestamped completion log in docs/ with progress indicators

It is intentionally deterministic and non-destructive for model checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Ensure project root is on sys.path when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backfill_standard_metrics_csv import backfill_run as backfill_csv_run
from scripts.backfill_standard_metric_visuals import process_run as backfill_visuals_run


@dataclass
class SpeedupAssumptions:
    capture_minutes: float
    min_grains_presence: int
    detect_ms_low: float
    detect_ms_high: float
    manual_best_case_minutes: float
    specialist_wait_days: float


def _fmt_num(value: float, decimals: int = 3) -> str:
    return f"{value:.{decimals}f}"


def _load_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "evaluation_metrics_summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_run(run_dir: Path) -> List[str]:
    required = [
        run_dir / "evaluation_metrics_summary.json",
        run_dir / "training_metrics.csv",
        run_dir / "training_report.md",
        run_dir / "images" / "post_training" / "classification_report.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return missing


def _compute_speedup(assumptions: SpeedupAssumptions) -> Dict[str, float]:
    # Total automatic processing time to reach minimum class declaration:
    # capture + (minimum grains * detector/classification ms per object)
    low_minutes = assumptions.capture_minutes + (
        assumptions.min_grains_presence * assumptions.detect_ms_low / 1000.0 / 60.0
    )
    high_minutes = assumptions.capture_minutes + (
        assumptions.min_grains_presence * assumptions.detect_ms_high / 1000.0 / 60.0
    )
    manual = assumptions.manual_best_case_minutes
    wait_minutes = assumptions.specialist_wait_days * 24.0 * 60.0
    return {
        "system_minutes_low": low_minutes,
        "system_minutes_high": high_minutes,
        "speedup_vs_manual_low": manual / high_minutes if high_minutes > 0 else 0.0,
        "speedup_vs_manual_high": manual / low_minutes if low_minutes > 0 else 0.0,
        "speedup_vs_wait_low": wait_minutes / high_minutes if high_minutes > 0 else 0.0,
        "speedup_vs_wait_high": wait_minutes / low_minutes if low_minutes > 0 else 0.0,
        "samples_per_hour_low": 60.0 / high_minutes if high_minutes > 0 else 0.0,
        "samples_per_hour_high": 60.0 / low_minutes if low_minutes > 0 else 0.0,
    }


def _write_run_update_summary(
    run_dir: Path,
    speedup: Dict[str, float],
    assumptions: SpeedupAssumptions,
    timestamp_tag: str,
) -> Path:
    summary = _load_summary(run_dir)
    std = summary.get("classification_standard", {})
    ml = summary.get("metric_learning", {})
    post_dir = run_dir / "images" / "post_training"
    standard_overview = post_dir / "standard_metrics_overview.png"
    per_class_matrix = post_dir / "per_class_precision_recall_f1.png"

    output_path = run_dir / f"paper_update_summary_{timestamp_tag}.md"
    lines = [
        f"# Paper update summary - {run_dir.name}",
        "",
        f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Run: `{run_dir}`",
        "",
        "## Standard metrics",
        "",
        f"- Accuracy: `{std.get('accuracy', 0.0):.6f}`",
        f"- Balanced Accuracy: `{std.get('balanced_accuracy', 0.0):.6f}`",
        f"- Precision Macro: `{std.get('precision_macro', 0.0):.6f}`",
        f"- Precision Weighted: `{std.get('precision_weighted', 0.0):.6f}`",
        f"- F1 Macro: `{std.get('f1_macro', 0.0):.6f}`",
        f"- F1 Weighted: `{std.get('f1_weighted', 0.0):.6f}`",
        "",
        "## Metric-learning indicators",
        "",
        f"- kNN Accuracy: `{ml.get('knn_accuracy', 0.0):.6f}`",
        f"- Distance Ratio (inter/intra): `{ml.get('distance_ratio_inter_intra', 0.0):.6f}`",
        "",
        "## Speedup assumptions used",
        "",
        f"- capture_minutes: `{assumptions.capture_minutes}`",
        f"- min_grains_presence: `{assumptions.min_grains_presence}`",
        f"- detect_ms_range: `{assumptions.detect_ms_low}-{assumptions.detect_ms_high}`",
        f"- manual_best_case_minutes: `{assumptions.manual_best_case_minutes}`",
        f"- specialist_wait_days: `{assumptions.specialist_wait_days}`",
        "",
        "## Derived speedup indicators",
        "",
        f"- system_minutes_range: `{speedup['system_minutes_low']:.2f} - {speedup['system_minutes_high']:.2f}`",
        f"- speedup_vs_manual_range: `{speedup['speedup_vs_manual_low']:.2f}x - {speedup['speedup_vs_manual_high']:.2f}x`",
        f"- speedup_vs_wait_range: `{speedup['speedup_vs_wait_low']:.1f}x - {speedup['speedup_vs_wait_high']:.1f}x`",
        "",
        "## Generated visual artifacts",
        "",
        f"- standard_metrics_overview: `{standard_overview}` (exists={standard_overview.exists()})",
        f"- per_class_precision_recall_f1: `{per_class_matrix}` (exists={per_class_matrix.exists()})",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _append_training_report_update(
    run_dir: Path,
    update_summary_path: Path,
) -> bool:
    report_path = run_dir / "training_report.md"
    if not report_path.exists():
        return False
    marker = update_summary_path.name
    text = report_path.read_text(encoding="utf-8")
    if marker in text:
        return True

    section = (
        "\n\n## Paper Improvements Update\n\n"
        f"- Generated summary: `{update_summary_path.name}`\n"
        "- Standard metric visuals regenerated:\n"
        "  - `images/post_training/standard_metrics_overview.png`\n"
        "  - `images/post_training/per_class_precision_recall_f1.png`\n"
    )
    report_path.write_text(text + section, encoding="utf-8")
    return True


def _build_comparison_rows(run_dirs: List[Path]) -> pd.DataFrame:
    rows: List[Dict] = []
    for run_dir in run_dirs:
        summary = _load_summary(run_dir)
        std = summary.get("classification_standard", {})
        ml = summary.get("metric_learning", {})
        rows.append(
            {
                "run": run_dir.name,
                "knn_accuracy": float(ml.get("knn_accuracy", 0.0)),
                "distance_ratio_inter_intra": float(
                    ml.get("distance_ratio_inter_intra", 0.0)
                ),
                "accuracy": float(std.get("accuracy", 0.0)),
                "balanced_accuracy": float(std.get("balanced_accuracy", 0.0)),
                "precision_macro": float(std.get("precision_macro", 0.0)),
                "precision_weighted": float(std.get("precision_weighted", 0.0)),
                "f1_macro": float(std.get("f1_macro", 0.0)),
                "f1_weighted": float(std.get("f1_weighted", 0.0)),
            }
        )

    df = pd.DataFrame(rows)
    # Primary ranking: accuracy, f1_weighted, distance ratio.
    df = df.sort_values(
        by=["accuracy", "f1_weighted", "distance_ratio_inter_intra"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def _write_completion_markdown(
    path: Path,
    completion_steps: List[Tuple[str, bool]],
    run_dirs: List[Path],
    comparison_df: pd.DataFrame,
    speedup: Dict[str, float],
    assumptions: SpeedupAssumptions,
    artifact_paths: Dict[str, Path],
) -> None:
    completed = sum(1 for _, ok in completion_steps if ok)
    total = len(completion_steps)
    pct = (completed / total) * 100.0 if total else 0.0
    now = datetime.now()

    lines: List[str] = []
    lines.append("# Registro de implementación 0% -> 100%")
    lines.append("")
    lines.append(f"**Fecha:** {now.strftime('%Y-%m-%d')}")
    lines.append(f"**Hora:** {now.strftime('%H:%M:%S')}")
    lines.append("**Zona horaria:** UTC-4")
    lines.append("")
    lines.append("## Estado global")
    lines.append("")
    lines.append(f"- Progreso total: **{_fmt_num(pct, 1)}%** ({completed}/{total} hitos)")
    lines.append(
        f"- Runs objetivo: {', '.join(f'`{r.name}`' for r in run_dirs)}"
    )
    lines.append("")
    lines.append("## Indicadores de completación")
    lines.append("")
    for idx, (title, ok) in enumerate(completion_steps, start=1):
        mark = "OK" if ok else "PENDIENTE"
        lines.append(f"- [{mark}] Hito {idx:02d}: {title}")
    lines.append("")
    lines.append("## Resultados comparativos (4 runs)")
    lines.append("")
    lines.append("| Rank | Run | Accuracy | F1 weighted | Distance ratio |")
    lines.append("|---:|---|---:|---:|---:|")
    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['run']} | "
            f"{row['accuracy']:.6f} | {row['f1_weighted']:.6f} | "
            f"{row['distance_ratio_inter_intra']:.6f} |"
        )
    lines.append("")
    lines.append("## Indicadores de speedup (supuestos explícitos)")
    lines.append("")
    lines.append(
        f"- Captura por muestra: `{assumptions.capture_minutes}` min"
    )
    lines.append(
        f"- Mínimo granos para declarar presencia de clase: `{assumptions.min_grains_presence}`"
    )
    lines.append(
        f"- Detección/clasificación por objeto: `{assumptions.detect_ms_low}-{assumptions.detect_ms_high}` ms"
    )
    lines.append(
        f"- Manual (mejor caso): `{assumptions.manual_best_case_minutes}` min"
    )
    lines.append(
        f"- Espera especialista (primera lectura): `{assumptions.specialist_wait_days}` días"
    )
    lines.append("")
    lines.append(
        f"- Tiempo sistema (rango): "
        f"`{_fmt_num(speedup['system_minutes_low'], 2)} - {_fmt_num(speedup['system_minutes_high'], 2)} min`"
    )
    lines.append(
        f"- Speedup vs manual (rango): "
        f"`{_fmt_num(speedup['speedup_vs_manual_low'], 2)}x - {_fmt_num(speedup['speedup_vs_manual_high'], 2)}x`"
    )
    lines.append(
        f"- Speedup vs espera especialista (rango): "
        f"`{_fmt_num(speedup['speedup_vs_wait_low'], 1)}x - {_fmt_num(speedup['speedup_vs_wait_high'], 1)}x`"
    )
    lines.append(
        f"- Throughput equivalente (muestras/h): "
        f"`{_fmt_num(speedup['samples_per_hour_low'], 2)} - {_fmt_num(speedup['samples_per_hour_high'], 2)}`"
    )
    lines.append("")
    lines.append("## Artefactos generados")
    lines.append("")
    for key, value in artifact_paths.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Observaciones para paper/LaTeX")
    lines.append("")
    lines.append(
        "- El claim `6x` debe reportarse como rango con fórmula y supuestos; "
        "con los parámetros actuales el rango esperado es mayor."
    )
    lines.append(
        "- La selección de backbone puede reportarse en dos ejes: "
        "rendimiento global y separación geométrica."
    )
    lines.append(
        "- Las figuras de métricas estándar por run fueron regeneradas/aseguradas."
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    run_dirs: List[Path],
    docs_dir: Path,
    assumptions: SpeedupAssumptions,
) -> Dict[str, Path]:
    completion_steps: List[Tuple[str, bool]] = []

    # 1) Validate run structure.
    all_valid = True
    missing_by_run: Dict[str, List[str]] = {}
    for run_dir in run_dirs:
        missing = _validate_run(run_dir)
        if missing:
            all_valid = False
            missing_by_run[run_dir.name] = missing
    completion_steps.append(("Validación de artefactos base por run", all_valid))
    if not all_valid:
        missing_lines = []
        for run_name, missing in missing_by_run.items():
            missing_lines.append(f"{run_name}: {missing}")
        raise FileNotFoundError("Faltan artefactos requeridos: " + " | ".join(missing_lines))

    # 2) Backfill CSV metrics.
    csv_ok = True
    for run_dir in run_dirs:
        try:
            backfill_csv_run(run_dir)
        except Exception:
            csv_ok = False
    completion_steps.append(("Backfill de métricas estándar en training_metrics.csv", csv_ok))

    # 3) Backfill visuals.
    vis_ok = True
    for run_dir in run_dirs:
        try:
            backfill_visuals_run(run_dir)
        except Exception:
            vis_ok = False
    completion_steps.append(("Backfill de visuales estándar por run", vis_ok))

    # 4) Build comparison artifacts.
    comparison_ok = True
    comparison_df = pd.DataFrame()
    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = run_dirs[0].parent
    comparison_csv = runs_dir / f"backbone_comparison_{now_tag}.csv"
    comparison_md = runs_dir / f"backbone_comparison_{now_tag}.md"
    try:
        comparison_df = _build_comparison_rows(run_dirs)
        comparison_df.to_csv(comparison_csv, index=False, encoding="utf-8")

        md_lines = [
            "# Backbone comparison (4 runs)",
            "",
            "| Rank | Run | Accuracy | Balanced Accuracy | Precision Macro | Precision Weighted | F1 Macro | F1 Weighted | Distance Ratio |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in comparison_df.iterrows():
            md_lines.append(
                f"| {int(row['rank'])} | {row['run']} | {row['accuracy']:.6f} | "
                f"{row['balanced_accuracy']:.6f} | {row['precision_macro']:.6f} | "
                f"{row['precision_weighted']:.6f} | {row['f1_macro']:.6f} | "
                f"{row['f1_weighted']:.6f} | {row['distance_ratio_inter_intra']:.6f} |"
            )
        comparison_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    except Exception:
        comparison_ok = False
    completion_steps.append(("Generación de comparación consolidada de backbones", comparison_ok))

    # 5) Speedup indicators.
    speedup_ok = True
    speedup = {}
    try:
        speedup = _compute_speedup(assumptions)
    except Exception:
        speedup_ok = False
    completion_steps.append(("Cálculo de indicadores de speedup y throughput", speedup_ok))

    # 6) Per-run update summaries + report patching.
    run_update_ok = True
    run_update_paths: Dict[str, Path] = {}
    run_reports_patched = True
    for run_dir in run_dirs:
        try:
            update_path = _write_run_update_summary(
                run_dir=run_dir,
                speedup=speedup,
                assumptions=assumptions,
                timestamp_tag=now_tag,
            )
            run_update_paths[run_dir.name] = update_path
            if not _append_training_report_update(run_dir, update_path):
                run_reports_patched = False
        except Exception:
            run_update_ok = False
            run_reports_patched = False
    completion_steps.append(("Generación de resumen de actualización por run", run_update_ok))
    completion_steps.append(("Actualización de training_report.md por run", run_reports_patched))

    # 7) Final timestamped completion log.
    completion_ok = True
    completion_md = docs_dir / f"{now_tag}_registro_completacion_0_100.md"
    artifacts = {
        "comparison_csv": comparison_csv,
        "comparison_md": comparison_md,
        "completion_log_md": completion_md,
    }
    for run_name, run_path in run_update_paths.items():
        artifacts[f"run_update_{run_name}"] = run_path
    try:
        _write_completion_markdown(
            path=completion_md,
            completion_steps=completion_steps + [("Registro final timestampado en docs", True)],
            run_dirs=run_dirs,
            comparison_df=comparison_df,
            speedup=speedup,
            assumptions=assumptions,
            artifact_paths=artifacts,
        )
    except Exception:
        completion_ok = False
    completion_steps.append(("Registro final timestampado en docs", completion_ok))

    # 8) Final status JSON (optional machine-readable).
    status_json = docs_dir / f"{now_tag}_registro_completacion_0_100.json"
    status_payload = {
        "generated_at": datetime.now().isoformat(),
        "runs": [str(r) for r in run_dirs],
        "completion_steps": [
            {"step": title, "ok": ok} for title, ok in completion_steps
        ],
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "assumptions": assumptions.__dict__,
    }
    status_json.write_text(json.dumps(status_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts["completion_status_json"] = status_json

    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete paper-improvement implementation for selected runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            "runs/2026-03-20_ConvNextV2-Tiny_00",
            "runs/2026-03-20_DeiT-s_03",
            "runs/2026-03-20_DINOv2 ViT-s_00",
            "runs/2026-03-20_ResNet-50_00",
        ],
        help="Run directories to process.",
    )
    parser.add_argument("--capture-minutes", type=float, default=30.0)
    parser.add_argument("--min-grains-presence", type=int, default=1200)
    parser.add_argument("--detect-ms-low", type=float, default=15.0)
    parser.add_argument("--detect-ms-high", type=float, default=18.0)
    parser.add_argument("--manual-best-case-minutes", type=float, default=1440.0)
    parser.add_argument("--specialist-wait-days", type=float, default=30.0)
    parser.add_argument("--docs-dir", type=str, default="docs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(r) for r in args.runs]
    assumptions = SpeedupAssumptions(
        capture_minutes=args.capture_minutes,
        min_grains_presence=args.min_grains_presence,
        detect_ms_low=args.detect_ms_low,
        detect_ms_high=args.detect_ms_high,
        manual_best_case_minutes=args.manual_best_case_minutes,
        specialist_wait_days=args.specialist_wait_days,
    )
    artifacts = run_pipeline(
        run_dirs=run_dirs,
        docs_dir=Path(args.docs_dir),
        assumptions=assumptions,
    )
    print("Pipeline completado. Artefactos:")
    for key, value in artifacts.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()

