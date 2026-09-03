#!/usr/bin/env python3
"""Export LaTeX tables for IEEE TPAMI manuscript from frozen evidence JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def pct(v: float, digits: int = 1) -> str:
    return f"{100.0 * v:.{digits}f}\\%"


def write_dataset_table(out: Path, stats: dict) -> None:
    train = stats["train"]
    val_n = 4362
    test_n = 4353
    total = train["total_samples"] + val_n + test_n
    lines = [
        r"\begin{table}[!t]",
        r"\caption{PollenBB16 stratified split (28,816 grain instances from 16 species~\cite{pollenbb162026}).}",
        r"\label{tab:dataset}",
        r"\centering",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Split & Images / grains & Role \\",
        r"\midrule",
        f"Train & {train['total_samples']:,} grains & Metric-learning training \\\\",
        f"Validation & {val_n:,} grains & Classifier checkpoint (mAP@R) \\\\",
        f"Test & {test_n:,} grains & Held-out classification \\\\",
        r"\midrule",
        f"Total & {total:,} grains & 16 classes, stratified split \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_detection_table(out: Path, det: dict) -> None:
    lines = [
        r"\begin{table}[!t]",
        r"\caption{ViT-dense detector on held-out test split (2,375 full-field images).}",
        r"\label{tab:detection}",
        r"\centering",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Det-F1 (IoU $\\geq$ 0.5) & {pct(det['det_f1'])} \\\\",
        f"Det-Precision & {pct(det['det_precision'])} \\\\",
        f"Det-Recall & {pct(det['det_recall'])} \\\\",
        f"Fm (SOD, $\\beta$=0.3) & {pct(det['Fm'])} \\\\",
        f"Mask IoU & {pct(det['mask_iou'])} \\\\",
        f"SOD MAE & {det['sod_mae']:.4f} \\\\",
        f"Grains MAE & {det['grains_mae']:.3f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_classification_table(out: Path, cls: dict) -> None:
    ml = cls["metric_learning"]
    sa = cls["classification_standard"]["slice_aware"]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Classifier on held-out test crops (4,353 grains, MLRC + slice-aware).}",
        r"\label{tab:classification}",
        r"\centering",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"R@1 (1-NN LOO, MLRC) & {pct(ml['R@1'])} \\\\",
        f"R@5 & {pct(ml['R@5'])} \\\\",
        f"mAP@R & {pct(ml['mAP@R'])} \\\\",
        f"NMI & {pct(ml['NMI'])} \\\\",
        f"Macro-F1 (1-NN LOO) & {pct(ml['F1_macro'])} \\\\",
        f"Slice-aware accuracy & {pct(ml['slice_accuracy'])} \\\\",
        f"Balanced accuracy (slice) & {pct(sa['balanced_accuracy'])} \\\\",
        f"Inter/intra distance ratio & {ml['distance_ratio_inter_intra']:.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_e2e_table(out: Path, e2e: dict) -> None:
    t = e2e["totals"]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{End-to-end audit on validation gallery (46 images, ViT-dense coarse-fine).}",
        r"\label{tab:e2e}",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Value & Count \\",
        r"\midrule",
        f"Detection recall (annotated, IoU $\\geq$ 0.3) & {pct(e2e['detection_recall_on_annotated'])} & {t['matched']}/{t['annotated']} \\\\",
        f"Classification on matched & {pct(e2e['classification_on_matched_only'])} & {t['matched_cls_ok']}/{t['matched']} \\\\",
        f"Legacy E2E accuracy & {pct(e2e['legacy_e2e_accuracy'])} & {t['legacy_ok']}/{t['detected']} \\\\",
        f"Missed annotations & -- & {t['missed']} \\\\",
        f"Extra false positives & -- & {t['extra_fp']} \\\\",
        f"Matched but wrong class & -- & {t['matched_cls_bad']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_perclass_table(out: Path, cls: dict) -> None:
    rows = cls["per_class"]["slice_aware"]
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Per-class slice-aware test performance (precision / recall / F1).}",
        r"\label{tab:perclass}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Species & Prec. & Rec. & F1 & Support \\",
        r"\midrule",
    ]
    for r in rows:
        name = r["class"].replace("_", r"\_")
        lines.append(
            f"\\textit{{{name}}} & {r['precision']:.3f} & {r['recall']:.3f} & "
            f"{r['f1']:.3f} & {r['support']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_embedding_metrics_table(out: Path, cls: dict, spectral: dict) -> None:
    ml = cls["metric_learning"]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Embedding geometry and spectral health (test, $N{=}4353$).}",
        r"\label{tab:embedding_metrics}",
        r"\centering",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Inter/intra distance ratio & {ml['distance_ratio_inter_intra']:.2f} \\\\",
        f"NMI & {pct(ml['NMI'])} \\\\",
        f"Global effective rank & {spectral['global_effective_rank']}/{spectral['max_dim']} \\\\",
        f"Collapse severity ($\\lambda_1/\\Sigma\\lambda$) & {spectral['collapse_severity']:.3f} \\\\",
        f"Collapsed class pairs & {spectral['collapsed_pairs']}/{spectral['total_pairs']} \\\\",
        f"Mean pair effective rank & {spectral['mean_pair_rank']:.1f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_spectral_pairs_table(out: Path, report_path: Path) -> None:
    import re

    pairs = []
    if report_path.exists():
        text = report_path.read_text(encoding="utf-8")
        blocks = re.split(r"\n(?=\d+\.\s)", text)
        for block in blocks:
            m = re.search(
                r"(.+?)\s+vs\s+(.+?):\s+lambda_1\s+=\s+([\d.]+)\s+\(([\d.]+)%\)\s+Effective Rank\s+=\s+(\d+)",
                block,
            )
            if m:
                a = re.sub(r"^\d+\.\s*", "", m.group(1).strip())
                b = re.sub(r"^\d+\.\s*", "", m.group(2).strip())
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "pct": float(m.group(4)),
                        "rank": int(m.group(5)),
                        "collapsed": "COLLAPSED" in block,
                    }
                )
    pairs = pairs[:6]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Top class pairs by spectral dominance ($\lambda_1$ share).}",
        r"\label{tab:spectral_pairs}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Class A & Class B & $\lambda_1$ (\%) & Eff.\ rank & Status \\",
        r"\midrule",
    ]
    for p in pairs:
        a = p["a"].replace("_", r"\_")
        b = p["b"].replace("_", r"\_")
        status = "collapsed" if p["collapsed"] else "healthy"
        lines.append(
            f"\\textit{{{a}}} & \\textit{{{b}}} & {p['pct']:.1f} & {p['rank']} & {status} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_error_budget_table(out: Path, e2e: dict) -> None:
    t = e2e["totals"]
    ann = t["annotated"]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{E2E error decomposition (validation gallery).}",
        r"\label{tab:error_budget}",
        r"\centering",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Error type & Count & \% of annotated \\",
        r"\midrule",
        f"Missed detection & {t['missed']} & {100*t['missed']/ann:.1f}\\% \\\\",
        f"Extra FP (no annotation) & {t['extra_fp']} & -- \\\\",
        f"Matched, wrong class & {t['matched_cls_bad']} & {100*t['matched_cls_bad']/ann:.1f}\\% \\\\",
        f"Correct E2E & {t['matched_cls_ok']} & {100*t['matched_cls_ok']/ann:.1f}\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-dir",
        default="paper/ieee_tpami_preliminary/evidence",
    )
    parser.add_argument(
        "--output-dir",
        default="paper/ieee_tpami_preliminary/tables",
    )
    args = parser.parse_args()
    evidence = Path(args.evidence_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = load_json(evidence / "dataset_stats_train.json")
    det = load_json(evidence / "detection_test.json")
    cls = load_json(evidence / "classification_test.json")
    e2e = load_json(evidence / "e2e_gallery_val.json")
    spectral_path = evidence / "spectral_summary.json"
    spectral = load_json(spectral_path) if spectral_path.exists() else {
        "global_effective_rank": 29,
        "max_dim": 128,
        "collapse_severity": 0.156,
        "collapsed_pairs": 2,
        "total_pairs": 120,
        "mean_pair_rank": 25.5,
    }
    report_path = Path("runs/2026-06-12_17-16-54_BEST_LAST/images/post_training/test/spectral_analysis/spectral_analysis_report.txt")

    write_dataset_table(out_dir / "table_dataset.tex", stats)
    write_detection_table(out_dir / "table_detection.tex", det)
    write_classification_table(out_dir / "table_classification.tex", cls)
    write_e2e_table(out_dir / "table_e2e.tex", e2e)
    write_perclass_table(out_dir / "table_perclass.tex", cls)
    write_error_budget_table(out_dir / "table_error_budget.tex", e2e)
    write_embedding_metrics_table(out_dir / "table_embedding_metrics.tex", cls, spectral)
    write_spectral_pairs_table(out_dir / "table_spectral_pairs.tex", report_path)
    print(f"Wrote LaTeX tables to {out_dir}")


if __name__ == "__main__":
    main()
