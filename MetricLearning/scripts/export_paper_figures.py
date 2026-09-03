#!/usr/bin/env python3
"""Generate IEEE-formatted figures for the TPAMI manuscript from frozen CSV/JSON."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import image as mpimg
from matplotlib.colors import LinearSegmentedColormap

# IEEE compsoc single-column figure width (inches); full width = 7.16
IEEE_COL_W = 3.5
IEEE_PAGE_W = 7.16

CLASS_SHORT = {
    "Anthemis_cotula": "Ant.cot",
    "Aristotelia_chilensis": "Ari.chi",
    "Brassica_rapa": "Bra.rap",
    "Castanea_sativa": "Cas.sat",
    "Conium_maculatum": "Con.mac",
    "Escallonia_pulverulenta": "Esc.pul",
    "Eucryphia_glutinosa": "Euc.glu",
    "Galega_officinalis": "Gal.off",
    "Gevuina_avellana": "Gev.ave",
    "Lomatia_hirsuta": "Lom.hir",
    "Medicago_sativa": "Med.sat",
    "Mentha_pulegium": "Men.pul",
    "Otholobium_glandulosum": "Oth.gla",
    "Quillaja_saponaria": "Qui.sap",
    "Rumex_acetosella": "Rum.ace",
    "Schinus_polygamus": "Sch.pol",
}


def apply_ieee_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.0,
        }
    )


def save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {path.with_suffix('.pdf')}")


def short_label(name: str) -> str:
    return CLASS_SHORT.get(name, name.replace("_", " ")[:10])


def error_budget_figure(e2e_path: Path, out_path: Path) -> None:
    with e2e_path.open(encoding="utf-8") as f:
        e2e = json.load(f)
    t = e2e["totals"]
    labels = ["Correct", "Missed", "Cls. err.", "Extra FP"]
    values = [t["matched_cls_ok"], t["missed"], t["matched_cls_bad"], t["extra_fp"]]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Instances")
    ax.set_title("E2E error budget (46-image gallery)")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            str(val),
            ha="center",
            va="bottom",
            fontsize=7,
        )
    save_fig(fig, out_path)


def pipeline_figure(out_path: Path) -> None:
    """Three-phase architecture diagram (IEEE full width)."""
    fig, ax = plt.subplots(figsize=(IEEE_PAGE_W, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef6fb", ec="#1f77b4"):
        rect = plt.Rectangle((x, y), w, h, fill=True, facecolor=fc, edgecolor=ec, linewidth=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.8)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#333", lw=0.8),
        )

    for x, label in [
        (0.2, "PHASE A — Annotation"),
        (4.9, "PHASE B — Offline training"),
        (9.6, "PHASE C — E2E deploy"),
    ]:
        ax.text(x, 6.55, label, fontsize=7.5, fontweight="bold", color="#333")

    box(0.2, 5.5, 2.0, 0.75, "PollenBB16\nfull-field $\\mathbf{I}$")
    box(2.4, 5.5, 2.2, 0.75, "U$^2$-NetP / expert\n$\\rightarrow$ \\texttt{.seg}")
    box(0.2, 4.3, 4.4, 0.9, "Contour $\\mathcal{C}_i$, bbox, class\n(on-the-fly crops)")

    box(4.9, 5.5, 2.1, 0.75, "Union mask $G$")
    box(7.2, 5.5, 2.1, 0.75, "ViT-dense $f_{\\mathrm{det}}$\n$\\hat{S}$ @ 504px")
    box(4.9, 4.3, 4.4, 0.9, "$\\mathcal{L}_{\\mathrm{det}}=$BCE+Dice\nPollenViTDetector ckpt")

    box(4.9, 2.8, 2.1, 0.75, "Crop $(\\mathbf{x},M)$\n252px + mask")
    box(7.2, 2.8, 2.1, 0.75, "AnalogyNet $f_{\\mathrm{cls}}$\n$\\mathbb{S}^{127}$")
    box(4.9, 1.6, 4.4, 0.9, "Sliced MS + slice-aware\nprototypes $\\boldsymbol{\\mu}_c^{(s)}$")

    box(9.6, 5.5, 2.0, 0.75, "Camera frame\n$\\mathbf{I}$")
    box(11.8, 5.5, 2.0, 0.75, "Coarse-fine\n$f_{\\mathrm{det}}$")
    box(9.6, 4.3, 4.2, 0.75, "NMS + quality gates\n\\texttt{classification\\_bbox}")
    box(9.6, 3.2, 4.2, 0.75, "\\texttt{grain\\_crop\\_utils}\n$(\\mathbf{x}_i,M_i)$")
    box(9.6, 2.1, 4.2, 0.75, "$f_{\\mathrm{cls}}\\rightarrow$\nSliceAware $\\hat{y}_i$")
    box(9.6, 0.9, 4.2, 0.75, "Species counts +\nannotated overlay")

    arrow(2.2, 5.85, 2.4, 5.85)
    arrow(3.5, 5.5, 3.5, 5.2)
    arrow(4.6, 4.75, 4.9, 5.85)
    arrow(4.6, 4.75, 4.9, 3.15)
    arrow(7.0, 5.85, 7.2, 5.85)
    arrow(8.25, 5.5, 8.25, 5.2)
    arrow(7.0, 3.15, 7.2, 3.15)
    arrow(8.25, 2.8, 8.25, 2.5)
    arrow(9.3, 4.75, 9.6, 5.85)
    arrow(9.3, 2.05, 9.6, 2.45)
    arrow(11.6, 5.5, 11.6, 5.05)
    arrow(11.6, 4.3, 11.6, 3.95)
    arrow(11.6, 3.2, 11.6, 2.85)
    arrow(11.6, 2.1, 11.6, 1.65)

    ax.set_title("MeliVision three-phase architecture", fontsize=9, fontweight="bold", pad=8)
    save_fig(fig, out_path)


def recall_curves_figure(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.4))
    ax.plot(df["epoch"], df["recall_at_1"] * 100, "o-", label="R@1", markersize=3)
    ax.plot(df["epoch"], df["recall_at_5"] * 100, "s-", label="R@5", markersize=3)
    if "best_selection_value" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["epoch"], df["best_selection_value"] * 100, "^--", color="#888", label="mAP@R", markersize=3)
        ax2.set_ylabel("mAP@R (%)", fontsize=7)
        ax2.tick_params(labelsize=6.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Validation retrieval metrics")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    save_fig(fig, out_path)


def per_class_f1_figure(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path).sort_values("f1", ascending=True)
    labels = [short_label(c) for c in df["class"]]
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 3.8))
    colors = ["#d62728" if f < 0.94 else "#1f77b4" for f in df["f1"]]
    ax.barh(labels, df["f1"] * 100, color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Slice-aware F1 (%)")
    ax.set_title("Per-class F1 (16 species, test)")
    ax.set_xlim(88, 100)
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(df["f1"] * 100):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=6)
    save_fig(fig, out_path)


def tsne_figure(csv_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, 2.8))
    classes = sorted(df["label_name"].unique())
    cmap = plt.get_cmap("tab20", len(classes))
    for i, cls in enumerate(classes):
        sub = df[df["label_name"] == cls]
        ax.scatter(
            sub["tsne_2d_dim1"],
            sub["tsne_2d_dim2"],
            s=4,
            alpha=0.45,
            color=cmap(i),
            label=short_label(cls),
            linewidths=0,
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Test embeddings (N={len(df)})")
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=5.5,
        frameon=False,
        markerscale=1.5,
        ncol=1,
    )
    save_fig(fig, out_path)


def embedding_geometry_figure(
    cossim_csv: Path, compact_csv: Path, metrics_json: Path, out_path: Path
) -> None:
    sim = pd.read_csv(cossim_csv, index_col=0)
    compact = pd.read_csv(compact_csv).sort_values("compactness", ascending=True)
    with metrics_json.open(encoding="utf-8") as f:
        metrics = json.load(f)
    ratio = metrics["metric_learning"]["distance_ratio_inter_intra"]

    labels = [short_label(c) for c in sim.index]
    sim_vals = sim.values.copy()
    np.fill_diagonal(sim_vals, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_PAGE_W, 2.8))

    cmap = LinearSegmentedColormap.from_list("cos", ["#2166ac", "#f7f7f7", "#b2182b"])
    im = axes[0].imshow(sim_vals, vmin=-0.35, vmax=0.35, cmap=cmap, aspect="auto")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=90, fontsize=5.5)
    axes[0].set_yticklabels(labels, fontsize=5.5)
    axes[0].set_title("(a) Centroid cosine (off-diagonal)")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

    clabels = [short_label(c) for c in compact["class_name"]]
    axes[1].barh(clabels, compact["compactness"], color="#4c72b0", edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Intra-class compactness (cosine)")
    axes[1].set_title(f"(b) Per-class compactness; inter/intra={ratio:.2f}")
    axes[1].set_xlim(0.82, 0.96)
    axes[1].grid(axis="x", alpha=0.25)

    fig.tight_layout(w_pad=1.2)
    save_fig(fig, out_path)


def spectral_figure(global_csv: Path, report_txt: Path, out_path: Path) -> None:
    gdf = pd.read_csv(global_csv)
    top = min(40, len(gdf))
    gdf = gdf.head(top)

    pairs = []
    if report_txt.exists():
        text = report_txt.read_text(encoding="utf-8")
        for m in re.finditer(
            r"(\d+)\.\s+(.+?)\s+vs\s+(.+?):\s+lambda_1\s+=\s+([\d.]+)\s+\(([\d.]+)%\)\s+Effective Rank\s+=\s+(\d+)",
            text,
        ):
            pairs.append(
                {
                    "rank": int(m.group(1)),
                    "a": m.group(2).strip(),
                    "b": m.group(3).strip(),
                    "lambda1": float(m.group(4)),
                    "pct": float(m.group(5)),
                    "eff_rank": int(m.group(6)),
                    "collapsed": "COLLAPSED" in text[m.start() : m.start() + 200],
                }
            )
    pairs = pairs[:10]

    fig, axes = plt.subplots(1, 2, figsize=(IEEE_PAGE_W, 2.8))

    axes[0].bar(gdf["component"], gdf["explained_variance"] * 100, color="#2ca02c", width=0.8)
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("(a) Global spectrum on $\\mathbb{S}^{127}$")
    axes[0].grid(axis="y", alpha=0.25)

    if pairs:
        ylab = [f"{short_label(p['a'])}–{short_label(p['b'])}" for p in pairs]
        colors = ["#d62728" if p.get("collapsed") else "#1f77b4" for p in pairs]
        axes[1].barh(ylab, [p["pct"] for p in pairs], color=colors, edgecolor="black", linewidth=0.3)
        axes[1].set_xlabel("$\\lambda_1$ share (%)")
        axes[1].set_title("(b) Top class-pair spectral dominance")
        axes[1].invert_yaxis()
        axes[1].grid(axis="x", alpha=0.25)
    else:
        axes[1].axis("off")

    fig.tight_layout(w_pad=1.0)
    save_fig(fig, out_path)


def confusion_matrix_figure(src_png: Path, out_path: Path) -> None:
    img = mpimg.imread(src_png)
    fig, ax = plt.subplots(figsize=(IEEE_COL_W, IEEE_COL_W))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Normalized confusion (slice-aware)")
    save_fig(fig, out_path)


def gradcam_montage_figure(gradcam_dir: Path, out_path: Path, classes: list[str]) -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # trusted local run artefacts
    max_side = 1200
    imgs = []
    for cls in classes:
        p = gradcam_dir / f"{cls}_gradcam.png"
        if p.exists():
            img = Image.open(p).convert("RGB")
            w, h = img.size
            scale = min(1.0, max_side / max(w, h))
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            imgs.append(img)
    if not imgs:
        print(f"Skip gradcam montage: no images in {gradcam_dir}")
        return

    n = len(imgs)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(IEEE_PAGE_W, rows * 1.8))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for ax, cls, img in zip(axes.flat, classes, imgs):
        ax.imshow(img)
        ax.set_title(short_label(cls), fontsize=7)
    save_fig(fig, out_path)


def copy_evidence(run_test: Path, evidence_dir: Path) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    copies = {
        "classification_test.json": run_test.parent.parent.parent / "evaluation_metrics_summary_test.json",
        "spectral_summary.json": run_test / "spectral_analysis" / "spectral_summary.json",
    }
    for dst, src in copies.items():
        if src.exists():
            shutil.copy2(src, evidence_dir / dst)
            print(f"Copied evidence/{dst}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classifier-run",
        default="runs/2026-06-12_17-16-54_BEST_LAST",
    )
    parser.add_argument("--evidence-dir", default="paper/ieee_tpami_preliminary/evidence")
    parser.add_argument("--output-generated", default="paper/ieee_tpami_preliminary/figures/generated")
    parser.add_argument("--output-results", default="paper/ieee_tpami_preliminary/figures/results")
    args = parser.parse_args()

    apply_ieee_style()
    root = Path(__file__).resolve().parent.parent
    run_test = root / args.classifier_run / "images" / "post_training" / "test"
    gen_dir = root / args.output_generated
    res_dir = root / args.output_results
    evidence = root / args.evidence_dir

    error_budget_figure(evidence / "e2e_gallery_val.json", gen_dir / "error_budget")
    pipeline_figure(gen_dir / "pipeline_overview")

    recall_curves_figure(run_test / "recall_curves.csv", res_dir / "recall_curves")
    per_class_f1_figure(run_test / "per_class_f1.csv", res_dir / "per_class_f1")
    tsne_figure(run_test / "tsne_2d.csv", res_dir / "tsne_2d")
    embedding_geometry_figure(
        run_test / "embedding_space_analysis_panel2_cossim.csv",
        run_test / "embedding_space_analysis_panel3_compactness.csv",
        root / args.classifier_run / "evaluation_metrics_summary_test.json",
        res_dir / "embedding_geometry",
    )
    spectral_figure(
        run_test / "spectral_analysis" / "spectral_analysis_GLOBAL.csv",
        run_test / "spectral_analysis" / "spectral_analysis_report.txt",
        res_dir / "spectral_analysis",
    )
    confusion_matrix_figure(run_test / "confusion_matrix_normalized.png", res_dir / "confusion_matrix")
    gradcam_montage_figure(
        run_test / "gradcam_gallery",
        res_dir / "gradcam_montage",
        ["Castanea_sativa", "Mentha_pulegium", "Gevuina_avellana", "Anthemis_cotula"],
    )
    copy_evidence(run_test, evidence)


if __name__ == "__main__":
    main()
