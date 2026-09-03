"""
Generate an academic-style equation sheet for evaluation metrics.

Outputs a high-resolution PNG that can be embedded in markdown docs.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "docs" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metric_equations_academic_style.png"

    # Serif look similar to academic papers.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "mathtext.fontset": "stix",
        }
    )

    fig = plt.figure(figsize=(12, 14), facecolor="white")
    ax = fig.add_axes([0.06, 0.04, 0.88, 0.92])
    ax.axis("off")

    y = 0.98
    dy_title = 0.055
    dy_eq = 0.07
    dy_note = 0.04

    ax.text(0.0, y, "Metric Definitions (Academic Notation)", fontsize=20, fontweight="bold", va="top")
    y -= 0.06

    sections = [
        (
            "1) Accuracy",
            r"$\mathrm{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i=y_i)$",
            "(1)",
            "Proportion of correctly classified samples.",
        ),
        (
            "2) Balanced Accuracy",
            r"$\mathrm{BalancedAccuracy}=\frac{1}{C}\sum_{c=1}^{C}\mathrm{Recall}_c,\quad \mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c}$",
            "(2)",
            "Average class recall, robust under class imbalance.",
        ),
        (
            "3) Precision (class / macro / weighted)",
            r"$\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c}$",
            "(3)",
            "",
        ),
        (
            "",
            r"$\mathrm{Precision}_{macro}=\frac{1}{C}\sum_{c=1}^{C}\mathrm{Precision}_c,\quad \mathrm{Precision}_{weighted}=\sum_{c=1}^{C}\frac{n_c}{N}\mathrm{Precision}_c$",
            "(4)",
            "",
        ),
        (
            "4) F1-score (class / macro / weighted)",
            r"$F1_c=\frac{2\,\mathrm{Precision}_c\,\mathrm{Recall}_c}{\mathrm{Precision}_c+\mathrm{Recall}_c}$",
            "(5)",
            "",
        ),
        (
            "",
            r"$F1_{macro}=\frac{1}{C}\sum_{c=1}^{C}F1_c,\quad F1_{weighted}=\sum_{c=1}^{C}\frac{n_c}{N}F1_c$",
            "(6)",
            "",
        ),
        (
            "5) Confusion Matrix",
            r"$M_{a,b}=|\{i:\,y_i=a\wedge \hat{y}_i=b\}|,\quad \tilde{M}_{a,b}=\frac{M_{a,b}}{\sum_{j=1}^{C}M_{a,j}}$",
            "(7)",
            "Rows: true class, columns: predicted class.",
        ),
        (
            "6) Recall@K",
            r"$R@K=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\left(\exists j\in \mathcal{N}_K(i):y_j=y_i\right)$",
            "(8)",
            r"$\mathcal{N}_K(i)$: top-K nearest neighbors of query $i$.",
        ),
        (
            "7) mAP@R",
            r"$mAP@R=\frac{1}{N}\sum_{i=1}^{N}AP@R_i(i)$",
            "(9)",
            r"$R_i$: number of relevant positives for query $i$.",
        ),
        (
            "8) NMI",
            r"$\mathrm{NMI}(Y,\hat{Y})=\frac{2\,I(Y;\hat{Y})}{H(Y)+H(\hat{Y})}$",
            "(10)",
            r"$I(\cdot;\cdot)$: mutual information, $H(\cdot)$: entropy.",
        ),
        (
            "9) Distance Ratio",
            r"$\rho=\frac{\bar{d}_{inter}}{\bar{d}_{intra}}$",
            "(11)",
            r"$\bar{d}_{inter}$: mean inter-class distance, $\bar{d}_{intra}$: mean intra-class distance.",
        ),
    ]

    for title, eq, eq_id, note in sections:
        if title:
            ax.text(0.0, y, title, fontsize=15, fontweight="bold", va="top")
            y -= dy_title

        ax.text(0.03, y, eq, fontsize=20, va="center")
        ax.text(0.97, y, eq_id, fontsize=14, ha="right", va="center")
        y -= dy_eq

        if note:
            ax.text(0.03, y, note, fontsize=11.5, color="#333333", va="top")
            y -= dy_note

        y -= 0.01

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

