"""
Generate a formal, publication-style pipeline diagram for MetricLearning.

Outputs:
  - Docs/pipeline_training_results_overview.svg
  - Docs/pipeline_training_results_overview.png
"""

from pathlib import Path
from textwrap import fill, wrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def _add_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    fc="#F8FAFC",
    ec="#1F2937",
    title_color="#0F172A",
    fs_title=16.0,
    fs_body=16.0,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.013",
        linewidth=1.15,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(patch)
    wrapped_title = fill(title, width=36, break_long_words=False, break_on_hyphens=False)
    wrapped_body = _format_body_lines(body, width=47)

    title_artist = ax.text(
        x + 0.010,
        y + h - 0.020,
        wrapped_title,
        ha="left",
        va="top",
        fontsize=fs_title,
        fontweight="bold",
        color=title_color,
    )
    title_lines = wrapped_title.count("\n") + 1
    body_top_offset = 0.020 + (title_lines * 0.024)
    body_artist = ax.text(
        x + 0.010,
        y + h - body_top_offset,
        wrapped_body,
        ha="left",
        va="top",
        fontsize=fs_body,
        color="#111827",
        linespacing=1.0,
    )
    # Hard clipping guarantees no glyph escapes box boundaries.
    title_artist.set_clip_path(patch)
    body_artist.set_clip_path(patch)
    return {"x": x, "y": y, "w": w, "h": h}


def _format_body_lines(body, width):
    """
    Format block body with predictable wrapping and no artificial blank spaces.
    Keeps Input/Process labels aligned and avoids awkward spacing artifacts.
    """
    rendered_lines = []
    for raw_line in body.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        if ":" in line:
            label, content = line.split(":", 1)
            prefix = f"{label.strip()}: "
            wrapped = wrap(
                content.strip(),
                width=max(12, width - len(prefix)),
                break_long_words=True,
                break_on_hyphens=False,
            )
            if wrapped:
                rendered_lines.append(prefix + wrapped[0])
                continuation_prefix = " " * len(prefix)
                rendered_lines.extend(continuation_prefix + part for part in wrapped[1:])
            else:
                rendered_lines.append(prefix)
        else:
            rendered_lines.extend(
                wrap(
                    line,
                    width=width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )
    return "\n".join(rendered_lines)


def _measure_text_width_axes(ax, text, fontsize, fontweight="normal"):
    """Return text width in axis coordinates (0..1)."""
    artist = ax.text(0.0, 0.0, text, fontsize=fontsize, fontweight=fontweight, alpha=0.0)
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_bbox = artist.get_window_extent(renderer=renderer)
    ax_bbox = ax.get_window_extent(renderer=renderer)
    artist.remove()
    if ax_bbox.width == 0:
        return 0.0
    return text_bbox.width / ax_bbox.width


def _compute_uniform_box_width(ax, columns_text, fs_title, fs_body):
    """
    Compute one shared box width for all columns using the maximum
    required content width found across the three columns.
    """
    max_width = 0.0
    for column in columns_text:
        for title, body in column:
            max_width = max(
                max_width,
                _measure_text_width_axes(ax, title, fs_title, fontweight="bold"),
            )
            for line in body.split("\n"):
                max_width = max(max_width, _measure_text_width_axes(ax, line, fs_body))
    return max_width + 0.050  # content + horizontal padding safety


def _down_arrow(ax, box_top, box_bottom, color="#334155"):
    x = box_top["x"] + box_top["w"] / 2
    y1 = box_top["y"]
    y2 = box_bottom["y"] + box_bottom["h"]
    arr = FancyArrowPatch(
        (x, y1 - 0.0015),
        (x, y2 + 0.0015),
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=2.0,
        color=color,
        zorder=6,
    )
    ax.add_patch(arr)


def _orthogonal_link(ax, source_box, target_box, color="#2563EB", label=None):
    sx = source_box["x"] + source_box["w"]
    sy = source_box["y"] + source_box["h"] * 0.52
    tx = target_box["x"]
    ty = target_box["y"] + target_box["h"] * 0.52
    gap = tx - sx
    midx = sx + max(0.006, gap * 0.55)
    if midx >= tx - 0.006:
        midx = (sx + tx) / 2

    # Strict orthogonal routing: horizontal -> vertical -> horizontal
    end_x = tx - 0.004
    ax.plot([sx, midx], [sy, sy], color=color, linewidth=2.0, zorder=7)
    ax.plot([midx, midx], [sy, ty], color=color, linewidth=2.0, zorder=7)
    ax.plot([midx, end_x], [ty, ty], color=color, linewidth=2.0, zorder=7)
    arr = FancyArrowPatch(
        (end_x, ty),
        (tx - 0.001, ty),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=2.0,
        color=color,
        zorder=8,
    )
    ax.add_patch(arr)
    if label:
        ax.text(
            midx + 0.006,
            (sy + ty) / 2,
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=16.0,
            color=color,
        )


def build_diagram(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(24, 13), constrained_layout=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Global title and subtitle
    ax.text(
        0.5,
        0.975,
        "General Metric Learning Diagram",
        ha="center",
        va="top",
        fontsize=30,
        fontweight="bold",
        color="#0F172A",
    )

    # Column geometry
    col_titles = [
        "I. Data Ingestion and Preparation",
        "II. Training and Validation",
        "III. Results, Reporting, and Inference",
    ]
    panel_bg = ["#F8FBFF", "#FBF9FF", "#F7FFFA"]

    # Color gradients (monotonic by column)
    left_colors = ["#ECF5FF", "#DCEEFF", "#C9E3FF", "#AFD5FF", "#8FC4F8", "#6EADE4"]
    mid_colors = ["#F3EDFF", "#E8DCFF", "#DBC8FF", "#CCB2FF", "#BA99F7", "#A47FE6"]
    right_colors = ["#E8FBF1", "#D5F6E4", "#BFF0D6", "#A2E8C2", "#84DDAE", "#63CC95"]

    # Column I: data ingestion
    left_texts = [
        (
            "S1. Data Entry",
            "Input: class-organized raw microscopy image folders\n"
            "Process: source normalization and class mapping resolution",
        ),
        (
            "S2. Dataset Census and Quality Control",
            "Input: discovered image inventory\n"
            "Process: scanning, invalid-path filtering, baseline statistics",
        ),
        (
            "S3. Stratified Split Construction",
            "Input: class-level metadata\n"
            "Process: balanced train/val/test split, materialized in data/processed",
        ),
        (
            "S4. Grain Segmentation and Annotation",
            "Input: split images + U2-Net detector\n"
            "Process: contour extraction, morphology filters, .seg generation",
        ),
        (
            "S5. Centralized Annotation Registry",
            "Input: per-sample .seg files\n"
            "Process: consolidation in data/annotations with full traceability",
        ),
        (
            "S6. Training Dataset Assembly",
            "Input: data/processed + data/annotations\n"
            "Process: on-the-fly crops, masking, augmentations, tensor export",
        ),
    ]

    # Column II: training
    mid_texts = [
        (
            "T1. DataLoader and Balanced Sampling",
            "Input: segmented grain dataset\n"
            "Process: MPerClassSampler for stable class-balanced mini-batches",
        ),
        (
            "T2. AnalogyNet Forward Pass",
            "Input: grain-crop mini-batch\n"
            "Process: backbone pass, projection head, normalized embedding",
        ),
        (
            "T3. Numerically Stable Optimization",
            "Input: embeddings + labels\n"
            "Process: AMP, AdamW, cosine schedule, warmup, gradient clipping",
        ),
        (
            "T4. Advanced Metric-Learning Objective",
            "Input: implicit positive/negative relations in batch\n"
            "Process: MS/Triplet/ArcFace/Proxy losses plus regularizers",
        ),
        (
            "T5. Per-Epoch Validation",
            "Input: validation embeddings\n"
            "Process: Recall@K, mAP@R, NMI, macro-F1, and accuracy",
        ),
        (
            "T6. Best-State Selection and Persistence",
            "Input: metric trajectory and early-stopping criterion\n"
            "Process: persist best model checkpoint and metrics log",
        ),
    ]

    # Column III: results and inference
    right_texts = [
        (
            "R1. Reference Artifacts",
            "Input: selected checkpoint\n"
            "Process: generate class centroids and reference embeddings",
        ),
        (
            "R2. Post-Training Analysis",
            "Input: training history and evaluation sets\n"
            "Process: curves, t-SNE, confusion analysis, distances, separability",
        ),
        (
            "R3. Auditable Technical Reporting",
            "Input: aggregated metrics + visual evidence\n"
            "Process: technical report, summary JSON, publication figures",
        ),
        (
            "R4. Full-Image Inference",
            "Input: non-segmented microscopy image\n"
            "Process: grain detection, embedding extraction, then kNN inference",
        ),
        (
            "R5. Batch Inference and Exportables",
            "Input: evaluation directory\n"
            "Process: CSV and HTML exports, annotated samples, comparative tables",
        ),
        (
            "R6. Final Operational Outcome",
            "Input: per-grain predictions with confidence\n"
            "Process: species counting, visual evidence, and full traceability",
        ),
    ]

    # Shared row layout
    # Keep a small, clean header band under titles (half of prior adjustment).
    y_start = 0.763
    row_h = 0.132
    row_gap = 0.009
    ys = [y_start - i * (row_h + row_gap) for i in range(6)]

    # Auto-size column/box width from max content among all columns.
    fs_title = 16.0
    fs_body = 16.0
    requested_box_w = _compute_uniform_box_width(
        ax,
        [left_texts, mid_texts, right_texts],
        fs_title=fs_title,
        fs_body=fs_body,
    )

    outer_margin = 0.02
    inter_col_gap = 0.014
    panel_inner_pad_x = 0.004
    col_w = (requested_box_w + 2 * panel_inner_pad_x) * 0.67

    # Preserve content-driven width as much as possible; shrink margins/gaps first.
    min_outer_margin = 0.006
    min_inter_col_gap = 0.004
    total_required = 3 * col_w + 2 * inter_col_gap + 2 * outer_margin
    if total_required > 1.0:
        excess = total_required - 1.0
        reduce_outer = min(excess / 2, outer_margin - min_outer_margin)
        outer_margin -= reduce_outer
        excess -= reduce_outer * 2

        if excess > 0:
            reduce_gap = min(excess / 2, inter_col_gap - min_inter_col_gap)
            inter_col_gap -= reduce_gap
            excess -= reduce_gap * 2

        # Last resort only if still impossible to fit.
        if excess > 0:
            col_w -= excess / 3

    box_w = col_w - 2 * panel_inner_pad_x

    total_width = 3 * col_w + 2 * inter_col_gap
    start_x = (1.0 - total_width) / 2.0
    col_x = [start_x, start_x + col_w + inter_col_gap, start_x + 2 * (col_w + inter_col_gap)]
    box_x = [x + panel_inner_pad_x for x in col_x]

    for i in range(3):
        panel = FancyBboxPatch(
            (col_x[i], 0.03),
            col_w,
            0.92,
            boxstyle="round,pad=0.010,rounding_size=0.015",
            linewidth=0.95,
            facecolor=panel_bg[i],
            edgecolor="#CBD5E1",
        )
        ax.add_patch(panel)
        ax.text(
            col_x[i] + col_w / 2,
            0.928,
            col_titles[i],
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color="#0F172A",
        )

    left_boxes = []
    mid_boxes = []
    right_boxes = []

    for i in range(6):
        left_boxes.append(
            _add_box(
                ax,
                box_x[0],
                ys[i],
                box_w,
                row_h,
                title=left_texts[i][0],
                body=left_texts[i][1],
                fc=left_colors[i],
                ec="#1E3A8A",
                title_color="#0B2A63",
            )
        )
        mid_boxes.append(
            _add_box(
                ax,
                box_x[1],
                ys[i],
                box_w,
                row_h,
                title=mid_texts[i][0],
                body=mid_texts[i][1],
                fc=mid_colors[i],
                ec="#5B21B6",
                title_color="#3B1576",
            )
        )
        right_boxes.append(
            _add_box(
                ax,
                box_x[2],
                ys[i],
                box_w,
                row_h,
                title=right_texts[i][0],
                body=right_texts[i][1],
                fc=right_colors[i],
                ec="#166534",
                title_color="#0B4D2A",
            )
        )

    # Vertical flow inside each column
    flow_color = "#1E293B"
    for i in range(5):
        _down_arrow(ax, left_boxes[i], left_boxes[i + 1], color=flow_color)
        _down_arrow(ax, mid_boxes[i], mid_boxes[i + 1], color=flow_color)
        _down_arrow(ax, right_boxes[i], right_boxes[i + 1], color=flow_color)

    # Cross-column links (strictly orthogonal, no diagonals)
    _orthogonal_link(
        ax,
        left_boxes[5],
        mid_boxes[0],
        color="#1D4ED8",
        label="Training-dataset handoff",
    )
    _orthogonal_link(
        ax,
        mid_boxes[5],
        right_boxes[0],
        color="#047857",
        label="Trained-model handoff",
    )

    svg_path = output_dir / "pipeline_training_results_overview.svg"
    png_path = output_dir / "pipeline_training_results_overview.png"
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def main():
    repo_root = Path(__file__).resolve().parents[1]
    docs_dir = repo_root / "Docs"
    svg_path, png_path = build_diagram(docs_dir)
    print(f"Generated diagram:\n- {svg_path}\n- {png_path}")


if __name__ == "__main__":
    main()

