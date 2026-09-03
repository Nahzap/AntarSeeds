from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from matplotlib.lines import Line2D

matplotlib.use("Agg")


PRIMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "precision_weighted",
    "F1_macro",
    "F1_weighted",
    "NMI",
    "mAP@R",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
]

SECONDARY_METRICS = [
    "train_loss",
    "val_loss",
    "learning_rate",
    "epoch_time_sec",
    "best_recall_at_1",
]

THREED_METRICS = [
    "train_loss",
    "val_loss",
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
]

LOSS_METRICS = {"train_loss", "val_loss"}


def prettify_backbone_name(raw_name: str) -> str:
    mapping = {
        "convnext_v2_tiny": "ConvNeXt V2 Tiny",
        "resnet50": "ResNet-50",
        "dinov2_vits14": "DINOv2 ViT-S/14",
        "deit_small_patch16_224": "DeiT-S",
    }
    return mapping.get(raw_name, raw_name)


def sanitize_metric_name(metric: str) -> str:
    name = metric.lower().replace("@", "at").replace("/", "_").replace("-", "_")
    name = name.replace(" ", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


def get_model_color(model_name: str, fallback_idx: int = 0) -> str:
    fixed = {
        "ConvNeXt V2 Tiny": "#2ca02c",   # green
        "DINOv2 ViT-S/14": "#1f77b4",    # blue
        "DeiT-S": "#f1c40f",             # yellow
        "ResNet-50": "#d62728",          # red
    }
    if model_name in fixed:
        return fixed[model_name]
    fallback = ["#17becf", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    return fallback[fallback_idx % len(fallback)]


def is_probability_metric(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool((numeric >= 0).all() and (numeric <= 1.05).all())


def reorder_metrics(metrics: list[str]) -> list[str]:
    preferred = [m for m in PRIMARY_METRICS + SECONDARY_METRICS if m in metrics]
    extras = sorted([m for m in metrics if m not in preferred])
    return preferred + extras


def compute_epoch_ticks(epoch_series: pd.Series, min_ticks: int = 5) -> list[int]:
    epochs = sorted(pd.to_numeric(epoch_series, errors="coerce").dropna().astype(int).unique().tolist())
    if not epochs:
        return []
    if len(epochs) <= min_ticks:
        return epochs

    start = epochs[0]
    end = epochs[-1]
    positions = [int(round(start + (end - start) * i / (min_ticks - 1))) for i in range(min_ticks)]
    ticks = sorted(set(positions + [start, end]))
    if len(ticks) < 4:
        # Fallback that guarantees enough visible divisions.
        step = max(1, (end - start) // 4)
        ticks = list(range(start, end + 1, step))
        if ticks[-1] != end:
            ticks.append(end)
    return ticks


def compute_dynamic_ylim(series: pd.Series, clamp_to_percent: bool = False) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return (0.0, 100.0) if clamp_to_percent else (0.0, 1.0)

    y_min = float(values.min())
    y_max = float(values.max())
    span = y_max - y_min

    if span <= 0:
        pad = max(abs(y_min) * 0.05, 0.5 if clamp_to_percent else 0.05)
    else:
        pad = max(span * 0.12, 0.35 if clamp_to_percent else span * 0.02)

    lower = y_min - pad
    upper = y_max + pad

    if clamp_to_percent:
        lower = max(0.0, lower)
        upper = min(100.0, upper)
        if upper - lower < 2.0:
            center = (upper + lower) / 2.0
            lower = max(0.0, center - 1.0)
            upper = min(100.0, center + 1.0)
    elif upper - lower < 1e-9:
        upper = lower + 1.0

    return lower, upper


def build_3d_metric_frame(epoch_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available_metrics = [m for m in THREED_METRICS if m in epoch_df.columns]
    if not available_metrics:
        return pd.DataFrame(), []

    long_rows: list[dict] = []
    for metric in available_metrics:
        series = pd.to_numeric(epoch_df[metric], errors="coerce")
        metric_min = float(series.min())
        metric_max = float(series.max())
        denom = metric_max - metric_min

        for _, row in epoch_df.iterrows():
            value = pd.to_numeric(row[metric], errors="coerce")
            if pd.isna(value):
                continue

            if denom <= 0:
                score = 100.0
            elif metric in LOSS_METRICS:
                # Invert losses so higher is better in a unified performance axis.
                score = ((metric_max - float(value)) / denom) * 100.0
            else:
                score = ((float(value) - metric_min) / denom) * 100.0

            long_rows.append(
                {
                    "model_name": row["model_name"],
                    "epoch": int(row["epoch"]),
                    "metric_name": metric,
                    "score_norm_pct": score,
                }
            )

    frame = pd.DataFrame(long_rows)
    metric_to_idx = {m: i for i, m in enumerate(available_metrics)}
    frame["metric_idx"] = frame["metric_name"].map(metric_to_idx)
    return frame, available_metrics


def build_3d_surface_payload(
    epoch_df: pd.DataFrame,
) -> tuple[list[str], list[int], dict[str, list[list[float]]], dict[str, int]]:
    available_metrics = [m for m in THREED_METRICS if m in epoch_df.columns]
    if not available_metrics:
        return [], [], {}, {}

    epochs = sorted(pd.to_numeric(epoch_df["epoch"], errors="coerce").dropna().astype(int).unique().tolist())
    model_names = sorted(epoch_df["model_name"].unique().tolist())

    # Global per-metric normalization for cross-model comparability.
    metric_bounds: dict[str, tuple[float, float]] = {}
    for metric in available_metrics:
        vals = pd.to_numeric(epoch_df[metric], errors="coerce").dropna()
        metric_bounds[metric] = (float(vals.min()), float(vals.max()))

    model_z_map: dict[str, list[list[float]]] = {}
    for model_name in model_names:
        model_df = epoch_df[epoch_df["model_name"] == model_name].copy()
        model_df = model_df.sort_values("epoch").set_index("epoch").reindex(epochs)
        z_matrix: list[list[float]] = []
        for metric in available_metrics:
            mmin, mmax = metric_bounds[metric]
            denom = mmax - mmin
            vals = pd.to_numeric(model_df[metric], errors="coerce")
            if denom <= 0:
                norm = pd.Series([100.0] * len(vals), index=vals.index)
            elif metric in LOSS_METRICS:
                norm = ((mmax - vals) / denom) * 100.0
            else:
                norm = ((vals - mmin) / denom) * 100.0
            norm = norm.clip(lower=0.0, upper=100.0)
            z_matrix.append([float(v) if pd.notna(v) else float("nan") for v in norm.values])
        model_z_map[model_name] = z_matrix

    metric_idx_map = {metric: i for i, metric in enumerate(available_metrics)}
    return available_metrics, epochs, model_z_map, metric_idx_map


def save_3d_multimetric_plot(epoch_df: pd.DataFrame, output_path: Path) -> None:
    metrics, epochs, model_z_map, _metric_idx_map = build_3d_surface_payload(epoch_df)
    if not metrics or not epochs or not model_z_map:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    model_names = sorted(model_z_map.keys())
    color_map = {model: get_model_color(model, i) for i, model in enumerate(model_names)}

    for model in model_names:
        z_matrix = np.array(model_z_map[model], dtype=float)
        # Mesh-style mantle: fibers along epochs (per metric)
        for metric_idx in range(len(metrics)):
            ax.plot(
                epochs,
                [metric_idx] * len(epochs),
                z_matrix[metric_idx, :],
                color=color_map[model],
                linewidth=1.2,
                alpha=0.98,
            )
        # Cross-fibers along metrics (per epoch)
        for epoch_idx, epoch in enumerate(epochs):
            ax.plot(
                [epoch] * len(metrics),
                list(range(len(metrics))),
                z_matrix[:, epoch_idx],
                color=color_map[model],
                linewidth=0.8,
                alpha=0.55,
            )

    xticks = compute_epoch_ticks(epoch_df["epoch"], min_ticks=5)
    if xticks:
        ax.set_xticks(xticks)
    ax.set_xlabel("Epoca")

    ax.set_yticks(list(range(len(metrics))))
    ax.set_yticklabels(metrics)
    ax.set_ylabel("Metrica")

    ax.set_zlim(0, 100)
    ax.set_zlabel("Rendimiento normalizado (%)")

    ax.view_init(elev=24, azim=-56)
    ax.set_title(
        "Comparacion 3D tipo malla por epoca (fibras por modelo)",
        pad=18,
        fontweight="bold",
    )

    legend_handles = [
        Line2D([0], [0], color=color_map[m], lw=3, label=m) for m in model_names
    ]
    ax.legend(handles=legend_handles, title="Modelo", loc="upper left", bbox_to_anchor=(0.0, 1.02))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.03, right=0.86, bottom=0.05, top=0.90)
    plt.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close()


def save_interactive_3d_mantle_html(epoch_df: pd.DataFrame, output_html: Path) -> None:
    metrics, epochs, model_z_map, metric_idx_map = build_3d_surface_payload(epoch_df)
    if not metrics or not epochs or not model_z_map:
        return

    model_names = sorted(model_z_map.keys())
    color_map = {model: get_model_color(model, i) for i, model in enumerate(model_names)}
    metric_label_map = {
        "train_loss": "Train Loss",
        "val_loss": "Val Loss",
        "recall_at_1": "Recall@1",
        "recall_at_5": "Recall@5",
        "recall_at_10": "Recall@10",
        "mAP@R": "mAP@R",
        "NMI": "NMI",
        "F1_macro": "F1 Macro",
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
        "precision_macro": "Precision Macro",
        "precision_weighted": "Precision Weighted",
        "F1_weighted": "F1 Weighted",
    }

    fig = go.Figure()
    base_z_by_model = {model: model_z_map[model] for model in model_names}
    surface_trace_map: dict[str, int] = {}
    mesh_metric_trace_map: dict[str, dict[str, int]] = {metric: {} for metric in metrics}
    mesh_epoch_trace_map: dict[str, list[int]] = {model: [] for model in model_names}
    mesh_epoch_base_z_map: dict[str, list[float]] = {}
    trace_idx = 0

    # Add smooth mantle traces (surface mode)
    x_grid = [epochs for _ in metrics]
    y_grid = [[metric_idx_map[m] for _ in epochs] for m in metrics]
    for model in model_names:
        custom_metric_grid = [[metric_label_map.get(m, m) for _ in epochs] for m in metrics]
        fig.add_trace(
            go.Surface(
                x=x_grid,
                y=y_grid,
                z=base_z_by_model[model],
                customdata=custom_metric_grid,
                surfacecolor=[[0.0 for _ in epochs] for _ in metrics],
                cmin=0,
                cmax=1,
                colorscale=[[0.0, color_map[model]], [1.0, color_map[model]]],
                showscale=False,
                opacity=0.50,
                hovertemplate=(
                    "Modelo: " + model
                    + "<br>Metrica: %{customdata}"
                    + "<br>Epoca: %{x}"
                    + "<br>Score norm.: %{z:.2f}%<extra></extra>"
                ),
                name=model,
                showlegend=False,
                visible=True,
            )
        )
        surface_trace_map[model] = trace_idx
        trace_idx += 1

    # Add fibrous mantle traces (mesh mode)
    for model in model_names:
        z_grid = np.array(base_z_by_model[model], dtype=float)
        for metric in metrics:
            m_idx = metric_idx_map[metric]
            fig.add_trace(
                go.Scatter3d(
                    x=epochs,
                    y=[m_idx] * len(epochs),
                    z=z_grid[m_idx, :].tolist(),
                    mode="lines",
                    line=dict(color=color_map[model], width=5),
                    opacity=0.98,
                    name=model,
                    showlegend=False,
                    hovertemplate=(
                        "Modelo: " + model
                        + "<br>Metrica: " + metric_label_map.get(metric, metric)
                        + "<br>Epoca: %{x}"
                        + "<br>Score norm.: %{z:.2f}%<extra></extra>"
                    ),
                    visible=False,
                )
            )
            mesh_metric_trace_map[metric][model] = trace_idx
            trace_idx += 1

        for e_idx, epoch in enumerate(epochs):
            base_row = z_grid[:, e_idx].tolist()
            fig.add_trace(
                go.Scatter3d(
                    x=[epoch] * len(metrics),
                    y=list(range(len(metrics))),
                    z=base_row,
                    mode="lines",
                    line=dict(color=color_map[model], width=2),
                    opacity=0.55,
                    name=model,
                    showlegend=False,
                    hoverinfo="skip",
                    visible=False,
                )
            )
            mesh_epoch_trace_map[model].append(trace_idx)
            mesh_epoch_base_z_map[str(trace_idx)] = base_row
            trace_idx += 1

    xticks = compute_epoch_ticks(epoch_df["epoch"], min_ticks=5)
    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="Comparacion 3D: manto suave o manto fibroso",
            x=0.5,
            xanchor="center",
            font=dict(size=24, family="Times New Roman, serif", color="#111827"),
        ),
        font=dict(family="Times New Roman, serif", size=13, color="#111827"),
        scene=dict(
            xaxis=dict(
                title="Epoca",
                tickmode="array",
                tickvals=xticks,
                showbackground=True,
                backgroundcolor="rgba(245,247,250,1)",
                gridcolor="rgba(180,180,180,0.45)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Metrica",
                tickmode="array",
                tickvals=list(range(len(metrics))),
                ticktext=[metric_label_map.get(m, m) for m in metrics],
                showbackground=True,
                backgroundcolor="rgba(245,247,250,1)",
                gridcolor="rgba(180,180,180,0.45)",
                zeroline=False,
            ),
            zaxis=dict(
                title="Rendimiento normalizado (%)",
                range=[0, 100],
                showbackground=True,
                backgroundcolor="rgba(245,247,250,1)",
                gridcolor="rgba(180,180,180,0.45)",
                zeroline=False,
            ),
            camera=dict(eye=dict(x=1.55, y=-1.6, z=1.05)),
            aspectmode="manual",
            aspectratio=dict(x=1.35, y=1.15, z=0.95),
        ),
        margin=dict(l=10, r=10, t=70, b=10),
        width=1500,
        height=900,
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id="metric3d_plot",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "metriclearning_3d_visualization",
                "height": 1200,
                "width": 1800,
                "scale": 2,
            },
        },
    )
    metric_key_map = {metric: f"metric_{sanitize_metric_name(metric)}" for metric in metrics}
    model_key_map = {model: f"model_{sanitize_metric_name(model)}" for model in model_names}

    model_legend_items = "".join(
        [
            (
                f"<span style=\"display:inline-flex;align-items:center;margin-right:14px;\">"
                f"<span style=\"width:12px;height:12px;border-radius:2px;background:{color_map[m]};display:inline-block;margin-right:6px;\"></span>"
                f"{m}</span>"
            )
            for m in model_names
        ]
    )

    model_checkbox_items = "\n".join(
        [
            (
                f"<label style=\"display:inline-flex;align-items:center;margin:4px 12px 4px 0;\">"
                f"<input type=\"checkbox\" id=\"{model_key_map[model]}\" checked style=\"margin-right:6px;\">"
                f"{model}</label>"
            )
            for model in model_names
        ]
    )

    metric_checkbox_items = "\n".join(
        [
            (
                f"<label style=\"display:inline-flex;align-items:center;margin:4px 12px 4px 0;\">"
                f"<input type=\"checkbox\" id=\"{metric_key_map[metric]}\" checked style=\"margin-right:6px;\">"
                f"{metric_label_map.get(metric, metric)}</label>"
            )
            for metric in metrics
        ]
    )
    render_mode_controls = (
        "<label style=\"display:inline-flex;align-items:center;margin-right:14px;\">"
        "<input type=\"radio\" name=\"render_mode\" id=\"mode_surface\" value=\"surface\" checked style=\"margin-right:6px;\">"
        "Manto suave</label>"
        "<label style=\"display:inline-flex;align-items:center;\">"
        "<input type=\"radio\" name=\"render_mode\" id=\"mode_mesh\" value=\"mesh\" style=\"margin-right:6px;\">"
        "Manto fibroso</label>"
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <title>3D Metric Mantle Comparison</title>
  <style>
    body {{
      font-family: "Times New Roman", serif;
      margin: 10px 14px;
      color: #111827;
      background: #ffffff;
    }}
    .panel {{
      border: 1px solid #d1d5db;
      border-radius: 6px;
      padding: 10px 12px;
      margin-bottom: 8px;
      background: #fcfcfd;
    }}
    .title {{
      font-weight: 700;
      margin-bottom: 6px;
      font-size: 15px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .btn {{
      border: 1px solid #9ca3af;
      border-radius: 4px;
      background: white;
      padding: 6px 10px;
      margin-right: 6px;
      cursor: pointer;
      font-family: "Times New Roman", serif;
      font-size: 13px;
    }}
    .btn:hover {{
      background: #f3f4f6;
    }}
    .caption {{
      border-top: 1px solid #d1d5db;
      margin-top: 8px;
      padding-top: 8px;
      font-size: 13px;
      line-height: 1.35;
    }}
  </style>
</head>
<body>
  <div class="row">
    <div class="panel">
      <div class="title">Controles de modelo</div>
      <div>{model_checkbox_items}</div>
    </div>
    <div class="panel">
      <div class="title">Modo de visualizacion</div>
      <div>{render_mode_controls}</div>
    </div>
    <div class="panel">
      <div class="title">Controles de metricas</div>
      <div>{metric_checkbox_items}</div>
    </div>
    <div class="panel">
      <div class="title">Guardar visualizacion actual</div>
      <button class="btn" id="btn_save_png">Guardar PNG</button>
      <button class="btn" id="btn_save_state">Guardar estado JSON</button>
    </div>
  </div>
  <div class="panel">
    <div class="title">Modelos</div>
    <div>{model_legend_items}</div>
    <div class="caption">
      Figura 3D para reporte cientifico con malla de fibras por modelo.
      Eje X: epoca, Eje Y: metrica, Eje Z: rendimiento normalizado (%).
      Para comparabilidad, train/val loss se invierten en la normalizacion.
    </div>
  </div>
  {plot_div}
  <script>
    const surfaceTraceMap = {json.dumps(surface_trace_map)};
    const meshMetricTraceMap = {json.dumps(mesh_metric_trace_map)};
    const meshEpochTraceMap = {json.dumps(mesh_epoch_trace_map)};
    const meshEpochBaseZMap = {json.dumps(mesh_epoch_base_z_map)};
    const metricKeyMap = {json.dumps(metric_key_map)};
    const modelKeyMap = {json.dumps(model_key_map)};
    const metricIdxMap = {json.dumps(metric_idx_map)};
    const metricLabelMap = {json.dumps(metric_label_map)};
    const epochs = {json.dumps(epochs)};
    const modelNames = {json.dumps(model_names)};
    const baseZByModel = {json.dumps(base_z_by_model)};

    function clone2D(arr) {{
      return arr.map(row => row.slice());
    }}

    function currentMode() {{
      const selected = document.querySelector('input[name="render_mode"]:checked');
      return selected ? selected.value : 'surface';
    }}

    function selectedMetrics() {{
      return Object.keys(metricIdxMap)
        .sort((a, b) => metricIdxMap[a] - metricIdxMap[b])
        .filter(metric => {{
          const id = metricKeyMap[metric];
          const el = document.getElementById(id);
          return !!(el && el.checked);
        }});
    }}

    function applyMetricFilter() {{
      const gd = document.getElementById('metric3d_plot');
      const totalTraces = (gd && gd.data) ? gd.data.length : 0;
      const visibility = Array(totalTraces).fill(false);
      const mode = currentMode();
      const selected = selectedMetrics();

      if (mode === 'surface') {{
        const surfIndices = [];
        const surfZ = [];
        const surfX = [];
        const surfY = [];
        modelNames.forEach(model => {{
          const modelOn = document.getElementById(modelKeyMap[model]).checked;
          const idx = surfaceTraceMap[model];
          if (idx === undefined) return;
          if (modelOn) {{
            visibility[idx] = true;
          }}
          const zGrid = selected.length > 0
            ? selected.map(metric => {{
                const rowIdx = metricIdxMap[metric];
                return (baseZByModel[model][rowIdx] || []).slice();
              }})
            : [[null]];
          const xGrid = selected.length > 0
            ? selected.map(_ => epochs.slice())
            : [[null]];
          const yGrid = selected.length > 0
            ? selected.map((_, i) => Array(epochs.length).fill(i))
            : [[null]];
          surfIndices.push(idx);
          surfZ.push(zGrid);
          surfX.push(xGrid);
          surfY.push(yGrid);
        }});
        Plotly.restyle('metric3d_plot', {{ visible: visibility }});
        if (surfIndices.length > 0) {{
          Plotly.restyle('metric3d_plot', {{ x: surfX, y: surfY, z: surfZ }}, surfIndices);
        }}
        Plotly.relayout('metric3d_plot', {{
          'scene.yaxis.tickvals': selected.map((_, i) => i),
          'scene.yaxis.ticktext': selected.map(m => metricLabelMap[m] || m),
        }});
        return;
      }}

      // Mesh mode
      const epochTraceIndices = [];
      const epochTraceZ = [];
      const selectedSet = new Set(selected);
      modelNames.forEach(model => {{
        const modelOn = document.getElementById(modelKeyMap[model]).checked;
        if (!modelOn) return;

        // Metric fibers
        selected.forEach(metric => {{
          const idx = (meshMetricTraceMap[metric] || {{}})[model];
          if (idx !== undefined) {{
            visibility[idx] = true;
          }}
        }});

        // Cross-fibers: keep model visible but hide unchecked metric dimensions.
        (meshEpochTraceMap[model] || []).forEach(idx => {{
          visibility[idx] = true;
          const baseRow = (meshEpochBaseZMap[String(idx)] || []).slice();
          Object.keys(metricIdxMap).forEach(metric => {{
            if (!selectedSet.has(metric)) {{
              const rowIdx = metricIdxMap[metric];
              if (rowIdx !== undefined && rowIdx < baseRow.length) {{
                baseRow[rowIdx] = null;
              }}
            }}
          }});
          epochTraceIndices.push(idx);
          epochTraceZ.push(baseRow);
        }});
      }});

      Plotly.restyle('metric3d_plot', {{ visible: visibility }});
      if (epochTraceIndices.length > 0) {{
        Plotly.restyle('metric3d_plot', {{ z: epochTraceZ }}, epochTraceIndices);
      }}
      Plotly.relayout('metric3d_plot', {{
        'scene.yaxis.tickvals': selected.map(m => metricIdxMap[m]),
        'scene.yaxis.ticktext': selected.map(m => metricLabelMap[m] || m),
      }});
    }}
    Object.values(metricKeyMap).forEach(id => {{
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', applyMetricFilter);
    }});
    Object.values(modelKeyMap).forEach(id => {{
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', applyMetricFilter);
    }});
    document.querySelectorAll('input[name="render_mode"]').forEach(el => {{
      el.addEventListener('change', applyMetricFilter);
    }});

    function downloadBlob(filename, content, mime) {{
      const blob = new Blob([content], {{ type: mime }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    document.getElementById('btn_save_png').addEventListener('click', () => {{
      Plotly.downloadImage('metric3d_plot', {{
        format: 'png',
        filename: 'metriclearning_3d_current_view',
        width: 1800,
        height: 1200,
        scale: 2
      }});
    }});

    document.getElementById('btn_save_state').addEventListener('click', () => {{
      const metricState = {{}};
      Object.entries(metricKeyMap).forEach(([metric, id]) => {{
        metricState[metric] = document.getElementById(id).checked;
      }});
      const modelState = {{}};
      Object.entries(modelKeyMap).forEach(([model, id]) => {{
        modelState[model] = document.getElementById(id).checked;
      }});
      const gd = document.getElementById('metric3d_plot');
      const camera = gd && gd._fullLayout && gd._fullLayout.scene ? gd._fullLayout.scene.camera : null;
      const state = {{
        saved_at: new Date().toISOString(),
        metric_visibility: metricState,
        model_visibility: modelState,
        camera: camera
      }};
      downloadBlob('metriclearning_3d_state.json', JSON.stringify(state, null, 2), 'application/json');
    }});

    applyMetricFilter();
  </script>
</body>
</html>"""

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")


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


def build_epoch_tables(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    epoch_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    common_numeric_metrics: set[str] | None = None

    for run_dir in run_dirs:
        metrics_df, config = load_run_data(run_dir)

        backbone_raw = str(config.get("model", {}).get("backbone_name", run_dir.name))
        model_name = prettify_backbone_name(backbone_raw)
        train_cfg = config.get("training", {})
        data_cfg = config.get("data", {})
        loss_cfg = train_cfg.get("loss", {})
        hard_mining_cfg = train_cfg.get("hard_mining", {})

        numeric_cols = [
            c
            for c in metrics_df.columns
            if c not in {"epoch", "is_best"} and pd.api.types.is_numeric_dtype(metrics_df[c])
        ]
        if common_numeric_metrics is None:
            common_numeric_metrics = set(numeric_cols)
        else:
            common_numeric_metrics &= set(numeric_cols)

        df = metrics_df.copy()
        df["model_name"] = model_name
        df["backbone_name"] = backbone_raw
        df["run_dir"] = str(run_dir)
        epoch_frames.append(df)

        best_idx = pd.to_numeric(df.get("recall_at_1"), errors="coerce").idxmax()
        best_row = df.loc[best_idx]
        final_row = df.iloc[-1]

        if "val_loss" in df.columns:
            val_series = pd.to_numeric(df["val_loss"], errors="coerce")
            min_val_idx = int(val_series.idxmin())
            min_val_loss = float(val_series.min())
            min_val_loss_epoch = int(df.loc[min_val_idx, "epoch"])
        else:
            min_val_loss = None
            min_val_loss_epoch = None

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
                "best_epoch_recall_at_1": int(best_row["epoch"]),
                "best_recall_at_1_pct": float(best_row.get("recall_at_1", 0.0)) * 100.0,
                "best_mAP@R_pct": float(best_row.get("mAP@R", 0.0)) * 100.0,
                "best_nmi_pct": float(best_row.get("NMI", 0.0)) * 100.0,
                "best_f1_macro_pct": float(best_row.get("F1_macro", 0.0)) * 100.0,
                "final_recall_at_1_pct": float(final_row.get("recall_at_1", 0.0)) * 100.0,
                "final_mAP@R_pct": float(final_row.get("mAP@R", 0.0)) * 100.0,
                "final_nmi_pct": float(final_row.get("NMI", 0.0)) * 100.0,
                "final_f1_macro_pct": float(final_row.get("F1_macro", 0.0)) * 100.0,
                "min_val_loss": min_val_loss,
                "epoch_min_val_loss": min_val_loss_epoch,
            }
        )

    epoch_df = pd.concat(epoch_frames, ignore_index=True)
    metric_cols = reorder_metrics(list(common_numeric_metrics or []))
    probability_metrics = [m for m in metric_cols if is_probability_metric(epoch_df[m])]

    for metric in probability_metrics:
        epoch_df[f"{metric}_pct"] = pd.to_numeric(epoch_df[metric], errors="coerce") * 100.0

    ordered_cols = ["model_name", "backbone_name", "run_dir", "epoch", "is_best"] + metric_cols
    ordered_cols += [f"{m}_pct" for m in probability_metrics]
    ordered_cols = [c for c in ordered_cols if c in epoch_df.columns]
    epoch_df = epoch_df[ordered_cols].sort_values(["model_name", "epoch"]).reset_index(drop=True)

    summary_df = pd.DataFrame(summary_rows).sort_values("best_recall_at_1_pct", ascending=False).reset_index(drop=True)
    return epoch_df, summary_df, metric_cols, probability_metrics


def save_primary_metrics_plot(epoch_df: pd.DataFrame, output_path: Path) -> None:
    plot_metrics = [m for m in PRIMARY_METRICS if f"{m}_pct" in epoch_df.columns]
    if not plot_metrics:
        return

    n = len(plot_metrics)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows), sharex=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    xticks = compute_epoch_ticks(epoch_df["epoch"], min_ticks=5)

    for idx, metric in enumerate(plot_metrics):
        ax = axes_list[idx]
        mcol = f"{metric}_pct"
        for model_name, model_df in epoch_df.groupby("model_name"):
            plot_df = model_df.sort_values("epoch")
            ax.plot(plot_df["epoch"], plot_df[mcol], marker="o", linewidth=1.8, markersize=3, label=model_name)
        ax.set_title(f"{metric} (%)")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Valor (%)")
        y_low, y_high = compute_dynamic_ylim(epoch_df[mcol], clamp_to_percent=True)
        ax.set_ylim(y_low, y_high)
        if xticks:
            ax.set_xticks(xticks)
        ax.grid(alpha=0.25)

    handles, labels = axes_list[0].get_legend_handles_labels()
    for j in range(len(plot_metrics), len(axes_list)):
        axes_list[j].axis("off")

    fig.legend(handles, labels, loc="upper center", ncol=4, title="Modelo")
    fig.suptitle("Comparacion por epoca - indicadores principales", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_secondary_metrics_plot(epoch_df: pd.DataFrame, output_path: Path) -> None:
    plot_metrics = [m for m in SECONDARY_METRICS if m in epoch_df.columns]
    if not plot_metrics:
        return

    n = len(plot_metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), sharex=True)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    xticks = compute_epoch_ticks(epoch_df["epoch"], min_ticks=5)

    for idx, metric in enumerate(plot_metrics):
        ax = axes_list[idx]
        for model_name, model_df in epoch_df.groupby("model_name"):
            plot_df = model_df.sort_values("epoch")
            ax.plot(plot_df["epoch"], plot_df[metric], marker="o", linewidth=1.8, markersize=3, label=model_name)
        ax.set_title(metric)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Valor")
        y_low, y_high = compute_dynamic_ylim(epoch_df[metric], clamp_to_percent=False)
        ax.set_ylim(y_low, y_high)
        if xticks:
            ax.set_xticks(xticks)
        ax.grid(alpha=0.25)

    handles, labels = axes_list[0].get_legend_handles_labels()
    for j in range(len(plot_metrics), len(axes_list)):
        axes_list[j].axis("off")

    fig.legend(handles, labels, loc="upper center", ncol=4, title="Modelo")
    fig.suptitle("Comparacion por epoca - indicadores secundarios", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_per_metric_tables(epoch_df: pd.DataFrame, metric_cols: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric in metric_cols:
        if metric not in epoch_df.columns:
            continue
        use_col = f"{metric}_pct" if f"{metric}_pct" in epoch_df.columns else metric
        pivot_df = (
            epoch_df.pivot_table(index="epoch", columns="model_name", values=use_col, aggfunc="mean")
            .sort_index()
            .reset_index()
        )
        pivot_df.to_csv(
            output_dir / f"epoch_comparison_{sanitize_metric_name(metric)}.csv",
            index=False,
            float_format="%.6f",
        )


def build_a4_wide_table(epoch_df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    compact = epoch_df[["epoch", "model_name"]].copy()
    for metric in metric_cols:
        if metric in epoch_df.columns:
            col = f"{metric}_pct" if f"{metric}_pct" in epoch_df.columns else metric
            compact[metric] = epoch_df[col]

    wide_df = compact.pivot_table(index="epoch", columns="model_name", aggfunc="mean").sort_index()
    wide_df.columns = [f"{sanitize_metric_name(m)}__{sanitize_metric_name(model)}" for m, model in wide_df.columns]
    return wide_df.reset_index()


def write_readme(
    output_dir: Path,
    run_dirs: list[Path],
    long_csv: Path,
    wide_csv: Path,
    summary_csv: Path,
    threed_plot: Path,
    interactive_html: Path,
    per_metric_dir: Path,
    metric_cols: list[str],
) -> None:
    text = [
        "# Comparacion de 4 modelos por epoca",
        "",
        "Este reporte compara entrenamientos por epoca (sin normalizar por % de completacion).",
        "",
        "## Runs incluidos",
    ]
    text.extend([f"- `{run_dir}`" for run_dir in run_dirs])
    text.extend(
        [
            "",
            "## Archivos generados",
            f"- `{long_csv.name}`: tabla larga con todos los indicadores comparables por epoca y modelo.",
            f"- `{wide_csv.name}`: tabla ancha orientada a reporte LaTeX A4 (epoch + metrica_modelo).",
            f"- `{summary_csv.name}`: configuraciones y resumen de mejores/finales indicadores.",
            f"- `{threed_plot.name}`: grafico 3D unico con 13 metricas por 20 epocas y 4 modelos.",
            f"- `{interactive_html.name}`: grafico 3D interactivo con checkboxes para mostrar/ocultar metricas.",
            f"- `{per_metric_dir.name}/`: CSV por metrica (epoch x 4 modelos), ideal para A4.",
            "",
            "## Indicadores comparados",
            f"- {', '.join(metric_cols)}",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate full epoch-based comparison report for multiple runs."
    )
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories to compare")
    parser.add_argument("--reports-root", default="reports", help="Reports root directory")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.runs]
    for run_dir in run_dirs:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.reports_root) / f"model_comparison_epochs_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_df, summary_df, metric_cols, _probability_metrics = build_epoch_tables(run_dirs)

    long_csv = output_dir / "epoch_model_metrics_long.csv"
    epoch_df.to_csv(long_csv, index=False, float_format="%.6f")

    wide_df = build_a4_wide_table(epoch_df, metric_cols)
    wide_csv = output_dir / "epoch_model_metrics_wide_a4.csv"
    wide_df.to_csv(wide_csv, index=False, float_format="%.6f")

    summary_csv = output_dir / "model_configuration_and_summary.csv"
    summary_df.to_csv(summary_csv, index=False, float_format="%.6f")

    threed_plot = output_dir / "all_models_3d_multimetric_by_epoch.png"
    save_3d_multimetric_plot(epoch_df, threed_plot)
    interactive_html = output_dir / "all_models_3d_multimetric_interactive.html"
    save_interactive_3d_mantle_html(epoch_df, interactive_html)

    per_metric_dir = output_dir / "tables_per_metric_a4"
    write_per_metric_tables(epoch_df, metric_cols, per_metric_dir)

    write_readme(
        output_dir,
        run_dirs,
        long_csv,
        wide_csv,
        summary_csv,
        threed_plot,
        interactive_html,
        per_metric_dir,
        metric_cols,
    )

    print(f"Report generated successfully: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
