"""
Standalone modular tool to visualize model embedding spaces in 3D.

Main goals:
- Let the user select a results folder.
- Discover model artifacts automatically.
- Let the user select checkpoint/reference/centroids.
- Render an interactive 3D view of the hyperspherical embedding space.

Run example:
    python scripts/model_hypersphere_viewer.py --source-dir runs/2026-02-06_20-17-07
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class ArtifactSelection:
    checkpoint_path: Optional[Path]
    reference_path: Optional[Path]
    centroids_path: Optional[Path]


@dataclass
class LoadedEmbeddingData:
    embeddings: np.ndarray
    labels: np.ndarray
    class_names: List[str]
    centroids: Optional[np.ndarray]
    source_summary: Dict[str, str]


class HyperSphereViewer:
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        method: str,
        seed: int,
        perplexity: float,
        umap_neighbors: int,
        umap_min_dist: float,
        max_points: int,
        non_interactive: bool,
    ):
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.method = method.lower()
        self.seed = seed
        self.perplexity = perplexity
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.max_points = max_points
        self.non_interactive = non_interactive

    @staticmethod
    def _normalize_to_sphere(points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.clip(norms, eps, None)
        return points / norms

    @staticmethod
    def _turbo_palette_hex(n_classes: int) -> List[str]:
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        if n_classes <= 0:
            return ["#1f77b4"]
        cmap = plt.get_cmap("turbo")
        if n_classes == 1:
            return [mcolors.to_hex(cmap(0.5))]
        return [mcolors.to_hex(cmap(i / (n_classes - 1))) for i in range(n_classes)]

    # -------------------------
    # Artifact discovery/select
    # -------------------------
    def discover_artifacts(self) -> Dict[str, List[Path]]:
        checkpoints = self._sorted_paths(self.source_dir.rglob("*.pth"))
        references = self._sorted_paths(self.source_dir.rglob("reference_embeddings.pt"))
        centroids = self._sorted_paths(self.source_dir.rglob("class_centroids.pt"))
        return {
            "checkpoints": checkpoints,
            "references": references,
            "centroids": centroids,
        }

    @staticmethod
    def _sorted_paths(paths: Sequence[Path]) -> List[Path]:
        return sorted((p.resolve() for p in paths), key=lambda p: str(p).lower())

    def choose_artifacts(
        self,
        artifacts: Dict[str, List[Path]],
        checkpoint_hint: Optional[Path],
        reference_hint: Optional[Path],
        centroids_hint: Optional[Path],
    ) -> ArtifactSelection:
        checkpoint = self._choose_single(
            "checkpoint (.pth)",
            artifacts["checkpoints"],
            checkpoint_hint,
            allow_none=True,
        )
        reference = self._choose_single(
            "reference embeddings (reference_embeddings.pt)",
            artifacts["references"],
            reference_hint,
            allow_none=True,
        )
        centroids = self._choose_single(
            "centroids (class_centroids.pt)",
            artifacts["centroids"],
            centroids_hint,
            allow_none=True,
        )
        return ArtifactSelection(
            checkpoint_path=checkpoint,
            reference_path=reference,
            centroids_path=centroids,
        )

    def _choose_single(
        self,
        label: str,
        candidates: List[Path],
        hint: Optional[Path],
        allow_none: bool,
    ) -> Optional[Path]:
        if hint is not None:
            resolved = hint.resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Path provided for {label} does not exist: {resolved}")
            return resolved

        if len(candidates) == 0:
            if allow_none:
                return None
            raise FileNotFoundError(f"No candidates found for {label} in: {self.source_dir}")

        if len(candidates) == 1 or self.non_interactive:
            return candidates[0]

        print(f"\nFound multiple candidates for {label}:")
        for idx, path in enumerate(candidates):
            print(f"  [{idx}] {path}")
        print("  [n] none")
        while True:
            selected = input(f"Select {label} index: ").strip().lower()
            if allow_none and selected == "n":
                return None
            if selected.isdigit():
                pos = int(selected)
                if 0 <= pos < len(candidates):
                    return candidates[pos]
            print("Invalid selection. Try again.")

    # -------------------------
    # Data loading
    # -------------------------
    def load_embedding_data(self, selected: ArtifactSelection) -> LoadedEmbeddingData:
        checkpoint_data: Dict = {}
        reference_data: Dict = {}
        centroids_data: Dict = {}
        source_summary: Dict[str, str] = {}

        if selected.checkpoint_path is not None:
            checkpoint_data = torch.load(
                selected.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            source_summary["checkpoint"] = str(selected.checkpoint_path)

        if selected.reference_path is not None:
            reference_data = torch.load(
                selected.reference_path,
                map_location="cpu",
                weights_only=False,
            )
            source_summary["reference_embeddings"] = str(selected.reference_path)

        if selected.centroids_path is not None:
            centroids_data = torch.load(
                selected.centroids_path,
                map_location="cpu",
                weights_only=False,
            )
            source_summary["class_centroids"] = str(selected.centroids_path)

        embeddings, labels = self._resolve_embeddings_and_labels(checkpoint_data, reference_data)
        class_names = self._resolve_class_names(checkpoint_data, reference_data, centroids_data, labels)
        centroids = self._resolve_centroids(centroids_data, embeddings, labels, len(class_names))

        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Mismatch embeddings/labels lengths: {embeddings.shape[0]} vs {labels.shape[0]}"
            )

        if self.max_points > 0 and embeddings.shape[0] > self.max_points:
            embeddings, labels = self._downsample(embeddings, labels, self.max_points, self.seed)
            source_summary["downsampled"] = f"true ({self.max_points} points)"
        else:
            source_summary["downsampled"] = "false"

        return LoadedEmbeddingData(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names,
            centroids=centroids,
            source_summary=source_summary,
        )

    def _resolve_embeddings_and_labels(
        self,
        checkpoint_data: Dict,
        reference_data: Dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if "reference_embeddings" in checkpoint_data and "reference_labels" in checkpoint_data:
            embeddings = self._to_numpy(checkpoint_data["reference_embeddings"])
            labels = self._to_numpy(checkpoint_data["reference_labels"]).astype(int)
            return embeddings, labels

        if "embeddings" in reference_data and "labels" in reference_data:
            embeddings = self._to_numpy(reference_data["embeddings"])
            labels = self._to_numpy(reference_data["labels"]).astype(int)
            return embeddings, labels

        raise ValueError(
            "Could not load embeddings+labels. Provide a checkpoint with reference embeddings "
            "or a reference_embeddings.pt file."
        )

    def _resolve_class_names(
        self,
        checkpoint_data: Dict,
        reference_data: Dict,
        centroids_data: Dict,
        labels: np.ndarray,
    ) -> List[str]:
        if "class_names" in checkpoint_data:
            return [str(x) for x in checkpoint_data["class_names"]]
        if "class_names" in reference_data:
            return [str(x) for x in reference_data["class_names"]]
        if "class_names" in centroids_data:
            return [str(x) for x in centroids_data["class_names"]]

        n_classes = int(np.max(labels)) + 1 if labels.size > 0 else 0
        return [f"class_{idx}" for idx in range(n_classes)]

    def _resolve_centroids(
        self,
        centroids_data: Dict,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_classes: int,
    ) -> Optional[np.ndarray]:
        if "centroids" in centroids_data:
            arr = self._to_numpy(centroids_data["centroids"])
            if arr.ndim != 2:
                raise ValueError(f"Centroids must be 2D. Got shape: {arr.shape}")
            return arr

        if n_classes == 0:
            return None

        centroids = []
        for class_id in range(n_classes):
            mask = labels == class_id
            if np.any(mask):
                centroids.append(np.mean(embeddings[mask], axis=0))
            else:
                centroids.append(np.zeros(embeddings.shape[1], dtype=np.float32))
        return np.vstack(centroids)

    @staticmethod
    def _downsample(
        embeddings: np.ndarray,
        labels: np.ndarray,
        max_points: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(embeddings.shape[0], size=max_points, replace=False)
        return embeddings[idx], labels[idx]

    @staticmethod
    def _to_numpy(obj) -> np.ndarray:
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy()
        arr = np.asarray(obj)
        return arr

    # -------------------------
    # Projection
    # -------------------------
    def project_to_3d(
        self,
        embeddings: np.ndarray,
        centroids: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, str]]:
        projection_info: Dict[str, str] = {
            "method": self.method,
            "seed": str(self.seed),
        }

        if centroids is not None and centroids.shape[1] != embeddings.shape[1]:
            centroids = None

        if centroids is None:
            full = embeddings
            n_embeddings = embeddings.shape[0]
        else:
            full = np.vstack([embeddings, centroids])
            n_embeddings = embeddings.shape[0]

        if self.method == "pca":
            reducer = PCA(n_components=3, random_state=self.seed)
            projected = reducer.fit_transform(full)
            projection_info["explained_variance_ratio"] = np.array2string(
                reducer.explained_variance_ratio_, precision=6, separator=","
            )
        elif self.method == "tsne":
            perpl = min(self.perplexity, max(5.0, (full.shape[0] - 1) / 3))
            reducer = TSNE(
                n_components=3,
                perplexity=perpl,
                learning_rate="auto",
                init="pca",
                random_state=self.seed,
                max_iter=1200,
            )
            projected = reducer.fit_transform(full)
            projection_info["perplexity"] = f"{perpl:.2f}"
        elif self.method == "umap":
            try:
                import umap
            except ImportError as exc:
                raise RuntimeError(
                    "UMAP selected but package 'umap-learn' is not installed. "
                    "Install with: pip install umap-learn"
                ) from exc
            reducer = umap.UMAP(
                n_components=3,
                metric="cosine",
                n_neighbors=self.umap_neighbors,
                min_dist=self.umap_min_dist,
                random_state=self.seed,
            )
            projected = reducer.fit_transform(full)
            projection_info["n_neighbors"] = str(self.umap_neighbors)
            projection_info["min_dist"] = str(self.umap_min_dist)
        else:
            raise ValueError(f"Unknown projection method: {self.method}")

        emb_3d = self._normalize_to_sphere(projected[:n_embeddings])
        cent_3d = (
            self._normalize_to_sphere(projected[n_embeddings:])
            if centroids is not None
            else None
        )
        return emb_3d, cent_3d, projection_info

    # -------------------------
    # Plot/export
    # -------------------------
    def build_and_save_visualization(
        self,
        data: LoadedEmbeddingData,
        emb_3d: np.ndarray,
        cent_3d: Optional[np.ndarray],
        projection_info: Dict[str, str],
        open_browser: bool,
    ) -> Tuple[Path, Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = self._build_plot(
            data=data,
            emb_3d=emb_3d,
            cent_3d=cent_3d,
            open_browser=open_browser,
        )

        projected_npz_path = self.output_dir / "hypersphere_projection_3d.npz"
        np.savez_compressed(
            projected_npz_path,
            embeddings_original=data.embeddings,
            labels=data.labels,
            projected_embeddings=emb_3d,
            projected_centroids=cent_3d if cent_3d is not None else np.array([]),
            class_names=np.array(data.class_names, dtype=object),
        )

        norms_all = np.linalg.norm(data.embeddings, axis=1)
        report = {
            "source_dir": str(self.source_dir),
            "output_dir": str(self.output_dir),
            "source_summary": data.source_summary,
            "projection": projection_info,
            "visualization_file": str(plot_path),
            "n_samples": int(data.embeddings.shape[0]),
            "embedding_dim": int(data.embeddings.shape[1]),
            "n_classes": int(len(data.class_names)),
            "embedding_norm_mean": float(norms_all.mean()),
            "embedding_norm_std": float(norms_all.std()),
        }
        report_path = self.output_dir / "hypersphere_view_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return plot_path, projected_npz_path, report_path

    def _build_plot(
        self,
        data: LoadedEmbeddingData,
        emb_3d: np.ndarray,
        cent_3d: Optional[np.ndarray],
        open_browser: bool,
    ) -> Path:
        try:
            return self._build_plotly_plot(data, emb_3d, cent_3d, open_browser)
        except ImportError:
            print(
                "WARNING: plotly is not installed. Falling back to static matplotlib figure.\n"
                "Install plotly for full interactivity: pip install plotly"
            )
            return self._build_matplotlib_plot(data, emb_3d, cent_3d)

    def _build_plotly_plot(
        self,
        data: LoadedEmbeddingData,
        emb_3d: np.ndarray,
        cent_3d: Optional[np.ndarray],
        open_browser: bool,
    ) -> Path:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        labels = data.labels
        class_names = data.class_names
        n_classes = len(class_names)

        available_methods = ["pca", "tsne"]
        try:
            import umap  # noqa: F401
            available_methods.insert(1, "umap")
        except Exception:
            pass

        projections: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        projections[self.method] = (emb_3d, cent_3d)
        for method in available_methods:
            if method == self.method:
                continue
            alt_emb, alt_cent, _ = self._project_with_method(
                method=method,
                embeddings=data.embeddings,
                centroids=data.centroids,
            )
            projections[method] = (alt_emb, alt_cent)

        if self.method not in projections:
            self.method = available_methods[0]

        method_order = [m for m in ["pca", "umap", "tsne"] if m in projections]
        class_colors = self._turbo_palette_hex(n_classes)
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "table"}]],
            column_widths=[0.78, 0.22],
            horizontal_spacing=0.02,
        )

        class_ids_sorted = sorted(int(x) for x in np.unique(labels))
        initial_emb, initial_cent = projections[self.method]

        # Sphere (single persistent trace)
        fig.add_trace(self._create_sphere_trace(go, visible=True), row=1, col=1)

        class_trace_indices: List[int] = []
        class_hover_templates_by_method: Dict[str, List[str]] = {}
        for class_id in class_ids_sorted:
            mask = labels == class_id
            class_name = class_names[class_id] if class_id < n_classes else f"class_{class_id}"
            norms = np.linalg.norm(data.embeddings[mask], axis=1)
            custom_data = np.column_stack(
                [
                    np.where(mask)[0],
                    labels[mask],
                    np.array([class_name] * int(np.sum(mask)), dtype=object),
                    norms,
                ]
            )
            color = class_colors[class_id % len(class_colors)]

            fig.add_trace(
                go.Scatter3d(
                    x=initial_emb[mask, 0],
                    y=initial_emb[mask, 1],
                    z=initial_emb[mask, 2],
                    mode="markers",
                    name=class_name,
                    showlegend=False,
                    marker=dict(size=4.6, opacity=0.9, color=color),
                    customdata=custom_data,
                    hovertemplate=(
                        "<b>Punto de embedding</b><br>"
                        "Clase: %{customdata[2]}<br>"
                        "Sample ID: %{customdata[0]}<br>"
                        "Clase ID: %{customdata[1]}<br>"
                        "Norma radial: %{customdata[3]:.5f}<br>"
                        "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            class_trace_indices.append(len(fig.data) - 1)

        centroid_trace_index: Optional[int] = None
        centroid_hover_templates_by_method: Dict[str, str] = {}
        if initial_cent is not None:
            centroid_colors = []
            centroid_labels = []
            for idx in range(initial_cent.shape[0]):
                if idx < n_classes:
                    centroid_colors.append(class_colors[idx % len(class_colors)])
                    centroid_labels.append(class_names[idx])
                else:
                    centroid_colors.append("#334155")
                    centroid_labels.append(f"class_{idx}")
            fig.add_trace(
                go.Scatter3d(
                    x=initial_cent[:, 0],
                    y=initial_cent[:, 1],
                    z=initial_cent[:, 2],
                    mode="markers",
                    name="centroides",
                    showlegend=False,
                    marker=dict(
                        size=8,
                        symbol="circle",
                        color=centroid_colors,
                        line=dict(color="#ffffff", width=1.7),
                    ),
                    customdata=np.array(centroid_labels, dtype=object).reshape(-1, 1),
                    hovertemplate=(
                        "<b>Punto centroide</b><br>"
                        "Clase: %{customdata[0]}<br>"
                        "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=1,
            )
            centroid_trace_index = len(fig.data) - 1

        # Right-side fixed table (always visible)
        row_names: List[str] = []
        row_counts: List[int] = []
        color_cells: List[str] = []
        for class_id in class_ids_sorted:
            class_name = class_names[class_id] if class_id < n_classes else f"class_{class_id}"
            row_names.append(class_name)
            row_counts.append(int(np.sum(labels == class_id)))
            color_cells.append(class_colors[class_id % len(class_colors)])

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Color", "Clase", "Muestras"],
                    fill_color="#0f172a",
                    font=dict(color="white", size=12),
                    align=["center", "left", "right"],
                    height=30,
                ),
                cells=dict(
                    values=[[""] * len(row_names), row_names, row_counts],
                    fill_color=[
                        color_cells,
                        ["#ffffff"] * len(row_names),
                        ["#ffffff"] * len(row_names),
                    ],
                    align=["center", "left", "right"],
                    font=dict(color=["#ffffff", "#111827", "#111827"], size=11),
                    height=28,
                ),
            ),
            row=1,
            col=2,
        )

        # Radial reference axes (informative, no euclidean grid)
        axis_len = 1.1
        axis_specs = [
            ([-axis_len, axis_len], [0, 0], [0, 0], "#64748b", "r-x"),
            ([0, 0], [-axis_len, axis_len], [0, 0], "#64748b", "r-y"),
            ([0, 0], [0, 0], [-axis_len, axis_len], "#64748b", "r-z"),
        ]
        for x_line, y_line, z_line, color, name in axis_specs:
            fig.add_trace(
                go.Scatter3d(
                    x=x_line,
                    y=y_line,
                    z=z_line,
                    mode="lines",
                    line=dict(color=color, width=2),
                    showlegend=False,
                    hoverinfo="skip",
                    name=name,
                ),
                row=1,
                col=1,
            )

        # Text labels for radial axes
        axis_label_specs = [
            (axis_len + 0.02, 0, 0, "r-x"),
            (0, axis_len + 0.02, 0, "r-y"),
            (0, 0, axis_len + 0.02, "r-z"),
        ]
        for x_t, y_t, z_t, txt in axis_label_specs:
            fig.add_trace(
                go.Scatter3d(
                    x=[x_t],
                    y=[y_t],
                    z=[z_t],
                    mode="text",
                    text=[txt],
                    textfont=dict(size=11, color="#334155"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

        norms_all = np.linalg.norm(data.embeddings, axis=1)
        base_title = (
            "3D Hyperspherical Embedding Viewer"
            f"<br>points={data.embeddings.shape[0]}, dim={data.embeddings.shape[1]}, "
            f"norm_mean={norms_all.mean():.5f}, norm_std={norms_all.std():.6f}"
        )

        # Precompute per-method hover text so the selected method is always correct.
        for method in method_order:
            method_templates: List[str] = []
            for class_id in class_ids_sorted:
                class_name = class_names[class_id] if class_id < n_classes else f"class_{class_id}"
                method_templates.append(
                    (
                        "<b>Punto de embedding</b><br>"
                        "Metodo: " + method.upper() + "<br>"
                        "Clase: %{customdata[2]}<br>"
                        "Sample ID: %{customdata[0]}<br>"
                        "Clase ID: %{customdata[1]}<br>"
                        "Norma radial: %{customdata[3]:.5f}<br>"
                        "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
                    )
                )
            class_hover_templates_by_method[method] = method_templates
            centroid_hover_templates_by_method[method] = (
                "<b>Punto centroide</b><br>"
                "Metodo: " + method.upper() + "<br>"
                "Clase: %{customdata[0]}<br>"
                "x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>"
            )

        buttons = []
        for method in method_order:
            method_emb, method_cent = projections[method]
            update_indices = list(class_trace_indices)
            x_values = []
            y_values = []
            z_values = []
            hover_values = list(class_hover_templates_by_method[method])

            for class_id in class_ids_sorted:
                mask = labels == class_id
                x_values.append(method_emb[mask, 0])
                y_values.append(method_emb[mask, 1])
                z_values.append(method_emb[mask, 2])

            if centroid_trace_index is not None and method_cent is not None:
                update_indices.append(centroid_trace_index)
                x_values.append(method_cent[:, 0])
                y_values.append(method_cent[:, 1])
                z_values.append(method_cent[:, 2])
                hover_values.append(centroid_hover_templates_by_method[method])

            buttons.append(
                dict(
                    label=method.upper(),
                    method="update",
                    args=[
                        {
                            "x": x_values,
                            "y": y_values,
                            "z": z_values,
                            "hovertemplate": hover_values,
                        },
                        {
                            "title": base_title + f"<br>method={method}",
                        },
                        update_indices,
                    ],
                )
            )

        fig.update_layout(
            title=base_title + f"<br>method={self.method}",
            scene=dict(
                xaxis_title="",
                yaxis_title="",
                zaxis_title="",
                xaxis=dict(range=[-1.2, 1.2]),
                yaxis=dict(range=[-1.2, 1.2]),
                zaxis=dict(range=[-1.2, 1.2]),
                aspectmode="cube",
                xaxis_showbackground=False,
                yaxis_showbackground=False,
                zaxis_showbackground=False,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                zaxis_showgrid=False,
                xaxis_zeroline=False,
                yaxis_zeroline=False,
                zaxis_zeroline=False,
                xaxis_visible=False,
                yaxis_visible=False,
                zaxis_visible=False,
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False,
            margin=dict(l=8, r=8, t=95, b=6),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Segoe UI, Inter, Arial", color="#111827"),
            annotations=[
                dict(
                    text="Selecciona un punto para ver su clase y su significado radial en el hover",
                    x=0.01,
                    y=1.03,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=11, color="#475569"),
                    align="left",
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=buttons,
                    x=0.01,
                    y=1.10,
                    xanchor="left",
                    yanchor="top",
                    showactive=True,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="#cbd5e1",
                    borderwidth=1,
                )
            ],
        )

        html_path = self.output_dir / "hypersphere_embedding_3d.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn", auto_open=open_browser)
        return html_path

    def _project_with_method(
        self,
        method: str,
        embeddings: np.ndarray,
        centroids: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, str]]:
        original = self.method
        self.method = method
        try:
            return self.project_to_3d(embeddings=embeddings, centroids=centroids)
        finally:
            self.method = original

    @staticmethod
    def _create_sphere_trace(go_module, visible: bool):
        phi = np.linspace(0, np.pi, 36)
        theta = np.linspace(0, 2 * np.pi, 72)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        x = np.sin(phi_grid) * np.cos(theta_grid)
        y = np.sin(phi_grid) * np.sin(theta_grid)
        z = np.cos(phi_grid)
        return go_module.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.11,
            showscale=False,
            name="unit_sphere",
            visible=visible,
            hoverinfo="skip",
            colorscale=[[0, "#94A3B8"], [1, "#94A3B8"]],
        )

    def _build_matplotlib_plot(
        self,
        data: LoadedEmbeddingData,
        emb_3d: np.ndarray,
        cent_3d: Optional[np.ndarray],
    ) -> Path:
        import matplotlib.pyplot as plt

        labels = data.labels
        class_names = data.class_names
        n_classes = len(class_names)

        fig = plt.figure(figsize=(13, 10))
        ax = fig.add_subplot(111, projection="3d")
        class_colors = self._turbo_palette_hex(n_classes)

        for class_id in np.unique(labels):
            mask = labels == class_id
            class_name = class_names[int(class_id)] if int(class_id) < n_classes else f"class_{int(class_id)}"
            ax.scatter(
                emb_3d[mask, 0],
                emb_3d[mask, 1],
                emb_3d[mask, 2],
                s=8,
                alpha=0.72,
                color=class_colors[int(class_id) % len(class_colors)],
                label=class_name,
            )

        if cent_3d is not None:
            centroid_colors = []
            for idx in range(cent_3d.shape[0]):
                if idx < n_classes:
                    centroid_colors.append(class_colors[idx % len(class_colors)])
                else:
                    centroid_colors.append("#334155")
            ax.scatter(
                cent_3d[:, 0],
                cent_3d[:, 1],
                cent_3d[:, 2],
                s=56,
                marker="o",
                c=centroid_colors,
                edgecolors="white",
                label="class_centroids",
            )

        # Draw unit sphere wireframe so the hypersphere is explicit.
        phi = np.linspace(0, np.pi, 30)
        theta = np.linspace(0, 2 * np.pi, 60)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        x = np.sin(phi_grid) * np.cos(theta_grid)
        y = np.sin(phi_grid) * np.sin(theta_grid)
        z = np.cos(phi_grid)
        ax.plot_wireframe(x, y, z, color="#8892a0", linewidth=0.25, alpha=0.35)

        norms_all = np.linalg.norm(data.embeddings, axis=1)
        ax.set_title(
            "3D Hyperspherical Embedding Viewer (static fallback)\n"
            f"method={self.method}, points={data.embeddings.shape[0]}, dim={data.embeddings.shape[1]}, "
            f"norm_mean={norms_all.mean():.5f}, norm_std={norms_all.std():.6f}"
        )
        ax.set_xlabel("X (unit sphere)")
        ax.set_ylabel("Y (unit sphere)")
        ax.set_zlabel("Z (unit sphere)")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_zlim(-1.2, 1.2)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

        png_path = self.output_dir / "hypersphere_embedding_3d.png"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return png_path

    # -------------------------
    # Run
    # -------------------------
    def run(
        self,
        checkpoint_hint: Optional[Path],
        reference_hint: Optional[Path],
        centroids_hint: Optional[Path],
        open_browser: bool,
    ) -> Tuple[Path, Path, Path]:
        artifacts = self.discover_artifacts()
        selected = self.choose_artifacts(
            artifacts=artifacts,
            checkpoint_hint=checkpoint_hint,
            reference_hint=reference_hint,
            centroids_hint=centroids_hint,
        )
        data = self.load_embedding_data(selected)
        emb_3d, cent_3d, projection_info = self.project_to_3d(
            embeddings=data.embeddings,
            centroids=data.centroids,
        )
        return self.build_and_save_visualization(
            data=data,
            emb_3d=emb_3d,
            cent_3d=cent_3d,
            projection_info=projection_info,
            open_browser=open_browser,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone 3D hypersphere embedding visualizer from model result folders."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Folder that contains model/result artifacts (checkpoint, reference embeddings, centroids).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder. Default: <source-dir>/visualization_3d",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional explicit checkpoint path (.pth).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional explicit reference_embeddings.pt path.",
    )
    parser.add_argument(
        "--centroids",
        type=str,
        default=None,
        help="Optional explicit class_centroids.pt path.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["pca", "umap", "tsne"],
        help="3D projection method.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE (used only with --method tsne).",
    )
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=25,
        help="n_neighbors for UMAP (used only with --method umap).",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="min_dist for UMAP (used only with --method umap).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=30000,
        help="Optional downsampling cap to keep interactive rendering fluid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Use first discovered artifact when multiple candidates exist.",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open generated HTML automatically in browser.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        print(f"ERROR: source directory does not exist: {source_dir}")
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else source_dir / "visualization_3d"

    viewer = HyperSphereViewer(
        source_dir=source_dir,
        output_dir=output_dir,
        method=args.method,
        seed=args.seed,
        perplexity=args.perplexity,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        max_points=args.max_points,
        non_interactive=args.non_interactive,
    )

    checkpoint_hint = Path(args.checkpoint) if args.checkpoint else None
    reference_hint = Path(args.reference) if args.reference else None
    centroids_hint = Path(args.centroids) if args.centroids else None

    try:
        html_path, npz_path, report_path = viewer.run(
            checkpoint_hint=checkpoint_hint,
            reference_hint=reference_hint,
            centroids_hint=centroids_hint,
            open_browser=args.open_browser,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2

    print("\nHypersphere visualization generated successfully:")
    print(f"- interactive_html: {html_path}")
    print(f"- projection_data: {npz_path}")
    print(f"- run_report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

