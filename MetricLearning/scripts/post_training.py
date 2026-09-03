"""
Post-Training Analysis Module
Generates all result visualizations and reports after training completes.

Functions:
    - plot_training_curves: Loss and Recall curves from CSV
    - plot_tsne_3d: 3D t-SNE embedding visualization
    - plot_confusion_matrix: kNN-based confusion matrix
    - plot_distance_distribution: Intra vs inter-class distances
    - generate_summary_report: Markdown summary of training results
    - run_post_training_analysis: Orchestrator that calls all above

Author: MetricLearning Project
Date: 2026-02-06
"""

import shutil
from typing import Optional

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from datetime import datetime
import logging
import json
from PIL import Image

# Análisis espectral (SOTA 2026)
from src.visualization.embedding_analysis import EmbeddingSpaceAnalyzer
from src.utils.unicode_utils import sanitize_for_ascii
from src.visualization.save_utils import save_plot_with_data
from src.utils.metrics_utils import compute_all_metrics
from src.utils.mlrc_protocol import (
    resolve_map_at_r_config,
    resolve_nmi_config,
    resolve_nn_metric,
    resolve_n_runs,
)
from src.inference.slice_classifier import (
    SliceAwareClassifier,
    SliceRepresentatives,
    compute_slice_centroids_from_embeddings,
    export_proxies_tensor_from_loss_state,
    load_slice_classifier_from_run,
    load_slice_representatives,
    save_slice_representatives,
)

logger = logging.getLogger(__name__)

# Paper-ready defaults for generated figures.
# Journal requirement: minimum 600 dpi; preferred 1200 dpi.
DEFAULT_FIG_DPI = 1200


def _free_step_memory(stage: str) -> None:
    """Release GPU/RAM between post-training steps to avoid OOM cascades."""
    from src.data.dataloader_utils import free_training_memory
    free_training_memory(stage)


def _resolve_run_config_path(run_dir: Path) -> Path:
    """Return the run-specific config path (config/config.yaml preferred)."""
    nested = run_dir / "config" / "config.yaml"
    if nested.exists():
        return nested
    flat = run_dir / "config.yaml"
    if flat.exists():
        return flat
    return nested


def publish_final_evaluation_summary(run_dir: Path, prefix: str = "test") -> Optional[Path]:
    """Copy hold-out evaluation JSON to evaluation_metrics_summary_final.json (MLRC)."""
    run_dir = Path(run_dir)
    src = run_dir / f"evaluation_metrics_summary_{prefix}.json"
    dst = run_dir / "evaluation_metrics_summary_final.json"
    if not src.exists():
        logger.warning("No se publicó resumen final: %s no existe", src)
        return None
    shutil.copy2(src, dst)
    logger.info("Evaluación final (hold-out) publicada: %s", dst)
    return dst


def _clear_prefix_artifacts(run_dir: Path, prefix: str) -> Path:
    """Remove all post-training artifacts for a prefix before regenerating."""
    run_dir = Path(run_dir)
    post_dir = run_dir / "images" / "post_training" / prefix

    if post_dir.exists():
        shutil.rmtree(post_dir)
        logger.info(f"Cleared post-training directory: {post_dir}")

    post_dir.mkdir(parents=True, exist_ok=True)

    for artifact in (
        run_dir / f"evaluation_metrics_summary_{prefix}.json",
        run_dir / f"training_report_{prefix}.md",
    ):
        if artifact.exists():
            artifact.unlink()
            logger.info(f"Removed: {artifact}")

    if prefix == "val":
        protocol = run_dir / "training_protocol.md"
        if protocol.exists():
            protocol.unlink()
            logger.info(f"Removed: {protocol}")

    return post_dir


def create_eval_dataloaders(config: dict):
    """Create inference loaders in the main process (no worker pools)."""
    import copy
    from src.data.data_engine import GrainDataEngine

    eval_config = copy.deepcopy(config)
    data = eval_config.setdefault("data", {})
    data["pin_memory"] = False
    data["num_workers"] = 0
    data["val_num_workers"] = 0
    policy = data.setdefault("loader_policy", {})
    policy["persistent_workers"] = False
    engine = GrainDataEngine.from_config(eval_config)
    return (
        engine.train_loader,
        engine.val_loader,
        getattr(engine, "test_loader", None),
        engine,
    )


def finalize_self_contained_checkpoint(run_dir: Path) -> None:
    """Re-save best_model.pth with embedded reference embeddings and centroids."""
    run_dir = Path(run_dir)
    best_ckpt = run_dir / "checkpoints" / "best_model.pth"
    ref_path = run_dir / "reference_embeddings.pt"
    cent_path = run_dir / "class_centroids.pt"
    config_path = _resolve_run_config_path(run_dir)

    if not (best_ckpt.exists() and ref_path.exists() and cent_path.exists()):
        logger.warning("Cannot finalize self-contained checkpoint — missing artifacts")
        return

    import yaml

    best_data = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
    ref_data = torch.load(str(ref_path), map_location="cpu", weights_only=False)
    cent_data = torch.load(str(cent_path), map_location="cpu", weights_only=False)

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    best_data["reference_embeddings"] = (
        ref_data["embeddings"].numpy()
        if torch.is_tensor(ref_data["embeddings"])
        else ref_data["embeddings"]
    )
    best_data["reference_labels"] = (
        ref_data["labels"].numpy()
        if torch.is_tensor(ref_data["labels"])
        else ref_data["labels"]
    )
    best_data["class_names"] = cent_data["class_names"]
    best_data["centroids"] = cent_data["centroids"]
    best_data["config"] = config

    slice_path = run_dir / "slice_representatives.pt"
    if slice_path.exists():
        slice_data = torch.load(str(slice_path), map_location="cpu", weights_only=False)
        best_data["slice_representatives"] = slice_data["representatives"]
        best_data["slice_metadata"] = {
            "num_slices": slice_data["num_slices"],
            "slice_dim": slice_data["slice_dim"],
            "mode": slice_data.get("mode", "slice_centroids"),
            "loss_type": slice_data.get("loss_type", "sliced_ms"),
        }
    proxy_path = run_dir / "class_proxies.pt"
    if proxy_path.exists():
        proxy_data = torch.load(str(proxy_path), map_location="cpu", weights_only=False)
        best_data["class_proxies"] = proxy_data["proxies"]

    torch.save(best_data, str(best_ckpt))
    logger.info(
        f"Self-contained checkpoint saved: {best_ckpt} "
        f"({len(best_data['class_names'])} classes, "
        f"{len(best_data['reference_embeddings'])} ref embeddings)"
    )


def generate_retrieval_epoch_curves(csv_path: str, output_dir: str):
    """
    Plot retrieval and classification metrics by epoch from training_metrics.csv.

    Does NOT mix Precision_macro (1-NN classifier) with Recall@1 (retrieval) on a
    fake PR plane — that was scientifically invalid (Musgrave et al., ECCV 2020).
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    if not csv_path.exists():
        logger.warning("Retrieval epoch curves skipped: metrics CSV not found")
        return artifacts

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.warning("Retrieval epoch curves skipped: training_metrics.csv is empty")
        return artifacts

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"])
    if df.empty:
        return artifacts

    retrieval_cols = [
        ("recall_at_1", "Recall@1", "#2ecc71"),
        ("recall_at_5", "Recall@5", "#9b59b6"),
        ("mAP@R", "mAP@R", "#3498db"),
    ]
    present_ret = [(c, l, col) for c, l, col in retrieval_cols if c in df.columns]
    for col, _, _ in present_ret:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if present_ret:
        fig, ax = plt.subplots(figsize=(12, 6))
        for col, label, color in present_ret:
            ax.plot(df["epoch"], df[col], "o-", color=color, linewidth=2, markersize=4, label=label)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Retrieval score", fontsize=12)
        ax.set_title("Retrieval Metrics by Epoch (validation)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        ret_path = output_dir / "retrieval_metrics_by_epoch.png"
        save_cols = ["epoch"] + [c for c, _, _ in present_ret]
        save_plot_with_data(ret_path, df[save_cols], dpi=DEFAULT_FIG_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Retrieval epoch curves saved: {ret_path}")
        artifacts["retrieval_metrics_by_epoch"] = str(ret_path)

    cls_cols = [
        ("F1_macro", "F1-macro", "#e67e22"),
        ("precision_macro", "Precision-macro", "#1abc9c"),
        ("balanced_accuracy", "Balanced Acc", "#34495e"),
    ]
    present_cls = [(c, l, col) for c, l, col in cls_cols if c in df.columns]
    for col, _, _ in present_cls:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if present_cls:
        fig, ax = plt.subplots(figsize=(12, 6))
        for col, label, color in present_cls:
            ax.plot(df["epoch"], df[col], "s-", color=color, linewidth=2, markersize=4, label=label)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Classification score", fontsize=12)
        ax.set_title("Classification Metrics by Epoch (validation)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        cls_path = output_dir / "classification_metrics_by_epoch.png"
        save_cols = ["epoch"] + [c for c, _, _ in present_cls]
        save_plot_with_data(cls_path, df[save_cols], dpi=DEFAULT_FIG_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Classification epoch curves saved: {cls_path}")
        artifacts["classification_metrics_by_epoch"] = str(cls_path)

    return artifacts


def generate_pr_f1_epoch_curves(csv_path: str, output_dir: str):
    """
    Deprecated alias — use generate_retrieval_epoch_curves (Phase A integrity).
    """
    logger.info(
        "generate_pr_f1_epoch_curves redirigido a generate_retrieval_epoch_curves "
        "(curva PR precision_macro vs R@1 eliminada por invalidez científica)"
    )
    return generate_retrieval_epoch_curves(csv_path, output_dir)


def extract_embeddings(model, dataloader, device, grain_config=None):
    """Extract all embeddings and labels from a dataloader (canonical forward + masks)."""
    from src.models.embedding_extractor import extract_embeddings as _extract

    embeddings, labels = _extract(
        model,
        dataloader,
        device,
        grain_config=grain_config,
        show_progress=True,
    )

    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(
        f"[extract_embeddings] {embeddings.shape[0]} samples, dim={embeddings.shape[1]}, "
        f"norm: mean={norms.mean():.4f} std={norms.std():.4f}, "
        f"classes={len(np.unique(labels))}"
    )
    return embeddings, labels


def _best_recall_running_max(df: pd.DataFrame) -> pd.Series:
    """Cumulative best R@1 — legacy CSV had best_recall_at_1; new CSV uses best_selection_value."""
    if "best_recall_at_1" in df.columns:
        return pd.to_numeric(df["best_recall_at_1"], errors="coerce")
    if "recall_at_1" in df.columns:
        return pd.to_numeric(df["recall_at_1"], errors="coerce").cummax()
    return pd.Series(dtype=float)


def plot_training_curves(csv_path: str, output_dir: str):
    """
    Generate training curves from the metrics CSV.
    Produces: loss_curves.png, recall_curves.png, lr_curve.png
    """
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine whether training stopped before configured max epoch.
    final_epoch = int(df['epoch'].iloc[-1]) if not df.empty else 0
    max_epoch_seen = int(df['epoch'].max()) if not df.empty else final_epoch

    # --- Loss Curves ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['epoch'], df['train_loss'], 'b-o', markersize=4, label='Train Loss', linewidth=2)
    ax.plot(df['epoch'], df['val_loss'], 'r-o', markersize=4, label='Val Loss', linewidth=2)

    # Mark best epoch
    best_rows = df[df['is_best'] == True]
    if not best_rows.empty:
        last_best = best_rows.iloc[-1]
        ax.axvline(x=last_best['epoch'], color='green', linestyle='--', alpha=0.5, label=f"Best (epoch {int(last_best['epoch'])})")
        if int(last_best['epoch']) < max_epoch_seen:
            ax.axvline(
                x=final_epoch,
                color='orange',
                linestyle=':',
                alpha=0.7,
                label=f"Stop (epoch {final_epoch})"
            )

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_with_data(output_dir / 'loss_curves.png', df[['epoch', 'train_loss', 'val_loss']], dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curves saved: {output_dir / 'loss_curves.png'}")

    # --- Recall Curves ---
    if "recall_at_1" in df.columns:
        best_r1 = _best_recall_running_max(df)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df["epoch"], df["recall_at_1"], "g-o", markersize=4, label="Recall@1", linewidth=2)
        if "recall_at_5" in df.columns:
            ax.plot(df["epoch"], df["recall_at_5"], "m-o", markersize=4, label="Recall@5", linewidth=2)
        if not best_r1.empty:
            ax.plot(df["epoch"], best_r1, "k--", alpha=0.5, label="Best R@1", linewidth=1)
        if "best_selection_value" in df.columns and "selection_metric" in df.columns:
            sel_metric = str(df["selection_metric"].iloc[0])
            if sel_metric != "recall_at_1":
                ax.plot(
                    df["epoch"],
                    pd.to_numeric(df["best_selection_value"], errors="coerce"),
                    "b:",
                    alpha=0.6,
                    label=f"Best {sel_metric}",
                    linewidth=1,
                )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Recall@K over Training", fontsize=16, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        recall_cols = ["epoch", "recall_at_1"]
        if "recall_at_5" in df.columns:
            recall_cols.append("recall_at_5")
        recall_export = df[recall_cols].copy()
        recall_export["best_recall_at_1"] = best_r1
        if "best_selection_value" in df.columns:
            recall_export["best_selection_value"] = df["best_selection_value"]
        save_plot_with_data(
            output_dir / "recall_curves.png",
            recall_export,
            dpi=DEFAULT_FIG_DPI,
            bbox_inches="tight",
        )
        plt.close()
        logger.info(f"Recall curves saved: {output_dir / 'recall_curves.png'}")

    # --- SOTA validation metrics (retrieval + clustering) ---
    sota_cols = [
        ('recall_at_1', 'Recall@1', 'g'),
        ('recall_at_5', 'Recall@5', 'm'),
        ('mAP@R', 'mAP@R', 'b'),
        ('NMI', 'NMI', 'c'),
        ('F1_macro', 'F1-macro', 'r'),
    ]
    present = [(col, label, color) for col, label, color in sota_cols if col in df.columns]
    if present:
        fig, ax = plt.subplots(figsize=(12, 6))
        for col, label, color in present:
            ax.plot(df['epoch'], df[col], f'{color}-o', markersize=4, label=label, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Validation Metrics (retrieval + clustering)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        metrics_cols = [col for col, _, _ in present]
        save_plot_with_data(output_dir / 'validation_metrics.png', df[['epoch'] + metrics_cols], dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
        plt.close()
        logger.info(f"Validation metrics curve saved: {output_dir / 'validation_metrics.png'}")

    # --- Learning Rate Curve ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['epoch'], df['learning_rate'], 'c-o', markersize=3, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot_with_data(output_dir / 'lr_curve.png', df[['epoch', 'learning_rate']], dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"LR curve saved: {output_dir / 'lr_curve.png'}")

    # --- Epoch Time ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(df['epoch'], df['epoch_time_sec'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Epoch Duration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_plot_with_data(output_dir / 'epoch_times.png', df[['epoch', 'epoch_time_sec']], dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Epoch times saved: {output_dir / 'epoch_times.png'}")

    plt.close("all")
    return df


def plot_tsne_3d(embeddings, labels, output_path, class_names, max_samples=3000):
    """Generate and save a 3D t-SNE plot and a 2D t-SNE plot of the embeddings."""
    logger.info("Generating 3D t-SNE visualization...")
    
    # --- Subsampling for TSNE performance (O(N^2) complexity) ---
    max_tsne_samples = 3000
    if len(embeddings) > max_tsne_samples:
        logger.info(f"Subsampling TSNE from {len(embeddings)} to {max_tsne_samples} points to speed up calculations...")
        np.random.seed(42)
        indices = np.random.choice(len(embeddings), max_tsne_samples, replace=False)
        tsne_embeddings = embeddings[indices]
        tsne_labels = labels[indices]
    else:
        tsne_embeddings = embeddings
        tsne_labels = labels

    embed_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1)
    logger.info(f"Embeddings shape: {embeddings.shape}, norms: mean={norms.mean():.4f} std={norms.std():.4f}")

    # Determine safe perplexity (must be < n_samples)
    perplexity = min(30, max(5, len(tsne_embeddings) // 3))

    # Run t-SNE with 3 components
    tsne = TSNE(n_components=3, perplexity=perplexity, init='pca', max_iter=1000, random_state=42, n_jobs=-1)
    emb_3d = tsne.fit_transform(tsne_embeddings)

    # Color palette
    n_classes = len(class_names)
    cmap = cm.get_cmap('tab10' if n_classes <= 10 else 'tab20')
    colors = [cmap(i / n_classes) for i in range(n_classes)]

    # --- 3D Interactive-style plot (multiple angles) ---
    fig = plt.figure(figsize=(20, 16))

    angles = [(30, 45), (30, 135), (60, 0), (10, 90)]
    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        for class_id, class_name in enumerate(class_names):
            mask = tsne_labels == class_id
            ax.scatter(
                emb_3d[mask, 0], emb_3d[mask, 1], emb_3d[mask, 2],
                c=[colors[class_id]], label=class_name, alpha=0.6, s=15
            )
        ax.set_title(f'View: elev={elev}, azim={azim}', fontsize=11)
        ax.view_init(elev=elev, azim=azim)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8, markerscale=2)

    fig.suptitle(
        f'Espacio de Embeddings $\mathbb{{S}}^{{{embed_dim}}}$ — t-SNE 3D\n'
        f'{n_samples} muestras, {len(class_names)} clases, '
        f'‖emb‖ = {norms.mean():.3f} ± {norms.std():.4f}',
        fontsize=16, fontweight='bold', y=1.0
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_3d = pd.DataFrame({
        'tsne_3d_dim1': emb_3d[:, 0],
        'tsne_3d_dim2': emb_3d[:, 1],
        'tsne_3d_dim3': emb_3d[:, 2],
        'label_id': tsne_labels,
        'label_name': [class_names[l] for l in tsne_labels]
    })
    save_plot_with_data(output_path, df_3d, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"3D t-SNE saved: {output_path}")

    # Also save 2D t-SNE
    tsne_2d = TSNE(n_components=2, perplexity=perplexity, init='pca', max_iter=1000, random_state=42, n_jobs=-1)
    emb_2d = tsne_2d.fit_transform(tsne_embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for class_id, class_name in enumerate(class_names):
        mask = tsne_labels == class_id
        axes[0].scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[colors[class_id]],
                        label=class_name, alpha=0.6, s=20)
    axes[0].set_title(
        f'S^{embed_dim} -> t-SNE 2D (por Clase)',
        fontsize=14, fontweight='bold'
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Density plot
    from scipy.stats import gaussian_kde
    # Cap KDE samples at 3000 to prevent infinite hanging
    kde_emb = emb_2d[:3000]
    xy = np.vstack([kde_emb[:, 0], kde_emb[:, 1]])
    z = gaussian_kde(xy)(xy)
    
    # Plot KDE points (only the sampled ones to match z)
    scatter = axes[1].scatter(kde_emb[:, 0], kde_emb[:, 1], c=z, s=20, cmap='viridis', alpha=0.6)
    
    # Plot remaining points in light gray if any
    if len(emb_2d) > 3000:
        axes[1].scatter(emb_2d[3000:, 0], emb_2d[3000:, 1], c='gray', s=10, alpha=0.1)
        
    axes[1].set_title(
        f'S^{embed_dim} -> t-SNE 2D Densidad (KDE max 3000 pts)',
        fontsize=14, fontweight='bold'
    )
    plt.colorbar(scatter, ax=axes[1], label='Density')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    tsne_2d_path = output_path.parent / 'tsne_2d.png'
    df_2d = pd.DataFrame({
        'tsne_2d_dim1': emb_2d[:, 0],
        'tsne_2d_dim2': emb_2d[:, 1],
        'label_id': tsne_labels,
        'label_name': [class_names[l] for l in tsne_labels],
        'density': np.pad(z, (0, len(emb_2d) - len(z)), constant_values=0)
    })
    save_plot_with_data(tsne_2d_path, df_2d, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"2D t-SNE saved: {tsne_2d_path}")


def plot_embedding_space_analysis(embeddings, labels, class_names, output_path):
    """
    Generate a comprehensive embedding space analysis visualization.
    Shows: norm distribution, angular separation, centroid structure, cosine similarity.
    This plot demonstrates the properties of the learned S^D hypersphere.
    """
    embed_dim = embeddings.shape[1]
    n_samples = embeddings.shape[0]
    n_classes = len(class_names)
    norms = np.linalg.norm(embeddings, axis=1)

    # Normalize for angular/cosine analysis
    emb_normed = embeddings / (norms[:, None] + 1e-8)

    # Per-class centroids (on normalized embeddings)
    centroids = np.zeros((n_classes, embed_dim))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            centroids[c] = emb_normed[mask].mean(axis=0)
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normed = centroids / (centroid_norms + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f'Análisis del Espacio de Embeddings $\\mathbb{{S}}^{{{embed_dim}}}$\n'
        f'{n_samples} muestras, {n_classes} clases',
        fontsize=18, fontweight='bold', y=0.98
    )

    # --- Panel 1: Norm distribution (or cosine distance distribution if L2-normalized) ---
    ax = axes[0, 0]
    norm_range = norms.max() - norms.min()
    if norm_range < 1e-6:
        # All embeddings are L2-normalized (norm ≈ 1.0) — show pairwise cosine
        # distance distribution instead (more informative)
        rng = np.random.RandomState(42)
        idx = rng.choice(n_samples, size=min(n_samples, 500), replace=False)
        sample_embs = emb_normed[idx]
        cos_dists = 1.0 - (sample_embs @ sample_embs.T)
        triu_idx = np.triu_indices(len(idx), k=1)
        cos_dist_vals = cos_dists[triu_idx]
        n_bins = max(5, min(50, len(cos_dist_vals) // 20))
        ax.hist(cos_dist_vals, bins=n_bins, color='steelblue', edgecolor='black', alpha=0.8, density=True)
        ax.axvline(cos_dist_vals.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Media = {cos_dist_vals.mean():.4f}')
        ax.set_xlabel('Distancia Coseno (1 - cos θ)', fontsize=12)
        ax.set_ylabel('Densidad', fontsize=12)
        ax.set_title(
            f'Distribución de Distancias Coseno\n'
            f'‖emb‖ = {norms.mean():.4f} (L2-normalizado, σ ≈ 0)',
            fontsize=13, fontweight='bold'
        )
    else:
        n_bins = max(5, min(50, int(n_samples / 10)))
        ax.hist(norms, bins=n_bins, color='steelblue', edgecolor='black', alpha=0.8, density=True)
        ax.axvline(norms.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Media = {norms.mean():.4f}')
        ax.axvline(1.0, color='green', linestyle=':', linewidth=2, label='Norma ideal = 1.0')
        ax.set_xlabel('‖embedding‖₂', fontsize=12)
        ax.set_ylabel('Densidad', fontsize=12)
        ax.set_title(
            f'Distribución de Normas (σ = {norms.std():.4f})',
            fontsize=13, fontweight='bold'
        )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Cosine similarity heatmap (centroids) ---
    ax = axes[0, 1]
    cos_sim = centroids_normed @ centroids_normed.T
    im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=-0.3, vmax=1.0, aspect='auto')
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if abs(cos_sim[i, j]) > 0.6 else 'black'
            ax.text(j, i, f'{cos_sim[i, j]:.2f}', ha='center', va='center',
                    fontsize=8, color=color)
    ax.set_title('Similitud Coseno entre Centroides', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine Similarity', shrink=0.8)

    # --- Panel 3: Per-class compactness (intra-class cosine sim) ---
    ax = axes[1, 0]
    compactness = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 1:
            cls_embs = emb_normed[mask]
            cls_cos = cls_embs @ centroids_normed[c]
            compactness.append(cls_cos.mean())
        else:
            compactness.append(0)

    colors_bar = ['#2ecc71' if v >= 0.8 else ('#f39c12' if v >= 0.6 else '#e74c3c')
                  for v in compactness]
    bars = ax.barh(class_names, compactness, color=colors_bar, edgecolor='#333', linewidth=0.5)
    for bar, v in zip(bars, compactness):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Similitud Coseno media al Centroide', fontsize=11)
    ax.set_title('Compacidad Intra-clase', fontsize=13, fontweight='bold')
    ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    # --- Panel 4: Angular separation between class centroids ---
    ax = axes[1, 1]
    # Compute angular distances (arccos of cosine similarity)
    cos_sim_clipped = np.clip(cos_sim, -1.0, 1.0)
    angular_dist = np.degrees(np.arccos(cos_sim_clipped))
    np.fill_diagonal(angular_dist, 0)

    # Off-diagonal angular distances
    mask_off = ~np.eye(n_classes, dtype=bool)
    off_diag = angular_dist[mask_off]

    n_bins_ang = max(5, min(30, len(off_diag) // 3))
    ax.hist(off_diag, bins=n_bins_ang, color='darkorange', edgecolor='black', alpha=0.8)
    ax.axvline(off_diag.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Media = {off_diag.mean():.1f}deg')
    ax.axvline(90, color='green', linestyle=':', linewidth=2, label='Separacion ideal = 90deg')
    ax.set_xlabel('Separación Angular (grados)', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title(
        f'Separacion Angular entre Centroides\n'
        f'Rango: [{off_diag.min():.1f}deg, {off_diag.max():.1f}deg]',
        fontsize=13, fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -- NEW CODE TO EXPORT ALL 4 GRAPHS TO SEPARATE CSVs --
    base_dir = output_path.parent
    base_name = output_path.stem

    # Panel 1: Norms or Cosine Distances
    if norm_range < 1e-6:
        pd.DataFrame({'cosine_distance': cos_dist_vals}).to_csv(base_dir / f"{base_name}_panel1_cos_dist.csv", index=False)
    else:
        pd.DataFrame({'norm': norms}).to_csv(base_dir / f"{base_name}_panel1_norms.csv", index=False)

    # Panel 2: Cosine Similarity Matrix
    pd.DataFrame(cos_sim, index=class_names, columns=class_names).to_csv(base_dir / f"{base_name}_panel2_cossim.csv")

    # Panel 3: Compactness (already present in analysis_data, but good to have explicit)
    pd.DataFrame({
        'class_name': class_names,
        'compactness': compactness
    }).to_csv(base_dir / f"{base_name}_panel3_compactness.csv", index=False)

    # Panel 4: Angular Separation Matrix
    pd.DataFrame(angular_dist, index=class_names, columns=class_names).to_csv(base_dir / f"{base_name}_panel4_angular.csv")
    # -----------------------------------------------------

    analysis_data = pd.DataFrame({
        'class_name': class_names,
        'compactness': compactness,
    })
    save_plot_with_data(output_path, analysis_data, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Embedding space analysis saved: {output_path}")

    return {
        'embed_dim': embed_dim,
        'n_samples': n_samples,
        'norm_mean': float(norms.mean()),
        'norm_std': float(norms.std()),
        'mean_angular_sep': float(off_diag.mean()),
        'min_angular_sep': float(off_diag.min()),
        'max_angular_sep': float(off_diag.max()),
        'compactness': {class_names[i]: float(compactness[i]) for i in range(n_classes)},
    }


def compute_slice_predictions(embeddings, slice_classifier: SliceAwareClassifier):
    """Classify numpy embeddings with slice-aware classifier."""
    device = slice_classifier.reps.representatives.device
    emb_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
    preds, _, _ = slice_classifier.predict(emb_t)
    return preds.cpu().numpy()


def compute_knn_predictions(embeddings, labels, n_classes, k=1, nn_metric='euclidean'):
    """Compute leave-one-out k-NN predictions in embedding space (MLRC protocol)."""
    from src.utils.metrics_utils import _compute_1nn_predictions

    if k != 1:
        logger.warning("compute_knn_predictions: only k=1 LOO supported; using k=1")
    return _compute_1nn_predictions(embeddings, labels, metric=nn_metric)


def _unpack_loader_batch(batch_data):
    if len(batch_data) == 3:
        return batch_data[0], batch_data[1], batch_data[2]
    return batch_data[0], batch_data[1], None


def _forward_embeddings(model, images, masks, grain_config, device):
    images = images.to(device)
    masked_pooling = bool((grain_config or {}).get("masked_pooling", False))
    if masked_pooling and masks is not None:
        return model(images, mask=masks.to(device))
    return model(images)


def _resolve_denorm_tensors(config):
    from src.utils.dataset_norm import resolve_normalization_config

    resolved = resolve_normalization_config(config)
    mean = torch.tensor(resolved["mean"], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(resolved["std"], dtype=torch.float32).view(1, 3, 1, 1)
    return mean, std


def _resolve_best_training_row(df, config):
    """Select best epoch row using training.validation_metric (B-01)."""
    validation_metric = (
        config.get("training", {}).get("validation_metric", "mAP@R") or "mAP@R"
    )
    metric_aliases = {
        "recall_at_1": "recall_at_1",
        "mAP@R": "mAP@R",
        "F1_macro": "F1_macro",
        "val_loss": "val_loss",
    }
    col = metric_aliases.get(validation_metric, "selection_value")
    if col in df.columns:
        return df.loc[df[col].astype(float).idxmax()], validation_metric
    if "is_best" in df.columns:
        flagged = df[df["is_best"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if len(flagged):
            return flagged.iloc[-1], validation_metric
    if "selection_value" in df.columns:
        return df.loc[df["selection_value"].astype(float).idxmax()], "selection_value"
    return df.loc[df["recall_at_1"].astype(float).idxmax()], "recall_at_1"


def compute_standard_metrics_from_predictions(labels, predictions, class_names):
    """Compute standard classification metrics from y_true/y_pred."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(len(class_names))), zero_division=0
    )
    per_class_df = pd.DataFrame({
        'class': class_names,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support
    })
    summary = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'precision_macro': float(precision_score(labels, predictions, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(labels, predictions, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(labels, predictions, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(labels, predictions, average='weighted', zero_division=0)),
    }
    return summary, per_class_df


def plot_per_class_f1(per_class_df, output_path, metric_label="Slice-aware LOO"):
    """Generate F1-by-class horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_df = per_class_df.sort_values('f1', ascending=True)
    bars = ax.barh(sorted_df['class'], sorted_df['f1'], color='#4ec9b0', edgecolor='#333', linewidth=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1-score', fontsize=12)
    ax.set_title(f'F1-score por Clase ({metric_label})', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for bar, value in zip(bars, sorted_df['f1']):
        ax.text(min(value + 0.01, 1.02), bar.get_y() + bar.get_height() / 2,
                f'{value:.3f}', va='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_plot_with_data(output_path, sorted_df[['class', 'f1']], dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Per-class F1 chart saved: {output_path}")


def generate_training_protocol(config, run_dir):
    """
    Build a detailed training protocol markdown file for traceability.
    Returns: (protocol_path, protocol_markdown)
    """
    run_dir = Path(run_dir)
    protocol_path = run_dir / 'training_protocol.md'
    dataset_stats_path = run_dir / 'images' / 'pre_training' / 'dataset_stats.json'

    stats = None
    if dataset_stats_path.exists():
        try:
            with open(dataset_stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
        except Exception as exc:
            logger.warning(f"Could not load dataset stats for protocol: {exc}")

    # Best-effort split metadata
    split_note = "No explicit split metadata found in run artifacts."
    if stats is not None:
        split_note = (
            f"Train={stats.get('train', {}).get('total_samples', 'N/A')}, "
            f"Val={stats.get('val', {}).get('total_samples', 'N/A')}."
        )

    lines = [
        "# Training Protocol",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        "",
        "## Dataset Split Strategy",
        "",
        f"- Source paths: train=`{config.get('data', {}).get('train_dir', 'N/A')}`, "
        f"val=`{config.get('data', {}).get('val_dir', 'N/A')}`, "
        f"test=`{config.get('data', {}).get('test_dir', 'N/A')}`",
        f"- Split summary: {split_note}",
        "- Stratification by class: expected from Dataset Manager workflow (to be confirmed by metadata/maps).",
        "- Cross-validation: not applied in current pipeline (single train/val/test split).",
        "",
        "## Training Hyperparameters",
        "",
        f"- Epochs: {config.get('training', {}).get('epochs', 'N/A')}",
        f"- Batch size: {config.get('data', {}).get('batch_size', 'N/A')}",
        f"- Optimizer: {config.get('training', {}).get('optimizer', 'N/A')}",
        f"- Learning rate: {config.get('training', {}).get('learning_rate', 'N/A')}",
        f"- Weight decay: {config.get('training', {}).get('weight_decay', 'N/A')}",
        f"- Scheduler: {config.get('training', {}).get('scheduler', 'N/A')}",
        f"- Warmup epochs: {config.get('training', {}).get('warmup_epochs', 'N/A')}",
        f"- Loss: {config.get('training', {}).get('loss', {}).get('type', 'N/A')}",
        f"- Validation metric for best model: {config.get('training', {}).get('validation_metric', 'recall_at_1')}",
        "",
        "## Hyperparameter Optimization Notes",
        "",
        "- Current run stores explicit final hyperparameters in `config/config.yaml`.",
        "- Automated cross-validated HPO is not part of this pipeline yet.",
        "- Comparative optimization is performed through multiple named runs.",
        "",
        "## Reproducibility",
        "",
        f"- Seed: {config.get('reproducibility', {}).get('seed', 'N/A')}",
        f"- Deterministic mode: {config.get('reproducibility', {}).get('deterministic', 'N/A')}",
        f"- cuDNN benchmark: {config.get('reproducibility', {}).get('benchmark', 'N/A')}",
    ]

    protocol_text = "\n".join(lines)
    with open(protocol_path, 'w', encoding='utf-8') as f:
        f.write(protocol_text)
    logger.info(f"Training protocol saved: {protocol_path}")
    return protocol_path, protocol_text


def plot_confusion_matrix_knn(embeddings, labels, class_names, output_path, k=1, nn_metric='euclidean'):
    """Generate confusion matrix using leave-one-out 1-NN (MLRC nn_metric)."""
    logger.info(f"Generating confusion matrix (1-NN LOO, metric={nn_metric})...")

    predictions = compute_knn_predictions(
        embeddings, labels, len(class_names), k=k, nn_metric=nn_metric
    )

    # Compute confusion matrix
    cm = sk_confusion_matrix(labels, predictions, labels=list(range(len(class_names))))

    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    # Accuracy
    accuracy = np.trace(cm) / cm.sum()
    std_metrics, per_class_df = compute_standard_metrics_from_predictions(labels, predictions, class_names)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'Confusion Matrix (Counts) — 1-NN LOO ({nn_metric}) Acc: {accuracy:.1%}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)

    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd', xticklabels=class_names,
                yticklabels=class_names, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved: {output_path}")

    # Save normalized confusion matrix as standalone artifact
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    norm_path = output_path.parent / 'confusion_matrix_normalized.png'
    plt.savefig(norm_path, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Normalized confusion matrix saved: {norm_path}")

    # Per-class accuracy
    per_class_acc = cm_norm.diagonal()
    for i, name in enumerate(class_names):
        logger.info(f"  {name}: {per_class_acc[i]:.1%}")

    return accuracy, per_class_acc, std_metrics, per_class_df, predictions


def plot_distance_distribution(embeddings, labels, class_names, output_path, max_pairs=50000):
    """Plot intra-class vs inter-class distance distributions."""
    logger.info("Generating distance distribution...")

    n = len(labels)
    intra_dists = []
    inter_dists = []

    # Sample pairs to avoid O(n^2) explosion
    rng = np.random.RandomState(42)
    indices = rng.choice(n, size=min(n, 2000), replace=False)

    for i in indices:
        for j in indices:
            if i >= j:
                continue
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if labels[i] == labels[j]:
                intra_dists.append(dist)
            else:
                inter_dists.append(dist)
            if len(intra_dists) + len(inter_dists) >= max_pairs:
                break
        if len(intra_dists) + len(inter_dists) >= max_pairs:
            break

    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)

    mean_intra = np.mean(intra_dists) if len(intra_dists) > 0 else 0
    mean_inter = np.mean(inter_dists) if len(inter_dists) > 0 else 0
    ratio = mean_inter / mean_intra if mean_intra > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Histogram
    axes[0].hist(intra_dists, bins=60, alpha=0.6, label='Intra-class', color='green', density=True)
    axes[0].hist(inter_dists, bins=60, alpha=0.6, label='Inter-class', color='red', density=True)
    axes[0].set_xlabel('Euclidean Distance', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Distance Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Box plot per class
    class_intra_dists = {name: [] for name in class_names}
    for i in indices:
        for j in indices:
            if i >= j or labels[i] != labels[j]:
                continue
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            class_intra_dists[class_names[labels[i]]].append(dist)

    box_data = [class_intra_dists[name] for name in class_names if class_intra_dists[name]]
    box_labels = [name for name in class_names if class_intra_dists[name]]
    axes[1].boxplot(box_data, labels=box_labels)
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Intra-class Distance', fontsize=12)
    axes[1].set_title('Intra-class Distance by Class', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()

    mean_intra = np.mean(intra_dists) if len(intra_dists) > 0 else 0
    mean_inter = np.mean(inter_dists) if len(inter_dists) > 0 else 0
    ratio = mean_inter / mean_intra if mean_intra > 0 else 0

    logger.info(f"Distance stats — Intra: {mean_intra:.4f}, Inter: {mean_inter:.4f}, Ratio: {ratio:.2f}")
    logger.info(f"Distance distribution saved: {output_path}")

    return mean_intra, mean_inter, ratio


def _classify_embeddings(embeddings, slice_classifier: SliceAwareClassifier):
    """Slice-aware classification; returns preds and pseudo-distances (1 - cosine)."""
    preds, scores, _ = slice_classifier.predict(embeddings)
    return preds, 1.0 - scores


def plot_inference_preview(model, slice_classifier, val_loader, device, output_path, class_names,
                          n_samples=10, grain_config=None, config=None):
    """
    Generate an inference preview grid: random val images with predictions,
    true labels, distances, and confidence color coding.

    Args:
        model: Trained model on device
        slice_classifier: SliceAwareClassifier with per-slice representatives
        val_loader: Validation DataLoader
        device: torch.device
        output_path: Path to save the image
        class_names: List of class names
        n_samples: Number of samples to show
    """
    logger.info(f"Generating inference preview ({n_samples} random samples)...")

    model.eval()

    # 1. Collect val images (we need raw tensors for display)
    all_images = []
    all_labels = []
    all_masks = []
    for batch_data in val_loader:
        images, labels, masks = _unpack_loader_batch(batch_data)
        all_images.append(images)
        all_labels.append(labels)
        if masks is not None:
            all_masks.append(masks)
        if sum(len(x) for x in all_images) >= max(n_samples * 5, 200):
            break
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_masks = torch.cat(all_masks, dim=0) if all_masks else None

    # 3. Sample n_samples random indices (balanced across classes)
    rng = np.random.RandomState(42)
    unique_labels = torch.unique(all_labels).numpy()
    per_class = max(1, -(-n_samples // len(unique_labels)))  # ceil division
    selected_idx = []
    for lbl in unique_labels:
        mask = (all_labels.numpy() == lbl).nonzero()[0]
        chosen = rng.choice(mask, size=min(per_class, len(mask)), replace=False)
        selected_idx.extend(chosen.tolist())
    rng.shuffle(selected_idx)
    selected_idx = selected_idx[:n_samples]
    n_samples = len(selected_idx)  # adjust if fewer samples available

    sel_images = all_images[selected_idx]
    sel_labels = all_labels[selected_idx]
    sel_masks = all_masks[selected_idx] if all_masks is not None else None

    with torch.no_grad():
        sel_embs = _forward_embeddings(
            model, sel_images, sel_masks, grain_config, device
        )
        pred_classes, min_dists = _classify_embeddings(sel_embs, slice_classifier)

    if config is not None:
        mean, std = _resolve_denorm_tensors(config)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    sel_denorm = (sel_images * std + mean).clamp(0, 1)

    # 6. Create the visualization grid
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 6 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else [axes]
    if n_rows == 1:
        axes = np.array([axes])

    for idx in range(n_samples):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[0][col]

        img_np = sel_denorm[idx].permute(1, 2, 0).numpy()
        true_lbl = sel_labels[idx].item()
        pred_lbl = pred_classes[idx].item()
        dist_val = min_dists[idx].item()
        correct = true_lbl == pred_lbl

        # Add colored border
        border = 6
        color = [0.2, 0.8, 0.2] if correct else [0.9, 0.2, 0.2]
        bordered = np.ones((img_np.shape[0] + 2 * border, img_np.shape[1] + 2 * border, 3))
        bordered[:, :] = color
        bordered[border:-border, border:-border] = img_np

        ax.imshow(bordered)
        ax.set_xticks([])
        ax.set_yticks([])

        status = "OK" if correct else "WRONG"
        title = (f"Pred: {class_names[pred_lbl]}\n"
                 f"True: {class_names[true_lbl]}\n"
                 f"Dist: {dist_val:.3f} [{status}]")
        title_color = 'green' if correct else 'red'
        ax.set_title(title, fontsize=10, fontweight='bold', color=title_color)

    # Hide unused axes
    for idx in range(n_samples, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col] if n_rows > 1 else axes[0][col]
        ax.axis('off')

    n_correct = sum(1 for i in range(n_samples) if sel_labels[i].item() == pred_classes[i].item())
    fig.suptitle(
        f'Inference Preview — {n_correct}/{n_samples} correct '
        f'(Best Model, Slice-aware Classification)',
        fontsize=16, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()

    logger.info(f"Inference preview saved: {output_path} ({n_correct}/{n_samples} correct)")
    return n_correct, n_samples


def compute_and_save_centroids(
    model,
    train_loader,
    device,
    save_dir,
    class_names,
    num_slices: int = 4,
    loss_state_dict=None,
    loss_type: str = None,
):
    """
    Compute class centroids, slice representatives, and reference embeddings.

    Saves:
      - slice_representatives.pt  (S, C, slice_dim) — primary slice classifier
      - class_centroids.pt          (C, embed_dim) — kNN / legacy compat
      - class_proxies.pt            (optional, sliced_proxy learned proxies)
      - reference_embeddings.pt

    Returns:
        (centroids, slice_classifier)
    """
    logger.info("Computing class centroids from training set (one-time)...")
    model.eval()

    all_embs = []
    all_labels = []

    with torch.no_grad():
        for batch_data in tqdm(train_loader, desc="Extracting train embeddings", leave=False):
            if len(batch_data) == 3:
                images, labels, masks = batch_data
                masks = masks.to(device, non_blocking=True)
            else:
                images, labels = batch_data[0], batch_data[1]
                masks = None
            images = images.to(device, non_blocking=True)
            embs = model(images, mask=masks)
            all_embs.append(embs.cpu())
            all_labels.append(labels)

    all_embs = torch.cat(all_embs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Compute centroids (mean embedding per class)
    n_classes = len(class_names)
    embed_dim = all_embs.shape[1]
    centroids = torch.zeros(n_classes, embed_dim)
    n_samples_per_class = torch.zeros(n_classes, dtype=torch.long)

    for c in range(n_classes):
        mask = (all_labels == c)
        if mask.sum() > 0:
            centroids[c] = all_embs[mask].mean(dim=0)
            n_samples_per_class[c] = mask.sum().item()

    # Save artifacts
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Log per-class stats and inter-centroid distances
    for c in range(n_classes):
        cnorm = torch.norm(centroids[c]).item()
        logger.info(f"  Centroid [{class_names[c]}]: n={n_samples_per_class[c]}, norm={cnorm:.4f}")
    inter_dists = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0)).squeeze(0)
    mask_triu = torch.triu(torch.ones(n_classes, n_classes, dtype=torch.bool), diagonal=1)
    mean_inter = inter_dists[mask_triu].mean().item()
    min_inter = inter_dists[mask_triu].min().item()
    logger.info(f"  Inter-centroid dist: mean={mean_inter:.4f}, min={min_inter:.4f}")

    centroid_data = {
        'centroids': centroids,
        'class_names': list(class_names),
        'n_samples_per_class': n_samples_per_class,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    torch.save(centroid_data, save_dir / 'class_centroids.pt')
    logger.info(f"Class centroids saved: {save_dir / 'class_centroids.pt'} "
                f"({n_classes} classes, {embed_dim}D)")

    ref_data = {
        'embeddings': all_embs,
        'labels': all_labels,
        'class_names': list(class_names),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    torch.save(ref_data, save_dir / 'reference_embeddings.pt')
    logger.info(f"Reference embeddings saved: {save_dir / 'reference_embeddings.pt'} "
                f"({all_embs.shape[0]} samples)")

    slice_reps = compute_slice_centroids_from_embeddings(
        all_embs, all_labels, class_names, num_slices=num_slices
    )
    save_slice_representatives(save_dir / 'slice_representatives.pt', slice_reps)
    slice_classifier = SliceAwareClassifier(slice_reps.to(device))

    if loss_type == 'sliced_proxy' and loss_state_dict:
        try:
            slice_dim = embed_dim // num_slices
            proxies = export_proxies_tensor_from_loss_state(
                loss_state_dict, num_slices, n_classes, slice_dim
            )
            torch.save(
                {
                    'proxies': proxies,
                    'class_names': list(class_names),
                    'num_slices': num_slices,
                    'slice_dim': slice_dim,
                    'embedding_dim': embed_dim,
                    'loss_type': 'sliced_proxy',
                },
                save_dir / 'class_proxies.pt',
            )
            logger.info(
                "Learned slice proxies saved: %s (%d slices x %d classes x %dD)",
                save_dir / 'class_proxies.pt',
                num_slices,
                n_classes,
                slice_dim,
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Could not export learned proxies: %s", exc)

    return centroids.to(device), slice_classifier


def load_centroids(centroids_path, device):
    """
    Load pre-computed centroids from file.

    Returns:
        centroids: tensor (n_classes, embed_dim) on device
        class_names: list of class name strings
    """
    data = torch.load(centroids_path, map_location=device, weights_only=True)
    centroids = data['centroids'].to(device)
    class_names = data['class_names']
    logger.info(f"Centroids loaded: {centroids_path} "
                f"({centroids.shape[0]} classes, {centroids.shape[1]}D)")
    return centroids, class_names


def generate_inference_gallery(model, slice_classifier, val_loader, device, output_dir, class_names,
                               n_per_class=10, grain_config=None, config=None):
    """
    Generate a per-class inference gallery with n_per_class random images per class.
    Each class gets its own image grid saved in output_dir/inference_gallery/.
    Also generates a combined summary page.

    Args:
        model: Trained model on device
        slice_classifier: SliceAwareClassifier
        val_loader: Validation DataLoader
        device: torch.device
        output_dir: Directory to save gallery images
        class_names: List of class names
        n_per_class: Number of samples per class to show
    """
    gallery_dir = Path(output_dir) / 'inference_gallery'
    gallery_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Pass 1: inference metadata only (no images stored)
    logger.info("  Running inference on full validation set...")
    all_labels = []
    all_preds = []
    all_dists = []
    all_all_dists = []

    if config is not None:
        mean, std = _resolve_denorm_tensors(config)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Gallery inference", leave=False):
            images, labels, masks = _unpack_loader_batch(batch_data)
            embs = _forward_embeddings(model, images, masks, grain_config, device)
            preds, min_d = _classify_embeddings(embs, slice_classifier)
            _, _, class_scores = slice_classifier.predict(embs)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_dists.append(min_d.cpu())
            all_all_dists.append(class_scores.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_dists = torch.cat(all_dists)
    all_all_dists = torch.cat(all_all_dists)

    rng = np.random.RandomState(42)
    class_stats = {}
    needed_indices = set()
    class_selected = {}

    # Select samples per class (same RNG sequence as original single-pass code)
    for cls_idx, cls_name in enumerate(class_names):
        mask = (all_labels == cls_idx).numpy()
        cls_indices = np.where(mask)[0]

        if len(cls_indices) == 0:
            continue

        cls_preds = all_preds[cls_indices]
        cls_correct = (cls_preds == cls_idx).numpy()
        n_correct = int(cls_correct.sum())
        n_total = len(cls_indices)
        cls_acc = n_correct / n_total if n_total > 0 else 0
        class_stats[cls_name] = {'correct': n_correct, 'total': n_total, 'accuracy': cls_acc}

        correct_idx = cls_indices[cls_correct]
        wrong_idx = cls_indices[~cls_correct]

        n_wrong_show = min(len(wrong_idx), max(3, n_per_class // 3))
        n_correct_show = min(len(correct_idx), n_per_class - n_wrong_show)
        if n_wrong_show + n_correct_show < n_per_class:
            n_correct_show = min(len(correct_idx), n_per_class - n_wrong_show)

        selected = []
        if len(wrong_idx) > 0:
            chosen_wrong = rng.choice(wrong_idx, size=min(n_wrong_show, len(wrong_idx)), replace=False)
            selected.extend(chosen_wrong.tolist())
        if len(correct_idx) > 0:
            chosen_correct = rng.choice(correct_idx, size=min(n_correct_show, len(correct_idx)), replace=False)
            selected.extend(chosen_correct.tolist())

        rng.shuffle(selected)
        if not selected:
            continue

        class_selected[cls_name] = {
            'indices': selected,
            'cls_idx': cls_idx,
            'n_correct': n_correct,
            'n_total': n_total,
            'cls_acc': cls_acc,
        }
        needed_indices.update(selected)

    logger.info(f"  Fetching {len(needed_indices)} images for inference gallery (2-pass, memory-safe)...")
    image_cache = {}
    global_idx = 0
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Gallery image fetch", leave=False):
            images, _ = batch_data[0], batch_data[1]
            denorm = (images * std + mean).clamp(0, 1)
            batch_size = images.shape[0]
            for i in range(batch_size):
                if global_idx in needed_indices:
                    image_cache[global_idx] = denorm[i].cpu()
                global_idx += 1

    _free_step_memory("inference_gallery_pass1")

    # 3. Generate per-class gallery
    for cls_name, sel_info in class_selected.items():
        cls_idx = sel_info['cls_idx']
        selected = sel_info['indices']
        n_correct = sel_info['n_correct']
        n_total = sel_info['n_total']
        cls_acc = sel_info['cls_acc']
        n_show = len(selected)

        n_cols = min(5, n_show)
        n_rows = (n_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5.5 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, idx in enumerate(selected):
            row, col = divmod(i, n_cols)
            ax = axes[row][col]

            img_np = image_cache[idx].permute(1, 2, 0).numpy()
            pred_lbl = all_preds[idx].item()
            dist_val = all_dists[idx].item()
            correct = pred_lbl == cls_idx

            # Colored border
            border = 5
            color = [0.15, 0.75, 0.15] if correct else [0.9, 0.15, 0.15]
            bordered = np.ones((img_np.shape[0] + 2 * border, img_np.shape[1] + 2 * border, 3))
            bordered[:, :] = color
            bordered[border:-border, border:-border] = img_np

            ax.imshow(bordered)
            ax.set_xticks([])
            ax.set_yticks([])

            if correct:
                title = f"Pred: {class_names[pred_lbl]}\nDist: {dist_val:.3f} [OK]"
                ax.set_title(title, fontsize=9, fontweight='bold', color='green')
            else:
                # Show top-3 distances for wrong predictions
                sample_dists = all_all_dists[idx]
                top3 = torch.topk(sample_dists, 3, largest=False)
                top3_str = ", ".join([f"{class_names[t.item()]}: {d.item():.3f}"
                                     for t, d in zip(top3.indices, top3.values)])
                title = (f"Pred: {class_names[pred_lbl]}\n"
                         f"Dist: {dist_val:.3f} [WRONG]\n"
                         f"Top3: {top3_str}")
                ax.set_title(title, fontsize=8, fontweight='bold', color='red')

        # Hide unused axes
        for i in range(n_show, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row][col].axis('off')

        status_emoji = "OK" if cls_acc >= 0.9 else ("~" if cls_acc >= 0.7 else "!!")
        fig.suptitle(
            f'{cls_name} — {n_correct}/{n_total} correct ({cls_acc:.1%}) [{status_emoji}]',
            fontsize=15, fontweight='bold',
            color='green' if cls_acc >= 0.9 else ('orange' if cls_acc >= 0.7 else 'red'),
            y=1.01
        )

        plt.tight_layout()
        save_path = gallery_dir / f'{cls_name}.png'
        plt.savefig(save_path, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"  {cls_name}: {n_correct}/{n_total} ({cls_acc:.1%}) -> {save_path.name}")

    # 4. Generate summary overview
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(class_stats.keys())
    accs = [class_stats[n]['accuracy'] for n in names]
    totals = [class_stats[n]['total'] for n in names]
    corrects = [class_stats[n]['correct'] for n in names]

    colors_bar = ['#2ecc71' if a >= 0.9 else ('#f39c12' if a >= 0.7 else '#e74c3c') for a in accs]
    bars = ax.barh(names, accs, color=colors_bar, edgecolor='#333', linewidth=0.5)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Per-Class Accuracy (Centroid Classification)', fontsize=14, fontweight='bold')
    ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
    ax.legend(loc='lower right', fontsize=9)

    for bar, acc, c, t in zip(bars, accs, corrects, totals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{acc:.1%} ({c}/{t})', va='center', fontsize=10, fontweight='bold')

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(gallery_dir / 'summary.png', dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()

    total_correct = sum(class_stats[n]['correct'] for n in names)
    total_samples = sum(class_stats[n]['total'] for n in names)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0

    logger.info(f"Inference gallery saved: {gallery_dir} ({total_correct}/{total_samples} = {overall_acc:.1%})")

    del image_cache, all_labels, all_preds, all_dists, all_all_dists
    _free_step_memory("inference_gallery_complete")
    return class_stats


def generate_summary_report(csv_path, config, output_path, slice_accuracy=0, knn_accuracy=0,
                            per_class_acc=None, dist_ratio=0, class_names=None, emb_stats=None,
                            std_metrics=None, per_class_df=None, training_protocol_path=None,
                            accuracy=0):
    """Generate a markdown summary report of training results."""
    df = pd.read_csv(csv_path)
    output_path = Path(output_path)

    best_row, selection_metric = _resolve_best_training_row(df, config)
    final_row = df.iloc[-1]
    total_time = df['epoch_time_sec'].sum()
    sel_value = float(
        best_row.get('selection_value', best_row.get(selection_metric, 0.0))
    )

    def fmt_time(s):
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {sec}s"
        return f"{m}m {sec}s"

    lines = [
        f"# Training Results Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Total training time:** {fmt_time(total_time)}  ",
        f"**Epochs:** {len(df)}  ",
        "",
        "---",
        "",
        "## Best Model Performance",
        "",
        f"Checkpoint seleccionado por **`{selection_metric}`** (protocolo MLRC).",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Best Epoch | {int(best_row['epoch'])} |",
        f"| Selection metric ({selection_metric}) | {sel_value:.4f} |",
        f"| Recall@1 (val epoch) | {float(best_row['recall_at_1']):.4f} ({float(best_row['recall_at_1']):.1%}) |",
        f"| mAP@R (val epoch) | {float(best_row.get('mAP@R', 0.0)):.4f} |",
        f"| Recall@5 | {float(best_row['recall_at_5']):.4f} ({float(best_row['recall_at_5']):.1%}) |",
        f"| Val Loss (aux. P×K) | {float(best_row['val_loss']):.4f} |",
        f"| Train Loss | {float(best_row['train_loss']):.4f} |",
        f"| Slice-aware accuracy (post-hoc LOO) | {slice_accuracy:.4f} ({slice_accuracy:.1%}) |",
        f"| 1-NN LOO accuracy (MLRC) | {knn_accuracy:.4f} ({knn_accuracy:.1%}) |",
        f"| Distance Ratio (inter/intra) | {dist_ratio:.2f} |",
        "",
        "## Final Epoch Metrics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Epoch | {int(final_row['epoch'])} |",
        f"| Train Loss | {float(final_row['train_loss']):.4f} |",
        f"| Val Loss | {float(final_row['val_loss']):.4f} |",
        f"| Recall@1 | {float(final_row['recall_at_1']):.4f} |",
        f"| Recall@5 | {float(final_row['recall_at_5']):.4f} |",
        f"| Learning Rate | {final_row['learning_rate']} |",
        "",
    ]

    # Standard classification metrics
    if std_metrics:
        lines.extend([
            "## Standard Classification Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {std_metrics.get('accuracy', 0.0):.4f} ({std_metrics.get('accuracy', 0.0):.1%}) |",
            f"| Balanced Accuracy | {std_metrics.get('balanced_accuracy', 0.0):.4f} ({std_metrics.get('balanced_accuracy', 0.0):.1%}) |",
            f"| Precision Macro | {std_metrics.get('precision_macro', 0.0):.4f} |",
            f"| Precision Weighted | {std_metrics.get('precision_weighted', 0.0):.4f} |",
            f"| F1 Macro | {std_metrics.get('f1_macro', 0.0):.4f} |",
            f"| F1 Weighted | {std_metrics.get('f1_weighted', 0.0):.4f} |",
            "",
        ])

    # Per-class accuracy table
    if per_class_acc is not None and class_names is not None:
        lines.append("## Per-Class Accuracy (Slice-aware LOO)")
        lines.append("")
        lines.append("| Class | Accuracy |")
        lines.append("|-------|----------|")
        for i, name in enumerate(class_names):
            lines.append(f"| {name} | {per_class_acc[i]:.1%} |")
        lines.append("")

    if per_class_df is not None and not per_class_df.empty:
        lines.append("## Per-Class Precision/Recall/F1")
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|-------|-----------|--------|----|---------|")
        for _, row in per_class_df.iterrows():
            lines.append(
                f"| {row['class']} | {float(row['precision']):.4f} | {float(row['recall']):.4f} | "
                f"{float(row['f1']):.4f} | {int(row['support'])} |"
            )
        lines.append("")

    # Training configuration
    if config:
        lines.extend([
            "## Training Configuration",
            "",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| Backbone | {config.get('model', {}).get('backbone_name', config.get('model', {}).get('backbone', 'N/A'))} |",
            f"| Embedding Dim | {config.get('model', {}).get('embedding_dim', 'N/A')} |",
            f"| Loss | {config.get('training', {}).get('loss', {}).get('type', 'N/A')} |",
            f"| Optimizer | {config.get('training', {}).get('optimizer', 'N/A')} |",
            f"| LR | {config.get('training', {}).get('learning_rate', 'N/A')} |",
            f"| Batch Size | {config.get('data', {}).get('batch_size', 'N/A')} |",
            f"| Mixed Precision | {config.get('hardware', {}).get('use_amp', 'N/A')} |",
            "",
        ])

    # Embedding space analysis
    if emb_stats:
        lines.extend([
            f"## Embedding Space Analysis ($\\mathbb{{S}}^{{{emb_stats['embed_dim']}}}$)",
            "",
            f"| Propiedad | Valor |",
            f"|-----------|-------|",
            f"| Dimensión | {emb_stats['embed_dim']} |",
            f"| Muestras | {emb_stats['n_samples']} |",
            f"| Norma media | {emb_stats['norm_mean']:.4f} ± {emb_stats['norm_std']:.4f} |",
            f"| Separacion angular media | {emb_stats['mean_angular_sep']:.1f}deg |",
            f"| Separacion angular rango | [{emb_stats['min_angular_sep']:.1f}deg, {emb_stats['max_angular_sep']:.1f}deg] |",
            "",
        ])
        if emb_stats.get('compactness'):
            lines.append("### Compacidad por Clase")
            lines.append("")
            lines.append("| Clase | Cosine Sim al Centroide |")
            lines.append("|-------|------------------------|")
            for cls_name, comp in emb_stats['compactness'].items():
                emoji = '🟢' if comp >= 0.8 else ('🟡' if comp >= 0.6 else '🔴')
                lines.append(f"| {cls_name} | {comp:.4f} {emoji} |")
            lines.append("")

    # Quality assessment
    r1 = float(best_row['recall_at_1'])
    if r1 >= 0.95:
        quality = "EXCELLENT"
    elif r1 >= 0.90:
        quality = "VERY GOOD"
    elif r1 >= 0.80:
        quality = "GOOD"
    elif r1 >= 0.70:
        quality = "ACCEPTABLE"
    else:
        quality = "NEEDS IMPROVEMENT"

    lines.extend([
        "## Quality Assessment",
        "",
        f"**Overall: {quality}** (R@1 = {r1:.1%})",
        "",
        "## Training Protocol",
        "",
    ])
    if training_protocol_path:
        lines.extend([
            f"- Detailed protocol: `{training_protocol_path}`",
            "- Includes split details, stratification/cross-validation statement, and effective hyperparameters.",
            "",
        ])
    else:
        lines.extend([
            "- Detailed protocol file not available in this run.",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Generated Files",
        "",
        "- `training_metrics.csv` — Epoch-by-epoch metrics",
        "- `images/post_training/loss_curves.png` — Train/Val loss over epochs",
        "- `images/post_training/recall_curves.png` — Recall@1 and Recall@5",
        "- `images/post_training/lr_curve.png` — Learning rate schedule",
        "- `images/post_training/epoch_times.png` — Per-epoch duration",
        "- `images/post_training/tsne_3d.png` — 3D t-SNE embedding clusters (S^D)",
        "- `images/post_training/tsne_2d.png` — 2D t-SNE + density (S^D)",
        "- `images/post_training/embedding_space_analysis.png` — Análisis S^D: normas, coseno, compacidad, separación angular",
        "- `images/post_training/confusion_matrix.png` — kNN confusion matrix",
        "- `images/post_training/confusion_matrix_normalized.png` — normalized confusion matrix",
        "- `images/post_training/classification_report.csv` — per-class precision/recall/F1/support",
        "- `images/post_training/per_class_f1.png` — F1-score per class",
        "- `images/post_training/precision_recall_by_epoch.png` — trayectoria Precision vs Recall por época",
        "- `images/post_training/f1_curves_by_epoch.png` — F1 Macro/Weighted por época",
        "- `images/post_training/f1_from_precision_recall_by_epoch.txt` — F1 calculado con la ecuación 2PR/(P+R)",
        "- `images/post_training/distance_distribution.png` — Intra/Inter distances",
        "- `evaluation_metrics_summary.json` — resumen estructurado de métricas",
        "- `training_protocol.md` — protocolo detallado del entrenamiento",
    ])

    report_text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    logger.info(f"Summary report saved: {output_path}")
    return report_text


def generate_gradcam_gallery(model, slice_classifier, val_loader, device, output_dir, class_names,
                              n_per_class=4, grain_config=None, config=None):
    """
    Generate per-class Grad-CAM heatmap gallery.
    For each class, shows n_per_class images with 3 panels:
      [Original | Heatmap (jet colormap) | Overlay]

    Uses EmbeddingSimilarityTarget to produce meaningful heatmaps for Metric Learning
    by maximizing cosine similarity with the predicted class centroid.

    Args:
        model: Trained model on device (eval mode)
        slice_classifier: SliceAwareClassifier
        val_loader: Validation DataLoader
        device: torch.device
        output_dir: Base directory to save gallery (creates gradcam_gallery/ subfolder)
        class_names: List of class names
        n_per_class: Number of samples per class to visualize
    """
    from src.utils.inference_utils import overlay_gradcam_on_image
    from src.utils.visual_explainer import create_explainer

    gallery_dir = Path(output_dir) / 'gradcam_gallery'
    gallery_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    explainer = create_explainer(model, device)  # created once, reused for all images

    if config is not None:
        mean, std = _resolve_denorm_tensors(config)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    logger.info("  Collecting validation predictions for Grad-CAM...")
    all_labels = []
    all_preds = []
    all_dists = []

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Grad-CAM inference", leave=False):
            images, labels, masks = _unpack_loader_batch(batch_data)
            embs = _forward_embeddings(model, images, masks, grain_config, device)
            preds, min_d = _classify_embeddings(embs, slice_classifier)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())
            all_dists.append(min_d.cpu())

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_dists = torch.cat(all_dists)

    rng = np.random.RandomState(42)
    n_classes = len(class_names)

    needed_indices = set()
    class_selected = {}
    summary_indices = {}
    for c_idx in range(n_classes):
        mask = (all_labels == c_idx)
        indices = torch.where(mask)[0].numpy()
        if len(indices) == 0:
            continue
        sel_count = min(n_per_class, len(indices))
        sel_indices = rng.choice(indices, size=sel_count, replace=False)
        class_selected[c_idx] = sel_indices
        needed_indices.update(sel_indices.tolist())
        summary_indices[c_idx] = int(rng.choice(indices))

    # Pass 2: fetch only the images required for visualization
    logger.info(f"  Fetching {len(needed_indices)} images for Grad-CAM (2-pass, memory-safe)...")
    image_cache = {}
    mask_cache = {}
    global_idx = 0
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Grad-CAM image fetch", leave=False):
            images, _, masks = _unpack_loader_batch(batch_data)
            batch_size = images.shape[0]
            for i in range(batch_size):
                if global_idx in needed_indices or global_idx in summary_indices.values():
                    image_cache[global_idx] = images[i].cpu()
                    if masks is not None:
                        mask_cache[global_idx] = masks[i].cpu()
                global_idx += 1

    _free_step_memory("gradcam_pass1")

    # 2. For each class, generate Grad-CAM for pre-selected samples
    for c_idx in range(n_classes):
        class_name = class_names[c_idx]
        if c_idx not in class_selected:
            logger.warning(f"  No samples for class {class_name}, skipping.")
            continue

        sel_indices = class_selected[c_idx]
        sel_count = len(sel_indices)

        fig, axes = plt.subplots(sel_count, 3, figsize=(15, 5 * sel_count))
        if sel_count == 1:
            axes = axes[np.newaxis, :]

        for row, idx in enumerate(sel_indices):
            img_tensor = image_cache[int(idx)]
            true_lbl = c_idx
            pred_lbl = all_preds[int(idx)].item()
            dist_val = all_dists[int(idx)].item()
            correct = true_lbl == pred_lbl

            # Denormalize for display
            img_denorm = (img_tensor.unsqueeze(0) * std + mean).clamp(0, 1)[0]
            img_np = img_denorm.permute(1, 2, 0).numpy()

            pred_target = slice_classifier.get_target_for_gradcam(pred_lbl, device)
            mask_t = mask_cache.get(int(idx))
            heatmap = explainer.generate_heatmap(
                img_tensor.unsqueeze(0),
                centroid=pred_target,
                mask=mask_t.unsqueeze(0) if mask_t is not None else None,
                grain_config=grain_config,
            )

            # Overlay
            overlay = overlay_gradcam_on_image(
                (img_np * 255).astype(np.uint8), heatmap, alpha=0.5
            )

            # Panel 1: Original
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title('Original', fontsize=11, fontweight='bold')
            axes[row, 0].axis('off')

            # Panel 2: Heatmap (jet colormap)
            axes[row, 1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
            axes[row, 1].set_title('Grad-CAM Heatmap', fontsize=11, fontweight='bold')
            axes[row, 1].axis('off')

            # Panel 3: Overlay
            axes[row, 2].imshow(overlay)
            status = "OK" if correct else "WRONG"
            title_color = 'green' if correct else 'red'
            axes[row, 2].set_title(
                f'Pred: {class_names[pred_lbl]} | Dist: {dist_val:.3f} [{status}]',
                fontsize=10, fontweight='bold', color=title_color
            )
            axes[row, 2].axis('off')

        fig.suptitle(
            f'Grad-CAM — {class_name} ({sel_count} samples)',
            fontsize=16, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        safe_name = class_name.replace(' ', '_').replace('/', '_')
        save_path = gallery_dir / f'{safe_name}_gradcam.png'
        plt.savefig(save_path, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"  Grad-CAM gallery saved: {save_path}")

    # 3. Summary: one sample per class in a single figure
    logger.info("  Generating Grad-CAM summary...")
    fig, axes = plt.subplots(n_classes, 3, figsize=(15, 4 * n_classes))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for c_idx in range(n_classes):
        class_name = class_names[c_idx]
        if c_idx not in summary_indices:
            for col in range(3):
                axes[c_idx, col].axis('off')
            continue

        idx = summary_indices[c_idx]
        img_tensor = image_cache[idx]
        pred_lbl = all_preds[idx].item()
        dist_val = all_dists[idx].item()
        correct = c_idx == pred_lbl

        img_denorm = (img_tensor.unsqueeze(0) * std + mean).clamp(0, 1)[0]
        img_np = img_denorm.permute(1, 2, 0).numpy()

        pred_target = slice_classifier.get_target_for_gradcam(pred_lbl, device)
        mask_t = mask_cache.get(int(idx))
        heatmap = explainer.generate_heatmap(
            img_tensor.unsqueeze(0),
            centroid=pred_target,
            mask=mask_t.unsqueeze(0) if mask_t is not None else None,
            grain_config=grain_config,
        )
        overlay = overlay_gradcam_on_image(
            (img_np * 255).astype(np.uint8), heatmap, alpha=0.5
        )

        axes[c_idx, 0].imshow(img_np)
        axes[c_idx, 0].set_ylabel(class_name, fontsize=11, fontweight='bold', rotation=0,
                                   labelpad=100, va='center')
        axes[c_idx, 0].set_xticks([])
        axes[c_idx, 0].set_yticks([])

        axes[c_idx, 1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[c_idx, 1].axis('off')

        axes[c_idx, 2].imshow(overlay)
        status = "OK" if correct else "WRONG"
        title_color = 'green' if correct else 'red'
        axes[c_idx, 2].set_title(
            f'{class_names[pred_lbl]} | d={dist_val:.3f} [{status}]',
            fontsize=9, fontweight='bold', color=title_color
        )
        axes[c_idx, 2].set_xticks([])
        axes[c_idx, 2].set_yticks([])

    # Column headers
    axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')

    fig.suptitle('Grad-CAM Summary — All Classes', fontsize=18, fontweight='bold', y=1.005)
    plt.tight_layout()
    summary_path = gallery_dir / 'gradcam_summary.png'
    plt.savefig(summary_path, dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
    plt.close()

    logger.info(f"  Grad-CAM summary saved: {summary_path}")
    logger.info(f"  Grad-CAM gallery complete: {gallery_dir}")

    del image_cache, explainer, all_preds, all_dists
    _free_step_memory("gradcam_complete")


def _fullimage_bbox_iou(a, b, iou_threshold: float = 0.3) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-6)


def _draw_fullimage_grain_labels(
    display_img,
    grains,
    *,
    expected_cls_idx: int,
    class_names,
    h_img: int,
    w_img: int,
):
    """Dibuja bbox + clase + confianza sobre la imagen (después del Grad-CAM)."""
    import cv2

    font_scale = max(0.45, min(h_img, w_img) / 1800)
    thickness = max(1, int(min(h_img, w_img) / 700))

    for grain in grains:
        bx, by, bw, bh = grain["bbox"]
        pred_cls_name = grain["predicted_class"]
        pred_cls = class_names.index(pred_cls_name) if pred_cls_name in class_names else -1
        confidence = float(grain.get("confidence", 0.0))
        correct = pred_cls == expected_cls_idx

        box_color = (0, 200, 0) if correct else (0, 0, 220)
        cv2.rectangle(display_img, (bx, by), (bx + bw, by + bh), box_color, max(2, thickness))

        label = f"{pred_cls_name} {confidence:.0%}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        ty = by - 6 if by - th - 8 > 0 else by + bh + th + 6
        cv2.rectangle(
            display_img,
            (bx, ty - th - baseline - 2),
            (bx + tw + 6, ty + baseline + 2),
            box_color,
            -1,
        )
        cv2.putText(
            display_img,
            label,
            (bx + 3, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def generate_fullimage_inference(model, slice_classifier, device, config, output_dir, class_names,
                                  n_images_per_class=3, with_gradcam=True):
    """
    Run inference on FULL-RESOLUTION original images using FullImageClassifier.
    
    Pipeline per image:
      1. Localize grains with PollenGrainDetector (U²-Net SOD + pollen morphology)
      2. Crop each grain from the original image
      3. Transform + embed each crop with the trained AnalogyNet (ViT)
      4. Classify each crop via slice-aware classifier
      5. Draw annotated bounding boxes + labels on the full image
      6. Optionally overlay Grad-CAM heatmaps per grain
    
    NUEVO (2026-03-07): Usa FullImageClassifier en lugar de requerir .seg files.
    Lee imágenes desde annotation_root usando headers de .seg para encontrar paths originales.
    
    Args:
        model: Trained AnalogyNet on device (eval mode)
        slice_classifier: SliceAwareClassifier for Grad-CAM targets
        device: torch.device
        config: Full configuration dict
        output_dir: Directory to save results
        class_names: List of class names
        n_images_per_class: Number of full images per class to process
        with_gradcam: Whether to generate Grad-CAM overlays per grain
    """
    import cv2
    from src.grain_detection.seg_format import SegFileReader
    from src.grain_detection.full_image_classifier import FullImageClassifier
    from src.utils.inference_utils import overlay_gradcam_on_image
    from src.utils.visual_explainer import create_explainer
    from src.data.transforms_enhanced import get_val_transforms

    gallery_dir = Path(output_dir) / 'fullimage_inference'
    gallery_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    explainer = create_explainer(model, device) if with_gradcam else None

    # Get image size and build validation transform
    image_size = config['data'].get('image_size', 252)
    val_config = {**config.get('augmentation', {}), **config['data']}
    val_transform = get_val_transforms(val_config)
    
    # Cargar reference embeddings para FullImageClassifier
    run_dir = Path(output_dir).parent.parent.parent
    ref_path = run_dir / 'reference_embeddings.pt'
    if not ref_path.exists():
        logger.warning(f"  Reference embeddings not found at {ref_path}, skipping fullimage inference")
        return {}
    
    ref_data = torch.load(ref_path, map_location='cpu', weights_only=False)
    reference_embeddings = ref_data['embeddings'].cpu().numpy() if torch.is_tensor(ref_data['embeddings']) else ref_data['embeddings']
    reference_labels = ref_data['labels'].cpu().numpy() if torch.is_tensor(ref_data['labels']) else ref_data['labels']
    
    # Inicializar FullImageClassifier
    grain_config = config.get('grain_detection', {})
    full_classifier = FullImageClassifier(
        model=model,
        transforms=val_transform,
        reference_embeddings=reference_embeddings,
        reference_labels=reference_labels,
        class_names=class_names,
        device=device,
        grain_config=grain_config,
        k=5,
        slice_classifier=slice_classifier,
    )

    # Normalization constants for denormalization
    mean_t = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Color palette for classes (distinct colors)
    palette = [
        (46, 204, 113), (52, 152, 219), (231, 76, 60),
        (241, 196, 15), (155, 89, 182), (26, 188, 156),
        (230, 126, 34), (149, 165, 166), (192, 57, 43), (41, 128, 185),
    ]

    # Buscar imágenes desde annotation_root (donde están los .seg)
    annotation_root = Path(config['data'].get('annotation_root', 'data/annotations'))
    if not annotation_root.exists():
        logger.warning(f"  Annotation root not found: {annotation_root}")
        return {}
    
    logger.info(f"  Full-image inference from annotations: {annotation_root}")

    rng = np.random.RandomState(42)
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    all_results = {}
    total_grains = 0
    total_correct = 0
    total_annotated = 0
    total_matched_ann = 0
    total_missed_ann = 0
    total_extra_fp = 0
    total_matched_cls_correct = 0
    total_matched_cls_wrong = 0

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = annotation_root / cls_name
        if not cls_dir.is_dir():
            logger.warning(f"  Class annotation dir not found: {cls_dir}")
            continue

        # Find .seg files
        seg_files = sorted(list(cls_dir.glob('*.seg')))
        if len(seg_files) == 0:
            logger.warning(f"  No .seg files for {cls_name}")
            continue
        
        # Select random subset
        sel_count = min(n_images_per_class, len(seg_files))
        sel_seg_paths = [seg_files[i] for i in rng.choice(len(seg_files), sel_count, replace=False)]

        color = palette[cls_idx % len(palette)]

        for seg_path in sel_seg_paths:
            # Extraer path de imagen original desde header del .seg
            try:
                with open(seg_path, 'r') as f:
                    lines = f.readlines()
                img_name = None
                for line in lines[:10]:  # Buscar en primeras 10 líneas
                    if line.startswith('# Imagen:'):
                        img_name = line.split(':', 1)[1].strip()
                        break
                
                if not img_name:
                    logger.warning(f"  No image name found in {seg_path.name}")
                    continue
                
                # Buscar imagen en múltiples ubicaciones
                img_path = None
                search_paths = [
                    seg_path.parent / img_name,  # Mismo dir que .seg
                    Path('data/processed/train') / cls_name / img_name,
                    Path('data/processed/val') / cls_name / img_name,
                    Path('data/processed/test') / cls_name / img_name,
                ]
                for candidate in search_paths:
                    if candidate.exists():
                        img_path = candidate
                        break
                
                if img_path is None or not img_path.exists():
                    logger.warning(f"  Image not found for {seg_path.name}: {img_name}")
                    continue
                
                seg_data = SegFileReader.read(str(seg_path))
                annotated_grains = seg_data.get("grains", [])
                n_annotated = len(annotated_grains)

                discovery_mode = bool(
                    (config.get("grain_detection") or {}).get("discovery_mode", False)
                )
                results = full_classifier.classify_full_image(
                    str(img_path),
                    return_crops=True,
                    discovery_mode=discovery_mode,
                )
                grains = results.get("grains", [])

                full_img = cv2.imread(str(img_path))
                if full_img is None:
                    continue
                h_img, w_img = full_img.shape[:2]
                display_img = full_img.copy()

            except Exception as e:
                logger.warning(f"  Error processing {seg_path.name}: {e}")
                continue

            if len(grains) == 0:
                logger.warning(f"  No grains detected in {img_path.name}")
                continue

            img_grains_correct = 0
            img_grains_total = len(grains)
            heatmaps_and_bboxes = []
            font_scale = max(0.4, min(h_img, w_img) / 2000)
            thickness = max(1, int(min(h_img, w_img) / 800))

            matched_ann = set()
            img_matched_cls_ok = 0
            img_matched_cls_bad = 0
            for g_idx, grain in enumerate(grains):
                pred_cls_name = grain["predicted_class"]
                pred_cls = (
                    class_names.index(pred_cls_name) if pred_cls_name in class_names else -1
                )
                if pred_cls == cls_idx:
                    img_grains_correct += 1

                best_iou = 0.0
                for ann_i, ann in enumerate(annotated_grains):
                    iou = _fullimage_bbox_iou(grain["bbox"], ann["bbox"])
                    if iou >= 0.3:
                        matched_ann.add(ann_i)
                    if iou > best_iou:
                        best_iou = iou
                if best_iou >= 0.3:
                    if pred_cls == cls_idx:
                        img_matched_cls_ok += 1
                    else:
                        img_matched_cls_bad += 1

                if grain.get("crop") is not None:
                    crops_dir = gallery_dir / "crops"
                    crops_dir.mkdir(parents=True, exist_ok=True)
                    safe_cls_name = cls_name.replace(" ", "_").replace("/", "_")
                    safe_pred = pred_cls_name.replace(" ", "_").replace("/", "_")
                    crop_save_path = crops_dir / (
                        f"{safe_cls_name}_{img_path.stem}_grain_{g_idx}_{safe_pred}.png"
                    )
                    cv2.imwrite(str(crop_save_path), grain["crop"])

                if with_gradcam and explainer is not None and grain.get("tensor") is not None:
                    target_cls = pred_cls if pred_cls != -1 else 0
                    pred_target = slice_classifier.get_target_for_gradcam(target_cls, device)
                    heatmap = explainer.generate_heatmap(
                        grain["tensor"].unsqueeze(0), centroid=pred_target
                    )
                    heatmaps_and_bboxes.append((heatmap, grain["bbox"]))

            # 1) Grad-CAM primero (no debe borrar etiquetas)
            if with_gradcam and explainer is not None and heatmaps_and_bboxes:
                from src.utils.inference_utils import overlay_full_image_gradcam

                full_img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                gradcam_bgr = cv2.cvtColor(
                    overlay_full_image_gradcam(
                        full_image=full_img_rgb,
                        heatmaps_and_bboxes=heatmaps_and_bboxes,
                        crop_padding=full_classifier.crop_padding,
                        alpha=0.45,
                    ),
                    cv2.COLOR_RGB2BGR,
                )
                display_img = cv2.addWeighted(gradcam_bgr, 0.85, full_img, 0.15, 0)

            # 2) Granos anotados en .seg no detectados (referencia visual)
            for ann_i, ann in enumerate(annotated_grains):
                if ann_i in matched_ann:
                    continue
                bx, by, bw, bh = ann["bbox"]
                cv2.rectangle(
                    display_img, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2, cv2.LINE_AA
                )
                cv2.putText(
                    display_img,
                    "anotado (no detectado)",
                    (bx, max(12, by - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 0.85,
                    (255, 200, 0),
                    thickness,
                    cv2.LINE_AA,
                )

            # 3) Etiquetas de clase + confianza encima de todo
            _draw_fullimage_grain_labels(
                display_img,
                grains,
                expected_cls_idx=cls_idx,
                class_names=class_names,
                h_img=h_img,
                w_img=w_img,
            )

            n_extra_discovered = sum(
                1
                for grain in grains
                if not any(
                    _fullimage_bbox_iou(grain["bbox"], ann["bbox"]) >= 0.3
                    for ann in annotated_grains
                )
            )
            n_missed_annotated = n_annotated - len(matched_ann)
            total_annotated += n_annotated
            total_matched_ann += len(matched_ann)
            total_missed_ann += n_missed_annotated
            total_extra_fp += n_extra_discovered
            total_matched_cls_correct += img_matched_cls_ok
            total_matched_cls_wrong += img_matched_cls_bad
            total_grains += img_grains_total
            total_correct += img_grains_correct
            acc = img_grains_correct / img_grains_total if img_grains_total > 0 else 0

            title = (
                f"{cls_name} | {img_path.name} | "
                f"Detectados: {img_grains_total} | Anotados .seg: {n_annotated} | "
                f"Correctos: {img_grains_correct}/{img_grains_total} ({acc:.0%}) | "
                f"Extra descubiertos: {n_extra_discovered} | "
                f"Anotados perdidos: {n_missed_annotated}"
            )
            bar_h = max(36, int(h_img * 0.045))
            title_bar = np.zeros((bar_h, w_img, 3), dtype=np.uint8)
            title_bar[:] = (40, 40, 40)
            cv2.putText(
                title_bar,
                title,
                (10, bar_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 1.1,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            display_img = np.vstack([title_bar, display_img])

            # Save
            safe_name = f"{cls_name}_{img_path.stem}".replace(' ', '_')
            save_path = gallery_dir / f"{safe_name}.png"
            cv2.imwrite(str(save_path), display_img)

            all_results.setdefault(cls_name, []).append({
                "image": img_path.name,
                "grains": img_grains_total,
                "annotated": n_annotated,
                "correct": img_grains_correct,
                "accuracy": acc,
                "discovered_extra": n_extra_discovered,
                "missed_annotated": n_missed_annotated,
            })

    # Generate summary figure
    if all_results:
        n_total_images = sum(len(v) for v in all_results.values())
        overall_acc = total_correct / total_grains if total_grains > 0 else 0

        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(all_results.keys())
        accs = []
        for name in names:
            c = sum(r['correct'] for r in all_results[name])
            t = sum(r['grains'] for r in all_results[name])
            accs.append(c / t if t > 0 else 0)

        colors_bar = ['#2ecc71' if a >= 0.9 else ('#f39c12' if a >= 0.7 else '#e74c3c') for a in accs]
        bars = ax.barh(names, accs, color=colors_bar, edgecolor='#333', linewidth=0.5)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel('Grain Classification Accuracy', fontsize=12)
        ax.set_title(f'Full-Image Inference — {total_correct}/{total_grains} grains ({overall_acc:.1%})',
                      fontsize=14, fontweight='bold')
        ax.axvline(x=0.9, color='green', linestyle='--', alpha=0.5)

        for bar, a in zip(bars, accs):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{a:.1%}', va='center', fontsize=10, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(gallery_dir / 'fullimage_summary.png', dpi=DEFAULT_FIG_DPI, bbox_inches='tight')
        plt.close()

        logger.info(f"  Full-image inference: {total_correct}/{total_grains} grains "
                     f"({overall_acc:.1%}) across {n_total_images} images -> {gallery_dir}")

        matched_cls_total = total_matched_cls_correct + total_matched_cls_wrong
        decomposed = {
            "legacy_e2e_accuracy": overall_acc,
            "legacy_correct": total_correct,
            "legacy_total_detected": total_grains,
            "detection_recall_on_annotated": (
                total_matched_ann / total_annotated if total_annotated else 0.0
            ),
            "classification_on_matched_only": (
                total_matched_cls_correct / matched_cls_total if matched_cls_total else 0.0
            ),
            "annotated_grains": total_annotated,
            "matched_annotations": total_matched_ann,
            "missed_annotations": total_missed_ann,
            "false_positive_detections": total_extra_fp,
            "matched_classification_correct": total_matched_cls_correct,
            "matched_classification_wrong": total_matched_cls_wrong,
        }
        from src.grain_detection.detector_registry import get_active_detector_config

        analysis_path = gallery_dir / "e2e_decomposed_metrics.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "active_detector": get_active_detector_config(config),
                    "decomposed": decomposed,
                    "per_class": all_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(
            "  E2E decomposed — det_recall=%.1f%% cls_matched=%.1f%% legacy=%.1f%% → %s",
            decomposed["detection_recall_on_annotated"] * 100,
            decomposed["classification_on_matched_only"] * 100,
            decomposed["legacy_e2e_accuracy"] * 100,
            analysis_path,
        )

    return all_results


def run_post_training_analysis(
    model,
    loader,
    config,
    run_dir,
    device,
    slice_classifier=None,
    centroids=None,
    prefix="val",
):
    """
    Full post-training analysis pipeline.
    Clears existing artifacts for the prefix, then regenerates all visualizations
    under run_dir/images/post_training/{prefix}/.
    Uses the BEST model (must be loaded before calling this).

    Args:
        model: Trained model (best checkpoint, already on device)
        loader: DataLoader (for evaluation, e.g. val_loader or test_loader)
        config: Full configuration dict
        run_dir: Run directory path
        device: torch.device
        slice_classifier: SliceAwareClassifier; loaded from run_dir if None
        centroids: Legacy full centroids (kNN compat only)
    """
    run_dir = Path(run_dir)
    from src.utils.config_utils import merge_global_active_detector
    from src.grain_detection.detector_registry import get_active_detector_config

    config = merge_global_active_detector(config, "config.yaml")
    active_detector = get_active_detector_config(config)
    logger.info(
        "Post-training E2E usará detector: %s (backend=%s)",
        active_detector.get("label"),
        active_detector.get("backend"),
    )
    class_names = loader.dataset.classes
    csv_path = run_dir / 'training_metrics.csv'

    if slice_classifier is None:
        embed_dim = config.get('model', {}).get('embedding_dim', 128)
        num_slices = config.get('training', {}).get('loss', {}).get('params', {}).get('num_slices', 4)
        slice_classifier = load_slice_classifier_from_run(run_dir, device, embed_dim, num_slices)

    if slice_classifier is None:
        logger.error("Slice classifier not available — run compute_and_save_centroids first")
        return

    if centroids is None and (run_dir / 'class_centroids.pt').exists():
        centroids, _ = load_centroids(str(run_dir / 'class_centroids.pt'), device)

    logger.info("=" * 70)
    logger.info(f"POST-TRAINING ANALYSIS (Best Model) - Prefix: {prefix.upper()}")
    logger.info("=" * 70)

    post_dir = _clear_prefix_artifacts(run_dir, prefix)
    emb_stats = None
    protocol_path = run_dir / 'training_protocol.md'

    # 1. Training curves from CSV
    if csv_path.exists():
        logger.info("[1/11] Generating training curves...")
        plot_training_curves(str(csv_path), str(post_dir))
    else:
        logger.warning(f"Metrics CSV not found: {csv_path}")

    # 2. Retrieval / classification metrics by epoch (validation log)
    if csv_path.exists():
        logger.info("[2/11] Generating retrieval epoch curves...")
        generate_retrieval_epoch_curves(str(csv_path), str(post_dir))
    else:
        logger.warning("Retrieval epoch curves skipped (no training_metrics.csv)")

    # 3. Embeddings + graphical analysis
    logger.info("[3/11] Extracting embeddings for graphical analysis...")
    embeddings, labels = extract_embeddings(
        model, loader, device, grain_config=config.get("grain_detection")
    )
    _free_step_memory("extract_embeddings")

    logger.info("[*] Generando gráficos secuencialmente (matplotlib state machine)...")
    plot_tsne_3d(embeddings, labels, str(post_dir / 'tsne_3d.png'), class_names)

    try:
        emb_stats = plot_embedding_space_analysis(
            embeddings, labels, class_names, str(post_dir / 'embedding_space_analysis.png')
        )
    except Exception as e:
        logger.error(f"Embedding space analysis failed: {e}")
        emb_stats = None

    from src.utils.mlrc_protocol import resolve_nn_metric

    nn_metric = resolve_nn_metric(config)
    grain_config = config.get("grain_detection", {})
    gallery_cfg = (config.get("evaluation") or {}).get("gallery") or {}

    knn_accuracy, knn_per_class_acc, knn_std_metrics, knn_per_class_df, _knn_predictions = (
        plot_confusion_matrix_knn(
            embeddings,
            labels,
            class_names,
            str(post_dir / 'confusion_matrix.png'),
            nn_metric=nn_metric,
        )
    )

    slice_predictions = compute_slice_predictions(embeddings, slice_classifier)
    slice_accuracy = accuracy_score(labels, slice_predictions)
    slice_std_metrics, slice_per_class_df = compute_standard_metrics_from_predictions(
        labels, slice_predictions, class_names
    )
    logger.info(
        "Slice-aware accuracy: %.4f | 1-NN LOO accuracy (%s): %.4f",
        slice_accuracy,
        nn_metric,
        knn_accuracy,
    )

    mean_intra, mean_inter, dist_ratio = plot_distance_distribution(
        embeddings, labels, class_names, str(post_dir / 'distance_distribution.png')
    )

    class_report_path = post_dir / 'classification_report_slice_aware.csv'
    slice_per_class_df.to_csv(class_report_path, index=False, encoding='utf-8')
    knn_report_path = post_dir / 'classification_report_1nn_loo.csv'
    knn_per_class_df.to_csv(knn_report_path, index=False, encoding='utf-8')
    logger.info(f"Classification reports saved: {class_report_path}, {knn_report_path}")
    plot_per_class_f1(
        slice_per_class_df,
        str(post_dir / 'per_class_f1.png'),
        metric_label="Slice-aware LOO",
    )
    plot_per_class_f1(
        knn_per_class_df,
        str(post_dir / 'per_class_f1_1nn_loo.png'),
        metric_label=f"1-NN LOO ({nn_metric})",
    )
    _free_step_memory("step3_plots")

    # 7. Inference preview
    logger.info("[7/11] Generating inference preview...")
    plot_inference_preview(
        model, slice_classifier, loader, device,
        str(post_dir / 'inference_preview.png'),
        class_names,
        n_samples=int(gallery_cfg.get("inference_preview", 10)),
        grain_config=grain_config,
        config=config,
    )
    _free_step_memory("inference_preview")

    # 8. Grad-CAM gallery
    logger.info("[8/11] Generating attention heatmap gallery...")
    try:
        generate_gradcam_gallery(
            model, slice_classifier, loader, device,
            str(post_dir), class_names,
            n_per_class=int(gallery_cfg.get("gradcam_per_class", 4)),
            grain_config=grain_config,
            config=config,
        )
    except Exception as e:
        logger.error(f"Attention/Grad-CAM generation FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
    _free_step_memory("gradcam_gallery")

    # 9. Per-class inference gallery
    logger.info("[9/11] Generating per-class inference gallery...")
    try:
        generate_inference_gallery(
            model, slice_classifier, loader, device,
            str(post_dir), class_names,
            n_per_class=int(gallery_cfg.get("inference_per_class", 10)),
            grain_config=grain_config,
            config=config,
        )
    except Exception as e:
        logger.error(f"Inference gallery generation FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
    _free_step_memory("inference_gallery")

    # 10. Full-image inference
    logger.info("[10/11] Generating full-image inference on test images...")
    try:
        generate_fullimage_inference(
            model, slice_classifier, device, config,
            str(post_dir), class_names,
            n_images_per_class=int(gallery_cfg.get("fullimage_per_class", 3)),
            with_gradcam=True
        )
    except Exception as e:
        logger.warning(f"Full-image inference failed (non-critical): {e}")
        import traceback
        logger.debug(traceback.format_exc())
    _free_step_memory("fullimage_inference")

    # 11. Training protocol (once per run, regenerated on val pass)
    if prefix == "val":
        logger.info("[11/12] Generating training protocol...")
        protocol_path, _ = generate_training_protocol(config, run_dir)
    elif not protocol_path.exists():
        logger.info("[11/12] Generating training protocol (missing)...")
        protocol_path, _ = generate_training_protocol(config, run_dir)
    else:
        logger.info("[11/12] Using existing training protocol")

    # 12. Evaluation summary JSON
    logger.info("[12/12] Generating evaluation summary JSON...")
    summary_json_path = run_dir / f'evaluation_metrics_summary_{prefix}.json'
    nn_metric = resolve_nn_metric(config)
    map_exact, map_multiplier = resolve_map_at_r_config(config)
    nmi_n_init, nmi_n_seeds, nmi_base_seed = resolve_nmi_config(config)
    mlrc_metrics = compute_all_metrics(
        embeddings,
        labels,
        nn_metric=nn_metric,
        k_list=[1, 5, 10],
        class_names=class_names,
        include_per_class=False,
        nmi_n_init=nmi_n_init,
        nmi_n_seeds=nmi_n_seeds,
        nmi_base_seed=nmi_base_seed,
        map_at_r_exact=map_exact,
        map_at_r_candidate_multiplier=map_multiplier,
    )
    n_runs = resolve_n_runs(config)
    mlrc_ci = {
        "protocol": "mlrc",
        "n_runs_configured": n_runs,
        "note": (
            "Single-run point estimates; aggregate mean±95% CI requires "
            f"{n_runs} independent training runs."
            if n_runs > 1
            else "Single-run point estimates (bootstrap CI optional)."
        ),
        "metrics": {
            "R@1": {"value": float(mlrc_metrics.get("R@1", 0.0))},
            "mAP@R": {"value": float(mlrc_metrics.get("mAP@R", 0.0))},
            "NMI": {
                "value": float(mlrc_metrics.get("NMI", 0.0)),
                "std_across_seeds": float(mlrc_metrics.get("NMI_std", 0.0)),
            },
            "F1_macro": {"value": float(mlrc_metrics.get("F1_macro", 0.0))},
        },
    }
    eval_summary = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_dir': str(run_dir),
        'prefix': prefix,
        'active_detector': active_detector,
        'mlrc_protocol': mlrc_ci,
        'metric_learning': {
            'slice_accuracy': float(slice_accuracy),
            'knn_accuracy': float(knn_accuracy),
            'distance_ratio_inter_intra': float(dist_ratio),
            'R@1': float(mlrc_metrics.get('R@1', 0.0)),
            'R@5': float(mlrc_metrics.get('R@5', 0.0)),
            'mAP@R': float(mlrc_metrics.get('mAP@R', 0.0)),
            'NMI': float(mlrc_metrics.get('NMI', 0.0)),
            'NMI_std': float(mlrc_metrics.get('NMI_std', 0.0)),
            'F1_macro': float(mlrc_metrics.get('F1_macro', 0.0)),
            'nn_metric': nn_metric,
            'map_at_r_exact': map_exact,
        },
        'classification_standard': {
            'slice_aware': slice_std_metrics,
            'knn_loo': knn_std_metrics,
        },
        'per_class': {
            'slice_aware': slice_per_class_df.to_dict(orient='records'),
            'knn_loo': knn_per_class_df.to_dict(orient='records'),
        },
    }
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation summary saved: {summary_json_path}")

    # 13. Summary report
    logger.info("[13/13] Generating summary report...")
    report_path = run_dir / f'training_report_{prefix}.md'
    generate_summary_report(
        csv_path=str(csv_path),
        config=config,
        output_path=str(report_path),
        slice_accuracy=float(slice_accuracy),
        knn_accuracy=float(knn_accuracy),
        per_class_acc=knn_per_class_acc,
        dist_ratio=dist_ratio,
        class_names=class_names,
        emb_stats=emb_stats,
        std_metrics=slice_std_metrics,
        per_class_df=slice_per_class_df,
        training_protocol_path=protocol_path.name if protocol_path.exists() else None,
    )

    # Spectral analysis
    logger.info("=" * 70)
    logger.info("SPECTRAL ANALYSIS - Dimensional Collapse Detection")
    logger.info("=" * 70)
    spectral_dir = post_dir / 'spectral_analysis'
    spectral_dir.mkdir(parents=True, exist_ok=True)
    try:
        analyzer = EmbeddingSpaceAnalyzer(
            embeddings=embeddings,
            labels=labels,
            class_names=class_names
        )

        logger.info("  -> Global spectrum analysis...")
        global_results = analyzer.analyze_global_spectrum(str(spectral_dir))

        msg1 = f"     Global λ₁: {global_results['collapse_severity']:.3f}"
        logger.info(sanitize_for_ascii(msg1))
        logger.info(f"     Global Effective Rank: {global_results['effective_rank']}/{analyzer.n_dim}")

        logger.info("  -> Analyzing all class pairs...")
        pair_results = analyzer.analyze_all_pairs(str(spectral_dir))

        collapsed_pairs = [
            (pair, res) for pair, res in pair_results.items()
            if res.get('is_collapsed', False)
        ]

        if collapsed_pairs:
            logger.warning(f"  ! COLLAPSED PAIRS DETECTED: {len(collapsed_pairs)}/{len(pair_results)}")
            for (ca, cb), res in collapsed_pairs[:3]:
                msg_warn = f"     - {ca} vs {cb}: λ₁={res['collapse_severity']:.3f}, Rank={res['effective_rank']}"
                logger.warning(sanitize_for_ascii(msg_warn))
        else:
            logger.info("  OK No collapsed pairs detected (all healthy)")

        embed_dim_cfg = int(config.get('model', {}).get('embedding_dim', analyzer.n_dim))
        spectral_summary = {
            'embedding_dim': embed_dim_cfg,
            'global_effective_rank': int(global_results['effective_rank']),
            'max_dim': int(analyzer.n_dim),
            'collapse_severity': float(global_results['collapse_severity']),
            'collapsed_pairs': len(collapsed_pairs),
            'total_pairs': len(pair_results),
            'mean_pair_rank': float(
                np.mean([res['effective_rank'] for res in pair_results.values()])
            ) if pair_results else 0.0,
        }
        summary_path = spectral_dir / 'spectral_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(spectral_summary, f, indent=2)
        logger.info(f"  Spectral summary: {summary_path}")

        logger.info(f"  Spectral analysis complete: {spectral_dir}")

    except Exception as e:
        logger.error(f"Spectral analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    del embeddings, labels
    _free_step_memory("post_training_complete")

    logger.info("=" * 70)
    logger.info(f"Post-training analysis complete!")
    logger.info(f"Results in: {post_dir}")
    logger.info("=" * 70)

    return {
        'output_dir': str(post_dir),
        'slice_accuracy': float(slice_accuracy),
        'knn_accuracy': float(knn_accuracy),
        'per_class_acc': knn_per_class_acc.tolist() if knn_per_class_acc is not None else [],
        'dist_ratio': dist_ratio,
        'class_names': class_names,
    }


def run_evaluation_from_checkpoint(
    checkpoint_path,
    config_path='config.yaml',
    output_dir='results',
    skip_test=False,
):
    """
    Run the exact post-training pipeline from train.py on an existing run.

    Clears and regenerates all artifacts for val (and test if available).
    Uses best_model.pth, class_centroids.pt, reference_embeddings.pt and the run's
    config/config.yaml.

    Args:
        checkpoint_path: Path to model checkpoint (.pth), typically checkpoints/best_model.pth
        config_path: Fallback config YAML (project root config.yaml)
        output_dir: Ignored — results always go to the run directory
        skip_test: When True, skip test post-training analysis (default False: val + test)

    Returns:
        dict with output paths and metrics
    """
    import yaml
    from src.data.dataloader_utils import dispose_data_engine, free_training_memory
    from scripts.train import create_model

    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent.parent
    run_config_path = _resolve_run_config_path(run_dir)

    logger.info("=" * 70)
    logger.info("POST-TRAINING EVALUATION (full regeneration)")
    logger.info("=" * 70)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    if run_config_path.exists():
        logger.info(f"Loading run config: {run_config_path}")
        with open(run_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Run config not found at {run_config_path}, using {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and config.get('hardware', {}).get('use_gpu', True) else 'cpu'
    )
    logger.info(f"Device: {device}")
    free_training_memory("pre_evaluation")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    config['model']['pretrained'] = False
    config['model']['freeze_backbone'] = False
    model = create_model(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    epoch_info = ckpt.get('epoch', '?')
    val_loss_info = ckpt.get('val_loss', '?')
    logger.info(f"Best model loaded (epoch {epoch_info}, val_loss={val_loss_info})")

    embed_dim = config.get('model', {}).get('embedding_dim', 128)
    num_slices = config.get('training', {}).get('loss', {}).get('params', {}).get('num_slices', 4)
    slice_classifier = load_slice_classifier_from_run(run_dir, device, embed_dim, num_slices)
    centroids = None

    if slice_classifier is None:
        logger.info("Slice artifacts missing — computing from training set...")
        from src.data.data_engine import GrainDataEngine
        import copy
        train_config = copy.deepcopy(config)
        train_config.setdefault("data", {})["pin_memory"] = False
        train_engine = GrainDataEngine.from_config(train_config)
        class_names_train = train_engine.train_loader.dataset.classes
        centroids, slice_classifier = compute_and_save_centroids(
            model,
            train_engine.train_loader,
            device,
            str(run_dir),
            class_names_train,
            num_slices=num_slices,
            loss_state_dict=ckpt.get('loss_state_dict'),
            loss_type=ckpt.get('loss_type'),
        )
        dispose_data_engine(train_engine)
    elif (run_dir / 'class_centroids.pt').exists():
        centroids, _ = load_centroids(str(run_dir / 'class_centroids.pt'), device)

    _, val_loader, test_loader, data_engine = create_eval_dataloaders(config)
    _free_step_memory("eval_dataloaders")

    logger.info("=" * 70)
    logger.info("RUNNING POST-TRAINING ANALYSIS (VAL)")
    logger.info("=" * 70)
    val_results = run_post_training_analysis(
        model, val_loader, config, str(run_dir), device,
        slice_classifier=slice_classifier, centroids=centroids, prefix="val"
    )

    if skip_test:
        logger.info("Test post-training skipped (--skip-test / skip_test=True)")
    elif test_loader is not None:
        logger.info("=" * 70)
        logger.info("RUNNING POST-TRAINING ANALYSIS (TEST)")
        logger.info("=" * 70)
        run_post_training_analysis(
            model, test_loader, config, str(run_dir), device,
            slice_classifier=slice_classifier, centroids=centroids, prefix="test"
        )
    else:
        test_dir = config.get("data", {}).get("test_dir")
        logger.warning(
            "Test post-training skipped: no test loader "
            f"(test_dir={test_dir!r})"
        )

    finalize_self_contained_checkpoint(run_dir)
    dispose_data_engine(data_engine)
    _free_step_memory("evaluation_complete")

    logger.info("=" * 70)
    logger.info(f"Post-training evaluation complete! Results in: {run_dir}")
    logger.info("=" * 70)

    if val_results is None:
        class_names = list(getattr(val_loader.dataset, 'classes', []))
        return {
            'output_dir': str(run_dir),
            'accuracy': 0,
            'per_class_acc': [],
            'dist_ratio': 0,
            'class_names': class_names,
        }

    val_results['output_dir'] = str(run_dir)
    return val_results

