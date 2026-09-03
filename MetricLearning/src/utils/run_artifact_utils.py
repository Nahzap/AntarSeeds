"""
Utilidades para resolver artefactos de inferencia desde un checkpoint de entrenamiento.

Usado por la GUI (pestaña Inferencia) y scripts CLI para verificar coherencia
train→inference con clasificación slice-aware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union


def resolve_run_dir_from_checkpoint(checkpoint_path: Union[str, Path]) -> Path:
    """
    Resuelve el directorio del run a partir de la ruta del checkpoint.

    Convención: ``runs/YYYY-MM-DD_HH-MM-SS/checkpoints/best_model.pth``
    → ``runs/YYYY-MM-DD_HH-MM-SS/``
    """
    ckpt = Path(checkpoint_path).resolve()
    if ckpt.parent.name == "checkpoints":
        return ckpt.parent.parent
    return ckpt.parent


def check_slice_inference_artifacts(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspecciona artefactos slice-aware disponibles para un checkpoint.

    Returns:
        Dict con run_dir, flags de artefactos y modo de clasificación esperado.
    """
    run_dir = resolve_run_dir_from_checkpoint(checkpoint_path)
    slice_path = run_dir / "slice_representatives.pt"
    proxy_path = run_dir / "class_proxies.pt"
    ref_path = run_dir / "reference_embeddings.pt"
    centroids_path = run_dir / "class_centroids.pt"

    has_slice = slice_path.exists()
    has_proxy = proxy_path.exists()
    has_ref = ref_path.exists()
    has_centroids = centroids_path.exists()

    if has_slice:
        mode = "slice-aware"
        primary_artifact = str(slice_path)
    elif has_proxy:
        mode = "slice-aware"
        primary_artifact = str(proxy_path)
    elif has_ref and has_centroids:
        mode = "knn_fallback"
        primary_artifact = str(ref_path)
    else:
        mode = "knn_fallback"
        primary_artifact = None

    return {
        "run_dir": run_dir,
        "has_slice_representatives": has_slice,
        "has_class_proxies": has_proxy,
        "has_reference_embeddings": has_ref,
        "has_class_centroids": has_centroids,
        "classification_mode": mode,
        "primary_artifact": primary_artifact,
        "slice_representatives_path": slice_path,
    }
