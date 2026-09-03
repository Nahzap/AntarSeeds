"""
Empirical calibration of inference thresholds on the validation split (B-07).

Calibrates k_neighbors and similarity_threshold from leave-one-out 1-NN scores
on validation embeddings, following pollen/ViT fine-tuning practice (MDPI 2026).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.embedding_extractor import extract_embeddings
from src.utils.mlrc_protocol import resolve_nn_metric

logger = logging.getLogger(__name__)

DEFAULT_K_CANDIDATES = [1, 3, 5, 7, 11, 15]


def _pairwise_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    nn_metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positive_scores, negative_scores) from leave-one-out 1-NN similarities."""
    n = len(embeddings)
    pos_scores: List[float] = []
    neg_scores: List[float] = []

    for i in range(n):
        query = embeddings[i]
        query_label = labels[i]

        if nn_metric == "euclidean":
            dists = np.linalg.norm(embeddings - query, axis=1)
            dists[i] = np.inf
            nn_idx = int(np.argmin(dists))
            score = float(-dists[nn_idx])
        elif nn_metric == "hyperbolic":
            q_norm2 = np.sum(query ** 2)
            u_norm2 = np.sum(embeddings ** 2, axis=1)
            dot = embeddings @ query
            sqdist = q_norm2 + u_norm2 - 2.0 * dot
            denom = np.clip((1.0 - q_norm2) * (1.0 - u_norm2), 1e-15, None)
            mobius = -(sqdist / denom)
            mobius[i] = -1e9
            nn_idx = int(np.argmax(mobius))
            score = float(mobius[nn_idx])
        else:
            sims = embeddings @ query
            sims[i] = -1e9
            nn_idx = int(np.argmax(sims))
            score = float(sims[nn_idx])

        if labels[nn_idx] == query_label:
            pos_scores.append(score)
        else:
            neg_scores.append(score)

    return np.asarray(pos_scores, dtype=np.float64), np.asarray(neg_scores, dtype=np.float64)


def _roc_optimal_threshold(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> Tuple[float, float]:
    """Youden's J statistic on pooled score distribution."""
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5, 0.0

    all_scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    thresholds = np.unique(all_scores)
    if len(thresholds) > 200:
        thresholds = np.quantile(all_scores, np.linspace(0.01, 0.99, 200))

    best_t = float(np.median(all_scores))
    best_j = -1.0
    for t in thresholds:
        pred_pos = all_scores >= t
        tpr = (pred_pos & (labels == 1)).sum() / max(1, (labels == 1).sum())
        fpr = (pred_pos & (labels == 0)).sum() / max(1, (labels == 0).sum())
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best_t = float(t)

    return best_t, float(best_j)


def _score_knn_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    nn_metric: str,
) -> float:
    """Leave-one-out k-NN accuracy."""
    n = len(embeddings)
    correct = 0
    for i in range(n):
        query = embeddings[i]
        query_label = labels[i]

        if nn_metric == "euclidean":
            dists = np.linalg.norm(embeddings - query, axis=1)
            dists[i] = np.inf
            nn_idx = np.argpartition(dists, min(k, n - 1) - 1)[:k]
        elif nn_metric == "hyperbolic":
            q_norm2 = np.sum(query ** 2)
            u_norm2 = np.sum(embeddings ** 2, axis=1)
            dot = embeddings @ query
            sqdist = q_norm2 + u_norm2 - 2.0 * dot
            denom = np.clip((1.0 - q_norm2) * (1.0 - u_norm2), 1e-15, None)
            mobius = -(sqdist / denom)
            mobius[i] = -1e9
            nn_idx = np.argpartition(-mobius, min(k, n - 1) - 1)[:k]
        else:
            sims = embeddings @ query
            sims[i] = -1e9
            nn_idx = np.argpartition(-sims, min(k, n - 1) - 1)[:k]

        votes = labels[nn_idx]
        pred = int(np.bincount(votes.astype(int)).argmax())
        if pred == query_label:
            correct += 1
    return correct / max(1, n)


def calibrate_inference_thresholds(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    config: Dict[str, Any],
    k_candidates: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Calibrate inference.k_neighbors and inference.similarity_threshold on val.

    Saves ``inference_calibration.json`` under *run_dir* and returns the payload.
    """
    run_dir = Path(run_dir)
    eval_cfg = config.get("evaluation", {}) or {}
    cal_cfg = eval_cfg.get("calibration", {}) or {}
    if not cal_cfg.get("enabled", True):
        logger.info("Threshold calibration disabled (evaluation.calibration.enabled=false)")
        return {}

    inference_cfg = config.get("inference", {}) or {}
    nn_metric = resolve_nn_metric(config)
    k_candidates = k_candidates or cal_cfg.get("k_candidates") or DEFAULT_K_CANDIDATES
    k_candidates = sorted({int(k) for k in k_candidates if int(k) >= 1})

    logger.info("Calibrating inference thresholds on validation set (nn_metric=%s)...", nn_metric)
    grain_config = config.get("grain_detection", {}) or {}
    embeddings, labels = extract_embeddings(
        model,
        val_loader,
        device,
        grain_config=grain_config,
        show_progress=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if nn_metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

    k_results = {
        int(k): _score_knn_accuracy(embeddings, labels, int(k), nn_metric)
        for k in k_candidates
    }
    best_k = max(k_results, key=k_results.get)

    pos_scores, neg_scores = _pairwise_scores(embeddings, labels, nn_metric)
    best_threshold, youden_j = _roc_optimal_threshold(pos_scores, neg_scores)

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": "val",
        "nn_metric": nn_metric,
        "defaults": {
            "k_neighbors": int(inference_cfg.get("k_neighbors", 5)),
            "similarity_threshold": float(inference_cfg.get("similarity_threshold", 0.6)),
        },
        "calibrated": {
            "k_neighbors": int(best_k),
            "similarity_threshold": float(best_threshold),
        },
        "diagnostics": {
            "knn_accuracy_by_k": {str(k): float(v) for k, v in k_results.items()},
            "youden_j": float(youden_j),
            "n_val_samples": int(len(labels)),
            "n_positive_pairs_scored": int(len(pos_scores)),
            "n_negative_pairs_scored": int(len(neg_scores)),
        },
    }

    out_path = run_dir / "inference_calibration.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(
        "Inference calibration saved: k_neighbors %s→%s, similarity_threshold %.3f→%.3f (%s)",
        payload["defaults"]["k_neighbors"],
        payload["calibrated"]["k_neighbors"],
        payload["defaults"]["similarity_threshold"],
        payload["calibrated"]["similarity_threshold"],
        out_path,
    )
    return payload
