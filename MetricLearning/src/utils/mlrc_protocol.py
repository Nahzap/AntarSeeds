"""
MLRC evaluation protocol helpers (Musgrave et al., ECCV 2020).

Centralizes validation-metric selection, nn_metric resolution, and
confidence-interval reporting for reproducible Deep Metric Learning evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Canonical validation metrics for checkpoint selection (B-01).
VALIDATION_METRIC_ALIASES = {
    "map@r": "mAP@R",
    "map_at_r": "mAP@R",
    "mAP@R": "mAP@R",
    "recall_at_1": "recall_at_1",
    "r@1": "recall_at_1",
    "val_loss": "val_loss",
    "f1_macro": "F1_macro",
    "F1_macro": "F1_macro",
}

HIGHER_IS_BETTER = {
    "mAP@R": True,
    "recall_at_1": True,
    "F1_macro": True,
    "val_loss": False,
}

METRIC_TO_VAL_KEY = {
    "mAP@R": "mAP@R",
    "recall_at_1": "R@1",
    "F1_macro": "F1_macro",
    "val_loss": "val_loss",
}


def normalize_validation_metric(name: str) -> str:
    """Map config aliases to canonical validation metric names."""
    key = str(name).strip()
    canonical = VALIDATION_METRIC_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unsupported training.validation_metric: {name!r}. "
            f"Allowed: {sorted(set(VALIDATION_METRIC_ALIASES.values()))}"
        )
    return canonical


def resolve_nn_metric(config: Dict[str, Any]) -> str:
    """
    Resolve nearest-neighbour metric from evaluation.nn_metric or sota.embedding_space.

    Priority: explicit evaluation.nn_metric > hyperbolic arcface > embedding_space.
    """
    eval_cfg = config.get("evaluation", {}) or {}
    explicit = eval_cfg.get("nn_metric")
    if explicit:
        metric = str(explicit).lower()
        if metric not in ("cosine", "euclidean", "hyperbolic"):
            raise ValueError(
                f"evaluation.nn_metric must be cosine|euclidean|hyperbolic, got {explicit!r}"
            )
        return metric

    sota = config.get("sota", {}) or {}
    embedding_space = str(sota.get("embedding_space", "euclidean")).lower()
    loss_type = (
        config.get("training", {}).get("loss", {}).get("type")
        or config.get("loss", {}).get("type", "")
    )
    if embedding_space == "hyperbolic" and loss_type == "arcface":
        return "hyperbolic"
    if embedding_space == "euclidean":
        return "euclidean"
    return "cosine"


def resolve_map_at_r_config(config: Dict[str, Any]) -> Tuple[bool, int]:
    """Return (exact, candidate_multiplier) from evaluation.map_at_r."""
    eval_cfg = config.get("evaluation", {}) or {}
    map_cfg = eval_cfg.get("map_at_r", {}) or {}
    exact = bool(map_cfg.get("exact", True))
    candidate_multiplier = int(map_cfg.get("candidate_multiplier", 12))
    return exact, candidate_multiplier


def resolve_nmi_config(config: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return (n_init, n_seeds, base_seed) from evaluation.nmi + reproducibility.seed."""
    eval_cfg = config.get("evaluation", {}) or {}
    nmi_cfg = eval_cfg.get("nmi", {}) or {}
    n_init = int(nmi_cfg.get("n_init", 10))
    n_seeds = int(nmi_cfg.get("n_seeds", 1))
    base_seed = int(config.get("reproducibility", {}).get("seed", 42))
    return n_init, max(1, n_seeds), base_seed


def resolve_n_runs(config: Dict[str, Any]) -> int:
    """Number of independent runs for CI reporting (Musgrave et al. BO+CV protocol)."""
    eval_cfg = config.get("evaluation", {}) or {}
    return max(1, int(eval_cfg.get("n_runs", 1)))


def metric_value_from_dict(metric: str, metrics: Dict[str, float]) -> float:
    """Extract the scalar used for checkpoint selection from a val metrics dict."""
    canonical = normalize_validation_metric(metric)
    key = METRIC_TO_VAL_KEY[canonical]
    return float(metrics.get(key, metrics.get(canonical, 0.0)))


def is_improvement(
    metric: str,
    current: float,
    best: float,
    min_delta: float = 0.0,
) -> bool:
    """Return True if *current* improves *best* for the given validation metric."""
    canonical = normalize_validation_metric(metric)
    higher = HIGHER_IS_BETTER[canonical]
    if higher:
        return current > best + min_delta
    return current < best - min_delta


def compute_confidence_interval(
    values: List[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Mean ± half-width for a list of scalar observations.

    Uses Student-t when n >= 2; otherwise reports point estimate with zero width.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci_half_width": 0.0, "n": 0}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci_half_width": 0.0, "n": 1}

    std = float(arr.std(ddof=1))
    try:
        from scipy import stats

        t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    except ImportError:
        # Normal approximation when scipy is unavailable (tests / minimal env).
        t_crit = 1.96
    ci_half = t_crit * std / np.sqrt(n)
    return {
        "mean": mean,
        "std": std,
        "ci_half_width": float(ci_half),
        "n": n,
        "confidence": confidence,
    }


def format_ci_summary(ci: Dict[str, float]) -> str:
    """Human-readable mean ± CI string."""
    mean = ci["mean"]
    half = ci.get("ci_half_width", 0.0)
    return f"{mean:.4f} ± {half:.4f}"
