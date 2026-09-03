"""
Config parsing and validation for spatial pooling modules.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_POOLING_CONFIG: Dict[str, Any] = {
    "strategy": "multi_spectral",
    "layers": [3, 11],
    "statistics": ["mean", "max", "std"],
    "gem": {"p": 3.0, "learnable": False, "eps": 1e-6},
    "masked": {"enabled": True, "threshold": 0.3, "weighting": "soft"},
    "drop_token": {"enabled": True, "prob": 0.3, "eval_mode": "off"},
    "norm_intermediate": True,
    "layer_fusion": "per_layer",  # per_layer | concat
}

VALID_STRATEGIES = {"mean", "max", "gem", "multi_spectral", "attention", "fsw"}
VALID_STATISTICS = {"mean", "max", "std"}
VALID_WEIGHTING = {"soft", "hard"}
VALID_EVAL_MODES = {"off", "same_as_train"}
VALID_FUSION = {"per_layer", "concat"}
VALID_LAYOUTS = {"lf_hf_dual", "slice_native", "multi_tower"}


def resolve_slice_layer_map(
    projection_cfg: dict,
    pooling_cfg: dict,
    num_slices: int,
) -> List[int]:
    """
    Map each slice tower to a ViT block index for pooled features.

    Default: first half of slices → deepest layer, second half → shallowest.
    Example num_slices=4, layers=[3,11] → [11, 11, 3, 3].
    """
    raw = projection_cfg.get("slice_layer_map")
    layers = list(pooling_cfg.get("layers") or [11])
    if not layers:
        raise ValueError("model.pooling.layers vacío — requerido para multi_tower")

    if raw is not None:
        if len(raw) != num_slices:
            raise ValueError(
                f"slice_layer_map length ({len(raw)}) debe igualar num_slices ({num_slices})"
            )
        for layer_idx in raw:
            if layer_idx not in layers:
                raise ValueError(
                    f"slice_layer_map layer {layer_idx} no está en pooling.layers {layers}"
                )
        return [int(x) for x in raw]

    deep = max(layers)
    shallow = min(layers)
    half = num_slices // 2
    return [deep] * half + [shallow] * (num_slices - half)


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_pooling_config(model_config: dict, grain_config: Optional[dict] = None) -> dict:
    """Merge model.pooling with defaults; sync masked.enabled from grain_detection."""
    grain_config = grain_config or {}
    raw = model_config.get("pooling") or {}
    cfg = _deep_merge(DEFAULT_POOLING_CONFIG, raw)

    if "masked" not in raw or "enabled" not in raw.get("masked", {}):
        if "masked_pooling" in grain_config:
            cfg["masked"]["enabled"] = bool(grain_config["masked_pooling"])

    # C-10: saliency_threshold drives masked pooling threshold unless explicitly set.
    saliency = grain_config.get("saliency_threshold")
    if saliency is not None and "threshold" not in raw.get("masked", {}):
        cfg["masked"]["threshold"] = float(saliency)

    return cfg


def resolve_projection_config(model_config: dict) -> dict:
    raw = model_config.get("projection") or {}
    layout = raw.get("layout", "lf_hf_dual")
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"model.projection.layout '{layout}' no soportado. Opciones: {VALID_LAYOUTS}")
    return {"layout": layout, **{k: v for k, v in raw.items() if k != "layout"}}


def validate_pooling_config(pooling: dict, projection_layout: str = "lf_hf_dual") -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    strategy = pooling.get("strategy", "multi_spectral")
    if strategy not in VALID_STRATEGIES:
        errors.append(f"model.pooling.strategy '{strategy}' inválida. Opciones: {sorted(VALID_STRATEGIES)}")

    layers = pooling.get("layers") or []
    if not layers:
        errors.append("model.pooling.layers no puede estar vacío para backbones ViT")

    stats = pooling.get("statistics") or []
    if strategy == "multi_spectral":
        if not stats:
            errors.append("model.pooling.statistics requiere al menos un estadístico")
        invalid = set(stats) - VALID_STATISTICS
        if invalid:
            errors.append(f"model.pooling.statistics contiene valores inválidos: {sorted(invalid)}")

    weighting = pooling.get("masked", {}).get("weighting", "soft")
    if weighting not in VALID_WEIGHTING:
        errors.append(f"model.pooling.masked.weighting '{weighting}' inválido")

    eval_mode = pooling.get("drop_token", {}).get("eval_mode", "off")
    if isinstance(eval_mode, bool):
        eval_mode = "same_as_train" if eval_mode else "off"
    if eval_mode not in VALID_EVAL_MODES:
        errors.append(f"model.pooling.drop_token.eval_mode '{eval_mode}' inválido")

    prob = float(pooling.get("drop_token", {}).get("prob", 0.0))
    if prob > 0 and eval_mode == "off":
        warnings.append(
            "drop_token.prob > 0 con eval_mode=off introduce shift train/eval en pooling"
        )

    fusion = pooling.get("layer_fusion", "per_layer")
    if fusion not in VALID_FUSION:
        errors.append(f"model.pooling.layer_fusion '{fusion}' inválido")

    if projection_layout == "slice_native" and fusion == "per_layer" and len(layers) > 1:
        warnings.append(
            "slice_native con layer_fusion=per_layer y múltiples capas usa cabezal único; "
            "considera layer_fusion=concat o una sola capa"
        )

    if projection_layout == "multi_tower" and fusion == "concat":
        warnings.append(
            "multi_tower requiere layer_fusion=per_layer para acceso por capa a cada torre"
        )

    if strategy == "fsw":
        fsw_cfg = pooling.get("fsw") or {}
        if int(fsw_cfg.get("num_iterations", 3)) < 1:
            errors.append("model.pooling.fsw.num_iterations debe ser >= 1")

    return errors, warnings


def validate_projection_config(
    projection_cfg: dict,
    pooling_cfg: dict,
    embedding_dim: int,
    num_slices: int,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    layout = projection_cfg.get("layout", "lf_hf_dual")

    if layout == "multi_tower":
        if embedding_dim % num_slices != 0:
            errors.append(
                f"multi_tower: embedding_dim ({embedding_dim}) no divisible por "
                f"num_slices ({num_slices})"
            )
        try:
            resolve_slice_layer_map(projection_cfg, pooling_cfg, num_slices)
        except ValueError as exc:
            errors.append(str(exc))

    return errors, warnings


def compute_pooling_output_dim(token_dim: int, strategy: str, statistics: List[str], num_layers: int, fusion: str) -> int:
    """Output feature size before projection."""
    if strategy == "multi_spectral":
        per_layer = token_dim * len(statistics)
    else:
        per_layer = token_dim

    if fusion == "concat":
        return per_layer * max(1, num_layers)
    return per_layer
