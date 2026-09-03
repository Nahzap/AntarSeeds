"""Factory helpers for building AnalogyNet from config."""

from __future__ import annotations

from typing import Any, Dict

from src.models.analogy_net import AnalogyNet
from src.models.pooling_config import resolve_pooling_config, resolve_projection_config
from src.utils.config_utils import get_backbone_name


def create_analogy_net_from_config(config: Dict[str, Any]) -> AnalogyNet:
    model_cfg = config["model"]
    grain_cfg = config.get("grain_detection", {})
    sota_cfg = config.get("sota", {})

    pooling_cfg = resolve_pooling_config(model_cfg, grain_cfg)
    projection_cfg = resolve_projection_config(model_cfg)
    loss_params = config.get("training", {}).get("loss", {}).get("params", {}) or {}
    num_slices = int(loss_params.get("num_slices", 4))

    return AnalogyNet(
        backbone=get_backbone_name(config),
        embedding_dim=model_cfg["embedding_dim"],
        pretrained=model_cfg.get("pretrained", True),
        projection_head_config=model_cfg.get("projection_head"),
        use_attention=model_cfg.get("use_attention", False),
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        embedding_space=sota_cfg.get("embedding_space", "euclidean"),
        hyperbolic_curvature=sota_cfg.get("hyperbolic_curvature", 1.0),
        pooling_config=pooling_cfg,
        projection_layout=projection_cfg["layout"],
        grain_config=grain_cfg,
        num_slices=num_slices,
        projection_config=projection_cfg,
        num_unfrozen_blocks=int(model_cfg.get("num_unfrozen_blocks", 2)),
        unfreeze_block_indices=model_cfg.get("unfreeze_block_indices"),
    )
