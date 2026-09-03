"""
Configurable spatial pooling over ViT patch tokens.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pooling_config import compute_pooling_output_dim

logger = logging.getLogger(__name__)


class PoolingModule(nn.Module):
    """
    Aggregates patch tokens from one or more transformer layers.

    Strategies: mean, max, gem, multi_spectral, attention.
    Supports masked (soft/hard) pooling and optional DropToken regularization.
    """

    def __init__(self, config: dict, token_dim: int, patch_size: int = 14):
        super().__init__()
        self.config = config
        self.token_dim = token_dim
        self.patch_size = patch_size

        self.strategy = config.get("strategy", "multi_spectral")
        self.statistics: List[str] = list(config.get("statistics") or ["mean", "max", "std"])
        self.mask_threshold = float(config.get("masked", {}).get("threshold", 0.01))
        self.mask_weighting = config.get("masked", {}).get("weighting", "soft")
        self.drop_enabled = bool(config.get("drop_token", {}).get("enabled", True))
        self.drop_prob = float(config.get("drop_token", {}).get("prob", 0.0))
        drop_eval = config.get("drop_token", {}).get("eval_mode", "off")
        if isinstance(drop_eval, bool):
            drop_eval = "same_as_train" if drop_eval else "off"
        self.drop_eval_mode = drop_eval
        self.layer_fusion = config.get("layer_fusion", "per_layer")

        gem_cfg = config.get("gem") or {}
        self.gem_eps = float(gem_cfg.get("eps", 1e-6))
        self.gem_learnable = bool(gem_cfg.get("learnable", False))
        init_p = float(gem_cfg.get("p", 3.0))
        if self.gem_learnable:
            self.gem_p = nn.Parameter(torch.tensor(init_p))
        else:
            self.register_buffer("gem_p", torch.tensor(init_p), persistent=False)

        fsw_cfg = config.get("fsw") or {}
        self.fsw_iterations = int(fsw_cfg.get("num_iterations", 3))
        self.fsw_temperature = float(fsw_cfg.get("temperature", 1.0))

        if self.strategy == "attention":
            self.attn_score = nn.Linear(token_dim, 1, bias=False)
        else:
            self.attn_score = None

        num_layers = len(config.get("layers") or [1])
        self.output_dim = compute_pooling_output_dim(
            token_dim, self.strategy, self.statistics, num_layers, self.layer_fusion
        )

    def build_patch_mask(
        self,
        spatial_mask: Optional[torch.Tensor],
        batch_size: int,
        num_patches: int,
        image_hw: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        if spatial_mask is None:
            return torch.ones((batch_size, num_patches), device=device)

        h_patches = image_hw[0] // self.patch_size
        w_patches = image_hw[1] // self.patch_size
        m = spatial_mask.unsqueeze(1).float()
        m = F.adaptive_avg_pool2d(m, (h_patches, w_patches))
        patch_mask = m.reshape(batch_size, -1)

        if self.mask_weighting == "hard":
            patch_mask = (patch_mask > self.mask_threshold).float()
        else:
            patch_mask = torch.where(
                patch_mask > self.mask_threshold,
                patch_mask,
                torch.zeros_like(patch_mask),
            )
        return patch_mask

    def _fsw_pool(
        self,
        tokens: torch.Tensor,
        patch_mask: torch.Tensor,
        active_count: torch.Tensor,
    ) -> torch.Tensor:
        """
        FSW-inspired iterative prototype pooling (Amir & Dym, ICLR 2025).

        Uses internal soft-assignment refinement; optional ``fswlib`` can be
        wired in when installed (see pooling.fsw.use_external).
        """
        fsw_cfg = self.config.get("fsw") or {}
        if fsw_cfg.get("use_external", False):
            try:
                import fswlib  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "model.pooling.strategy=fsw with fsw.use_external=true "
                    "requires: pip install fswlib"
                ) from exc
            raise NotImplementedError(
                "External fswlib integration hook reserved; use use_external=false "
                "for built-in iterative FSW approximation."
            )

        pm = patch_mask.unsqueeze(-1)
        proto = (tokens * pm).sum(dim=1) / active_count
        for _ in range(max(1, self.fsw_iterations)):
            dist_sq = ((tokens - proto.unsqueeze(1)) ** 2 * pm).sum(dim=-1)
            dist_sq = dist_sq.masked_fill(patch_mask <= 0, float("inf"))
            weights = F.softmax(-dist_sq / max(self.fsw_temperature, 1e-6), dim=1)
            weights = weights.unsqueeze(-1) * pm
            denom = weights.sum(dim=1).clamp(min=1e-6)
            proto = (tokens * weights).sum(dim=1) / denom
        return proto

    def _apply_drop_token(self, patch_mask: torch.Tensor) -> torch.Tensor:
        apply_drop = self.drop_enabled and self.drop_prob > 0
        if not apply_drop:
            return patch_mask
        if not self.training and self.drop_eval_mode != "same_as_train":
            return patch_mask
        random_mask = torch.empty_like(patch_mask).bernoulli_(1.0 - self.drop_prob)
        return patch_mask * random_mask

    def _pool_single_layer(self, tokens: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        B, N, D = tokens.shape
        pm = patch_mask.unsqueeze(-1)
        active_count = patch_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        parts = []

        if self.strategy in ("mean", "multi_spectral") and (
            self.strategy == "mean" or "mean" in self.statistics
        ):
            mean = (tokens * pm).sum(dim=1) / active_count
            if self.strategy == "mean":
                return mean
            parts.append(mean)

        if self.strategy in ("max", "multi_spectral") and (
            self.strategy == "max" or "max" in self.statistics
        ):
            neg_inf = torch.tensor(-float("inf"), device=tokens.device, dtype=tokens.dtype)
            masked = torch.where(pm > 0, tokens, neg_inf)
            max_val, _ = masked.max(dim=1)
            if self.strategy == "max":
                return max_val
            parts.append(max_val)

        if self.strategy == "multi_spectral" and "std" in self.statistics:
            mean = parts[0] if parts else (tokens * pm).sum(dim=1) / active_count
            var = (((tokens - mean.unsqueeze(1)) ** 2) * pm).sum(dim=1) / active_count
            parts.append(torch.sqrt(var + 1e-6))

        if self.strategy == "gem":
            clamped = tokens.clamp(min=self.gem_eps)
            powered = clamped.pow(self.gem_p)
            gem = ((powered * pm).sum(dim=1) / active_count).pow(1.0 / self.gem_p.clamp(min=self.gem_eps))
            return gem

        if self.strategy == "attention":
            scores = self.attn_score(tokens).squeeze(-1)
            scores = scores.masked_fill(patch_mask <= 0, float("-inf"))
            weights = F.softmax(scores, dim=1).unsqueeze(-1)
            return (tokens * weights).sum(dim=1)

        if self.strategy == "fsw":
            return self._fsw_pool(tokens, patch_mask, active_count)

        if not parts:
            raise ValueError(f"Estrategia de pooling sin salida: {self.strategy}")
        return torch.cat(parts, dim=1)

    def forward(
        self,
        layer_tokens: Dict[int, torch.Tensor],
        spatial_mask: Optional[torch.Tensor] = None,
        image_hw: Optional[tuple] = None,
    ) -> torch.Tensor | tuple:
        """
        Args:
            layer_tokens: {layer_idx: (B, N, D)}
            spatial_mask: optional (B, H, W)
            image_hw: (H, W) tensor spatial size

        Returns:
            per_layer fusion: tuple of per-layer vectors (len = num layers)
            concat fusion: single vector
        """
        if not layer_tokens:
            raise ValueError("layer_tokens vacío")

        first = next(iter(layer_tokens.values()))
        B, N, _ = first.shape
        device = first.device
        if image_hw is None:
            side = int(N ** 0.5)
            image_hw = (side * self.patch_size, side * self.patch_size)

        patch_mask = self.build_patch_mask(spatial_mask, B, N, image_hw, device)
        patch_mask = self._apply_drop_token(patch_mask)

        pooled = []
        for layer_idx in sorted(layer_tokens.keys()):
            feat = self._pool_single_layer(layer_tokens[layer_idx], patch_mask)
            pooled.append(feat)

        if self.layer_fusion == "concat":
            return torch.cat(pooled, dim=1)
        if len(pooled) == 1:
            return pooled[0]
        return tuple(pooled)
