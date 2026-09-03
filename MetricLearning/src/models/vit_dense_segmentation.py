"""
ViT-dense salient object detection model (DINOv2 encoder + dense decoder).

Ref: Caron et al. (2021) DINO; Oquab et al. (2023) DINOv2;
     Qin et al. (2020) U²-Net SOD formulation (pixel-wise saliency).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.detector_head import DenseSaliencyDecoder
from src.utils.config_utils import get_backbone_name

logger = logging.getLogger(__name__)

_HUB_MAP = {
    "dinov2_vits14": "dinov2_vits14",
    "dinov2_vitb14": "dinov2_vitb14",
    "dinov2_vits14_reg": "dinov2_vits14_reg4",
    "dinov2_vitb14_reg": "dinov2_vitb14_reg4",
}

_EMBED_DIM = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
}


def _load_raw_dinov2(backbone_name: str, pretrained: bool = True) -> nn.Module:
    hub_name = _HUB_MAP.get(backbone_name, backbone_name)
    return torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=pretrained)


def _patch_grid_hw(image_hw: Tuple[int, int], patch_size: int) -> Tuple[int, int]:
    h, w = image_hw
    return h // patch_size, w // patch_size


class ViTDenseSegmentationModel(nn.Module):
    """
    DINOv2 patch encoder + dense saliency decoder for pollen grain localization.

    Partial fine-tuning: last ``num_unfrozen_blocks`` transformer blocks trainable.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        *,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        num_unfrozen_blocks: int = 2,
        patch_size: int = 14,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.patch_size = patch_size
        self.raw_vit = _load_raw_dinov2(backbone_name, pretrained=pretrained)
        embed_dim = _EMBED_DIM.get(backbone_name, getattr(self.raw_vit, "embed_dim", 384))
        self.decoder = DenseSaliencyDecoder(embed_dim)
        self._configure_freeze(freeze_backbone, num_unfrozen_blocks)
        logger.info(
            f"[ViTDenseSeg] backbone={backbone_name}, embed={embed_dim}, "
            f"unfrozen_blocks={num_unfrozen_blocks if freeze_backbone else 'all'}"
        )

    def _configure_freeze(self, freeze_backbone: bool, num_unfrozen_blocks: int) -> None:
        if not freeze_backbone:
            return
        for p in self.raw_vit.parameters():
            p.requires_grad = False
        blocks = getattr(self.raw_vit, "blocks", None)
        if blocks is None or num_unfrozen_blocks <= 0:
            return
        n = min(num_unfrozen_blocks, len(blocks))
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens reshaped to [B, C, Hp, Wp]."""
        feats = self.raw_vit.forward_features(x)
        if isinstance(feats, dict):
            patch_tokens = feats.get("x_norm_patchtokens")
            if patch_tokens is None:
                pre = feats.get("x_prenorm")
                n_reg = getattr(self.raw_vit, "num_register_tokens", 0)
                patch_tokens = pre[:, 1 + n_reg :, :]
        else:
            out = self.raw_vit.get_intermediate_layers(x, n=1, reshape=False, norm=True)
            tokens = out[0] if isinstance(out, (list, tuple)) else out
            n_reg = getattr(self.raw_vit, "num_register_tokens", 0)
            patch_tokens = tokens[:, 1 + n_reg :, :]

        b, n, c = patch_tokens.shape
        side = math.isqrt(n)
        if side * side == n:
            hp = wp = side
        else:
            hp = max(1, x.shape[2] // self.patch_size)
            wp = n // hp
            if hp * wp != n:
                raise RuntimeError(
                    f"Patch count {n} no coincide con rejilla {hp}x{wp} "
                    f"(input {tuple(x.shape)})"
                )
        spatial = patch_tokens.transpose(1, 2).reshape(b, c, hp, wp)
        return spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] with H,W multiples of patch_size
        Returns:
            Saliency [B, 1, H, W]
        """
        spatial = self._patch_tokens(x)
        return self.decoder(spatial, (x.shape[2], x.shape[3]))

    @classmethod
    def from_config(cls, config: dict) -> "ViTDenseSegmentationModel":
        det_cfg = config.get("detection_training", {}) or {}
        model_cfg = config.get("model", {}) or {}
        return cls(
            backbone_name=get_backbone_name(config),
            pretrained=bool(model_cfg.get("pretrained", True)),
            freeze_backbone=bool(det_cfg.get("freeze_backbone", True)),
            num_unfrozen_blocks=int(det_cfg.get("num_unfrozen_blocks", 2)),
            patch_size=int(det_cfg.get("patch_size", 14)),
        )
