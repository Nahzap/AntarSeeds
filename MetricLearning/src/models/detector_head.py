"""
Dense decoder head for ViT patch-token saliency maps.

Maps spatial patch features to a full-resolution saliency probability map
via lightweight convolutions and bilinear upsampling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseSaliencyDecoder(nn.Module):
    """Patch-grid → dense saliency logits (sigmoid applied at inference)."""

    def __init__(self, in_channels: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, patch_map: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        """
        Args:
            patch_map: [B, C, Hp, Wp] spatial patch tokens
            target_hw: (H, W) output resolution
        Returns:
            Saliency logits [B, 1, H, W] (apply sigmoid for probabilities)
        """
        x = self.net(patch_map)
        x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return x
