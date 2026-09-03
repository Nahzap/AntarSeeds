"""
Mask-aware transform helpers for on-the-fly grain crops.

Albumentations applies the same geometric ops to image and mask when
additional_targets={'mask': 'mask'} is set on the Compose pipeline.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch


def to_mask_uint8_numpy(mask) -> np.ndarray:
    """Binary mask as H×W uint8 for Albumentations (accepts ndarray or Tensor)."""
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask)
    return ((m > 0).astype(np.uint8)) * 255


def to_mask_float_tensor(mask) -> torch.Tensor:
    """Binary mask as float32 tensor H×W (Albumentations may return Tensor after ToTensorV2)."""
    if isinstance(mask, torch.Tensor):
        return (mask > 0).float().squeeze()
    return torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))


def apply_transform(
    transform: Any,
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Run a train/val transform on RGB crop; optionally keep mask aligned.

    Returns:
        image tensor, or (image tensor, mask tensor) when mask is provided.
    """
    if mask is not None:
        mask_u8 = to_mask_uint8_numpy(mask)
        if hasattr(transform, "apply_with_mask"):
            return transform.apply_with_mask(image, mask_u8)

        pipeline = getattr(transform, "transform", None)
        if pipeline is not None:
            out = pipeline(image=image, mask=mask_u8)
            return out["image"], to_mask_float_tensor(out["mask"])

    if hasattr(transform, "__call__"):
        return transform(image)

    raise TypeError(f"Unsupported transform type: {type(transform)}")
