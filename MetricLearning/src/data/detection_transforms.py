"""Letterbox resize + ImageNet normalization for detector training."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_to_square(
    image: np.ndarray,
    mask: Optional[np.ndarray],
    target_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, float]]:
    """
    Resize preserving aspect ratio; pad to ``target_size`` square.

    Returns image BGR, mask (float 0-1), and scale metadata for inverse mapping.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w, 1)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_r = None

    pad_top = (target_size - new_h) // 2
    pad_left = (target_size - new_w) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_right = target_size - new_w - pad_left

    image_out = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(128, 128, 128),
    )
    if mask_r is not None:
        mask_out = cv2.copyMakeBorder(
            mask_r, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=0,
        )
    else:
        mask_out = None

    meta = {
        "scale": scale,
        "pad_top": float(pad_top),
        "pad_left": float(pad_left),
        "orig_h": float(h),
        "orig_w": float(w),
    }
    return image_out, mask_out, meta


def rasterize_grain_union_mask(
    grains: list,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Build binary union mask from .seg grain contours/bboxes."""
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for grain in grains:
        contour = grain.get("contour")
        if contour is not None and len(contour) >= 3:
            cv2.fillPoly(mask, [np.asarray(contour, dtype=np.int32)], 255)
        else:
            x, y, bw, bh = grain["bbox"]
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
    return (mask > 0).astype(np.float32)


def to_model_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 → normalized RGB float tensor [3,H,W]."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def train_augment(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    aug_cfg: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lightweight paired augmentations for detection training."""
    if aug_cfg.get("random_horizontal_flip", 0) and np.random.rand() < float(aug_cfg["random_horizontal_flip"]):
        image_bgr = cv2.flip(image_bgr, 1)
        mask = cv2.flip(mask, 1)
    if aug_cfg.get("random_vertical_flip", 0) and np.random.rand() < float(aug_cfg["random_vertical_flip"]):
        image_bgr = cv2.flip(image_bgr, 0)
        mask = cv2.flip(mask, 0)
    rot = int(aug_cfg.get("random_rotation", 0) or 0)
    if rot > 0:
        angle = np.random.uniform(-rot, rot)
        h, w = image_bgr.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image_bgr = cv2.warpAffine(image_bgr, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    return image_bgr, mask
