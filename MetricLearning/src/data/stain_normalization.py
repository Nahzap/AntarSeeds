"""
Stain normalization for microscopy RGB images (Phase E-02).

- Reinhard et al. (2001): color transfer in LAB space.
- Macenko et al. (ISBI 2009): stain matrix normalization in optical density space.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


class ReinhardStainNormalization:
    """
    Reinhard color normalization (Reinhard et al., 2001).
    Maps image LAB statistics to target mean/std per channel.
    """

    def __init__(self, target_mean: Optional[List[float]] = None, target_std: Optional[List[float]] = None):
        self.target_mean = target_mean or [70.0, 0.0, 0.0]
        self.target_std = target_std or [15.0, 5.0, 5.0]

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        for i in range(3):
            ch = lab[:, :, i]
            src_mean = ch.mean()
            src_std = ch.std() + 1e-6
            lab[:, :, i] = (ch - src_mean) * (self.target_std[i] / src_std) + self.target_mean[i]
            lab[:, :, i] = np.clip(lab[:, :, i], 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


class MacenkoStainNormalization:
    """
    Macenko stain normalization (Macenko et al., ISBI 2009).

    Normalizes H&E-style staining via optical density eigendecomposition.
    Suitable as optional pre-normalize step before ImageNet/dataset Normalize.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.15):
        self.alpha = float(alpha)
        self.beta = float(beta)

    @staticmethod
    def _rgb_to_od(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32) / 255.0
        img = np.maximum(img, 1e-6)
        return -np.log(img)

    @staticmethod
    def _od_to_rgb(od: np.ndarray) -> np.ndarray:
        rgb = np.exp(-od) * 255.0
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        od = self._rgb_to_od(image)
        od_flat = od.reshape(-1, 3)
        od_thresh = self.beta * np.max(od_flat, axis=1)
        keep = od_flat[np.max(od_flat, axis=1) > od_thresh]
        if keep.shape[0] < 10:
            return image

        cov = np.cov(keep.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        v = eigvecs[:, np.argsort(eigvals)[-1]]
        if v[0] < 0:
            v = -v

        proj = keep @ v
        h1 = proj.min()
        h2 = proj.max()
        if abs(h2 - h1) < 1e-6:
            return image

        stain = np.array([[h1, h2], [v[0], v[0]]])
        od_norm = od.copy()
        for i in range(3):
            od_norm[:, :, i] = (od[:, :, i] - h1) / (h2 - h1) * self.alpha + (1 - self.alpha) / 2
        return self._od_to_rgb(od_norm)


def build_stain_normalizer(method: Optional[str], config: Optional[dict] = None):
    if not method or str(method).lower() in ("none", "off", "", "null"):
        return None
    cfg = config or {}
    key = str(method).lower()
    if key == "reinhard":
        tcfg = cfg.get("reinhard") or {}
        return ReinhardStainNormalization(
            target_mean=tcfg.get("target_mean"),
            target_std=tcfg.get("target_std"),
        )
    if key == "macenko":
        mcfg = cfg.get("macenko") or {}
        return MacenkoStainNormalization(
            alpha=float(mcfg.get("alpha", 1.0)),
            beta=float(mcfg.get("beta", 0.15)),
        )
    raise ValueError(f"stain_method no soportado: {method!r} (reinhard|macenko)")
