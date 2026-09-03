"""
ModelGrainDetector — Localización de granos con el best model entrenado (AnalogyNet).

Usa mapas de saliencia derivados del propio modelo (atención ViT / Grad-CAM),
NO U²-Net. U²-Net queda reservado para preprocesamiento (.seg / recortes).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .detected_grain import DetectedGrain
from .saliency_grain_extraction import (
    extract_grains_from_saliency,
    nms_merge_grains,
)

logger = logging.getLogger(__name__)


class ModelGrainDetector:
    """
    Detector de granos basado en el modelo de metric learning entrenado.

    Pipeline: saliency del best model → umbral adaptativo → contornos → NMS.
    """

    def __init__(
        self,
        model: nn.Module,
        transforms,
        device: torch.device,
        config: Optional[Dict] = None,
        slice_classifier=None,
    ):
        config = config or {}
        self.model = model
        self.model.eval()
        self.transforms = transforms
        self.device = device
        self.slice_classifier = slice_classifier

        self.adaptive_k = float(config.get("adaptive_k", 0.30))
        self.min_area = int(config.get("min_area", 200))
        self.max_area = int(config.get("max_area", 200000))
        self.morph_kernel_size = int(config.get("morph_kernel_size", 3))
        self.min_circularity = float(config.get("min_circularity", 0.3))

        logger.info(
            "[ModelGrainDetector] Inicializado — detección vía best model "
            f"(adaptive_k={self.adaptive_k}, min_area={self.min_area})"
        )

    def detect(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[DetectedGrain]]:
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.float32), []

        t0 = time.perf_counter()
        saliency = self._compute_saliency(image)
        grains = extract_grains_from_saliency(
            saliency,
            image,
            adaptive_k=self.adaptive_k,
            min_area=self.min_area,
            max_area=self.max_area,
            morph_kernel_size=self.morph_kernel_size,
            min_circularity=self.min_circularity,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[ModelGrainDetector] {len(grains)} granos detectados en {elapsed:.0f}ms"
        )
        return saliency, grains

    def detect_exhaustive(
        self, image: np.ndarray, *, iou_threshold: float = 0.35
    ) -> Tuple[np.ndarray, List[DetectedGrain]]:
        """Multi-umbral sobre saliency del modelo + fusión NMS."""
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.float32), []

        t0 = time.perf_counter()
        saliency = self._compute_saliency(image)

        merged: List[DetectedGrain] = []
        k_scales = (1.0, 0.65, 0.4)
        circ_scales = (1.0, 0.75, 0.5)
        for k_scale in k_scales:
            for circ_scale in circ_scales:
                merged.extend(
                    extract_grains_from_saliency(
                        saliency,
                        image,
                        adaptive_k=self.adaptive_k,
                        min_area=self.min_area,
                        max_area=self.max_area,
                        morph_kernel_size=self.morph_kernel_size,
                        min_circularity=self.min_circularity,
                        adaptive_k_override=max(0.08, self.adaptive_k * k_scale),
                        min_circularity_override=max(
                            0.12, self.min_circularity * circ_scale
                        ),
                    )
                )

        grains = nms_merge_grains(merged, iou_threshold=iou_threshold)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[ModelGrainDetector] exhaustive: {len(grains)} granos "
            f"(candidatos={len(merged)}) en {elapsed:.0f}ms"
        )
        return saliency, grains

    def _image_to_tensor(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            tensor = self.transforms(rgb)
            if isinstance(tensor, tuple):
                tensor = tensor[0]
        else:
            from torchvision import transforms as T

            tensor = T.ToTensor()(rgb)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Transforms must return a torch.Tensor")
        return tensor.unsqueeze(0)

    def _mean_gradcam_centroid(self) -> Optional[torch.Tensor]:
        if self.slice_classifier is None:
            return None
        n_classes = len(getattr(self.slice_classifier, "class_names", []) or [])
        if n_classes <= 0:
            return None
        centroids = []
        for cls_idx in range(n_classes):
            c = self.slice_classifier.get_target_for_gradcam(cls_idx, self.device)
            centroids.append(c.detach())
        return torch.stack(centroids, dim=0).mean(dim=0)

    def _compute_saliency(self, image_bgr: np.ndarray) -> np.ndarray:
        from src.utils.inference_utils import (
            _is_vit_backbone,
            extract_dino_attention,
            extract_gradient_weighted_attention,
        )
        from src.utils.visual_explainer import create_explainer

        h, w = image_bgr.shape[:2]
        tensor = self._image_to_tensor(image_bgr).to(self.device)

        saliency: np.ndarray
        if _is_vit_backbone(self.model):
            centroid = self._mean_gradcam_centroid()
            if centroid is not None:
                try:
                    saliency = extract_gradient_weighted_attention(
                        self.model, tensor, self.device, centroid
                    )
                except Exception as exc:
                    logger.warning(
                        "[ModelGrainDetector] Grad-attention falló (%s), "
                        "usando CLS attention",
                        exc,
                    )
                    saliency, _ = extract_dino_attention(
                        self.model, tensor, self.device
                    )
            else:
                saliency, _ = extract_dino_attention(self.model, tensor, self.device)
        else:
            explainer = create_explainer(self.model, self.device)
            centroid = self._mean_gradcam_centroid()
            saliency = explainer.generate_heatmap(tensor, centroid=centroid)

        if saliency.shape[:2] != (h, w):
            saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(saliency.astype(np.float32), 0.0, 1.0)
