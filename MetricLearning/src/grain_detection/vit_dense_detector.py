"""
PollenViTDetector — ViT-dense saliency detector for inference.

Pipeline (default ``coarse_fine``):
    1. Coarse global pass
    2. Supplement discovery (sliding-window + saliency peaks)
    3. Fine verification per candidate
    4. Contextual bbox expansion before classification
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.data.detection_transforms import letterbox_to_square, to_model_tensor
from src.grain_detection.coarse_fine_detection import run_coarse_fine_detection
from src.grain_detection.detected_grain import DetectedGrain
from src.grain_detection.detection_profiles import PROFILES, resolve_grain_detection_config
from src.grain_detection.saliency_grain_extraction import (
    extract_grains_from_saliency,
    grain_extraction_kwargs,
)
from src.models.vit_dense_segmentation import ViTDenseSegmentationModel
from src.utils.detector_resolution_config import resolve_detector_input_size

logger = logging.getLogger(__name__)


class PollenViTDetector:
    """Inference wrapper with the same ``detect(image_bgr)`` contract as PollenGrainDetector."""

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        config = resolve_grain_detection_config(config)
        self.config = config
        self.device = torch.device(
            device or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.input_size = resolve_detector_input_size(
            {"detection_training": config, "grain_detection": config}
        )
        self.adaptive_k = float(config.get("adaptive_k", 0.30))
        self.min_area = int(config.get("min_area", 1000))
        self.max_area = int(config.get("max_area", 80000))
        self.morph_kernel_size = int(config.get("morph_kernel_size", 3))
        self.min_circularity = float(config.get("min_circularity", 0.45))
        self.max_bbox_side_frac = float(config.get("max_bbox_side_frac", 0.35))
        self.max_aspect_ratio = float(config.get("max_aspect_ratio", 2.5))

        strategy = str(config.get("detection_strategy", "coarse_fine")).strip().lower()
        self.detection_strategy = (
            strategy if strategy in ("coarse_fine", "single_pass") else "coarse_fine"
        )

        self.sliding_window_enabled = bool(config.get("sliding_window_enabled", True))
        overlap = max(0.0, min(0.75, float(config.get("sliding_window_overlap", 0.25))))
        self._sw_tile_size = (
            int(config["sliding_window_tile_size"])
            if config.get("sliding_window_tile_size")
            else None
        )
        self._sw_stride = (
            int(config["sliding_window_stride"])
            if config.get("sliding_window_stride") is not None
            else None
        )
        self._sw_overlap = overlap

        self.supplement_min_saliency = float(config.get("supplement_min_saliency", 0.42))
        self.supplement_min_peak = float(config.get("supplement_min_peak", 0.38))
        self.supplement_k_boost = float(config.get("sliding_window_k_boost", 0.05))
        self.verify_crop_padding = float(config.get("verify_crop_padding", 0.45))
        self.verify_min_saliency = float(config.get("verify_min_saliency", 0.32))
        self.verify_min_iou = float(config.get("verify_min_iou", 0.12))
        self.coarse_retain_min_saliency = float(config.get("coarse_retain_min_saliency", 0.36))
        self.detector_bbox_expand_frac = float(config.get("detector_bbox_expand_frac", 0.18))
        self.min_candidate_area = int(config.get("min_candidate_area", 1500))

        self._extraction_kwargs = grain_extraction_kwargs(config)

        sup_profile = str(config.get("supplement_detection_profile", "sensitive")).lower()
        self._supplement_preset = PROFILES.get(sup_profile, PROFILES["sensitive"])

        verify_profile = str(config.get("verify_detection_profile", sup_profile)).lower()
        self._verify_preset = PROFILES.get(verify_profile, self._supplement_preset)

        self.coarse_merge_iou = float(config.get("coarse_merge_iou", 0.28))
        self.coarse_supplement_merge_iou = float(config.get("coarse_supplement_merge_iou", 0.25))
        self.coarse_nms_iou = float(config.get("coarse_nms_iou", 0.32))
        self.peak_min_area = config.get("peak_min_area")
        self.peak_min_area = int(self.peak_min_area) if self.peak_min_area is not None else None
        self.peak_min_side = int(config.get("peak_min_side", 28))
        self.peak_morph_kernel_size = int(config.get("peak_morph_kernel_size", 5))
        reject_area = config.get("reject_max_area_frac")
        self.reject_max_area_frac = float(reject_area) if reject_area is not None else None
        reject_side = config.get("reject_max_side_frac")
        self.reject_max_side_frac = (
            float(reject_side)
            if reject_side is not None
            else float(config.get("max_bbox_side_frac", 0.40))
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_cfg = ckpt.get("config", config)
        if not isinstance(model_cfg, dict):
            model_cfg = config
        self.input_size = resolve_detector_input_size(model_cfg)
        self.model = ViTDenseSegmentationModel.from_config(model_cfg)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_path = str(checkpoint_path)
        logger.info(
            f"[PollenViTDetector] Loaded {Path(checkpoint_path).name} @ {self.input_size}px "
            f"(strategy={self.detection_strategy}, bbox_expand={self.detector_bbox_expand_frac:.0%})"
        )

    @staticmethod
    def _tile_starts(length: int, tile: int, stride: int) -> List[int]:
        if length <= tile:
            return [0]
        starts = list(range(0, length - tile + 1, stride))
        last = length - tile
        if not starts or starts[-1] != last:
            starts.append(last)
        return sorted(set(starts))

    def _resolve_sliding_params(self) -> Tuple[int, int]:
        tile = self._sw_tile_size or self.input_size
        tile = max(self.input_size, tile)
        stride = (
            max(1, self._sw_stride)
            if self._sw_stride is not None
            else max(1, int(tile * (1.0 - self._sw_overlap)))
        )
        return tile, stride

    def _needs_sliding_window(self, h: int, w: int) -> bool:
        if not self.sliding_window_enabled:
            return False
        tile, _ = self._resolve_sliding_params()
        return max(h, w) > tile

    @torch.inference_mode()
    def _predict_saliency_single(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        image_lb, _, meta = letterbox_to_square(image_bgr, None, self.input_size)
        tensor = to_model_tensor(image_lb).unsqueeze(0).to(self.device)
        pred = torch.sigmoid(self.model(tensor))[0, 0].cpu().numpy()

        scale = meta["scale"]
        pad_top = int(meta["pad_top"])
        pad_left = int(meta["pad_left"])
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        crop = pred[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
        saliency = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(saliency, 0.0, 1.0).astype(np.float32)

    @torch.inference_mode()
    def _predict_saliency_sliding(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        tile, stride = self._resolve_sliding_params()
        acc = np.zeros((h, w), dtype=np.float32)
        n_tiles = 0

        for y0 in self._tile_starts(h, tile, stride):
            for x0 in self._tile_starts(w, tile, stride):
                y1 = min(y0 + tile, h)
                x1 = min(x0 + tile, w)
                crop = image_bgr[y0:y1, x0:x1]
                ch, cw = crop.shape[:2]

                if ch < tile or cw < tile:
                    padded = np.full((tile, tile, 3), 128, dtype=np.uint8)
                    padded[:ch, :cw] = crop
                    tile_sal = self._predict_saliency_single(padded)[:ch, :cw]
                else:
                    tile_sal = self._predict_saliency_single(crop)

                acc[y0:y1, x0:x1] = np.maximum(acc[y0:y1, x0:x1], tile_sal)
                n_tiles += 1

        logger.info(
            f"[PollenViTDetector] sliding_window {w}x{h} -> {n_tiles} tiles "
            f"(tile={tile}, stride={stride})"
        )
        return np.clip(acc, 0.0, 1.0)

    def _extract_grains(
        self,
        saliency: np.ndarray,
        image: np.ndarray,
        *,
        k_override: Optional[float] = None,
        preset: Optional[Dict] = None,
    ) -> List[DetectedGrain]:
        p = {**self._extraction_kwargs, **(preset or {})}
        adaptive_k = float(p.pop("adaptive_k", self.adaptive_k))
        return extract_grains_from_saliency(
            saliency,
            image,
            adaptive_k=adaptive_k,
            min_area=int(p.get("min_area", self.min_area)),
            max_area=int(p.get("max_area", self.max_area)),
            max_area_frac=p.get("max_area_frac"),
            morph_kernel_size=int(p.get("morph_kernel_size", self.morph_kernel_size)),
            min_circularity=float(p.get("min_circularity", self.min_circularity)),
            max_bbox_side_frac=float(p.get("max_bbox_side_frac", self.max_bbox_side_frac)),
            max_aspect_ratio=float(p.get("max_aspect_ratio", self.max_aspect_ratio)),
            adaptive_k_override=k_override,
        )

    def _extract_grains_sensitive(
        self,
        saliency: np.ndarray,
        image: np.ndarray,
        *,
        k_override: Optional[float] = None,
    ) -> List[DetectedGrain]:
        return self._extract_grains(
            saliency, image, k_override=k_override, preset=self._supplement_preset
        )

    def _extract_grains_verify(self, saliency: np.ndarray, image: np.ndarray) -> List[DetectedGrain]:
        return self._extract_grains(saliency, image, preset=self._verify_preset)

    def _verify_pass(self, image: np.ndarray) -> Tuple[np.ndarray, List[DetectedGrain]]:
        saliency = self._predict_saliency_single(image)
        return saliency, self._extract_grains_verify(saliency, image)

    def _predict_and_extract(self, image: np.ndarray) -> Tuple[np.ndarray, List[DetectedGrain]]:
        saliency = self._predict_saliency_single(image)
        return saliency, self._extract_grains(saliency, image)

    def _detect_coarse_fine(self, image: np.ndarray) -> Tuple[np.ndarray, List[DetectedGrain]]:
        saliency, grains, _ = run_coarse_fine_detection(
            image,
            coarse_pass_fn=self._predict_saliency_single,
            sliding_saliency_fn=self._predict_saliency_sliding,
            extract_fn=self._extract_grains,
            extract_fn_sensitive=self._extract_grains_sensitive,
            verify_pass_fn=self._verify_pass,
            needs_sliding_fn=self._needs_sliding_window,
            supplement_enabled=self.sliding_window_enabled,
            supplement_min_saliency=self.supplement_min_saliency,
            supplement_min_peak=self.supplement_min_peak,
            verify_crop_padding=self.verify_crop_padding,
            verify_min_saliency=self.verify_min_saliency,
            verify_min_iou=self.verify_min_iou,
            coarse_retain_min_saliency=self.coarse_retain_min_saliency,
            supplement_k_boost=self.adaptive_k + self.supplement_k_boost,
            bbox_expand_frac=self.detector_bbox_expand_frac,
            min_candidate_area=self.min_candidate_area,
            supplement_merge_iou=self.coarse_supplement_merge_iou,
            merge_iou=self.coarse_merge_iou,
            nms_iou=self.coarse_nms_iou,
            peak_min_area=self.peak_min_area,
            peak_min_side=self.peak_min_side,
            peak_morph_kernel_size=self.peak_morph_kernel_size,
            reject_max_area_frac=self.reject_max_area_frac,
            reject_max_side_frac=self.reject_max_side_frac,
        )
        return saliency, grains

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, List[DetectedGrain]]:
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.float32), []

        t0 = time.perf_counter()
        if self.detection_strategy == "coarse_fine":
            saliency, grains = self._detect_coarse_fine(image)
        else:
            saliency, grains = self._predict_and_extract(image)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[PollenViTDetector] {len(grains)} granos en {elapsed:.0f}ms ({self.detection_strategy})"
        )
        return saliency, grains

    def is_ready(self) -> bool:
        return self.model is not None
