"""
DetectorLocalizedGrainDataset — DML crops from active detector localizations.

When ``data.crop_source: detector``, grains are extracted by running the active
detector once per image at index-build time (not per __getitem__).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.data.grain_dataset import SegmentedGrainDataset, IMAGE_EXTENSIONS
from src.grain_detection.detector_registry import load_active_detector, validate_active_detector

logger = logging.getLogger(__name__)


class DetectorLocalizedGrainDataset(SegmentedGrainDataset):
    """
    Same output contract as SegmentedGrainDataset but index built from detector.
    Class labels come from ImageFolder; geometry from active_detector.detect().
    """

    def __init__(
        self,
        root: str,
        config: dict,
        transform: Optional[Callable] = None,
        crop_padding: float = 0.2,
        crop_size: int = 256,
        use_mask: bool = False,
        mask_bg_mode: str = "gray",
        min_saliency: float = 0.0,
        return_metadata: bool = False,
        return_mask: bool = False,
        annotation_root: Optional[str] = None,
        **kwargs,
    ):
        self._full_config = config
        errors = validate_active_detector(config)
        if errors:
            raise ValueError("Detector activo inválido:\n" + "\n".join(errors))
        self._detector = load_active_detector(config)
        super().__init__(
            root=root,
            transform=transform,
            crop_padding=crop_padding,
            crop_size=crop_size,
            use_mask=use_mask,
            mask_bg_mode=mask_bg_mode,
            min_saliency=min_saliency,
            return_metadata=return_metadata,
            return_mask=return_mask,
            annotation_root=annotation_root,
            **kwargs,
        )

    def _build_index(self):
        """Override: scan images and localize grains with active detector."""
        if not self.root.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.root}")

        class_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        skipped_empty = 0
        active = self._full_config.get("grain_detection", {}).get("active_detector", {})
        logger.info(
            f"[DetectorLocalizedGrainDataset] Indexing with "
            f"{active.get('backend', 'u2net')} @ {self.root}"
        )

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                try:
                    _, grains = self._detector.detect(image)
                except Exception as exc:
                    logger.warning(f"[DetLocDataset] detect falló {img_path.name}: {exc}")
                    continue
                if not grains:
                    skipped_empty += 1
                    continue
                img_str = str(img_path)
                for grain in grains:
                    sal = float(getattr(grain, "saliency_prob", 0.0))
                    if sal < self.min_saliency:
                        continue
                    bbox = tuple(grain.bbox)
                    contour = grain.contour
                    self._grain_samples.append(
                        (img_str, class_idx, bbox, sal, contour)
                    )
                    self.samples.append((img_str, class_idx))
                    self.targets.append(class_idx)

        if skipped_empty:
            logger.info(f"[DetectorLocalizedGrainDataset] {skipped_empty} imágenes sin detecciones")
