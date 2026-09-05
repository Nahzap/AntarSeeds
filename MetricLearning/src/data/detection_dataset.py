"""
FullImageDetectionDataset — 1 sample = 1 field image + union saliency GT from .seg.

Task: class-agnostic pollen grain localization (saliency segmentation).
Splits align with DML via ``data.train_dir`` / ``val_dir`` / ``test_dir``.
"""

from __future__ import annotations

import collections
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.detection_transforms import (
    letterbox_to_square,
    rasterize_grain_union_mask,
    to_model_tensor,
    train_augment,
)
from src.grain_detection.annotation_paths import find_existing_seg_path, seg_path_for
from src.grain_detection.seg_format import SegFileReader
from src.utils.detector_resolution_config import resolve_detector_input_size

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class FullImageDetectionDataset(Dataset):
    """
    Image-level dataset for ViT-dense saliency training.

    Each index corresponds to one full microscopy frame with a binary union mask
    derived from ``.seg`` polygon annotations (oracle GT for localization).
    """

    def __init__(
        self,
        root: str,
        *,
        annotation_root: Optional[str] = None,
        input_size: int = 504,
        augment: bool = False,
        aug_config: Optional[dict] = None,
        min_grains: int = 1,
        max_images: int = 0,
        subset_seed: int = 42,
        image_cache_size: int = 8,
        image_cache_max_mb: int = 96,
        max_cached_image_mb: int = 32,
    ):
        self.root = Path(root)
        self.annotation_root = Path(annotation_root) if annotation_root else None
        self.input_size = int(input_size)
        self.augment = augment
        self.aug_config = aug_config or {}
        self.min_grains = min_grains
        self.max_images = int(max_images or 0)
        self.subset_seed = int(subset_seed)
        self.image_cache_size = max(1, int(image_cache_size))
        self.image_cache_max_mb = max(32, int(image_cache_max_mb))
        self.max_cached_image_mb = max(4, int(max_cached_image_mb))
        self._image_cache: collections.OrderedDict[str, Tuple[np.ndarray, List]] = collections.OrderedDict()
        self._image_cache_bytes = 0
        self.h5_path: str | None = None
        self._h5_file = None

        self.samples: List[Dict[str, Any]] = []
        self._build_index()
        self._apply_subset()

        logger.info(
            f"[FullImageDetectionDataset] {len(self.samples)} images @ {self.root} "
            f"(input_size={self.input_size}, augment={self.augment}, "
            f"max_images={self.max_images or 'all'})"
        )

    def _apply_subset(self) -> None:
        if self.max_images <= 0 or len(self.samples) <= self.max_images:
            return
        rng = np.random.default_rng(self.subset_seed)
        indices = sorted(rng.choice(len(self.samples), self.max_images, replace=False).tolist())
        self.samples = [self.samples[i] for i in indices]
        logger.info(
            f"[FullImageDetectionDataset] Subset: {len(self.samples)} imágenes "
            f"(max_images={self.max_images}, seed={self.subset_seed})"
        )

    def _resolve_seg_path(self, class_dir: Path, img_path: Path) -> Path:
        if self.annotation_root is not None:
            existing = find_existing_seg_path(img_path, self.annotation_root)
            return existing or seg_path_for(img_path, self.annotation_root)
        return img_path.with_suffix(".seg")

    def _build_index(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.root}")

        skipped_no_seg = 0
        skipped_empty = 0

        for class_dir in sorted(d for d in self.root.iterdir() if d.is_dir()):
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                seg_path = self._resolve_seg_path(class_dir, img_path)
                if not seg_path.exists():
                    skipped_no_seg += 1
                    continue
                try:
                    grains = SegFileReader.read_grains(str(seg_path))
                except Exception as exc:
                    logger.warning(f"[DetDataset] Error leyendo {seg_path.name}: {exc}")
                    continue
                if len(grains) < self.min_grains:
                    skipped_empty += 1
                    continue
                self.samples.append(
                    {
                        "image_path": str(img_path),
                        "seg_path": str(seg_path),
                        "class_name": class_dir.name,
                        "num_grains": len(grains),
                        "grains": grains,
                    }
                )

        if skipped_no_seg:
            logger.info(f"[DetDataset] {skipped_no_seg} imágenes sin .seg omitidas")
        if skipped_empty:
            logger.info(f"[DetDataset] {skipped_empty} imágenes sin granos omitidas")

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None

    def _read_from_h5(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._h5_file is None:
            import h5py

            self._h5_file = h5py.File(self.h5_path, "r")
        image_bgr = self._h5_file["images"][idx]
        mask_u8 = self._h5_file["masks"][idx]
        return np.asarray(image_bgr), np.asarray(mask_u8, dtype=np.uint8)

    def _load_letterboxed_pair(self, idx: int, meta: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if self.h5_path:
            return self._read_from_h5(idx)
        image, grains = self._load_image_and_grains(meta)
        mask = rasterize_grain_union_mask(grains, image.shape[:2])
        image_lb, mask_lb, _ = letterbox_to_square(image, mask, self.input_size)
        mask_u8 = (np.clip(mask_lb, 0.0, 1.0) * 255).astype(np.uint8)
        return image_lb, mask_u8

    def get_stats(self) -> Dict[str, Any]:
        if not self.samples:
            return {"num_images": 0}
        grains = [s["num_grains"] for s in self.samples]
        multi = sum(1 for g in grains if g > 1)
        return {
            "num_images": len(self.samples),
            "total_grains": int(sum(grains)),
            "grains_per_image_mean": float(np.mean(grains)),
            "grains_per_image_max": int(max(grains)),
            "multi_grain_images": multi,
            "multi_grain_pct": 100.0 * multi / len(self.samples),
        }

    def _cache_put(self, key: str, image: np.ndarray, grains: List) -> Tuple[np.ndarray, List]:
        nbytes = image.nbytes
        if nbytes > self.max_cached_image_mb * 1024 * 1024:
            return image, grains
        while (
            self._image_cache
            and (
                len(self._image_cache) >= self.image_cache_size
                or self._image_cache_bytes + nbytes > self.image_cache_max_mb * 1024 * 1024
            )
        ):
            _, evicted = self._image_cache.popitem(last=False)
            self._image_cache_bytes -= evicted[0].nbytes
        self._image_cache[key] = (image, grains)
        self._image_cache_bytes += nbytes
        return image, grains

    def _load_image_and_grains(self, meta: Dict[str, Any]) -> Tuple[np.ndarray, List]:
        key = meta["image_path"]
        if key in self._image_cache:
            self._image_cache.move_to_end(key)
            image, grains = self._image_cache[key]
            return image.copy(), grains
        image = cv2.imread(key)
        if image is None:
            raise IOError(f"No se pudo leer: {key}")
        grains = meta.get("grains")
        if grains is None:
            grains = SegFileReader.read_grains(meta["seg_path"])
        return self._cache_put(key, image, grains)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.samples[idx]
        image_lb, mask_u8 = self._load_letterboxed_pair(idx, meta)
        mask = mask_u8.astype(np.float32) / 255.0

        if self.augment:
            image_lb, mask = train_augment(image_lb, mask, self.aug_config)

        image_t = to_model_tensor(image_lb)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return {
            "image": image_t,
            "saliency_gt": mask_t,
            "image_path": meta["image_path"],
            "num_grains": meta["num_grains"],
        }
