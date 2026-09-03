"""DataLoader factory for ViT-dense detector training."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.data.detection_dataset import FullImageDetectionDataset
from src.data.detector_h5_cache import build_detector_h5_cache
from src.data.dataloader_utils import (
    apply_ram_safe_dataloader_limits,
    auto_tune_resources,
    dispose_dataloader_workers,
)
from src.utils.detector_resolution_config import resolve_detector_input_size

logger = logging.getLogger(__name__)


def _collate_detection(batch):
    images = torch.stack([b["image"] for b in batch])
    masks = torch.stack([b["saliency_gt"] for b in batch])
    return {
        "image": images,
        "saliency_gt": masks,
        "image_path": [b["image_path"] for b in batch],
        "num_grains": [b["num_grains"] for b in batch],
    }


def _dataloader_worker_init(worker_id: int) -> None:
    import numpy as np

    np.random.seed((np.random.get_state()[1][0] + worker_id) % (2**32 - 1))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass
    try:
        import torch as _torch

        _torch.set_num_threads(1)
    except ImportError:
        pass


class DetectorDataEngine:
    """Builds train/val/test loaders for full-image saliency detection."""

    def __init__(
        self,
        config: dict,
        train_dataset,
        val_dataset,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_dataset=None,
        test_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    @classmethod
    def from_config(cls, config: dict) -> "DetectorDataEngine":
        config = dict(config)
        auto_tune_resources(config, log=logger)
        apply_ram_safe_dataloader_limits(config, logger=logger)

        data_cfg = config["data"]
        det_cfg = config.get("detection_training", {}) or {}
        grain_cfg = config.get("grain_detection", {}) or {}
        input_size = resolve_detector_input_size(config)
        annotation_root = data_cfg.get("annotation_root")
        aug_cfg = det_cfg.get("augmentation", {}) or {}

        seed = int(config.get("reproducibility", {}).get("seed", 42))
        max_train = int(det_cfg.get("max_train_images", 0) or 0)
        max_val = int(det_cfg.get("max_val_images", 0) or 0)

        cache_kwargs = {
            "image_cache_size": int(grain_cfg.get("image_cache_size", 8)),
            "image_cache_max_mb": int(grain_cfg.get("image_cache_max_mb", 96)),
            "max_cached_image_mb": int(grain_cfg.get("max_cached_image_mb", 32)),
        }

        train_ds = FullImageDetectionDataset(
            data_cfg["train_dir"],
            annotation_root=annotation_root,
            input_size=input_size,
            augment=True,
            aug_config=aug_cfg,
            max_images=max_train,
            subset_seed=seed,
            **cache_kwargs,
        )
        val_ds = FullImageDetectionDataset(
            data_cfg["val_dir"],
            annotation_root=annotation_root,
            input_size=input_size,
            augment=False,
            max_images=max_val,
            subset_seed=seed + 1,
            **cache_kwargs,
        )
        if len(train_ds) == 0:
            raise RuntimeError(
                f"No hay imágenes con .seg en {data_cfg['train_dir']}. "
                "Ejecute Análisis de Contornos primero."
            )

        test_ds = None
        test_dir = data_cfg.get("test_dir")
        if test_dir and Path(test_dir).exists():
            test_ds = FullImageDetectionDataset(
                test_dir,
                annotation_root=annotation_root,
                input_size=input_size,
                augment=False,
                **cache_kwargs,
            )
            if len(test_ds) == 0:
                test_ds = None

        use_h5 = bool(det_cfg.get("use_h5_cache", True))
        if use_h5:
            build_detector_h5_cache(train_ds, "train")
            build_detector_h5_cache(val_ds, "val")
            if test_ds is not None:
                build_detector_h5_cache(test_ds, "test")
        else:
            logger.info("[DetectorDataEngine] use_h5_cache=false — lectura on-the-fly desde disco")

        batch_size = int(det_cfg.get("batch_size", 4))
        val_batch = int(det_cfg.get("val_batch_size", 0) or batch_size)
        workers = int(det_cfg.get("num_workers", data_cfg.get("num_workers", 4)))
        val_workers = int(data_cfg.get("val_num_workers", 0))
        if val_workers < 0:
            val_workers = max(0, workers // 2)
        pin_memory = bool(data_cfg.get("pin_memory", True))
        prefetch = max(1, int(data_cfg.get("prefetch_factor", 2)))
        persistent = workers > 0

        common_loader_kwargs = {
            "pin_memory": pin_memory,
            "collate_fn": _collate_detection,
        }
        if workers > 0:
            common_loader_kwargs.update(
                {
                    "num_workers": workers,
                    "persistent_workers": persistent,
                    "prefetch_factor": prefetch,
                    "worker_init_fn": _dataloader_worker_init,
                }
            )
        else:
            common_loader_kwargs["num_workers"] = 0

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **common_loader_kwargs,
        )

        val_kwargs = dict(common_loader_kwargs)
        if val_workers > 0:
            val_kwargs.update(
                {
                    "num_workers": val_workers,
                    "persistent_workers": val_workers > 0,
                    "prefetch_factor": prefetch,
                }
            )
        else:
            val_kwargs["num_workers"] = 0
            val_kwargs.pop("persistent_workers", None)
            val_kwargs.pop("prefetch_factor", None)
            val_kwargs.pop("worker_init_fn", None)

        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch,
            shuffle=False,
            **val_kwargs,
        )
        test_loader = None
        if test_ds is not None:
            test_loader = DataLoader(
                test_ds,
                batch_size=val_batch,
                shuffle=False,
                **val_kwargs,
            )

        logger.info(
            f"[DetectorDataEngine] train={len(train_ds)} val={len(val_ds)} "
            f"test={len(test_ds) if test_ds else 0} | input_size={input_size} | "
            f"workers train={workers} val={val_workers} prefetch={prefetch} | "
            f"h5_cache={use_h5}"
        )
        return cls(config, train_ds, val_ds, train_loader, val_loader, test_ds, test_loader)

    def release_workers(self) -> None:
        for loader in (self.train_loader, self.val_loader, self.test_loader):
            dispose_dataloader_workers(loader)
        for ds in (self.train_dataset, self.val_dataset, self.test_dataset):
            if ds is not None and hasattr(ds, "close"):
                ds.close()
