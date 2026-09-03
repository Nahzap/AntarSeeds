"""
GrainDataEngine — unified on-the-fly data pipeline (train + val).

Images stay on disk (symlinks / ImageFolder layout unchanged).
Labels (.seg) are read at index build time; crops are built per __getitem__.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import cv2
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import DataLoader

from src.data.grain_dataset import SegmentedGrainDataset
from src.data.dataloader_utils import (
    apply_ram_safe_dataloader_limits,
    auto_tune_resources,
    dispose_dataloader_workers,
    estimate_worker_ram_mb,
)
from src.data.transforms_enhanced import get_train_transforms, get_val_transforms
from src.utils.config_utils import get_backbone_name
from src.utils.resolution_config import resolve_effective_image_size

logger = logging.getLogger(__name__)


def _dataloader_worker_init(worker_id: int):
    import os
    import numpy as np

    # Prevent identical noise backgrounds across all data workers
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


def _build_h5_cache(dataset, prefix: str):
    import os
    import h5py
    import tempfile
    import atexit
    import hashlib
    import glob
    import signal
    from tqdm import tqdm
    from collections import defaultdict
    import numpy as np
    import concurrent.futures

    if len(dataset) == 0:
        return

    # 1. Config Hash para identificar si la configuracion cambio
    # Nota: omitimos dataset.mask_bg_mode porque esa politica se aplica on-the-fly
    # en __getitem__ y no altera los pixeles crudos guardados en el archivo .h5
    config_str = f"{dataset.root}_{len(dataset)}_{dataset.crop_size}_{dataset.crop_padding}_{dataset.use_mask}"
    config_hash = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:12]
    
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(dataset.root)), "cache")
    os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, f"metriclearning_cache_{prefix}_{config_hash}.h5")
    
    # Limpieza: borrar cachés anteriores si el hash cambió para no confundir ni acumular basura
    import glob
    for old_cache in glob.glob(os.path.join(temp_dir, f"metriclearning_cache_{prefix}_*.h5")):
        if old_cache != path:
            try:
                os.remove(old_cache)
                logger.warning(f"[{prefix.upper()}] Nueva configuración detectada. Eliminando caché antiguo para evitar confusiones: {old_cache}")
            except Exception:
                pass

    dataset.h5_path = path

    # 2. Reutilizacion Inteligente (Cache Hit)
    if os.path.exists(path):
        try:
            with h5py.File(path, 'r') as f:
                if 'crops' in f and f['crops'].shape == (len(dataset), dataset.crop_size, dataset.crop_size, 4):
                    logger.info(f"[{prefix.upper()}] CACHE HIT! Reutilizando {path} instantáneamente.")
                    return
        except Exception:
            logger.warning(f"[{prefix.upper()}] Caché {path} corrupto. Reconstruyendo...")
            try:
                os.remove(path)
            except Exception:
                pass

    logger.info(f"[{prefix.upper()}] Compilando HDF5 Cache ({len(dataset)} granos) en {path}...")
    
    image_to_indices = defaultdict(list)
    for idx in range(len(dataset._grain_samples)):
        img_path = dataset._grain_samples[idx][0]
        image_to_indices[img_path].append(idx)
        
    def process_image(img_path, indices):
        try:
            image, scale = dataset._load_image(img_path)
            results = []
            for idx in indices:
                _, _, bbox, saliency, contour = dataset._grain_samples[idx]
                bbox, contour = dataset._scale_geometry(bbox, contour, scale)
                grain = {"bbox": bbox, "contour": contour}
                
                crop = dataset._extract_crop(image, grain)
                if crop.shape[:2] != (dataset.crop_size, dataset.crop_size):
                    crop = cv2.resize(crop, (dataset.crop_size, dataset.crop_size), interpolation=cv2.INTER_CUBIC)
                    
                if dataset.use_mask and contour is not None:
                    mask_crop = dataset._generate_mask(grain, image.shape[:2])
                    if mask_crop.shape[:2] != (dataset.crop_size, dataset.crop_size):
                        mask_crop = cv2.resize(mask_crop, (dataset.crop_size, dataset.crop_size), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_crop = np.full((dataset.crop_size, dataset.crop_size), 255, dtype=np.uint8)
                    
                out = np.empty((dataset.crop_size, dataset.crop_size, 4), dtype=np.uint8)
                out[:,:,:3] = crop
                out[:,:,3] = mask_crop
                results.append((idx, out))
            return results
        except Exception as e:
            logger.error(f"Error caching {img_path}: {e}")
            return []

    with h5py.File(path, 'w') as f:
        dset = f.create_dataset(
            'crops', 
            shape=(len(dataset), dataset.crop_size, dataset.crop_size, 4),
            dtype=np.uint8,
            chunks=(1, dataset.crop_size, dataset.crop_size, 4),
            compression='lzf'
        )
        
        items = list(image_to_indices.items())
        chunk_size = 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            for i in tqdm(range(0, len(items), chunk_size), desc=f"Caching {prefix}"):
                chunk = items[i:i + chunk_size]
                futures = {executor.submit(process_image, p, idxs): (p, idxs) for p, idxs in chunk}
                for future in concurrent.futures.as_completed(futures):
                    for idx, out in future.result():
                        dset[idx] = out

    # Clear zombie memory (raw contours) from grain samples after HDF5 is successfully built
    for i in range(len(dataset._grain_samples)):
        lst = list(dataset._grain_samples[i])
        lst[4] = None  # contour
        dataset._grain_samples[i] = tuple(lst)


def resolve_dataloader_workers(config: dict) -> Tuple[int, int]:
    """
    Resolve train/val worker counts from config.

    val_num_workers:
      -1 (default): same pool size as train (shared regime)
       0: main-process val only
      >0: explicit val worker count (capped at train workers)
    """
    data_config = config["data"]
    train_workers = int(data_config.get("num_workers", 2))
    if train_workers < 0:
        raise ValueError(f"data.num_workers must be >= 0, got {train_workers}")

    val_raw = data_config.get("val_num_workers", -1)
    if val_raw is None:
        val_raw = -1
    val_workers = int(val_raw)
    if val_workers < 0:
        val_workers = train_workers
    val_workers = max(0, min(val_workers, train_workers))
    return train_workers, val_workers


def grain_collate_fn(batch):
    """Stack batches; resize masks to image spatial size if needed."""
    if not batch:
        raise ValueError("Empty batch in grain_collate_fn")

    if len(batch[0]) == 3:
        images, labels, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.long)
        masks = torch.stack(masks, dim=0)
        if masks.shape[-2:] != images.shape[-2:]:
            masks = F.interpolate(
                masks.unsqueeze(1).float(),
                size=images.shape[-2:],
                mode="nearest",
            ).squeeze(1)
        return images, labels, masks

    images, labels = zip(*batch)
    return torch.stack(images, dim=0), torch.as_tensor(labels, dtype=torch.long)


def _loader_policy(config: dict) -> dict:
    data_cfg = config.get("data", {})
    return data_cfg.get("loader_policy", {}) or {}


def _grain_dataset_kwargs(config: dict, transform, mask_bg_mode: str) -> dict:
    grain_cfg = config.get("grain_detection", {})
    data_cfg = config.get("data", {})
    image_size = resolve_effective_image_size(config)
    annotation_root = data_cfg.get("annotation_root", "data/annotations")

    return {
        "transform": transform,
        "crop_padding": float(grain_cfg.get("crop_padding", 0.2)),
        "crop_size": image_size,
        "use_mask": bool(grain_cfg.get("use_mask", True)),
        "mask_bg_mode": mask_bg_mode,
        "min_saliency": float(grain_cfg.get("min_saliency", 0.0)),
        "return_mask": bool(grain_cfg.get("masked_pooling", False)),
        "annotation_root": annotation_root,
        "image_cache_size": int(grain_cfg.get("image_cache_size", 4)),
        "image_cache_max_mb": int(grain_cfg.get("image_cache_max_mb", 48)),
        "max_cached_image_mb": int(grain_cfg.get("max_cached_image_mb", 12)),
    }


class GrainDataEngine:
    """
    Builds train/val datasets and DataLoaders with a shared worker policy.
    Supports pausing train prefetch during validation (exclusive phases).
    """

    def __init__(
        self,
        config: dict,
        train_dataset,
        val_dataset,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_workers: int,
        val_workers: int,
        test_dataset: Optional[Any] = None,
        test_loader: Optional[DataLoader] = None,
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_workers = train_workers
        self.val_workers = val_workers
        self._policy = _loader_policy(config)
        self._exclusive_val = bool(
            self._policy.get("exclusive_train_workers_during_val", True)
        )

    @classmethod
    def from_config(cls, config: dict) -> "GrainDataEngine":
        logger = logging.getLogger(__name__)
        auto_tune_resources(config, logger)
        ram_est = apply_ram_safe_dataloader_limits(config, logger)
        data_config = config["data"]
        aug_config = config["augmentation"]
        transform_config = {**aug_config, **data_config}
        aug_mode = config.get("sota", {}).get("augmentation_mode", "full")
        backbone_name = get_backbone_name(config)
        grain_cfg = config.get("grain_detection", {})

        train_transforms = get_train_transforms(
            transform_config, mode=aug_mode, backbone_name=backbone_name
        )
        val_transforms = get_val_transforms(transform_config, backbone_name=backbone_name)

        train_workers, val_workers = resolve_dataloader_workers(config)
        policy = _loader_policy(config)
        persistent = bool(policy.get("persistent_workers", True))
        prefetch_factor = int(data_config.get("prefetch_factor", 2))
        if train_workers > 0 or val_workers > 0:
            prefetch_factor = max(1, min(prefetch_factor, 4))
        pin_memory = bool(data_config.get("pin_memory", True))
        batch_size = int(data_config["batch_size"])
        val_batch_size = int(data_config.get("val_batch_size", 0) or batch_size)
        val_batch_size = max(batch_size, val_batch_size)
        use_grain_collate = False

        use_grain_collate = bool(grain_cfg.get("masked_pooling", False))
        train_mask_bg = str(grain_cfg.get("train_mask_bg_mode", "gray"))
        val_mask_bg = str(grain_cfg.get("val_mask_bg_mode", "gray"))

        logger.info(
            f"[GRAIN MODE] On-the-fly crops from .seg - train: {data_config['train_dir']}"
        )
        crop_source = str(data_config.get("crop_source", "seg")).strip().lower()
        if crop_source == "detector":
            from src.data.detector_localized_grain_dataset import DetectorLocalizedGrainDataset
            from src.grain_detection.detector_registry import get_active_detector_config

            active = get_active_detector_config(config)
            logger.info(
                f"[GRAIN MODE] crop_source=detector | active={active.get('backend')} "
                f"checkpoint={active.get('checkpoint')}"
            )
            DatasetCls = DetectorLocalizedGrainDataset
        else:
            DatasetCls = SegmentedGrainDataset
            logger.info(f"[GRAIN MODE] crop_source=seg (oracle .seg)")

        train_kw = _grain_dataset_kwargs(config, train_transforms, train_mask_bg)
        val_kw = _grain_dataset_kwargs(config, val_transforms, val_mask_bg)

        if DatasetCls is SegmentedGrainDataset:
            train_dataset = DatasetCls(root=data_config["train_dir"], **train_kw)
            val_dataset = DatasetCls(root=data_config["val_dir"], **val_kw)
        else:
            train_dataset = DatasetCls(root=data_config["train_dir"], config=config, **train_kw)
            val_dataset = DatasetCls(root=data_config["val_dir"], config=config, **val_kw)
        
        test_dataset = None
        test_dir = data_config.get("test_dir")
        if test_dir and Path(test_dir).exists():
            if DatasetCls is SegmentedGrainDataset:
                test_dataset = DatasetCls(root=test_dir, **val_kw)
            else:
                test_dataset = DatasetCls(root=test_dir, config=config, **val_kw)
            if len(test_dataset) == 0:
                test_dataset = None
            else:
                logger.info(f"[GRAIN MODE] Test set cargado: {test_dir} ({len(test_dataset)} samples)")

        if len(train_dataset) == 0:
            raise RuntimeError(
                f"No se encontraron granos en {data_config['train_dir']}. "
                "Ejecuta 'Análisis de Contornos' primero."
            )
        if len(val_dataset) == 0:
            logger.warning(f"[GRAIN MODE] Val set vacío: {data_config['val_dir']}")
        config["num_classes"] = len(train_dataset.classes)

        # Build HDF5 Cache sequentially by image to avoid redundant I/O
        _build_h5_cache(train_dataset, "train")
        _build_h5_cache(val_dataset, "val")
        if test_dataset is not None:
            _build_h5_cache(test_dataset, "test")

        sampler = MPerClassSampler(
            labels=train_dataset.targets,
            m=int(data_config["samples_per_class"]),
            batch_size=batch_size,
            length_before_new_iter=len(train_dataset),
        )

        collate = grain_collate_fn if use_grain_collate else None

        def _make_loader(
            dataset,
            workers: int,
            *,
            sampler=None,
            shuffle: bool = False,
            drop_last: bool = False,
            loader_batch_size: Optional[int] = None,
        ) -> DataLoader:
            bs = loader_batch_size if loader_batch_size is not None else batch_size
            kw: Dict[str, Any] = {
                "batch_size": bs,
                "num_workers": workers,
                "pin_memory": pin_memory,
                "drop_last": drop_last,
                "collate_fn": collate,
            }
            if sampler is not None:
                kw["sampler"] = sampler
            else:
                kw["shuffle"] = shuffle
            if workers > 0:
                kw["worker_init_fn"] = _dataloader_worker_init
                kw["prefetch_factor"] = prefetch_factor
                kw["persistent_workers"] = persistent
            return DataLoader(dataset, **kw)

        train_loader = _make_loader(
            train_dataset,
            train_workers,
            sampler=sampler,
            drop_last=bool(data_config.get("drop_last", True)),
        )
        val_loader = _make_loader(
            val_dataset,
            val_workers,
            shuffle=False,
            drop_last=False,
            loader_batch_size=val_batch_size,
        )
        
        test_loader = None
        if test_dataset is not None:
            test_loader = _make_loader(
                test_dataset,
                val_workers,
                shuffle=False,
                drop_last=False,
                loader_batch_size=val_batch_size,
            )
        if val_batch_size != batch_size:
            logger.info(
                f"Val DataLoader batch_size={val_batch_size} (train={batch_size})"
            )

        if train_workers > 0 or val_workers > 0:
            batch_mb = batch_size * 3 * int(data_config.get("image_size", 224)) ** 2 * 4 / (1024**2)
            cache_mb = int(config.get("grain_detection", {}).get("image_cache_max_mb", 96))
            est = (train_workers + val_workers) * cache_mb + (
                train_workers + val_workers
            ) * prefetch_factor * batch_mb
            logger.info(
                f"GrainDataEngine: train_workers={train_workers}, val_workers={val_workers}, "
                f"prefetch={prefetch_factor}, persistent={persistent}, exclusive_val={policy.get('exclusive_train_workers_during_val', True)}"
            )
            logger.info(
                f"  RAM estimate (rough): ~{est:.0f} MB batch/prefetch + "
                f"~{ram_est['total_workers_mb']:.0f} MB worker pools"
            )

        return cls(
            config,
            train_dataset,
            val_dataset,
            train_loader,
            val_loader,
            train_workers,
            val_workers,
            test_dataset,
            test_loader,
        )

    def release_train_workers(self):
        """Shut down train DataLoader workers when train prefetch is not needed."""
        if self.train_workers <= 0:
            return
        dispose_dataloader_workers(self.train_loader)
        import gc
        gc.collect()

    def release_val_workers(self):
        """Shut down val DataLoader workers when val prefetch is not needed."""
        if self.val_workers <= 0:
            return
        dispose_dataloader_workers(self.val_loader)
        if self.test_loader is not None:
            dispose_dataloader_workers(self.test_loader)
        import gc
        gc.collect()

    def release_all_workers(self, *, close_datasets: bool = False):
        """Shut down DataLoader pools. Dataset handles stay open during training."""
        from src.data.dataloader_utils import close_dataset_resources, free_training_memory

        self.release_train_workers()
        self.release_val_workers()
        if close_datasets:
            for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
                close_dataset_resources(dataset)
        import gc
        gc.collect()
        free_training_memory("data_engine_workers_released")


def create_dataloaders(config: dict):
    """Backward-compatible entry point used by train.py."""
    engine = GrainDataEngine.from_config(config)
    return engine.train_loader, engine.val_loader, engine
