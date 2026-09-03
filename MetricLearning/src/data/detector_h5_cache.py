"""HDF5 cache builder + persistent manifest for ViT-dense detector training."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import cv2
import numpy as np
from tqdm import tqdm

from src.data.detection_transforms import letterbox_to_square, rasterize_grain_union_mask

if TYPE_CHECKING:
    from src.data.detection_dataset import FullImageDetectionDataset

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1
MANIFEST_NAME = "detector_cache_manifest.json"


def _norm_path(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve()).lower()
    except OSError:
        return str(path).lower()


def _cache_root(dataset_root: Path) -> Path:
    return dataset_root.parent.parent / "cache"


def _manifest_path(dataset_root: Path) -> Path:
    return _cache_root(dataset_root) / MANIFEST_NAME


def _sample_paths_fingerprint(samples: list) -> str:
    paths = "\n".join(sorted(s["image_path"] for s in samples))
    return hashlib.md5(paths.encode("utf-8")).hexdigest()[:16]


def cache_fingerprint(dataset: "FullImageDetectionDataset", prefix: str) -> str:
    """Stable fingerprint from split config + exact image list (independent of cwd)."""
    payload = {
        "prefix": prefix,
        "root": _norm_path(dataset.root),
        "annotation_root": _norm_path(dataset.annotation_root),
        "input_size": int(dataset.input_size),
        "max_images": int(dataset.max_images),
        "subset_seed": int(dataset.subset_seed),
        "n_images": len(dataset.samples),
        "paths_fp": _sample_paths_fingerprint(dataset.samples),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]


def _cache_path(dataset: "FullImageDetectionDataset", prefix: str) -> Path:
    cache_dir = _cache_root(dataset.root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_fingerprint(dataset, prefix)
    return cache_dir / f"detector_cache_{prefix}_{fp}.h5"


def _load_manifest(dataset_root: Path) -> dict:
    path = _manifest_path(dataset_root)
    if not path.exists():
        return {"version": MANIFEST_VERSION, "entries": {}}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "entries" in data:
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"[DETECTOR-CACHE] Manifest ilegible ({path}): {exc}")
    return {"version": MANIFEST_VERSION, "entries": {}}


def _save_manifest(dataset_root: Path, manifest: dict) -> None:
    path = _manifest_path(dataset_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _validate_h5(path: Path, *, n: int, size: int) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import h5py

        with h5py.File(path, "r") as f:
            return (
                "images" in f
                and "masks" in f
                and f["images"].shape == (n, size, size, 3)
                and f["masks"].shape == (n, size, size)
            )
    except Exception:
        return False


def _register_manifest_entry(
    dataset: "FullImageDetectionDataset",
    prefix: str,
    h5_path: Path,
    fingerprint: str,
) -> None:
    manifest = _load_manifest(dataset.root)
    manifest["entries"][prefix] = {
        "fingerprint": fingerprint,
        "h5_path": str(h5_path.resolve()),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "num_images": len(dataset.samples),
        "input_size": int(dataset.input_size),
        "max_images": int(dataset.max_images),
        "subset_seed": int(dataset.subset_seed),
        "root": _norm_path(dataset.root),
        "annotation_root": _norm_path(dataset.annotation_root),
    }
    _save_manifest(dataset.root, manifest)


def _resolve_existing_cache(
    dataset: "FullImageDetectionDataset",
    prefix: str,
) -> Path | None:
    """Return valid cache path from manifest or canonical filename without rebuilding."""
    n = len(dataset.samples)
    size = dataset.input_size
    fingerprint = cache_fingerprint(dataset, prefix)

    manifest = _load_manifest(dataset.root)
    entry = manifest.get("entries", {}).get(prefix)
    if entry and entry.get("fingerprint") == fingerprint:
        cached = Path(entry["h5_path"])
        if _validate_h5(cached, n=n, size=size):
            return cached

    canonical = _cache_path(dataset, prefix)
    if _validate_h5(canonical, n=n, size=size):
        if not entry or entry.get("fingerprint") != fingerprint:
            _register_manifest_entry(dataset, prefix, canonical, fingerprint)
        return canonical

    if canonical.exists() and canonical.stat().st_size == 0:
        try:
            canonical.unlink()
        except OSError:
            pass
    return None


def _prepare_letterboxed_sample(meta: dict, input_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(meta["image_path"])
    if image is None:
        raise IOError(f"No se pudo leer: {meta['image_path']}")
    grains = meta.get("grains")
    if grains is None:
        from src.grain_detection.seg_format import SegFileReader

        grains = SegFileReader.read_grains(meta["seg_path"])
    mask = rasterize_grain_union_mask(grains, image.shape[:2])
    image_lb, mask_lb, _ = letterbox_to_square(image, mask, input_size)
    mask_u8 = (np.clip(mask_lb, 0.0, 1.0) * 255).astype(np.uint8)
    return image_lb.astype(np.uint8), mask_u8


def _release_grains_from_samples(dataset: "FullImageDetectionDataset") -> None:
    for sample in dataset.samples:
        sample.pop("grains", None)


def build_detector_h5_cache(dataset: "FullImageDetectionDataset", prefix: str) -> None:
    """
    Reuse letterboxed HDF5 when manifest + file match; build only on miss.
    """
    import concurrent.futures
    import h5py

    n = len(dataset.samples)
    if n == 0:
        return

    size = dataset.input_size
    fingerprint = cache_fingerprint(dataset, prefix)
    existing = _resolve_existing_cache(dataset, prefix)
    if existing is not None:
        dataset.h5_path = str(existing)
        _release_grains_from_samples(dataset)
        logger.info(
            f"[DETECTOR-{prefix.upper()}] CACHE HIT — {existing} "
            f"(fp={fingerprint}, n={n}, size={size})"
        )
        return

    path = _cache_path(dataset, prefix)
    dataset.h5_path = str(path)

    logger.info(
        f"[DETECTOR-{prefix.upper()}] CACHE MISS — compilando HDF5 "
        f"({n} imgs, {size}px, fp={fingerprint}) → {path}"
    )

    def _build_one(idx: int) -> tuple[int, np.ndarray, np.ndarray]:
        img_lb, mask_u8 = _prepare_letterboxed_sample(dataset.samples[idx], size)
        return idx, img_lb, mask_u8

    with h5py.File(path, "w") as f:
        img_dset = f.create_dataset(
            "images",
            shape=(n, size, size, 3),
            dtype=np.uint8,
            chunks=(1, size, size, 3),
            compression="lzf",
        )
        mask_dset = f.create_dataset(
            "masks",
            shape=(n, size, size),
            dtype=np.uint8,
            chunks=(1, size, size),
            compression="lzf",
        )
        f.attrs["input_size"] = size
        f.attrs["num_images"] = n
        f.attrs["fingerprint"] = fingerprint

        workers = min(8, max(2, (os.cpu_count() or 4) - 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_build_one, i) for i in range(n)]
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=n,
                desc=f"Det cache {prefix}",
            ):
                idx, img_lb, mask_u8 = fut.result()
                img_dset[idx] = img_lb
                mask_dset[idx] = mask_u8

    _register_manifest_entry(dataset, prefix, path, fingerprint)
    _release_grains_from_samples(dataset)
    logger.info(f"[DETECTOR-{prefix.upper()}] HDF5 listo: {path}")


# Backward-compatible alias used in tests
_config_hash = cache_fingerprint
