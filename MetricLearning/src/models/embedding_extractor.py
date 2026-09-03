"""
Canonical embedding extraction — single forward path for train/eval/inference/XAI.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


def extract_embeddings_from_batch(
    model: nn.Module,
    images: torch.Tensor,
    masks: Optional[torch.Tensor] = None,
    *,
    return_prenorm: bool = False,
    use_mask_from_config: bool = True,
    grain_config: Optional[dict] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Single canonical forward for one batch.

    Args:
        model: AnalogyNet (or compatible)
        images: (B, 3, H, W)
        masks: optional (B, H, W)
        return_prenorm: if True, returns (embeddings, prenorm_embeddings)
        use_mask_from_config: if False, never pass mask even if provided
        grain_config: used to check masked_pooling flag
    """
    grain_config = grain_config or {}
    masked_pooling = bool(grain_config.get("masked_pooling", False))
    effective_mask = masks if (use_mask_from_config and masked_pooling and masks is not None) else None

    if return_prenorm:
        return model(images, mask=effective_mask, return_prenorm=True)
    return model(images, mask=effective_mask)


def iter_batches_with_masks(dataloader: DataLoader):
    """Yield (images, labels, masks_or_none) from loader batches."""
    for batch in dataloader:
        if len(batch) == 3:
            images, labels, masks = batch
            yield images, labels, masks
        elif len(batch) >= 2:
            yield batch[0], batch[1], None
        else:
            raise ValueError(f"Batch inesperado con {len(batch)} elementos")


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    grain_config: Optional[dict] = None,
    return_paths: bool = False,
    show_progress: bool = True,
    return_prenorm: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, list],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Extract embeddings from a dataloader using the canonical forward.

    Returns:
        (embeddings, labels) or (embeddings, labels, paths) or with prenorm triple
    """
    model.eval()
    grain_config = grain_config or {}

    all_embeddings = []
    all_prenorm = []
    all_labels = []
    paths_list = [] if return_paths else None

    iterator = tqdm(dataloader, desc="Extracting embeddings") if show_progress else dataloader

    with torch.no_grad():
        for batch in iterator:
            if return_paths:
                if len(batch) == 3:
                    images, labels, extra = batch
                    if isinstance(extra, (list, tuple)) and extra and isinstance(extra[0], str):
                        paths_list.extend(extra)
                        masks = None
                    else:
                        masks = extra
                else:
                    raise ValueError("return_paths=True requiere paths en el batch")
            else:
                images, labels, masks = (
                    (batch[0], batch[1], batch[2]) if len(batch) == 3 else (batch[0], batch[1], None)
                )

            images = images.to(device, non_blocking=True)
            if masks is not None:
                masks = masks.to(device, non_blocking=True)

            out = extract_embeddings_from_batch(
                model,
                images,
                masks,
                return_prenorm=return_prenorm,
                grain_config=grain_config,
            )

            if return_prenorm:
                embeddings, prenorm = out
                all_prenorm.append(prenorm.cpu())
            else:
                embeddings = out

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels if isinstance(labels, torch.Tensor) else torch.as_tensor(labels))

    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()

    if return_prenorm:
        prenorm = torch.cat(all_prenorm, dim=0)
        if return_paths:
            return embeddings.numpy(), labels, prenorm.numpy(), paths_list
        return embeddings.numpy(), labels, prenorm.numpy()

    if return_paths:
        return embeddings.numpy(), labels, paths_list
    return embeddings.numpy(), labels
