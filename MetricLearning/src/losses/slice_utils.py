"""
Utilidades compartidas para dimensional slicing de embeddings.

Usado por SlicedMSLoss, SlicedProxyLoss y SliceAwareClassifier.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F


def resolve_slice_geometry(embedding_dim: int, num_slices: int) -> Tuple[int, int]:
    """Return (num_slices, slice_dim) after validating divisibility."""
    if embedding_dim % num_slices != 0:
        raise ValueError(
            f"embedding_dim ({embedding_dim}) must be divisible by "
            f"num_slices ({num_slices})"
        )
    return num_slices, embedding_dim // num_slices


def split_and_normalize_slices(
    embeddings: torch.Tensor,
    num_slices: int,
) -> List[torch.Tensor]:
    """
    Particiona (B, D) en S slices disjuntos y L2-normaliza cada uno.

    Returns:
        Lista de tensores (B, D/S), uno por slice.
    """
    _, slice_dim = resolve_slice_geometry(embeddings.shape[1], num_slices)
    slices = []
    for i in range(num_slices):
        start = i * slice_dim
        end = start + slice_dim
        sl = F.normalize(embeddings[:, start:end], p=2, dim=1)
        slices.append(sl)
    return slices
