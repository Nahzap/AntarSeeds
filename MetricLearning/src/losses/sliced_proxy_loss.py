"""
Sliced Proxy Loss for Deep Metric Learning.

Combines the "Divide and Conquer the Embedding Space" (Sanakoyeu et al., CVPR 2019)
technique with Stop-Gradient Softmax Loss (SG-Softmax, AAAI 2023) proxies.

By partitioning the embedding space into S disjoint slices and applying
a proxy-based loss independently to each slice, we break Neural Collapse
(increasing spectral rank) while maintaining extreme robustness to small
batch sizes (Batch Starvation) which normally breaks pairwise losses.
"""

import torch
import torch.nn as nn
import logging

from src.losses.sg_softmax import SGSoftmaxLoss
from src.losses.slice_utils import resolve_slice_geometry, split_and_normalize_slices

logger = logging.getLogger(__name__)


class SlicedProxyLoss(nn.Module):
    """
    Sliced Proxy Loss via SG-Softmax.
    
    Partitions the embedding vector into S equal-size disjoint slices and
    applies an independent SG-Softmax loss to each. Each slice is L2-normalized
    before loss computation, and has its own set of learned class proxies.
    
    Args:
        num_classes (int): Number of classes in the dataset.
        num_slices (int): Number of disjoint subspaces S. Default 4.
        embedding_dim (int): Total embedding dimension. Must be divisible by num_slices.
        scale (float): Scaling factor for logits in SG-Softmax.
        margin (float): Additive angular margin in SG-Softmax.
    """
    
    def __init__(self, num_classes: int, num_slices: int = 4, 
                 embedding_dim: int = 128, scale: float = 30.0,
                 margin: float = 0.2):
        super().__init__()
        
        self.num_slices, self.slice_dim = resolve_slice_geometry(embedding_dim, num_slices)
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # S independent SG-Softmax instances, each with its own
        # learnable proxy matrix of shape (num_classes, slice_dim)
        self.slice_losses = nn.ModuleList([
            SGSoftmaxLoss(
                embedding_dim=self.slice_dim,
                num_classes=num_classes,
                scale=scale,
                margin=margin
            )
            for i in range(num_slices)
        ])
        
        logger.info(
            f"SlicedProxyLoss initialized: {num_slices} slices × {self.slice_dim}D, "
            f"{num_classes} classes, scale={scale}, margin={margin}"
        )
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute the sliced proxy loss.
        
        Args:
            embeddings: (B, D) L2-normalized embeddings from the model.
            labels: (B,) integer class labels.
            
        Returns:
            Scalar loss: mean of S independent proxy losses.
        """
        total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        for loss_fn, slice_emb in zip(
            self.slice_losses,
            split_and_normalize_slices(embeddings, self.num_slices),
        ):
            slice_loss = loss_fn(slice_emb, labels)
            total_loss = total_loss + slice_loss
        
        # Return the mean across all slices
        return total_loss / self.num_slices
    
    def get_proxies(self) -> list:
        """
        Returns all learnable proxy parameters for optimizer registration.
        Each slice has its own `proxies` nn.Parameter matrix.
        """
        params = []
        for loss_fn in self.slice_losses:
            params.append(loss_fn.proxies)
        return params

    def export_proxies(self) -> torch.Tensor:
        """Stack learned proxies as (num_slices, num_classes, slice_dim)."""
        return torch.stack(
            [loss_fn.proxies.detach().cpu() for loss_fn in self.slice_losses],
            dim=0,
        )
