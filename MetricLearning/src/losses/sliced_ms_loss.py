"""
Spectral Embedding Expansion via Dimensional Slicing.

Applies Multi-Similarity Loss with Learnable Dynamic Margins (LDM)
independently over S orthogonal subspaces to break Neural Collapse
(Papyan et al., PNAS 2020) and force spectrally rich representations.

When embedding_dim >> num_classes, a single loss collapses the embedding
to a (C-1)-dimensional simplex. By computing S independent losses on
disjoint subspaces, we force S independent simplexes, raising the
effective spectral rank to S × (C-1).

References:
    - Multi-Similarity Loss: Wang et al., CVPR 2019
    - Embedding Expansion: Ko & Gu, CVPR 2020
    - Divide and Conquer the Embedding Space: Sanakoyeu et al., CVPR 2019
    - Neural Collapse: Papyan, Han & Donoho, PNAS 2020
"""

import torch
import torch.nn as nn
import logging

from src.losses.ldm_loss import LearnableDynamicMarginMSLoss
from src.losses.slice_utils import resolve_slice_geometry, split_and_normalize_slices

logger = logging.getLogger(__name__)


class SlicedMSLoss(nn.Module):
    """
    Sliced Multi-Similarity Loss with Learnable Dynamic Margins.
    
    Partitions the embedding vector into S equal-size disjoint slices and
    applies an independent LDM MS-Loss to each. Each slice is L2-normalized
    before loss computation, forcing it to live on its own unit hypersphere.
    
    Args:
        num_classes (int): Number of classes in the dataset.
        num_slices (int): Number of disjoint subspaces S. Default 4.
        embedding_dim (int): Total embedding dimension. Must be divisible by num_slices.
        alpha (float): MS-Loss positive pair weighting. Default 2.0.
        beta (float): MS-Loss negative pair weighting. Default 50.0.
        base_margin (float): Base positive margin for LDM. Default 0.5.
    """
    
    def __init__(self, num_classes: int, num_slices: int = 4, 
                 embedding_dim: int = 128, alpha: float = 2.0,
                 beta: float = 50.0, base_margin: float = 0.5):
        super().__init__()
        
        self.num_slices, self.slice_dim = resolve_slice_geometry(embedding_dim, num_slices)
        self.embedding_dim = embedding_dim
        
        # S independent LDM MS-Loss instances, each with its own
        # learnable per-class margins (pos_margin, neg_margin)
        self.slice_losses = nn.ModuleList([
            LearnableDynamicMarginMSLoss(
                num_classes=num_classes,
                alpha=alpha,
                beta=beta,
                base_margin=base_margin
            )
            for i in range(num_slices)
        ])
        
        logger.info(
            f"SlicedMSLoss initialized: {num_slices} slices × {self.slice_dim}D, "
            f"{num_classes} classes, α={alpha}, β={beta}, base_margin={base_margin}"
        )
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, 
                indices_tuple=None) -> torch.Tensor:
        """
        Compute the sliced MS loss.
        
        Args:
            embeddings: (B, D) L2-normalized embeddings from the model.
            labels: (B,) integer class labels.
            indices_tuple: Optional mined indices (passed to each slice loss).
            
        Returns:
            Scalar loss: mean of S independent slice losses.
        """
        total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        for i, (loss_fn, slice_emb) in enumerate(
            zip(self.slice_losses, split_and_normalize_slices(embeddings, self.num_slices))
        ):
            slice_loss = loss_fn(slice_emb, labels, indices_tuple=indices_tuple)
            total_loss = total_loss + slice_loss
        
        # Return the mean across all slices
        return total_loss / self.num_slices
    
    def get_learnable_margins(self) -> list:
        """
        Returns all learnable margin parameters for optimizer registration.
        Each slice has its own pos_margin and neg_margin Parameters.
        """
        params = []
        for loss_fn in self.slice_losses:
            params.append(loss_fn.pos_margin)
            params.append(loss_fn.neg_margin)
        return params
