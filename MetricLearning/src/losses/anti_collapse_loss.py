"""
Anti-Collapse regularization for deep metric learning (D-04).

Penalizes embedding uniformity collapse via hypersphere uniformity loss
(Wang & Isola, ICML 2020), aligned with anti-collapse objectives in
IEEE Trans. Multimedia 2024 literature.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AntiCollapseLoss(nn.Module):
    """
    Uniformity regularizer on L2-normalized embeddings.

    L = lambda * log E_{i!=j} exp(-t * ||z_i - z_j||^2)

    Higher embedding spread → lower loss contribution.
    """

    def __init__(self, lambda_ac: float = 0.1, t: float = 2.0):
        super().__init__()
        self.lambda_ac = float(lambda_ac)
        self.t = float(t)
        logger.info(f"AntiCollapseLoss enabled (lambda={self.lambda_ac}, t={self.t})")

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        z = F.normalize(embeddings, p=2, dim=1)
        n = z.shape[0]
        if n < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        pdist_sq = torch.pdist(z, p=2).pow(2)
        uniformity = torch.log(torch.exp(-self.t * pdist_sq).mean() + 1e-8)
        return self.lambda_ac * uniformity
