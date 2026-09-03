"""
Non-Isotropy Regularization (NIR) for Deep Metric Learning.
Based on: Roth, Vinyals & Akata (2022) "Non-isotropy Regularization for
Proxy-based Deep Metric Learning" (CVPR 2022)

Penalizes isotropic (spherical) local distributions per class,
encouraging anisotropic embeddings that preserve intra-class structure.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class NIRRegularizer(nn.Module):
    """
    Non-Isotropy Regularization.

    L_NIR = -lambda * mean_c[ log det(Sigma_c + eps*I) ]

    Skips classes with fewer than ``min_samples_per_class`` points in the
    current batch to avoid rank-deficient covariance crashes.
    """

    def __init__(self, lambda_nir=0.5, eps=1e-4, min_samples_per_class: int = 2):
        super().__init__()
        self.lambda_nir = lambda_nir
        self.eps = eps
        self.min_samples_per_class = max(2, int(min_samples_per_class))

    def forward(self, embeddings, labels):
        device = embeddings.device
        log_dets = []

        for class_id in labels.unique():
            mask = labels == class_id
            count = int(mask.sum().item())
            if count < self.min_samples_per_class:
                continue

            class_emb = embeddings[mask]
            centered = class_emb - class_emb.mean(dim=0, keepdim=True)
            n = centered.shape[0]
            cov = (centered.t() @ centered) / max(1, n - 1)
            dim = cov.shape[0]
            cov = cov + self.eps * torch.eye(dim, device=device, dtype=cov.dtype)

            sign, logdet = torch.linalg.slogdet(cov)
            if sign > 0 and torch.isfinite(logdet):
                log_dets.append(logdet)

        if not log_dets:
            return torch.zeros((), device=device, requires_grad=True)

        return -self.lambda_nir * torch.stack(log_dets).mean()
