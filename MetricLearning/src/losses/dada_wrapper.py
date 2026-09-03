"""
Data-Augmented Domain Adaptation (DADA) Wrapper for Proxy-based Losses.
Based on: Ren, Chen, Wang & Hua (2024) "Towards Improved Proxy-based Deep
Metric Learning via Data-Augmented Domain Adaptation" (AAAI 2024)

Wraps any existing loss and adds an MMD-based alignment term between
original and augmented embedding distributions.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class DADAWrapper(nn.Module):
    """
    Data-Augmented Domain Adaptation wrapper.
    
    Wraps a base metric learning loss and adds a Maximum Mean Discrepancy
    (MMD) term that aligns the distributions of original and augmented
    embeddings, reducing the domain gap between samples and proxies.
    
    L_total = L_base(embeddings, labels) + lambda_dada * MMD(embeddings, aug_embeddings)
    
    Args:
        base_loss (nn.Module): Base metric learning loss function.
        lambda_dada (float): Weight for the MMD alignment term. Default 0.1.
        kernel_bandwidth (float): Bandwidth for Gaussian kernel. Default 1.0.
    """
    
    def __init__(self, base_loss, lambda_dada=0.1, kernel_bandwidth=1.0):
        super().__init__()
        self.base_loss = base_loss
        self.lambda_dada = lambda_dada
        self.kernel_bandwidth = kernel_bandwidth
        
        logger.info(f"DADA wrapper initialized: lambda={lambda_dada}")
    
    def _gaussian_kernel(self, x, y):
        """Compute Gaussian kernel matrix between x and y."""
        # (B, 1, D) - (1, M, D) -> (B, M, D) -> (B, M)
        x_norm = (x ** 2).sum(-1).view(-1, 1)
        y_norm = (y ** 2).sum(-1).view(1, -1)
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        dist_sq = dist_sq.clamp(min=0.0)
        return torch.exp(-dist_sq / (2 * self.kernel_bandwidth ** 2))
    
    def _compute_mmd(self, x, y):
        """
        Compute Maximum Mean Discrepancy between two distributions.
        
        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
        
        Args:
            x: (B, D) original embeddings
            y: (M, D) augmented embeddings
        Returns:
            Scalar MMD^2 value
        """
        n = x.shape[0]
        m = y.shape[0]
        
        if n < 2 or m < 2:
            return torch.tensor(0.0, device=x.device)
        
        k_xx = self._gaussian_kernel(x, x)
        k_yy = self._gaussian_kernel(y, y)
        k_xy = self._gaussian_kernel(x, y)
        
        # Unbiased MMD estimator
        mmd = (k_xx.sum() - k_xx.diag().sum()) / (n * (n - 1)) + \
              (k_yy.sum() - k_yy.diag().sum()) / (m * (m - 1)) - \
              2 * k_xy.sum() / (n * m)
        
        return mmd.clamp(min=0.0)
    
    def forward(self, embeddings, labels, augmented_embeddings=None):
        """
        Compute base loss + optional DADA alignment.
        
        Args:
            embeddings: (B, D) embeddings from original images
            labels: (B,) class labels
            augmented_embeddings: (B, D) embeddings from augmented images.
                If None, only base loss is computed.
                
        Returns:
            Scalar total loss
        """
        # Base loss (works with any metric learning loss)
        base_loss = self.base_loss(embeddings, labels)
        
        if augmented_embeddings is not None and self.lambda_dada > 0:
            mmd_loss = self._compute_mmd(embeddings, augmented_embeddings)
            total_loss = base_loss + self.lambda_dada * mmd_loss
            return total_loss
        
        return base_loss
