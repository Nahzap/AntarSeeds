"""
Hyperbolic Embedding Space for Deep Metric Learning.
Based on: Ermolov et al. (2022) "Hyperbolic Vision Transformers" (CVPR 2022)

Maps Euclidean embeddings to the Poincaré Ball model of hyperbolic space,
which naturally captures hierarchical relationships between classes.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PoincareBallProjection(nn.Module):
    """
    Projects Euclidean embeddings onto the Poincaré Ball model.
    
    The Poincaré Ball is a model of hyperbolic space where points lie
    inside the unit ball {x : ||x|| < 1/sqrt(c)} with curvature -c.
    
    Args:
        curvature (float): Absolute value of negative curvature. Default 1.0.
        eps (float): Numerical stability epsilon.
    """
    
    def __init__(self, curvature=1.0, eps=1e-5):
        super().__init__()
        self.curvature = curvature
        self.eps = eps
        self.max_norm = 1.0 / (curvature ** 0.5) - eps
    
    def expmap0(self, u):
        """Exponential map from origin to Poincare ball.
        u: (B, D) point in tangent space at origin.
        """
        dtype = u.dtype
        u = u.float()
        # Zero-center to prevent angular collapse from ReLU layers
        u = u - u.mean(dim=0, keepdim=True)
        
        sqrt_c = self.curvature ** 0.5
        u_norm_sq = (u ** 2).sum(dim=-1, keepdim=True)
        u_norm = (u_norm_sq + self.eps).sqrt()
        
        # Clip norm to prevent tanh saturation (tanh(x) -> 1.0 for x > 10)
        # and vanishing gradients for the magnitude
        max_u_norm = 15.0 / sqrt_c
        u_norm_clipped = u_norm.clamp(max=max_u_norm)
        
        res = torch.tanh(sqrt_c * u_norm_clipped) * u / u_norm
        return res.to(dtype)
    
    def project(self, x):
        """Project point back into the Poincaré ball if it escaped."""
        dtype = x.dtype
        x = x.float()
        norm = x.norm(dim=-1, keepdim=True)
        cond = norm > self.max_norm
        projected = x / norm.clamp(min=self.eps) * self.max_norm
        res = torch.where(cond, projected, x)
        return res.to(dtype)
    
    def forward(self, x):
        """
        Map Euclidean embeddings to Poincaré ball.
        
        Args:
            x: (B, D) Euclidean embeddings
        Returns:
            (B, D) Poincaré ball embeddings
        """
        x = self.expmap0(x)
        x = self.project(x)
        return x


def poincare_distance(x, y, curvature=1.0, eps=1e-5):
    """
    Compute pairwise Poincaré distance between two sets of points.
    
    d(x, y) = (1/sqrt(c)) * arccosh(1 + 2c * ||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))
    
    Args:
        x: (B, D) points in Poincaré ball
        y: (N, D) points in Poincaré ball
        curvature: Absolute curvature value
    Returns:
        (B, N) pairwise distances
    """
    c = curvature
    sqrt_c = c ** 0.5
    
    # Compute in float32 to prevent AMP overflow (float16 max is 65504)
    orig_dtype = x.dtype
    x = x.float()
    y = y.float()
    
    # Direct computation of squared distance avoids catastrophic cancellation
    # (x.unsqueeze(1) - y.unsqueeze(0)) is (B, N, D)
    diff_sq = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1) # (B, N)
    
    x_norm_sq = (x * x).sum(dim=-1, keepdim=True).clamp(max=1.0/c - eps)  # (B, 1)
    y_norm_sq = (y * y).sum(dim=-1, keepdim=True).clamp(max=1.0/c - eps)  # (N, 1)
    
    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq.t())  # (B, N)
    denom = denom.clamp(min=eps)
    
    # Prevent exploding noise near boundary
    arg = 1 + 2 * c * diff_sq / denom
    
    # Relax clamp slightly to allow gradients for very close points
    arg = arg.clamp(min=1.0 + 1e-6)
    
    dist = (1.0 / sqrt_c) * torch.acosh(arg)
    return dist.to(orig_dtype)
