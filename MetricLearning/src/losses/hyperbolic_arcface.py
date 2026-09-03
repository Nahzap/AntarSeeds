import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.hyperbolic import poincare_distance, PoincareBallProjection

class HyperbolicArcFaceLoss(nn.Module):
    """
    Hyperbolic ArcFace Loss.
    Computes distances in the Poincaré ball between embeddings and class prototypes,
    and converts them to logits applying an additive distance margin to the true class.
    
    Logits formulation:
    logit = scale * (-distance)
    For true class:
    logit = scale * -(distance + margin)
    """
    def __init__(self, num_classes, embedding_size, margin=0.2, scale=30.0, curvature=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.curvature = curvature
        
        # Prototypes in Euclidean tangent space, then projected to hyperbolic
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.proj = PoincareBallProjection(curvature=curvature)

    def forward(self, embeddings, labels, *args, **kwargs):
        # embeddings are already in Poincare Ball (B, D)
        # weight must be projected to Poincare Ball (C, D)
        hyperbolic_weights = self.proj(self.weight)
        
        # Compute pairwise Poincare distances (B, C)
        dists = poincare_distance(embeddings, hyperbolic_weights, curvature=self.curvature)
        
        # Add margin to the positive class distance
        one_hot = F.one_hot(labels, num_classes=self.num_classes).bool()
        
        dists_with_margin = dists.clone()
        dists_with_margin[one_hot] = dists[one_hot] + self.margin
        
        # Convert distances to logits (smaller distance -> higher logit)
        logits = -dists_with_margin * self.scale
        
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def compute_loss(self, embeddings, labels, **kwargs):
        # Compatibility wrapper for Hard Negative Mining expecting compute_loss
        hyperbolic_weights = self.proj(self.weight)
        dists = poincare_distance(embeddings, hyperbolic_weights, curvature=self.curvature)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).bool()
        dists_with_margin = dists.clone()
        dists_with_margin[one_hot] = dists[one_hot] + self.margin
        logits = -dists_with_margin * self.scale
        
        # Per-sample loss for hard negative mining
        per_sample_loss = F.cross_entropy(logits, labels, reduction='none')
        return {
            'loss': {
                'losses': per_sample_loss
            }
        }
