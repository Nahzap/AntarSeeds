"""
Stop-Gradient Softmax Loss (SG-Softmax) for Deep Metric Learning.
Based on: Yang, Wang & Zhang (2023) "Stop-Gradient Softmax Loss for
Deep Metric Learning" (AAAI 2023)

Applies stop_gradient to the softmax denominator to stabilize gradients
and achieve superior performance over Multi-Similarity Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SGSoftmaxLoss(nn.Module):
    """
    Stop-Gradient Softmax Loss.
    
    A proxy-based loss that normalizes embeddings and proxies on the 
    hypersphere, adds an angular margin to positive pairs, and applies
    stop-gradient to the log-sum-exp denominator for stable optimization.
    
    Args:
        embedding_dim (int): Dimension of embeddings.
        num_classes (int): Number of classes.
        scale (float): Scaling factor for logits. Default 30.0.
        margin (float): Additive angular margin. Default 0.4.
    """
    
    def __init__(
        self,
        embedding_dim,
        num_classes,
        scale=30.0,
        margin=0.4,
        ortho_penalty_weight=1.0,
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ortho_penalty_weight = float(ortho_penalty_weight)
        self.num_classes = num_classes
        
        # Learnable class proxies
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies.unsqueeze(0))
        
        logger.info(
            f"SG-Softmax initialized: {num_classes} classes, "
            f"scale={scale}, margin={margin}"
        )
    
    def forward(self, embeddings, labels):
        """
        Compute SG-Softmax loss.
        
        Args:
            embeddings: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels
            
        Returns:
            Scalar loss
        """
        # Normalize embeddings and proxies
        embeddings = F.normalize(embeddings, p=2, dim=1)
        proxies = F.normalize(self.proxies, p=2, dim=1)
        
        # Cosine similarity: (B, C)
        sim = embeddings @ proxies.t()
        
        # 1. Corrección Matemática: ArcFace Angular Margin
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        import math
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        
        # Calcular sin(theta) de forma segura
        sin_theta = torch.sqrt(1.0 - torch.pow(sim, 2).clamp(0, 1) + 1e-6)
        cos_theta_m = sim * cos_m - sin_theta * sin_m
        
        # Target fallback para evitar inestabilidad si theta + m > pi
        threshold = math.cos(math.pi - self.margin)
        cond = sim > threshold
        cos_theta_m = torch.where(cond, cos_theta_m, sim - sin_m * self.margin)
        
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Aplicar margen angular solo a las clases positivas
        sim_with_margin = torch.where(one_hot.bool(), cos_theta_m, sim)
        
        # Scale logits
        logits = self.scale * sim_with_margin
        
        # Correct SG-Softmax: Apply stop-gradient to the positive logit in the denominator
        # to prevent gradients from penalizing the positive class in the logsumexp term.
        pos_mask = one_hot.bool()
        logits_sg = torch.where(pos_mask, logits.detach(), logits)
        
        log_numerator = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        log_denominator = torch.logsumexp(logits_sg, dim=1)
        
        loss_ce = -(log_numerator - log_denominator).mean()
        
        # 2. Corrección Matemática: Penalización de Ortogonalidad en Proxies
        # Penaliza fuertemente cualquier similitud proxy-proxy > 0 (es decir, ángulos < 90°)
        proxy_sim = proxies @ proxies.t()
        proxy_mask = ~torch.eye(self.num_classes, dtype=torch.bool, device=embeddings.device)
        ortho_penalty = torch.relu(proxy_sim[proxy_mask]).mean()
        
        # Sumar la penalización para forzar activamente que los centroides se repelan a >90°
        loss = loss_ce + self.ortho_penalty_weight * ortho_penalty
        return loss
