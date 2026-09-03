"""
Self-Distillation with EMA Teacher for Deep Metric Learning.
Based on: Zeng et al. (2024) "Improving Deep Metric Learning via
Self-Distillation and Online Batch Diffusion Process" (Visual Intelligence 2024)

Maintains an Exponential Moving Average (EMA) copy of the student model
as teacher, and uses KL divergence between their similarity distributions
as a regularization signal that captures intra-class structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging

logger = logging.getLogger(__name__)


class EMATeacher:
    """
    Exponential Moving Average Teacher for self-distillation.
    
    Maintains a slowly-updated copy of the student model:
        theta_teacher = alpha * theta_teacher + (1 - alpha) * theta_student
    
    Args:
        student_model (nn.Module): The student model to track.
        alpha (float): EMA decay rate. Higher = slower update. Default 0.999.
    """
    
    def __init__(self, student_model, alpha=0.999):
        self.alpha = alpha
        self.teacher = copy.deepcopy(student_model)
        # Freeze teacher parameters
        for param in self.teacher.parameters():
            param.requires_grad = False
        logger.info(f"EMA Teacher initialized (alpha={alpha})")
    
    @torch.no_grad()
    def update(self, student_model):
        """Update teacher weights with EMA of student weights."""
        for t_param, s_param in zip(self.teacher.parameters(),
                                     student_model.parameters()):
            t_param.data.mul_(self.alpha).add_(s_param.data, alpha=1.0 - self.alpha)
    
    @torch.no_grad()
    def get_teacher_embeddings(self, images, masks=None):
        """Get embeddings from teacher model (no grad)."""
        self.teacher.eval()
        if masks is not None:
            return self.teacher(images, mask=masks)
        return self.teacher(images)
    
    def to(self, device):
        """Move teacher to device."""
        self.teacher.to(device)
        return self


class SelfDistillationLoss(nn.Module):
    """
    Self-Distillation loss via KL divergence on similarity distributions.
    
    Computes pairwise similarity matrices from student and teacher embeddings,
    converts them to probability distributions via softmax with temperature,
    then minimizes KL divergence.
    
    Args:
        temperature (float): Softmax temperature. Lower = sharper. Default 0.1.
        lambda_sd (float): Weight of self-distillation loss. Default 0.5.
    """
    
    def __init__(self, temperature=0.1, lambda_sd=0.5):
        super().__init__()
        self.temperature = temperature
        self.lambda_sd = lambda_sd
    
    def forward(self, student_embeddings, teacher_embeddings):
        """
        Compute self-distillation loss.
        
        Args:
            student_embeddings: (B, D) from student model (with grad)
            teacher_embeddings: (B, D) from teacher model (no grad)
            
        Returns:
            Scalar self-distillation loss
        """
        # Normalize embeddings to properly compute cosine similarity
        student_norm = F.normalize(student_embeddings, p=2, dim=1, eps=1e-5)
        teacher_norm = F.normalize(teacher_embeddings, p=2, dim=1, eps=1e-5)
        
        # Pairwise cosine similarity matrices
        student_sim = student_norm @ student_norm.t()
        teacher_sim = teacher_norm @ teacher_norm.t()
        
        # Remove diagonal (self-similarity)
        B = student_sim.shape[0]
        mask = ~torch.eye(B, dtype=torch.bool, device=student_sim.device)
        student_sim = student_sim[mask].view(B, B - 1)
        teacher_sim = teacher_sim[mask].view(B, B - 1)
        
        # Softmax with temperature -> probability distributions
        student_log_prob = F.log_softmax(student_sim / self.temperature, dim=1)
        teacher_prob = F.softmax(teacher_sim / self.temperature, dim=1)
        
        # KL Divergence: D_KL(teacher || student)
        kl_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean')
        
        return self.lambda_sd * kl_loss
