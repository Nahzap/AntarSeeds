"""
Amplitude Penalty Loss / Clamping Penalty.
Penalizes extreme embedding values to constrain the dimensions.
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class AmplitudePenaltyLoss(nn.Module):
    def __init__(self, lambda_penalty=0.1, penalty_type='clamping', margin=0.5):
        super().__init__()
        self.lambda_penalty = lambda_penalty
        self.penalty_type = penalty_type
        self.margin = margin
        
        if penalty_type == 'huber':
            self.loss_fn = nn.HuberLoss(delta=margin, reduction='mean')
        elif penalty_type == 'l2':
            self.loss_fn = lambda x: torch.mean(x**2)
        elif penalty_type == 'l1':
            self.loss_fn = lambda x: torch.mean(torch.abs(x))
        elif penalty_type == 'l4':
            self.loss_fn = lambda x: torch.mean(x**4)
        elif penalty_type == 'clamping':
            self.loss_fn = lambda x: torch.mean(torch.relu(torch.abs(x) - margin))
        else:
            raise ValueError(f"Unknown penalty type: {penalty_type}")

    def forward(self, embeddings, labels=None):
        if self.penalty_type == 'huber':
            target = torch.zeros_like(embeddings)
            penalty = self.loss_fn(embeddings, target)
        else:
            penalty = self.loss_fn(embeddings)
            
        return self.lambda_penalty * penalty
