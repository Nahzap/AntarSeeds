import torch
import torch.nn as nn

class CovarianceDecorrelationLoss(nn.Module):
    """
    Covariance Decorrelation (Whitening) Loss (inspired by VICReg/Barlow Twins).
    
    Forces the covariance matrix of the batch embeddings to approximate the 
    identity matrix (I_D). This mathematically guarantees full rank (D dimensions) 
    utilization by penalizing redundant, correlated features.
    """
    def __init__(self, target_variance: float = 1.0, var_weight: float = 1.0, cov_weight: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.target_variance = target_variance
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.eps = eps

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            embeddings: Tensor of shape (B, D).
            labels: Ignored. Included for compatibility.
            
        Returns:
            Scalar loss tensor.
        """
        B, D = embeddings.shape
        if B <= 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Center the embeddings along the batch dimension
        x = embeddings - embeddings.mean(dim=0)

        # 1. Variance Loss: Push variance of each dimension to `target_variance`
        std = torch.sqrt(x.var(dim=0) + self.eps)
        loss_var = torch.mean(torch.relu(self.target_variance - std))

        # 2. Covariance Loss: Push off-diagonal elements of covariance matrix to 0
        cov = (x.T @ x) / (B - 1)
        
        # We want to sum the squared off-diagonal elements
        # Extract diagonal elements to subtract them later or just mask them
        mask = ~torch.eye(D, dtype=torch.bool, device=embeddings.device)
        off_diagonals = cov[mask]
        loss_cov = (off_diagonals.pow(2)).sum() / D

        loss = self.var_weight * loss_var + self.cov_weight * loss_cov
        return loss
