import torch
import torch.nn as nn

class HypersphericalUniformityLoss(nn.Module):
    """
    Hyperspherical Uniformity Loss (Wang & Isola, ICML 2020)
    
    Mathematically pushes all embeddings to distribute uniformly across the 
    hypersphere, avoiding dimensional collapse and naturally maximizing 
    pairwise angles without artificially hardcoded margins.
    
    L_uniform = log E_{x, y} [ e^{-t ||x - y||_2^2} ]
    """
    def __init__(self, t: float = 2.0):
        super().__init__()
        self.t = t

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            embeddings: Tensor of shape (B, D). Should be L2 normalized.
            labels: Ignored. Included for compatibility with trainer pipelines.
            
        Returns:
            Scalar loss tensor.
        """
        # Ensure representations are L2-normalized
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise Euclidean distances squared: ||x - y||^2 = 2 - 2 * (x dot y)
        # We can also compute it directly via torch.cdist or matrix multiplication.
        sq_pdist = torch.pdist(embeddings, p=2).pow(2)
        
        # Apply Gaussian RBF potential and compute the log mean
        loss = torch.log(torch.mean(torch.exp(-self.t * sq_pdist)))
        
        return loss
