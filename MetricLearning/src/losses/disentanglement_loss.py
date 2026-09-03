import torch
import torch.nn as nn

class DisentanglementLoss(nn.Module):
    """
    Mutual Information Minimization between Shape (LF) and Texture (HF) branches.
    
    Assuming `prenorm_embeddings` is formed by concatenating two L2-normalized vectors 
    of equal size (LF and HF), this loss computes the squared cosine similarity 
    between them. By minimizing this, we force the network to make the Texture 
    features mathematically orthogonal to the Shape features, preventing the 
    network from storing redundant information.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, prenorm_embeddings: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            prenorm_embeddings: Tensor of shape (B, D). E.g. D=128 (64 LF + 64 HF).
            labels: Ignored. Included for signature compatibility.
            
        Returns:
            Scalar loss tensor.
        """
        B, D = prenorm_embeddings.shape
        if D % 2 != 0:
            raise ValueError(f"DisentanglementLoss requires an even number of dimensions. Got D={D}")
        
        half_d = D // 2
        # Since AnalogyNet already L2-normalizes these halves, their dot product is the cosine sim
        lf = prenorm_embeddings[:, :half_d]
        hf = prenorm_embeddings[:, half_d:]
        
        cosine_sim = (lf * hf).sum(dim=1)
        
        # Penalize any non-zero similarity (we want orthogonality, cos_sim = 0)
        loss = (cosine_sim ** 2).mean()
        
        return self.weight * loss
