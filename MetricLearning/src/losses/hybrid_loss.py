import torch
import torch.nn as nn
from src.losses.sg_softmax import SGSoftmaxLoss
from src.losses.ldm_loss import LearnableDynamicMarginMSLoss
# Support standard MS Loss if pytorch_metric_learning is available
try:
    from pytorch_metric_learning.losses import MultiSimilarityLoss
except ImportError:
    MultiSimilarityLoss = None

class HybridProxyMSLoss(nn.Module):
    """
    Hybrid Proxy MS Loss.
    Combina la repulsión global inter-clase de los Proxies (SG-Softmax / ACML)
    con la dispersión fina intra-clase de la Multi-Similarity Loss (o LDM).
    
    Evita el colapso dimensional forzando repulsión local, y resuelve la dependencia
    del batch del MS Loss fijando un centro global para cada clase.
    """
    def __init__(self, embedding_dim: int, num_classes: int, 
                 proxy_scale: float = 30.0, proxy_margin: float = 0.4,
                 ms_alpha: float = 2.0, ms_beta: float = 50.0, ms_base_margin: float = 0.5,
                 w_global: float = 1.0, w_local: float = 1.0, w_cov: float = 2.0,
                 use_ldm: bool = True):
        super().__init__()
        
        # 1. Componente Global: Proxy con Stop-Gradient (variante estable de ACML)
        self.proxy_loss = SGSoftmaxLoss(
            embedding_dim=embedding_dim, 
            num_classes=num_classes, 
            scale=proxy_scale, 
            margin=proxy_margin
        )
        
        # 2. Componente Local: MS Loss tradicional o LDM
        if use_ldm:
            self.ms_loss = LearnableDynamicMarginMSLoss(
                num_classes=num_classes,
                alpha=ms_alpha,
                beta=ms_beta,
                base_margin=ms_base_margin
            )
        else:
            if MultiSimilarityLoss is None:
                raise ImportError("pytorch_metric_learning is required for traditional MS Loss.")
            self.ms_loss = MultiSimilarityLoss(alpha=ms_alpha, beta=ms_beta, base=ms_base_margin)
            
        self.w_global = w_global
        self.w_local = w_local
        self.w_cov = w_cov
        self.use_ldm = use_ldm

    def _covariance_decorrelation_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Penaliza la correlación cruzada entre las 128 dimensiones espectrales.
        Obliga a la red a no repetir características y usar todo el espacio latente.
        """
        B, D = embeddings.shape
        if B <= 1:
            return torch.tensor(0.0, device=embeddings.device)
            
        # Centrar los embeddings (Media 0 por columna)
        x_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        # Calcular covarianza
        cov = (x_centered.t() @ x_centered) / (B - 1)
        
        # Extraer elementos fuera de la diagonal
        off_diag = cov - torch.diag(torch.diag(cov))
        # Penalizar fuertemente la magnitud de la covarianza cruzada
        loss_cov = (off_diag ** 2).sum() / D
        
        # Inyectar Variance Preserving Loss (VICReg style)
        # Forzamos que la desviación estándar de cada canal sea de al menos 1.0
        # Esto previene que la red apague 124 dimensiones (varianza 0) para eludir la covarianza.
        std = torch.sqrt(x_centered.var(dim=0) + 1e-04)
        loss_var = torch.relu(1.0 - std).mean()
        
        return loss_cov + loss_var

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, indices_tuple=None):
        """
        Calcula la pérdida híbrida con penalización espectral.
        """
        # Pérdida global basada en proxies (evalúa todos los embeddings contra los proxies)
        loss_global = self.proxy_loss(embeddings, labels)
        
        # Pérdida local basada en pares
        if self.use_ldm:
            loss_local = self.ms_loss(embeddings, labels, indices_tuple=indices_tuple)
        else:
            loss_local = self.ms_loss(embeddings, labels, indices_tuple)
            
        # Covariance Decorrelation Penalty
        loss_cov = self._covariance_decorrelation_loss(embeddings)
        
        # FIX: Eliminamos el multiplicador * self.proxy_loss.scale para evitar 
        # gradientes excesivos (norm > 700) que activaban el grad_clip de forma innecesaria.
        # Ahora w_local y w_global se calibran empíricamente en config.yaml
        effective_w_local = self.w_local
        
        return self.w_global * loss_global + effective_w_local * loss_local + self.w_cov * loss_cov
