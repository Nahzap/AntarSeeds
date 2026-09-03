import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableDynamicMarginMSLoss(nn.Module):
    """
    Learnable Dynamic Margin Multi-Similarity Loss (LDM-MS).
    Para garantizar ortogonalidad (>90°), desacoplamos el margen positivo y negativo.
    - pos_margin: Atrae los positivos a ser muy similares (inicializado en base_margin).
    - neg_margin: Empuja los negativos hacia 0 o menos (inicializado en 0.0 para ortogonalidad global).
    """
    def __init__(self, num_classes: int, alpha: float = 2.0, beta: float = 50.0, base_margin: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base_margin = base_margin
        
        # Margen dinámico positivo
        self.pos_margin = nn.Parameter(torch.ones(num_classes) * base_margin)
        # Margen dinámico negativo, fijado inicialmente en 0.0 (ortogonalidad estricta 90°)
        self.neg_margin = nn.Parameter(torch.ones(num_classes) * 0.0)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, indices_tuple=None):
        """
        Calcula la pérdida. 
        Nota: el uso de mineros duros (indices_tuple) con LDM/MSLoss distorsiona el soft-mining 
        del LogSumExp y causa colapso dimensional y desbalanceo.
        """
        device = embeddings.device
        
        # Similitud coseno par a par (matriz NxN)
        sim_mat = torch.matmul(embeddings, embeddings.t())
        
        # Obtener los márgenes para los anchors en este batch
        # Asegurar restricciones estrictas:
        # 1. pos_margin >= base_margin (alta similitud intra-clase)
        # 2. neg_margin <= 0.0 (garantizar >90° de separación inter-clase)
        p_margins_i = torch.clamp(self.pos_margin[labels], min=self.base_margin).unsqueeze(1)
        n_margins_i = torch.clamp(self.neg_margin[labels], max=0.0).unsqueeze(1)
        
        # Crear máscaras globales
        labels_expanded = labels.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.t()) & ~torch.eye(labels.size(0), dtype=torch.bool, device=device)
        neg_mask = labels_expanded != labels_expanded.t()
        
        # Si se proporcionan tuplas minadas (a, p, a, n), aplicarlas como máscara adicional
        if indices_tuple is not None:
            if len(indices_tuple) == 3:
                a_p, p_p, n_n = indices_tuple
                a_n = a_p
            elif len(indices_tuple) == 4:
                a_p, p_p, a_n, n_n = indices_tuple
            else:
                raise ValueError(f"indices_tuple expected 3 or 4 elements, got {len(indices_tuple)}")
            
            # Limpiar máscaras y habilitar solo los pares minados
            mined_pos_mask = torch.zeros_like(pos_mask)
            if len(a_p) > 0:
                mined_pos_mask[a_p, p_p] = True
            pos_mask = pos_mask & mined_pos_mask
            
            mined_neg_mask = torch.zeros_like(neg_mask)
            if len(a_n) > 0:
                mined_neg_mask[a_n, n_n] = True
            neg_mask = neg_mask & mined_neg_mask
            
        # Diferencias con los márgenes desacoplados
        pos_diff = sim_mat - p_margins_i
        neg_diff = sim_mat - n_margins_i
        
        # Exponenciales con enmascaramiento seguro
        pos_exp = torch.exp(-self.alpha * pos_diff) * pos_mask.float()
        neg_exp = torch.exp(self.beta * neg_diff) * neg_mask.float()
        
        # Sumas
        pos_sum = pos_exp.sum(dim=1)
        neg_sum = neg_exp.sum(dim=1)
        
        has_pos = pos_mask.sum(dim=1) > 0
        has_neg = neg_mask.sum(dim=1) > 0
        
        # Calcular términos
        loss_pos = torch.where(has_pos, (1.0 / self.alpha) * torch.log(1.0 + pos_sum), torch.zeros_like(pos_sum))
        loss_neg = torch.where(has_neg, (1.0 / self.beta) * torch.log(1.0 + neg_sum), torch.zeros_like(neg_sum))
        
        loss = loss_pos + loss_neg
        
        # Retornar el promedio sobre los anclajes válidos (que tienen pos o neg)
        valid_anchors = has_pos | has_neg
        if valid_anchors.any():
            return loss[valid_anchors].mean()
        return torch.tensor(0.0, device=device, requires_grad=True)

