"""
Hard Negative Mining (OHEM) para Metric Learning

Implementa Online Hard Example Mining (Shrivastava et al., CVPR 2016) adaptado
para espacios embedding normalizados. Permite al modelo enfocarse en los ejemplos
más difíciles durante el entrenamiento.

Problema que resuelve:
- En datasets con clases morfológicamente similares, el gradiente de las clases
  fáciles ahoga la señal de error de las clases difíciles.
- OHEM computa gradientes SOLO sobre el top-k% de muestras con mayor pérdida.

Uso:
    miner = HardNegativeMiner(hard_ratio=0.3, min_samples=32)
    loss_per_sample = loss_func(embeddings, labels, reduction='none')
    hard_mask = miner.mine_hard_examples(loss_per_sample)
    loss = loss_per_sample[hard_mask].mean()

Referencias:
- Shrivastava et al., "Training Region-based Object Detectors with OHEM", CVPR 2016
- Adaptación para metric learning en hiperesfera S^{d-1}

Autor: MetricLearning Project
Fecha: 2026-03-06
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any


logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """
    Online Hard Example Mining para Metric Learning.
    
    Selecciona el top-k% de muestras con mayor pérdida para computar gradientes,
    enfocando el aprendizaje en los casos más difíciles.
    
    Args:
        hard_ratio (float): Fracción de samples más difíciles a seleccionar.
                           Default: 0.3 (top 30%)
        min_samples (int): Mínimo de samples a incluir, evita batches muy pequeños.
                          Default: 32
        warmup_epochs (int): Épocas sin mining (usa todas las muestras).
                            Default: 0
        adaptive_ratio (bool): Si True, ajusta ratio según dificultad promedio.
                              Default: False
    
    Attributes:
        hard_ratio (float): Fracción de hard examples
        min_samples (int): Mínimo de samples por batch
        warmup_epochs (int): Épocas de warmup
        adaptive_ratio (bool): Uso de ratio adaptativo
        current_epoch (int): Época actual (para warmup)
        stats (dict): Estadísticas de mining
    """
    
    def __init__(
        self,
        hard_ratio: float = 0.3,
        min_samples: int = 32,
        warmup_epochs: int = 0,
        adaptive_ratio: bool = False
    ):
        """Inicializa el Hard Negative Miner."""
        
        # Validaciones
        if not 0.0 < hard_ratio <= 1.0:
            raise ValueError(f"hard_ratio debe estar en (0, 1], got {hard_ratio}")
        if min_samples < 1:
            raise ValueError(f"min_samples debe ser >= 1, got {min_samples}")
        if warmup_epochs < 0:
            raise ValueError(f"warmup_epochs debe ser >= 0, got {warmup_epochs}")
        
        self.hard_ratio = hard_ratio
        self.min_samples = min_samples
        self.warmup_epochs = warmup_epochs
        self.adaptive_ratio = adaptive_ratio
        
        # Estado interno
        self.current_epoch = 0
        self.stats = {
            'total_batches': 0,
            'total_samples_seen': 0,
            'total_samples_mined': 0,
            'avg_loss_hard': 0.0,
            'avg_loss_easy': 0.0,
            'mining_active': False
        }
        
        logger.info(
            f"HardNegativeMiner initialized: ratio={hard_ratio}, "
            f"min_samples={min_samples}, warmup={warmup_epochs}, "
            f"adaptive={adaptive_ratio}"
        )
    
    def mine_hard_examples(
        self,
        loss_per_sample: torch.Tensor,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Selecciona hard examples basado en pérdida individual.
        
        Args:
            loss_per_sample (Tensor): Pérdida de cada sample, shape (B,)
            return_stats (bool): Si True, retorna también estadísticas.
        
        Returns:
            hard_mask (Tensor): Máscara booleana (B,) indicando hard examples.
                               Si return_stats=True, retorna (hard_mask, stats_dict)
        
        Example:
            >>> loss_per_sample = torch.tensor([0.5, 1.2, 0.3, 2.1, 0.8])
            >>> miner = HardNegativeMiner(hard_ratio=0.4)
            >>> hard_mask = miner.mine_hard_examples(loss_per_sample)
            >>> hard_mask
            tensor([False, True, False, True, False])  # Top 40%: 1.2, 2.1
        """
        
        if loss_per_sample.dim() != 1:
            raise ValueError(
                f"loss_per_sample debe ser 1D, got shape {loss_per_sample.shape}"
            )
        
        B = loss_per_sample.size(0)
        device = loss_per_sample.device
        
        # Durante warmup, usar todas las muestras
        if self.current_epoch < self.warmup_epochs:
            hard_mask = torch.ones(B, dtype=torch.bool, device=device)
            self.stats['mining_active'] = False
            
            if return_stats:
                stats = {'warmup': True, 'num_selected': B}
                return hard_mask, stats
            return hard_mask
        
        # Calcular k (número de samples a seleccionar)
        if self.adaptive_ratio:
            # Ratio adaptativo: más samples si pérdida promedio es alta
            avg_loss = loss_per_sample.mean().item()
            adjusted_ratio = min(1.0, self.hard_ratio * (1.0 + avg_loss))
            k = max(self.min_samples, int(adjusted_ratio * B))
        else:
            k = max(self.min_samples, int(self.hard_ratio * B))
        
        # Asegurar k <= B
        k = min(k, B)
        
        # Top-k pérdidas más altas
        if k == B:
            # Caso especial: todas las muestras son "hard"
            hard_mask = torch.ones(B, dtype=torch.bool, device=device)
            hard_indices = torch.arange(B, device=device)
        else:
            _, hard_indices = torch.topk(loss_per_sample, k, largest=True)
            hard_mask = torch.zeros(B, dtype=torch.bool, device=device)
            hard_mask[hard_indices] = True
        
        # Actualizar estadísticas
        self.stats['total_batches'] += 1
        self.stats['total_samples_seen'] += B
        self.stats['total_samples_mined'] += k
        self.stats['mining_active'] = True
        
        # Estadísticas de pérdida (hard vs easy)
        if k < B:
            easy_mask = ~hard_mask
            self.stats['avg_loss_hard'] = loss_per_sample[hard_mask].mean().item()
            self.stats['avg_loss_easy'] = loss_per_sample[easy_mask].mean().item()
        else:
            self.stats['avg_loss_hard'] = loss_per_sample.mean().item()
            self.stats['avg_loss_easy'] = 0.0
        
        if return_stats:
            stats = {
                'warmup': False,
                'num_selected': k,
                'num_total': B,
                'selection_ratio': k / B,
                'avg_loss_hard': self.stats['avg_loss_hard'],
                'avg_loss_easy': self.stats['avg_loss_easy'],
                'loss_ratio': (self.stats['avg_loss_hard'] / self.stats['avg_loss_easy']
                              if self.stats['avg_loss_easy'] > 0 else float('inf'))
            }
            return hard_mask, stats
        
        return hard_mask
    
    def set_epoch(self, epoch: int):
        """
        Actualiza la época actual (para warmup).
        
        Args:
            epoch (int): Número de época actual (0-indexed)
        """
        self.current_epoch = epoch
        
        if epoch == self.warmup_epochs:
            logger.info(
                f"HardNegativeMiner: Warmup completado en epoch {epoch}, "
                f"mining activado con ratio={self.hard_ratio}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas acumuladas de mining.
        
        Returns:
            dict: Estadísticas con keys:
                - total_batches: Total de batches procesados
                - total_samples_seen: Total de samples vistos
                - total_samples_mined: Total de samples seleccionados
                - avg_mining_ratio: Ratio promedio de mining
                - avg_loss_hard: Pérdida promedio de hard examples
                - avg_loss_easy: Pérdida promedio de easy examples
                - mining_active: Si el mining está activo
        """
        stats = self.stats.copy()
        
        if stats['total_samples_seen'] > 0:
            stats['avg_mining_ratio'] = (
                stats['total_samples_mined'] / stats['total_samples_seen']
            )
        else:
            stats['avg_mining_ratio'] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reinicia las estadísticas acumuladas."""
        self.stats = {
            'total_batches': 0,
            'total_samples_seen': 0,
            'total_samples_mined': 0,
            'avg_loss_hard': 0.0,
            'avg_loss_easy': 0.0,
            'mining_active': False
        }
        logger.debug("HardNegativeMiner: Estadísticas reiniciadas")
    
    def __repr__(self) -> str:
        """Representación string del miner."""
        return (
            f"HardNegativeMiner(hard_ratio={self.hard_ratio}, "
            f"min_samples={self.min_samples}, warmup_epochs={self.warmup_epochs}, "
            f"adaptive={self.adaptive_ratio}, current_epoch={self.current_epoch})"
        )
