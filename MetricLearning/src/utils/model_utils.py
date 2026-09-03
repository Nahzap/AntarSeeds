"""
Utilidades para carga y manejo de modelos.
Centraliza la lógica de carga de checkpoints usada en múltiples scripts.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Dict,
    device: torch.device,
    eval_mode: bool = True
) -> nn.Module:
    """
    Carga modelo AnalogyNet desde checkpoint.
    
    Args:
        checkpoint_path: Ruta al archivo checkpoint (.pth)
        config: Diccionario de configuración con 'model' key
        device: Dispositivo torch (cuda/cpu)
        eval_mode: Si True, pone modelo en modo evaluación
        
    Returns:
        Modelo cargado y listo para usar
        
    Example:
        >>> config = load_config('config.yaml')
        >>> device = torch.device('cuda')
        >>> model = load_model_from_checkpoint('models/final_model.pth', config, device)
    """
    from src.utils.model_factory import create_analogy_net_from_config

    logger.info(f"Loading model from {checkpoint_path}")

    model = create_analogy_net_from_config(config)
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Cargar state dict (compatible con ambos formatos)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    # Mover a device
    model = model.to(device)
    
    # Modo evaluación si se requiere
    if eval_mode:
        model.eval()
    
    logger.info("Model loaded successfully")
    
    return model


def save_model_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    val_loss: float,
    filepath: str,
    config: Optional[Dict] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    additional_info: Optional[Dict] = None
):
    """
    Guarda checkpoint del modelo.
    
    Args:
        model: Modelo a guardar
        optimizer: Optimizador (opcional)
        epoch: Número de época
        val_loss: Loss de validación
        filepath: Ruta donde guardar
        config: Configuración (opcional)
        scheduler: Scheduler (opcional)
        scaler: GradScaler para AMP (opcional)
        additional_info: Información adicional (opcional)
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    if config is not None:
        checkpoint['config'] = config
    
    if additional_info is not None:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved: {filepath}")


def get_model_info(model: nn.Module) -> Dict:
    """
    Obtiene información del modelo.
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        Dict con información del modelo
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'model_name': model.__class__.__name__,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }
    
    if hasattr(model, 'embedding_dim'):
        info['embedding_dim'] = model.embedding_dim
    
    if hasattr(model, 'backbone_name'):
        info['backbone'] = model.backbone_name
    
    return info
