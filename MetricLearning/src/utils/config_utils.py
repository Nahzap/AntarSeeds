"""
Utilidades para manejo de configuración.
Centraliza carga y procesamiento de config.yaml
"""

import yaml
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> Dict:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario de configuración
        
    Example:
        >>> config = load_config('config.yaml')
        >>> print(config['model']['backbone'])
        'resnet50'
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    
    return config


def resolve_run_config_path(checkpoint_path: str, fallback: str = "config.yaml") -> str:
    """Prefer run-specific config/config.yaml next to a checkpoint."""
    ckpt = Path(checkpoint_path).resolve()
    run_dir = ckpt.parent.parent
    for candidate in (run_dir / "config" / "config.yaml", run_dir / "config.yaml"):
        if candidate.exists():
            return str(candidate)
    fallback_path = Path(fallback)
    if fallback_path.exists():
        return str(fallback_path)
    return fallback


def merge_global_active_detector(config: Dict, global_config_path: str = "config.yaml") -> Dict:
    """
    Aplica grain_detection.active_detector del config.yaml del proyecto.

    El entrenamiento DML usa siempre .seg; el detector activo (U²-Net / ViT-denso)
    se elige en la pestaña Detección y debe usarse en inferencia y evaluación E2E,
    aunque el snapshot del run tenga otro valor.
    """
    path = Path(global_config_path)
    if not path.is_absolute():
        for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
            candidate = base / path
            if candidate.exists():
                path = candidate
                break
    if not path.exists():
        return config
    try:
        global_cfg = load_config(str(path))
    except FileNotFoundError:
        return config
    active = (global_cfg.get("grain_detection") or {}).get("active_detector")
    global_gd = global_cfg.get("grain_detection") or {}
    merged = dict(config)
    gd = dict(merged.get("grain_detection") or {})
    # Global grain_detection is the source of truth for inference (no key whitelist).
    for key, value in global_gd.items():
        if key == "active_detector":
            continue
        gd[key] = value
    if active and active.get("backend"):
        gd["active_detector"] = active
        backend = str(active.get("backend", "")).lower()
        if backend:
            gd["localization_detector"] = backend
    merged["grain_detection"] = gd
    if active and active.get("backend"):
        logger.info(
            "[config] active_detector desde %s → backend=%s",
            path,
            active.get("backend"),
        )
    return merged


def get_transforms_from_config(config: Dict, mode: str = 'val'):
    """
    Obtiene transforms desde configuración.
    
    Args:
        config: Configuración completa
        mode: 'train' o 'val'
        
    Returns:
        Transform pipeline
        
    Example:
        >>> config = load_config()
        >>> train_transforms = get_transforms_from_config(config, 'train')
        >>> val_transforms = get_transforms_from_config(config, 'val')
    """
    from src.data.transforms_enhanced import get_train_transforms, get_val_transforms
    
    transform_config = merge_configs(config['augmentation'], config['data'])
    backbone_name = get_backbone_name(config)
    
    if mode == 'train':
        aug_mode = config.get('sota', {}).get('augmentation_mode', 'full')
        return get_train_transforms(transform_config, mode=aug_mode, backbone_name=backbone_name)
    elif mode == 'val' or mode == 'test':
        return get_val_transforms(transform_config, backbone_name=backbone_name)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'train' or 'val'")


def merge_configs(*configs: Dict) -> Dict:
    """
    Merge múltiples diccionarios de configuración.
    
    Args:
        *configs: Diccionarios a mergear
        
    Returns:
        Diccionario mergeado
        
    Example:
        >>> config1 = {'a': 1, 'b': 2}
        >>> config2 = {'b': 3, 'c': 4}
        >>> merged = merge_configs(config1, config2)
        >>> print(merged)  # {'a': 1, 'b': 3, 'c': 4}
    """
    merged = {}
    for config in configs:
        if config is not None:
            merged.update(config)
    return merged


def get_backbone_name(config: Dict) -> str:
    """
    Obtiene nombre del backbone con compatibilidad hacia atrás.
    
    Args:
        config: Configuración completa
        
    Returns:
        Nombre del backbone
        
    Example:
        >>> config = load_config()
        >>> backbone = get_backbone_name(config)
        >>> print(backbone)  # 'dinov2_vits14'
    """
    # Compatibilidad hacia atrás: backbone_name (nuevo) o backbone (legacy)
    return config['model'].get('backbone_name', config['model'].get('backbone', 'dinov2_vits14'))


def get_device_from_config(config: Dict) -> str:
    """
    Obtiene dispositivo desde configuración.
    
    Args:
        config: Configuración completa
        
    Returns:
        'cuda' o 'cpu'
    """
    import torch
    
    use_gpu = config.get('hardware', {}).get('use_gpu', True)
    
    if use_gpu and torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def validate_config(config: Dict) -> bool:
    """
    Valida que la configuración tenga todos los campos requeridos.
    
    Args:
        config: Configuración a validar
        
    Returns:
        True si es válida
        
    Raises:
        ValueError si falta algún campo requerido
    """
    required_keys = ['model', 'data', 'training']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validar sección model
    if 'model' not in config:
        raise ValueError("Falta sección 'model' en config")
    # Aceptar backbone_name (nuevo) o backbone (legacy)
    if 'backbone_name' not in config['model'] and 'backbone' not in config['model']:
        raise ValueError("Falta 'model.backbone_name' o 'model.backbone' en config")
    
    if 'embedding_dim' not in config['model']:
        raise ValueError("Missing 'embedding_dim' in model config")
    
    data_keys = ['train_dir', 'val_dir', 'test_dir']
    for key in data_keys:
        if key not in config['data']:
            raise ValueError(f"Missing '{key}' in data config")
    
    logger.info("Configuration validated successfully")
    
    return True
