"""
Zero-Shot Evaluation Protocol for Deep Metric Learning.

Protocolo estándar en FGVC siguiendo:
- CUB-200-2011 (Wah et al. 2011)
- Stanford Online Products (Song et al. CVPR 2016)

División por CLASES, no por imágenes: entrena en K clases,
evalúa en C-K clases nunca vistas.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from src.utils.metrics_utils import compute_all_metrics
from src.utils.inference_utils import extract_embeddings

logger = logging.getLogger(__name__)


def split_dataset_by_classes(
    dataset,
    train_classes: List[int],
    test_classes: List[int]
) -> Tuple[Subset, Subset]:
    """
    Divide dataset por clases para evaluación Zero-Shot.
    
    Args:
        dataset: Dataset completo
        train_classes: Índices de clases para entrenamiento
        test_classes: Índices de clases para evaluación (nunca vistas)
        
    Returns:
        (train_subset, test_subset)
    """
    # Obtener todos los labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Iterar dataset para obtener labels
        labels = []
        for _, label in dataset:
            labels.append(label)
        labels = np.array(labels)
    
    # Índices de muestras por clase
    train_indices = []
    test_indices = []
    
    for idx, label in enumerate(labels):
        if label in train_classes:
            train_indices.append(idx)
        elif label in test_classes:
            test_indices.append(idx)
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    logger.info(f"Zero-Shot split: {len(train_indices)} train samples from {len(train_classes)} classes")
    logger.info(f"                 {len(test_indices)} test samples from {len(test_classes)} classes")
    
    return train_subset, test_subset


def evaluate_zero_shot(
    model: torch.nn.Module,
    test_dataset,
    device: torch.device,
    batch_size: int = 32,
    class_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Evaluación Zero-Shot en clases nunca vistas.
    
    El modelo fue entrenado en K clases, ahora se evalúa en C-K clases
    completamente nuevas. Demuestra capacidad de generalización del
    espacio de embeddings.
    
    Args:
        model: Modelo entrenado (congelado)
        test_dataset: Dataset con clases nunca vistas
        device: Device de cómputo
        batch_size: Batch size para inferencia
        class_names: Nombres de clases (opcional)
        
    Returns:
        Dict con métricas Zero-Shot
    """
    model.eval()
    
    logger.info("="*70)
    logger.info("ZERO-SHOT EVALUATION (Unseen Classes)")
    logger.info("="*70)
    
    # DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Extraer embeddings
    with torch.no_grad():
        embeddings, labels = extract_embeddings(model, test_loader, device)
    
    logger.info(f"Extracted {len(embeddings)} embeddings from {len(np.unique(labels))} unseen classes")
    
    # Calcular métricas SOTA
    metrics = compute_all_metrics(
        embeddings=embeddings,
        labels=labels,
        k_list=[1, 5, 10],
        class_names=class_names
    )
    
    # Añadir flag de Zero-Shot
    metrics['evaluation_type'] = 'zero_shot'
    metrics['num_unseen_classes'] = len(np.unique(labels))
    
    return metrics


def evaluate_zero_shot_protocol(
    model: torch.nn.Module,
    full_dataset,
    num_train_classes: int,
    device: torch.device,
    batch_size: int = 32,
    seed: int = 42
) -> Dict[str, any]:
    """
    Protocolo completo Zero-Shot siguiendo CUB-200/SOP.
    
    Divide dataset en K clases para "entrenamiento simulado" y C-K
    para evaluación Zero-Shot. Como el modelo ya está entrenado,
    solo evalúa en las clases reservadas.
    
    Args:
        model: Modelo pre-entrenado
        full_dataset: Dataset completo
        num_train_classes: Número de clases a "reservar" como vistas
        device: Device de cómputo
        batch_size: Batch size
        seed: Seed para reproducibilidad
        
    Returns:
        Dict con resultados del protocolo Zero-Shot
    """
    np.random.seed(seed)
    
    # Obtener clases únicas
    if hasattr(full_dataset, 'classes'):
        num_classes = len(full_dataset.classes)
        class_names = full_dataset.classes
    else:
        # Determinar clases del dataset
        labels = []
        for _, label in full_dataset:
            labels.append(label)
        unique_classes = np.unique(labels)
        num_classes = len(unique_classes)
        class_names = [f"Class_{i}" for i in unique_classes]
    
    if num_train_classes >= num_classes:
        raise ValueError(f"num_train_classes ({num_train_classes}) debe ser < num_classes ({num_classes})")
    
    # Dividir clases aleatoriamente
    all_classes = np.arange(num_classes)
    np.random.shuffle(all_classes)
    
    train_classes = all_classes[:num_train_classes].tolist()
    test_classes = all_classes[num_train_classes:].tolist()
    
    logger.info(f"Zero-Shot Protocol: {num_train_classes} seen classes, {len(test_classes)} unseen classes")
    logger.info(f"Seen classes: {train_classes}")
    logger.info(f"Unseen classes: {test_classes}")
    
    # Crear subset de clases no vistas
    _, test_subset = split_dataset_by_classes(full_dataset, train_classes, test_classes)
    
    # Evaluar en clases no vistas
    test_class_names = [class_names[i] for i in test_classes if i < len(class_names)]
    
    metrics = evaluate_zero_shot(
        model=model,
        test_dataset=test_subset,
        device=device,
        batch_size=batch_size,
        class_names=test_class_names
    )
    
    # Añadir información del protocolo
    metrics['protocol'] = {
        'num_seen_classes': num_train_classes,
        'num_unseen_classes': len(test_classes),
        'seen_classes': train_classes,
        'unseen_classes': test_classes,
        'seed': seed
    }
    
    return metrics


def compare_standard_vs_zeroshot(
    model: torch.nn.Module,
    standard_dataset,
    zeroshot_dataset,
    device: torch.device,
    batch_size: int = 32,
    class_names: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compara evaluación estándar vs Zero-Shot.
    
    Útil para demostrar capacidad de generalización:
    - Evaluación estándar: clases vistas durante entrenamiento
    - Evaluación Zero-Shot: clases nunca vistas
    
    Args:
        model: Modelo entrenado
        standard_dataset: Dataset con clases vistas
        zeroshot_dataset: Dataset con clases no vistas
        device: Device
        batch_size: Batch size
        class_names: Nombres de clases
        
    Returns:
        {'standard': metrics, 'zero_shot': metrics, 'comparison': delta}
    """
    logger.info("="*70)
    logger.info("STANDARD vs ZERO-SHOT COMPARISON")
    logger.info("="*70)
    
    # Evaluación estándar
    logger.info("\n--- STANDARD EVALUATION (Seen Classes) ---")
    standard_loader = DataLoader(
        standard_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    with torch.no_grad():
        std_embeddings, std_labels = extract_embeddings(model, standard_loader, device)
    
    standard_metrics = compute_all_metrics(std_embeddings, std_labels, class_names=class_names)
    standard_metrics['evaluation_type'] = 'standard'
    
    # Evaluación Zero-Shot
    logger.info("\n--- ZERO-SHOT EVALUATION (Unseen Classes) ---")
    zeroshot_metrics = evaluate_zero_shot(
        model, zeroshot_dataset, device, batch_size, class_names
    )
    
    # Comparación
    comparison = {
        'R@1_delta': standard_metrics.get('R@1', 0) - zeroshot_metrics.get('R@1', 0),
        'R@5_delta': standard_metrics.get('R@5', 0) - zeroshot_metrics.get('R@5', 0),
        'mAP@R_delta': standard_metrics.get('mAP@R', 0) - zeroshot_metrics.get('mAP@R', 0),
        'NMI_delta': standard_metrics.get('NMI', 0) - zeroshot_metrics.get('NMI', 0),
        'F1_macro_delta': standard_metrics.get('F1_macro', 0) - zeroshot_metrics.get('F1_macro', 0),
    }
    
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY (Standard - Zero-Shot)")
    logger.info("="*70)
    logger.info(f"R@1 delta:      {comparison['R@1_delta']:+.4f}")
    logger.info(f"R@5 delta:      {comparison['R@5_delta']:+.4f}")
    logger.info(f"mAP@R delta:    {comparison['mAP@R_delta']:+.4f}")
    logger.info(f"NMI delta:      {comparison['NMI_delta']:+.4f}")
    logger.info(f"F1-Macro delta: {comparison['F1_macro_delta']:+.4f}")
    logger.info("="*70)
    logger.info("Nota: Delta pequeño indica BUENA generalización a clases no vistas")
    logger.info("="*70)
    
    return {
        'standard': standard_metrics,
        'zero_shot': zeroshot_metrics,
        'comparison': comparison
    }
