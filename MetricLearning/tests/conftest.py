"""
Fixtures compartidos para tests unitarios del proyecto MetricLearning.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Dispositivo para tests (CPU para CI)."""
    return torch.device('cpu')


@pytest.fixture
def sample_batch():
    """Batch de imágenes sintéticas para tests."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def small_batch():
    """Batch pequeño para tests rápidos."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_embeddings():
    """Embeddings L2-normalizados sintéticos con 3 clases."""
    np.random.seed(42)
    n_per_class = 20
    dim = 128
    
    embeddings = []
    labels = []
    
    for cls_id in range(3):
        center = np.random.randn(dim)
        center = center / np.linalg.norm(center)
        
        for _ in range(n_per_class):
            noise = np.random.randn(dim) * 0.1
            emb = center + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
            labels.append(cls_id)
    
    return np.array(embeddings, dtype=np.float32), np.array(labels)


@pytest.fixture
def projection_head_config():
    """Config estándar para projection head (como en config.yaml)."""
    return {
        'hidden_dims': [512, 256],
        'use_batch_norm': True,
        'dropout': 0.0
    }


@pytest.fixture
def model_config():
    """Config mínima de modelo para tests."""
    return {
        'model': {
            'backbone': 'resnet18',
            'embedding_dim': 128,
            'pretrained': False,
            'projection_head': {
                'hidden_dims': [512, 256],
                'use_batch_norm': True,
                'dropout': 0.0
            }
        }
    }
