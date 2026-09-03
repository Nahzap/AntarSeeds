"""
SOTA Loss Functions and Regularizers for Deep Metric Learning.

Implements:
- NIR: Non-Isotropy Regularization (Roth et al., CVPR 2022)
- HIER: Hierarchical Regularization (Kim et al., CVPR 2023)
- SG-Softmax: Stop-Gradient Softmax Loss (Yang et al., AAAI 2023)
- DADA: Data-Augmented Domain Adaptation wrapper (Ren et al., AAAI 2024)
- SlicedMS: Spectral Embedding Expansion (Ko & Gu, CVPR 2020)
- SlicedProxy: Dimensional Slicing on Proxy-based loss (Sanakoyeu et al., CVPR 2019)
"""

from src.losses.nir_regularizer import NIRRegularizer
from src.losses.hier_regularizer import HIERRegularizer
from src.losses.sg_softmax import SGSoftmaxLoss
from src.losses.dada_wrapper import DADAWrapper
from src.losses.amplitude_penalty import AmplitudePenaltyLoss
from src.losses.sliced_ms_loss import SlicedMSLoss
from src.losses.sliced_proxy_loss import SlicedProxyLoss

__all__ = [
    'NIRRegularizer',
    'HIERRegularizer',
    'SGSoftmaxLoss',
    'DADAWrapper',
    'AmplitudePenaltyLoss',
    'SlicedMSLoss',
    'SlicedProxyLoss',
]

