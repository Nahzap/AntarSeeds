"""
Visualization module for Metric Learning system.
Provides tools for visual inspection of inputs, latent space, and predictions.
"""

from .visualization_utils import denormalize_image, tensor_to_pil
from .analytics import (
    visualize_input_batch,
    plot_embeddings_tsne,
    visualize_inference_batch,
    compute_class_centroids
)

__all__ = [
    'denormalize_image',
    'tensor_to_pil',
    'visualize_input_batch',
    'plot_embeddings_tsne',
    'visualize_inference_batch',
    'compute_class_centroids'
]
