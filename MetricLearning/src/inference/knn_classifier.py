"""
Clasificador k-NN para embeddings.
Unifica la lógica de clasificación usada en múltiples scripts.
"""

import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm
import logging


def euclidean_distances(X, Y):
    """
    Calcula distancias euclidianas entre X y Y.
    
    Args:
        X: Array (n_samples_X, n_features)
        Y: Array (n_samples_Y, n_features)
        
    Returns:
        Array (n_samples_X, n_samples_Y) con distancias
    """
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)[np.newaxis, :]
    distances = np.sqrt(np.maximum(XX - 2 * np.dot(X, Y.T) + YY, 0))
    return distances

def cosine_distances(X, Y):
    X_norm = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    Y_norm = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-8)
    return 1 - np.dot(X_norm, Y_norm.T)

def poincare_distances_np(X, Y, curvature=1.0):
    c = curvature
    eps = 1e-5
    X_norm_sq = np.clip(np.sum(X**2, axis=1, keepdims=True), 0, 1/c - eps)
    Y_norm_sq = np.clip(np.sum(Y**2, axis=1, keepdims=True), 0, 1/c - eps)
    
    X_n = np.sum(X**2, axis=1, keepdims=True)
    Y_n = np.sum(Y**2, axis=1, keepdims=True)
    diff_sq = np.maximum(X_n + Y_n.T - 2 * np.dot(X, Y.T), 0)
    
    denom = np.maximum((1 - c * X_norm_sq) * (1 - c * Y_norm_sq.T), eps)
    arg = np.maximum(1 + 2 * c * diff_sq / denom, 1 + eps)
    return (1 / np.sqrt(c)) * np.arccosh(arg)

logger = logging.getLogger(__name__)


class KNNClassifier:
    """
    Clasificador k-NN para embeddings de metric learning.
    
    Attributes:
        reference_embeddings: Embeddings de referencia (N, D)
        reference_labels: Labels de referencia (N,)
        class_names: Lista de nombres de clases
        k: Número de vecinos
        metric_type: 'euclidean', 'cosine', or 'hyperbolic'
    
    Example:
        >>> classifier = KNNClassifier(ref_embeddings, ref_labels, class_names, k=5)
        >>> result = classifier.predict(query_embedding)
        >>> print(result['predicted_class'])
        'Medicago_sativa'
    """
    
    def __init__(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: np.ndarray,
        class_names: List[str],
        k: int = 5,
        distance_threshold: float = 0.0,
        metric_type: str = 'euclidean',
    ):
        """
        Inicializa clasificador k-NN.
        
        Args:
            reference_embeddings: Embeddings de referencia (N, D)
            reference_labels: Labels de referencia (N,)
            class_names: Nombres de clases
            k: Número de vecinos
            distance_threshold: Si > 0, rechaza predicciones con mean_distance > threshold
                               (marca como UNKNOWN). 0 = sin rechazo.
            metric_type: Metric to use for nearest neighbors.
        """
        self.reference_embeddings = reference_embeddings
        self.reference_labels = reference_labels
        self.class_names = class_names
        self.k = k
        self.distance_threshold = distance_threshold
        self.metric_type = metric_type
        
        self.idx_to_class = {i: name for i, name in enumerate(class_names)}
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        
        logger.info(f"KNN Classifier initialized with k={k}, {len(reference_embeddings)} references, "
                    f"dist_threshold={distance_threshold}, metric_type={metric_type}")
    
    def _compute_distances(self, X, Y):
        if self.metric_type == 'cosine':
            return cosine_distances(X, Y)
        elif self.metric_type == 'hyperbolic':
            return poincare_distances_np(X, Y)
        else:
            return euclidean_distances(X, Y)

    def predict(
        self,
        query_embedding: np.ndarray,
        return_distances: bool = False,
        return_neighbors: bool = False
    ) -> Dict:
        """
        Predice clase de un embedding query.
        
        Args:
            query_embedding: Embedding a clasificar (1, D) o (D,)
            return_distances: Si True, retorna distancias
            return_neighbors: Si True, retorna índices de vecinos
            
        Returns:
            Dict con predicción, confianza, etc.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances = self._compute_distances(query_embedding, self.reference_embeddings)[0]
        
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.reference_labels[k_nearest_indices]
        k_nearest_distances = distances[k_nearest_indices]
        
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predicted_class = self.idx_to_class[predicted_label]
        
        confidence = np.max(counts) / self.k
        
        mean_distance = np.mean(k_nearest_distances)
        
        # S2: Reject if too far from all references (out-of-distribution)
        rejected = False
        if self.distance_threshold > 0 and mean_distance > self.distance_threshold:
            rejected = True
        
        result = {
            'predicted_class': 'UNKNOWN' if rejected else predicted_class,
            'predicted_label': int(predicted_label),
            'confidence': 0.0 if rejected else float(confidence),
            'mean_distance': float(mean_distance),
            'rejected': rejected,
            'nearest_class': predicted_class,
        }
        
        if return_distances:
            result['k_nearest_distances'] = k_nearest_distances
            result['all_distances'] = distances
        
        if return_neighbors:
            result['k_nearest_indices'] = k_nearest_indices
            result['k_nearest_labels'] = k_nearest_labels
        
        return result
    
    def predict_batch(
        self,
        query_embeddings: np.ndarray,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Predice clases para múltiples embeddings.
        
        Args:
            query_embeddings: Embeddings a clasificar (N, D)
            show_progress: Si True, muestra progreso
            
        Returns:
            Lista de diccionarios con predicciones
        """
        results = []
        
        iterator = tqdm(query_embeddings, desc="Classifying") if show_progress else query_embeddings
        
        for query_embedding in iterator:
            result = self.predict(query_embedding)
            results.append(result)
        
        return results
    
    def get_class_distribution(self, query_embedding: np.ndarray) -> Dict[str, float]:
        """
        Obtiene distribución de clases en k vecinos.
        
        Args:
            query_embedding: Embedding a clasificar (1, D) o (D,)
            
        Returns:
            Dict {clase: proporción}
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances = self._compute_distances(query_embedding, self.reference_embeddings)[0]
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.reference_labels[k_nearest_indices]
        
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        
        distribution = {}
        for label, count in zip(unique_labels, counts):
            class_name = self.idx_to_class[label]
            distribution[class_name] = float(count / self.k)
        
        return distribution
    
    def find_similar(
        self,
        query_embedding: np.ndarray,
        k: Optional[int] = None,
        return_distances: bool = True
    ) -> Dict:
        """
        Encuentra k imágenes más similares.
        
        Args:
            query_embedding: Embedding query (1, D) o (D,)
            k: Número de similares (si None, usa self.k)
            return_distances: Si True, retorna distancias
            
        Returns:
            Dict con índices, labels, distancias
        """
        if k is None:
            k = self.k
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        distances = self._compute_distances(query_embedding, self.reference_embeddings)[0]
        
        top_k_indices = np.argsort(distances)[:k]
        top_k_labels = self.reference_labels[top_k_indices]
        top_k_distances = distances[top_k_indices]
        
        result = {
            'indices': top_k_indices,
            'labels': top_k_labels,
            'class_names': [self.idx_to_class[label] for label in top_k_labels]
        }
        
        if return_distances:
            result['distances'] = top_k_distances
        
        return result
