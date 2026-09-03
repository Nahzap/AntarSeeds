"""
Utilidades para cálculo de métricas.
Centraliza operaciones comunes de evaluación.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    balanced_accuracy_score,
    normalized_mutual_info_score,
    f1_score,
    precision_score,
    precision_recall_fscore_support
)
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _compute_1nn_predictions(
    embeddings: np.ndarray,
    labels: np.ndarray,
    metric: str = 'cosine'
) -> np.ndarray:
    """
    Predice etiquetas con clasificador leave-one-out 1-NN en espacio de embeddings.

    Args:
        embeddings: Embeddings (N, D)
        labels: Labels verdaderas (N,)

        metric: 'cosine' (dot-product similarity) o 'euclidean'

    Returns:
        predicted_labels: Etiquetas predichas (N,)
    """
    if metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(embeddings, embeddings)
        np.fill_diagonal(distances, np.inf)
        nearest_neighbors = np.argmin(distances, axis=1)
    elif metric == 'cosine':
        # Keep consistency with compute_recall_at_k (dot-product retrieval).
        similarity = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(similarity, -1e9)
        nearest_neighbors = np.argmax(similarity, axis=1)
    else:
        raise ValueError(f"Unsupported 1-NN metric: {metric}")

    predicted_labels = labels[nearest_neighbors]
    return predicted_labels


def _topk_indices_from_embeddings(embeddings: np.ndarray, max_k: int, nn_metric: str = 'cosine') -> np.ndarray:
    """
    Obtiene top-k vecinos procesando en chunks para evitar OOM (matriz densa de 40GB).
    """
    if max_k <= 0:
        raise ValueError(f"max_k must be > 0, got {max_k}")
    
    n = len(embeddings)
    indices = np.zeros((n, max_k), dtype=int)
    chunk_size = 2000
    
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        
        if nn_metric == 'hyperbolic':
            c_emb = embeddings[i:end]
            c_norm2 = np.sum(c_emb**2, axis=1, keepdims=True)
            u_norm2 = np.sum(embeddings**2, axis=1, keepdims=True).T
            dot = np.dot(c_emb, embeddings.T)
            
            sqdist = c_norm2 + u_norm2 - 2 * dot
            denom = (1.0 - c_norm2) * (1.0 - u_norm2)
            denom = np.clip(denom, 1e-15, None)
            
            # mobius_sq is monotonically proportional to poincare distance
            # Negative because argpartition keeps largest values (we want smallest distance)
            chunk_sim = -(sqdist / denom)
        else:
            # Dot product para este chunk: (chunk_size, D) @ (D, N) -> (chunk_size, N)
            chunk_sim = np.dot(embeddings[i:end], embeddings.T)
        
        # Diagonal a -inf
        for j in range(end - i):
            chunk_sim[j, i + j] = -1e9
            
        part = np.argpartition(-chunk_sim, kth=max_k - 1, axis=1)[:, :max_k]
        part_scores = np.take_along_axis(chunk_sim, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        indices[i:end] = np.take_along_axis(part, order, axis=1)
        
    return indices


def _compute_1nn_predictions_from_similarity(
    similarity: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """Predicciones 1-NN a partir de matriz de similitud precomputada."""
    nearest_neighbors = np.argmax(similarity, axis=1)
    return labels[nearest_neighbors]


def compute_accuracy(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Calcula accuracy simple.
    
    Args:
        predictions: Predicciones (N,)
        ground_truth: Ground truth (N,)
        
    Returns:
        Accuracy (0-1)
    """
    return accuracy_score(ground_truth, predictions)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: List[str] = None,
    normalize: bool = False
) -> np.ndarray:
    """
    Calcula matriz de confusión.
    
    Args:
        y_true: Ground truth (N,)
        y_pred: Predicciones (N,)
        num_classes: Número de clases (opcional)
        class_names: Nombres de clases (opcional)
        normalize: Si True, normaliza por filas
        
    Returns:
        Matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm


def compute_class_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: List[str],
    confidences: np.ndarray = None
) -> pd.DataFrame:
    """
    Calcula métricas por clase.
    
    Args:
        predictions: Predicciones (N,)
        ground_truth: Ground truth (N,)
        class_names: Nombres de clases
        confidences: Confianzas (N,) opcional
        
    Returns:
        DataFrame con métricas por clase
    """
    results = []
    
    for i, class_name in enumerate(class_names):
        mask = ground_truth == i
        
        if mask.sum() == 0:
            continue
        
        class_predictions = predictions[mask]
        class_ground_truth = ground_truth[mask]
        
        correct = (class_predictions == class_ground_truth).sum()
        total = len(class_ground_truth)
        recall = correct / total
        
        avg_confidence = None
        if confidences is not None:
            avg_confidence = confidences[mask].mean()
        
        result = {
            'class': class_name,
            'total': total,
            'correct': correct,
            'recall': recall
        }
        
        if avg_confidence is not None:
            result['avg_confidence'] = avg_confidence
        
        results.append(result)
    
    return pd.DataFrame(results)


def compute_retrieval_metrics(
    query_labels: np.ndarray,
    retrieved_labels: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Calcula métricas de retrieval (Precision@K).
    
    Args:
        query_labels: Labels de queries (N,)
        retrieved_labels: Labels de retrieved (N, K)
        k_values: Valores de K para calcular
        
    Returns:
        Dict con Precision@K para cada K
    """
    metrics = {}
    
    for k in k_values:
        top_k = retrieved_labels[:, :k]
        
        correct_counts = np.sum(top_k == query_labels[:, None], axis=1)
        precision_at_k = np.mean(correct_counts / k)
        metrics[f'Precision@{k}'] = float(precision_at_k)
    
    return metrics


def generate_classification_report(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    class_names: List[str],
    confidences: np.ndarray = None
) -> str:
    """
    Genera reporte de clasificación en texto.
    
    Args:
        predictions: Predicciones (N,)
        ground_truth: Ground truth (N,)
        class_names: Nombres de clases
        confidences: Confianzas (N,) opcional
        
    Returns:
        String con reporte formateado
    """
    overall_acc = compute_accuracy(predictions, ground_truth)
    
    class_metrics = compute_class_metrics(predictions, ground_truth, class_names, confidences)
    
    report = []
    report.append("=" * 70)
    report.append("CLASSIFICATION REPORT")
    report.append("=" * 70)
    report.append("")
    report.append(f"Overall Accuracy: {overall_acc:.2%}")
    report.append(f"Total Samples: {len(predictions)}")
    report.append("")
    report.append("Per-Class Metrics:")
    report.append("-" * 70)
    
    for _, row in class_metrics.iterrows():
        report.append(f"\n{row['class']}:")
        report.append(f"  Total: {row['total']}")
        report.append(f"  Correct: {row['correct']}")
        report.append(f"  Recall: {row['recall']:.2%}")
        if 'avg_confidence' in row:
            report.append(f"  Avg Confidence: {row['avg_confidence']:.2%}")
    
    report.append("")
    report.append("=" * 70)
    
    return "\n".join(report)


def compute_recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: List[int] = [1, 5, 10],
    nn_metric: str = 'cosine',
    topk_indices: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula Recall@K para un conjunto de embeddings.
    
    Para cada query, verifica si al menos uno de los K vecinos mas cercanos
    comparte la misma clase. Metrica estandar en Deep Metric Learning
    (Wang et al. CVPR 2019, Kim et al. CVPR 2020).
    
    Args:
        embeddings: Embeddings (N, D), asumidos L2-normalizados
        labels: Labels (N,)
        k_list: Lista de valores K para calcular R@K
        
    Returns:
        Dict con R@K para cada K. Ej: {'R@1': 0.85, 'R@5': 0.95}
    """
    n = len(embeddings)
    max_k = max(k_list)
    if topk_indices is None:
        indices = _topk_indices_from_embeddings(embeddings, max_k=max_k, nn_metric=nn_metric)
    else:
        indices = topk_indices[:, :max_k]
    
    metrics = {}
    for k in k_list:
        top_k_labels = labels[indices[:, :k]]
        # Correct if at least one retrieved label matches query label
        correct_mask = (top_k_labels == labels[:, None]).any(axis=1)
        metrics[f'R@{k}'] = float(correct_mask.mean()) if n > 0 else 0.0
    
    return metrics


def _row_nn_scores(
    embeddings: np.ndarray,
    query_idx: int,
    nn_metric: str = "cosine",
) -> np.ndarray:
    """Per-neighbour scores for one query (higher = more similar for cosine/hyperbolic)."""
    n = len(embeddings)
    query = embeddings[query_idx]

    if nn_metric == "euclidean":
        scores = -np.linalg.norm(embeddings - query, axis=1)
    elif nn_metric == "hyperbolic":
        q_norm2 = np.sum(query ** 2)
        u_norm2 = np.sum(embeddings ** 2, axis=1)
        dot = embeddings @ query
        sqdist = q_norm2 + u_norm2 - 2.0 * dot
        denom = np.clip((1.0 - q_norm2) * (1.0 - u_norm2), 1e-15, None)
        scores = -(sqdist / denom)
    else:
        scores = embeddings @ query

    scores[query_idx] = -1e9
    return scores


def compute_nmi(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_init: int = 10,
    n_seeds: int = 1,
    base_seed: int = 42,
) -> Tuple[float, float]:
    """
    Calcula Normalized Mutual Information (NMI) entre clusters
    de embeddings y labels reales.
    
    Usa K-Means con K = numero de clases unicas.
    Metrica estandar en Deep Metric Learning (Wang et al. CVPR 2019).
    Con n_seeds > 1 reporta media sobre reinicios (Musgrave et al. MLRC).
    
    Args:
        embeddings: Embeddings (N, D)
        labels: Labels (N,)
        n_init: K-Means n_init per seed
        n_seeds: Number of random seeds to average
        base_seed: First random_state for K-Means
        
    Returns:
        (nmi_mean, nmi_std)
    """
    n_clusters = len(np.unique(labels))
    n_seeds = max(1, int(n_seeds))
    n_init = max(1, int(n_init))
    scores: List[float] = []

    for offset in range(n_seeds):
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=int(base_seed) + offset,
            n_init=n_init,
        )
        cluster_assignments = kmeans.fit_predict(embeddings)
        scores.append(float(normalized_mutual_info_score(labels, cluster_assignments)))

    nmi_mean = float(np.mean(scores))
    nmi_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    if n_seeds > 1:
        logger.info(
            f"NMI: {nmi_mean:.4f} ± {nmi_std:.4f} "
            f"(clusters={n_clusters}, seeds={n_seeds}, n_init={n_init})"
        )
    else:
        logger.info(f"NMI: {nmi_mean:.4f} (clusters={n_clusters})")

    return nmi_mean, nmi_std


def compute_embedding_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k_list: List[int] = [1, 5],
    nn_metric: str = 'cosine'
) -> Dict[str, float]:
    """
    Calcula metricas de calidad de embeddings completas.
    Combina Recall@K y NMI.
    
    Args:
        embeddings: Embeddings (N, D)
        labels: Labels (N,)
        k_list: Valores de K para Recall@K
        
    Returns:
        Dict con todas las metricas
    """
    metrics = compute_recall_at_k(embeddings, labels, k_list, nn_metric=nn_metric)
    nmi_mean, _ = compute_nmi(embeddings, labels)
    metrics["NMI"] = nmi_mean
    
    logger.info(f"Embedding quality: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    
    return metrics


def compute_map_at_r(
    embeddings: np.ndarray,
    labels: np.ndarray,
    nn_metric: str = 'cosine',
    similarity: Optional[np.ndarray] = None,
    sorted_indices: Optional[np.ndarray] = None,
    candidate_multiplier: int = 12,
    exact: bool = True,
) -> float:
    """
    Calcula Mean Average Precision at R (mAP@R).
    
    Metrica estandar en Deep Metric Learning (Song et al. CVPR 2016,
    Musgrave et al. ECCV 2020). R es el numero de ejemplos positivos
    por query (mismo label).
    
    Para cada query, calcula Average Precision hasta que se hayan
    recuperado todos los R ejemplos positivos, luego promedia
    sobre todas las queries.
    
    Args:
        embeddings: Embeddings L2-normalizados (N, D)
        labels: Labels (N,)
        
    Returns:
        mAP@R score (0-1)
    """
    n = len(embeddings)
    
    average_precisions = []
    
    for i in range(n):
        query_label = labels[i]
        
        # Encontrar todos los positivos (mismo label, excluyendo query)
        positives_mask = (labels == query_label)
        positives_mask[i] = False
        num_positives = positives_mask.sum()
        
        if num_positives == 0:
            continue

        if sorted_indices is not None:
            row_indices = sorted_indices[i]
        elif similarity is not None:
            row = similarity[i]
            if exact:
                cap = n - 1
            else:
                cap = min(
                    n - 1,
                    max(num_positives + 32, num_positives * max(2, candidate_multiplier)),
                )
            if cap < n - 1:
                row_indices = np.argpartition(-row, cap - 1)[:cap]
                row_indices = row_indices[np.argsort(-row[row_indices])]
            else:
                row_indices = np.argsort(-row)
        else:
            row = _row_nn_scores(embeddings, i, nn_metric=nn_metric)
            if exact:
                cap = n - 1
            else:
                cap = min(
                    n - 1,
                    max(num_positives + 32, num_positives * max(2, candidate_multiplier)),
                )
            if cap < n - 1:
                row_indices = np.argpartition(-row, cap - 1)[:cap]
                row_indices = row_indices[np.argsort(-row[row_indices])]
            else:
                row_indices = np.argsort(-row)
        
        # Recuperar indices ordenados por similitud
        retrieved_labels = labels[row_indices]
        
        # Calcular precision acumulativa hasta recuperar todos los R positivos
        precisions = []
        num_correct = 0
        
        for rank, retrieved_label in enumerate(retrieved_labels, start=1):
            if retrieved_label == query_label:
                num_correct += 1
                precision_at_k = num_correct / rank
                precisions.append(precision_at_k)
                
                # Detenerse cuando se han encontrado todos los positivos
                if num_correct == num_positives:
                    break
        
        # Average Precision para esta query
        if precisions:
            # mAP@R DIVIDE ESTRICTAMENTE POR num_positives (R), NO por len(precisions)
            ap = np.sum(precisions) / num_positives
            average_precisions.append(ap)
    
    # Mean Average Precision
    map_at_r = np.mean(average_precisions) if average_precisions else 0.0
    
    logger.info(f"mAP@R: {map_at_r:.4f}")
    
    return float(map_at_r)


def compute_f1_macro(
    embeddings: np.ndarray,
    labels: np.ndarray,
    nn_metric: str = 'cosine',
    predicted_labels: Optional[np.ndarray] = None
) -> float:
    """
    Calcula F1-Score Macro usando clasificacion 1-NN.
    
    Critico para datasets con clases desbalanceadas (Barz & Denzler, WACV 2020).
    Calcula F1 para cada clase y luego promedia sin ponderar,
    dando igual peso a clases minoritarias.
    
    Args:
        embeddings: Embeddings (N, D)
        labels: Labels (N,)
        
    Returns:
        F1-Score Macro (0-1)
    """
    if predicted_labels is None:
        predicted_labels = _compute_1nn_predictions(embeddings, labels, metric=nn_metric)
    
    # F1-Score Macro (promedio no ponderado)
    f1_macro = f1_score(labels, predicted_labels, average='macro', zero_division=0)
    
    logger.info(f"F1-Score Macro ({nn_metric} 1-NN): {f1_macro:.4f}")
    
    return float(f1_macro)


def compute_per_class_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    nn_metric: str = 'cosine',
    predicted_labels: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calcula metricas por clase para analisis detallado.
    
    Args:
        embeddings: Embeddings (N, D)
        labels: Labels (N,)
        class_names: Nombres de clases (opcional)
        
    Returns:
        Dict con metricas por clase: {class_name: {metric: value}}
    """
    if predicted_labels is None:
        predicted_labels = _compute_1nn_predictions(embeddings, labels, metric=nn_metric)
    
    # Calcular precision, recall, f1 por clase
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted_labels, average=None, zero_division=0
    )
    
    unique_labels = np.unique(labels)
    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_labels]
    
    results = {}
    for idx, label in enumerate(unique_labels):
        class_name = class_names[idx] if idx < len(class_names) else f"Class_{label}"
        results[class_name] = {
            'precision': float(precision[idx]),
            'recall': float(recall[idx]),
            'f1_score': float(f1[idx]),
            'support': int(support[idx])
        }
    
    return results


def compute_standard_classification_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    nn_metric: str = 'cosine',
    predicted_labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula métricas estándar de clasificación a partir de embeddings usando 1-NN.

    Métricas:
    - Accuracy
    - Balanced Accuracy
    - Precision Macro
    - Precision Weighted
    - F1 Macro
    - F1 Weighted
    """
    if predicted_labels is None:
        predicted_labels = _compute_1nn_predictions(embeddings, labels, metric=nn_metric)

    metrics = {
        'accuracy': float(accuracy_score(labels, predicted_labels)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predicted_labels)),
        'precision_macro': float(precision_score(labels, predicted_labels, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(labels, predicted_labels, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(labels, predicted_labels, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(labels, predicted_labels, average='weighted', zero_division=0)),
    }

    logger.info(
        f"Std Classification Metrics ({nn_metric} 1-NN): "
        + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    )
    return metrics


def compute_all_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    nn_metric: str = 'cosine',
    k_list: List[int] = [1, 5, 10],
    class_names: Optional[List[str]] = None,
    include_per_class: bool = True,
    nmi_n_init: int = 10,
    nmi_n_seeds: int = 1,
    nmi_base_seed: int = 42,
    map_at_r_exact: bool = True,
    map_at_r_candidate_multiplier: int = 12,
) -> Dict[str, any]:
    """
    Calcula TODAS las metricas SOTA para Deep Metric Learning.
    
    Protocolo completo siguiendo:
    - Song et al. (CVPR 2016): Recall@K, NMI
    - Musgrave et al. (ECCV 2020): mAP@R
    - Barz & Denzler (WACV 2020): F1-Score Macro
    
    Args:
        embeddings: Embeddings L2-normalizados (N, D)
        labels: Labels (N,)
        k_list: Valores de K para Recall@K
        class_names: Nombres de clases (opcional)
        
    Returns:
        Dict con todas las metricas: {'R@1', 'R@5', 'NMI', 'mAP@R', 'F1_macro', 'per_class'}
    """
    logger.info("Computing SOTA metrics for Deep Metric Learning...")
    
    metrics = {}
    max_k = max(k_list)
    
    # L2-normalize only for cosine retrieval (respect evaluation.nn_metric / embedding_space).
    if nn_metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)
        
    # Shared compute: batched top-k indices and 1-NN predictions to prevent RAM crash
    topk_indices = _topk_indices_from_embeddings(embeddings, max_k=max_k, nn_metric=nn_metric)
    predicted_labels = labels[topk_indices[:, 0]]

    # Recall@K (Song et al. CVPR 2016)
    recall_metrics = compute_recall_at_k(
        embeddings,
        labels,
        k_list,
        topk_indices=topk_indices
    )
    metrics.update(recall_metrics)
    
    # NMI (Song et al. CVPR 2016) — multi-seed when evaluation.nmi.n_seeds > 1
    nmi_mean, nmi_std = compute_nmi(
        embeddings,
        labels,
        n_init=nmi_n_init,
        n_seeds=nmi_n_seeds,
        base_seed=nmi_base_seed,
    )
    metrics["NMI"] = nmi_mean
    metrics["NMI_std"] = nmi_std

    # mAP@R (Musgrave et al. ECCV 2020)
    metrics["mAP@R"] = compute_map_at_r(
        embeddings,
        labels,
        nn_metric=nn_metric,
        similarity=None,
        exact=map_at_r_exact,
        candidate_multiplier=map_at_r_candidate_multiplier,
    )

    # F1-Score Macro (Barz & Denzler WACV 2020) — same nn_metric as R@K
    metrics['F1_macro'] = compute_f1_macro(
        embeddings,
        labels,
        nn_metric=nn_metric,
        predicted_labels=predicted_labels
    )

    # Métricas estándar de clasificación (1-NN leave-one-out)
    std_metrics = compute_standard_classification_metrics(
        embeddings,
        labels,
        class_names,
        nn_metric=nn_metric,
        predicted_labels=predicted_labels
    )
    metrics['Accuracy'] = std_metrics['accuracy']
    metrics['Balanced_Accuracy'] = std_metrics['balanced_accuracy']
    metrics['Precision_macro'] = std_metrics['precision_macro']
    metrics['Precision_weighted'] = std_metrics['precision_weighted']
    metrics['F1_weighted'] = std_metrics['f1_weighted']
    
    if include_per_class:
        metrics['per_class'] = compute_per_class_metrics(
            embeddings,
            labels,
            class_names,
            nn_metric=nn_metric,
            predicted_labels=predicted_labels,
        )
    
    logger.info("="*70)
    logger.info("SOTA Metrics Summary:")
    logger.info(f"  R@1:      {metrics.get('R@1', 0):.4f}")
    logger.info(f"  R@5:      {metrics.get('R@5', 0):.4f}")
    logger.info(f"  mAP@R:    {metrics['mAP@R']:.4f}")
    nmi_log = f"{metrics['NMI']:.4f}"
    if metrics.get("NMI_std", 0.0) > 0:
        nmi_log += f" ± {metrics['NMI_std']:.4f}"
    logger.info(f"  NMI:      {nmi_log}")
    logger.info(f"  F1-Macro: {metrics['F1_macro']:.4f}")
    logger.info(f"  Accuracy: {metrics['Accuracy']:.4f}")
    logger.info(f"  BalAcc:   {metrics['Balanced_Accuracy']:.4f}")
    logger.info(f"  Prec-M:   {metrics['Precision_macro']:.4f}")
    logger.info(f"  Prec-W:   {metrics['Precision_weighted']:.4f}")
    logger.info(f"  F1-W:     {metrics['F1_weighted']:.4f}")
    logger.info("="*70)
    
    return metrics
