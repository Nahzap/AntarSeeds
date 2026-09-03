"""
Análisis Espectral del Espacio de Embeddings

Herramientas para validación algebraica del colapso dimensional en espacios
embedding normalizados (hiperesfera S^{d-1}).

Problema que detecta:
- En topologías superpuestas, ocurre colapso dimensional: la red extrae UNA
  característica dominante (e.g., circularidad general), provocando que λ₁ ≈ 0.99.
- Aunque el vector reside en ℝ^128, su rango efectivo es ~1.

Protocolo de validación (de la investigación):
1. Extraer embeddings del validation set
2. Aislar muestras de clases problemáticas  
3. Computar SVD (Singular Value Decomposition)
4. Analizar distribución de varianza en componentes principales

Criterios de colapso (implementación):
- collapse_severity = λ₁ / Σλ (varianza total en covarianza de embeddings)
- is_collapsed si collapse_severity > 0.80
- effective_rank = mínimo de componentes para ≥95% varianza explicada acumulada
  (participation rank; no confundir con stable rank de Roy–Neill)

Referencias:
- Investigación ViTs (docs/20260305_173615_Research_ViTs.md)
- Análisis espectral de matrices de covarianza

Autor: MetricLearning Project
Fecha: 2026-03-06
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.decomposition import PCA
from itertools import combinations
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional, Any

from src.utils.unicode_utils import sanitize_for_ascii


logger = logging.getLogger(__name__)


class EmbeddingSpaceAnalyzer:
    """
    Analizador espectral para espacios de embeddings normalizados.
    
    Computa eigenvalues, effective rank y detecta colapso dimensional.
    
    Args:
        embeddings (np.ndarray): Embeddings normalizados, shape (N, D)
        labels (np.ndarray): Etiquetas de clase, shape (N,)
        class_names (List[str]): Nombres de las clases
    
    Attributes:
        embeddings: Matriz de embeddings
        labels: Vector de etiquetas
        class_names: Lista de nombres
        n_samples: Número de muestras
        n_dim: Dimensión del embedding
    
    Example:
        >>> analyzer = EmbeddingSpaceAnalyzer(val_embeddings, val_labels, class_names)
        >>> results = analyzer.analyze_class_pair("ClassA", "ClassB", output_dir)
        >>> if results['is_collapsed']:
        >>>     print(f"COLAPSO DETECTADO: λ₁={results['eigenvalues'][0]:.3f}")
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        class_names: List[str]
    ):
        """Inicializa el analizador con embeddings y labels."""
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings debe ser 2D, got shape {embeddings.shape}")
        if labels.ndim != 1:
            raise ValueError(f"labels debe ser 1D, got shape {labels.shape}")
        if len(embeddings) != len(labels):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings, {len(labels)} labels")
        
        self.embeddings = embeddings
        self.labels = labels
        self.class_names = class_names
        self.n_samples = len(embeddings)
        self.n_dim = embeddings.shape[1]
        
        logger.info(
            f"EmbeddingSpaceAnalyzer initialized: {self.n_samples} samples, "
            f"dim={self.n_dim}, {len(class_names)} classes"
        )
    
    def analyze_class_pair(
        self,
        class_a: str,
        class_b: str,
        output_dir: str
    ) -> Dict[str, Any]:
        """
        Análisis espectral de 2 clases específicas.
        
        Computa:
        1. Eigenvalues de la matriz de covarianza
        2. Varianza explicada acumulada
        3. Rango efectivo (95% varianza)
        4. Diagnóstico de colapso dimensional
        5. Visualización del espectro
        
        Args:
            class_a: Nombre de la primera clase
            class_b: Nombre de la segunda clase
            output_dir: Directorio para guardar gráficos
        
        Returns:
            dict con keys:
                - eigenvalues: np.array de eigenvalues
                - explained_variance: Varianza explicada por componente
                - cumulative_variance: Varianza acumulada
                - effective_rank: Número de componentes para 95% varianza
                - is_collapsed: bool, True si λ₁ > 0.80
                - collapse_severity: λ₁ / total_var (0-1)
        """
        # Obtener índices de clases
        try:
            idx_a_class = self.class_names.index(class_a)
            idx_b_class = self.class_names.index(class_b)
        except ValueError as e:
            raise ValueError(f"Clase no encontrada: {e}")
        
        # Filtrar embeddings de las 2 clases
        idx_a = self.labels == idx_a_class
        idx_b = self.labels == idx_b_class
        X_sub = self.embeddings[idx_a | idx_b]
        
        if len(X_sub) < 2:
            logger.warning(f"Insuficientes samples para {class_a} vs {class_b}: {len(X_sub)}")
            return {
                'error': 'Insufficient samples',
                'n_samples': len(X_sub)
            }
        
        # Matriz de covarianza
        X_centered = X_sub - X_sub.mean(axis=0)
        cov_matrix = (X_centered.T @ X_centered) / len(X_sub)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = eigenvalues[::-1]  # Orden descendente
        eigenvectors = eigenvectors[:, ::-1]
        
        # Varianza explicada
        total_var = eigenvalues.sum()
        if total_var == 0:
            logger.warning(f"Varianza total cero para {class_a} vs {class_b}")
            return {'error': 'Zero variance'}
        
        explained_var = eigenvalues / total_var
        cumulative_var = np.cumsum(explained_var)
        
        # Rango efectivo (95% varianza)
        effective_rank = np.searchsorted(cumulative_var, 0.95) + 1
        effective_rank = min(effective_rank, self.n_dim)  # Cap al máximo
        
        # Diagnóstico de colapso
        collapse_severity = eigenvalues[0] / total_var
        is_collapsed = collapse_severity > 0.80
        
        results = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var,
            'effective_rank': effective_rank,
            'is_collapsed': is_collapsed,
            'collapse_severity': collapse_severity,
            'n_samples': len(X_sub),
            'class_a': class_a,
            'class_b': class_b
        }
        
        # Visualización y exportación CSV
        output_path = Path(output_dir) / f"spectral_analysis_{class_a}_{class_b}.png"
        csv_path = Path(output_dir) / f"spectral_analysis_{class_a}_{class_b}.csv"
        
        df = pd.DataFrame({
            'component': range(1, len(eigenvalues) + 1),
            'eigenvalue': eigenvalues,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var
        })
        df.to_csv(csv_path, index=False)
        
        self._plot_spectrum(results, output_path)
        
        # Log resultado
        status = "COLLAPSED" if is_collapsed else "HEALTHY"
        msg = f"[{class_a} vs {class_b}] λ₁={collapse_severity:.3f}, " \
              f"Eff.Rank={effective_rank}/{self.n_dim}, Status={status}"
        logger.info(sanitize_for_ascii(msg))
        
        return results
    
    def _plot_spectrum(
        self,
        results: Dict[str, Any],
        output_path: Path
    ):
        """Genera gráficos de análisis espectral."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        class_a = results['class_a']
        class_b = results['class_b']
        eigenvalues = results['eigenvalues']
        cumulative_var = results['cumulative_variance']
        effective_rank = results['effective_rank']
        
        # 1. Eigenvalues (primeros 50)
        n_show = min(50, len(eigenvalues))
        axes[0].bar(range(n_show), eigenvalues[:n_show], color='steelblue', alpha=0.7)
        axes[0].set_title('Eigenvalues (lambda_1, lambda_2, ..., lambda_50)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Component Index', fontsize=12)
        axes[0].set_ylabel('Eigenvalue', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axhline(y=eigenvalues[0]*0.1, color='red', linestyle='--', alpha=0.5, label='10% of lambda_1')
        axes[0].legend()
        
        # 2. Varianza explicada acumulada
        axes[1].plot(cumulative_var, linewidth=2.5, color='darkgreen')
        axes[1].axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% threshold')
        axes[1].axvline(x=effective_rank, color='orange', linestyle='--', linewidth=2,
                       label=f"Effective rank = {effective_rank}")
        axes[1].fill_between(range(len(cumulative_var)), cumulative_var, alpha=0.3, color='green')
        axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Components', fontsize=12)
        axes[1].set_ylabel('Cumulative Variance', fontsize=12)
        axes[1].set_ylim([0, 1.05])
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Log-scale eigenvalues
        axes[2].semilogy(eigenvalues, 'o-', markersize=4, linewidth=1.5, color='purple')
        axes[2].set_title('Eigenvalue Spectrum (log scale)', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Component Index', fontsize=12)
        axes[2].set_ylabel('Eigenvalue (log)', fontsize=12)
        axes[2].grid(True, alpha=0.3, which='both')
        
        # Diagnóstico de colapso
        collapse_text = "! COLLAPSED" if results['is_collapsed'] else "OK HEALTHY"
        collapse_color = 'red' if results['is_collapsed'] else 'green'
        
        fig.suptitle(
            f"Spectral Analysis: {class_a} vs {class_b}\n"
            f"lambda_1={eigenvalues[0]:.4f} ({results['collapse_severity']*100:.1f}%) | "
            f"Effective Rank = {effective_rank} / {self.n_dim} | "
            f"Status: {collapse_text}",
            fontsize=16, fontweight='bold', color=collapse_color
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Spectrum plot saved: {output_path}")
    
    def analyze_all_pairs(
        self,
        output_dir: str,
        max_pairs: Optional[int] = None
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Análisis exhaustivo de todas las combinaciones de clases.
        Identifica automáticamente pares con colapso dimensional.
        
        Args:
            output_dir: Directorio para guardar resultados
            max_pairs: Máximo de pares a analizar (None = todos)
        
        Returns:
            dict: {(class_a, class_b): results}
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_all = {}
        pairs = list(combinations(self.class_names, 2))
        
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        
        logger.info(f"Analyzing {len(pairs)} class pairs...")
        
        for class_a, class_b in pairs:
            try:
                results = self.analyze_class_pair(class_a, class_b, output_dir)
                if 'error' not in results:
                    results_all[(class_a, class_b)] = results
            except Exception as e:
                logger.error(f"Error analyzing {class_a} vs {class_b}: {e}")
                continue
        
        # Ranking de pares más colapsados
        collapsed_pairs = sorted(
            [(pair, res) for pair, res in results_all.items()],
            key=lambda x: x[1]['collapse_severity'],
            reverse=True
        )
        
        # Reporte textual
        report_path = output_path / "spectral_analysis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("SPECTRAL ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Dataset: {self.n_samples} samples, {self.n_dim}-dimensional embeddings\n")
            f.write(f"Classes: {len(self.class_names)}\n")
            f.write(f"Pairs analyzed: {len(results_all)}\n\n")
            
            f.write("="*70 + "\n")
            f.write("TOP 10 MOST COLLAPSED CLASS PAIRS\n")
            f.write("="*70 + "\n\n")
            
            for i, ((ca, cb), res) in enumerate(collapsed_pairs[:10], 1):
                status = "! COLLAPSED" if res['is_collapsed'] else "OK HEALTHY"
                f.write(f"{i}. {ca} vs {cb}:\n")
                f.write(f"   lambda_1 = {res['eigenvalues'][0]:.4f} ({res['collapse_severity']*100:.1f}%)\n")
                f.write(f"   Effective Rank = {res['effective_rank']} / {self.n_dim}\n")
                f.write(f"   Status: {status}\n")
                f.write(f"   Samples: {res['n_samples']}\n\n")
            
            # Resumen estadístico
            all_severities = [res['collapse_severity'] for res in results_all.values()]
            all_ranks = [res['effective_rank'] for res in results_all.values()]
            
            f.write("="*70 + "\n")
            f.write("STATISTICAL SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Collapse Severity (lambda_1/total):\n")
            f.write(f"  Mean: {np.mean(all_severities):.3f}\n")
            f.write(f"  Std:  {np.std(all_severities):.3f}\n")
            f.write(f"  Min:  {np.min(all_severities):.3f}\n")
            f.write(f"  Max:  {np.max(all_severities):.3f}\n\n")
            
            f.write(f"Effective Rank:\n")
            f.write(f"  Mean: {np.mean(all_ranks):.1f} / {self.n_dim}\n")
            f.write(f"  Std:  {np.std(all_ranks):.1f}\n")
            f.write(f"  Min:  {np.min(all_ranks)}\n")
            f.write(f"  Max:  {np.max(all_ranks)}\n\n")
            
            n_collapsed = sum(1 for res in results_all.values() if res['is_collapsed'])
            f.write(f"Collapsed pairs: {n_collapsed} / {len(results_all)} ({n_collapsed/len(results_all)*100:.1f}%)\n")
        
        logger.info(f"Report saved: {report_path}")
        logger.info(f"Collapsed pairs: {n_collapsed}/{len(results_all)}")
        
        return results_all
    
    def analyze_global_spectrum(self, output_dir: str) -> Dict[str, Any]:
        """
        Análisis espectral del espacio completo (todas las clases juntas).
        
        Args:
            output_dir: Directorio para guardar resultados
        
        Returns:
            dict: Resultados del análisis global
        """
        logger.info("Analyzing global embedding space...")
        
        # Centrar embeddings
        X_centered = self.embeddings - self.embeddings.mean(axis=0)
        cov_matrix = (X_centered.T @ X_centered) / self.n_samples
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]
        
        # Análisis
        total_var = eigenvalues.sum()
        explained_var = eigenvalues / total_var
        cumulative_var = np.cumsum(explained_var)
        effective_rank = np.searchsorted(cumulative_var, 0.95) + 1
        collapse_severity = eigenvalues[0] / total_var
        is_collapsed = collapse_severity > 0.80
        
        results = {
            'eigenvalues': eigenvalues,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var,
            'effective_rank': effective_rank,
            'is_collapsed': is_collapsed,
            'collapse_severity': collapse_severity,
            'n_samples': self.n_samples,
            'class_a': 'ALL',
            'class_b': 'CLASSES'
        }
        
        # Plot y exportación CSV
        output_path = Path(output_dir) / "spectral_analysis_GLOBAL.png"
        csv_path = Path(output_dir) / "spectral_analysis_GLOBAL.csv"
        
        df = pd.DataFrame({
            'component': range(1, len(eigenvalues) + 1),
            'eigenvalue': eigenvalues,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var
        })
        df.to_csv(csv_path, index=False)
        
        self._plot_spectrum(results, output_path)
        
        status = "! COLLAPSED" if is_collapsed else "OK HEALTHY"
        msg = f"[GLOBAL] λ₁={collapse_severity:.3f}, " \
              f"Eff.Rank={effective_rank}/{self.n_dim}, Status={status}"
        logger.info(sanitize_for_ascii(msg))
        
        return results
