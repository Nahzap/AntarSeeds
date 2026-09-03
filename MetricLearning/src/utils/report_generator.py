"""
Sistema de Generación de Reportes de Ingeniería para Inferencia.
Genera reportes profesionales con gráficos estadísticos en formato HTML/PDF.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class InferenceReportGenerator:
    """
    Generador de reportes de ingeniería para inferencia en carpetas.
    Crea reportes profesionales con estadísticas y gráficos.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Inicializa generador de reportes.
        
        Args:
            output_dir: Directorio para guardar reportes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar estilo de gráficos
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def generate_report(
        self,
        predictions: List[Dict],
        folder_path: str,
        model_info: Dict,
        class_names: List[str],
        report_name: Optional[str] = None,
        visual_samples: Optional[List[Dict[str, Any]]] = None
    ) -> Path:
        """
        Genera reporte completo de inferencia.
        
        Args:
            predictions: Lista de predicciones con metadata
            folder_path: Ruta de la carpeta analizada
            model_info: Información del modelo usado
            class_names: Nombres de las clases
            report_name: Nombre del reporte (opcional)
            
        Returns:
            Path al archivo de reporte generado
        """
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"inference_report_{timestamp}"
        
        # Crear DataFrame con resultados
        df = self._create_dataframe(predictions)
        
        # Generar estadísticas
        stats = self._compute_statistics(df, class_names)
        
        # Generar gráficos
        figures_dir = self.output_dir / f"{report_name}_figures"
        figures_dir.mkdir(exist_ok=True)
        
        figure_paths = self._generate_figures(df, stats, class_names, figures_dir)
        
        # Generar reporte HTML
        html_path = self._generate_html_report(
            df, stats, figure_paths, folder_path, model_info, report_name, visual_samples
        )
        
        # Guardar CSV con resultados detallados
        csv_path = self.output_dir / f"{report_name}_results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info(f"Reporte generado: {html_path}")
        
        return html_path
    
    def _create_dataframe(self, predictions: List[Dict]) -> pd.DataFrame:
        """Crea DataFrame desde lista de predicciones"""
        data = []
        for pred in predictions:
            data.append({
                'image_path': pred.get('image_path', ''),
                'image_name': Path(pred.get('image_path', '')).name,
                'predicted_class': pred.get('predicted_class', ''),
                'confidence': pred.get('confidence', 0.0),
                'mean_distance': pred.get('mean_distance', pred.get('distance', 0.0)),
                'n_grains': pred.get('n_grains', 1),
                'true_class': pred.get('true_class', None),
                'image_width': pred.get('image_width', 0),
                'image_height': pred.get('image_height', 0),
                'image_area_px': pred.get('image_area_px', 0.0),
                'grain_area_sum_px': pred.get('grain_area_sum_px', 0.0),
                'grain_coverage_pct': pred.get('grain_coverage_pct', 0.0),
                'detect_ms': pred.get('detect_ms', 0.0),
                'classify_ms': pred.get('classify_ms', 0.0),
                'elapsed_ms': pred.get('elapsed_ms', 0.0),
            })
        
        return pd.DataFrame(data)
    
    def _compute_statistics(self, df: pd.DataFrame, class_names: List[str]) -> Dict:
        """Calcula estadísticas del dataset de inferencia"""
        stats = {
            'total_images': len(df),
            'num_classes_predicted': df['predicted_class'].nunique(),
            'class_distribution': df['predicted_class'].value_counts().to_dict(),
            'confidence_stats': {
                'mean': df['confidence'].mean(),
                'std': df['confidence'].std(),
                'min': df['confidence'].min(),
                'max': df['confidence'].max(),
                'median': df['confidence'].median()
            },
            'distance_stats': {
                'mean': df['mean_distance'].mean(),
                'std': df['mean_distance'].std(),
                'min': df['mean_distance'].min(),
                'max': df['mean_distance'].max(),
                'median': df['mean_distance'].median()
            },
            'total_grains': int(df['n_grains'].sum()),
            'images_without_grains': int((df['n_grains'] == 0).sum()),
            'grains_per_image_stats': {
                'mean': df['n_grains'].mean(),
                'std': df['n_grains'].std(),
                'min': df['n_grains'].min(),
                'max': df['n_grains'].max(),
                'median': df['n_grains'].median()
            },
            'coverage_stats': {
                'mean': df['grain_coverage_pct'].mean(),
                'std': df['grain_coverage_pct'].std(),
                'min': df['grain_coverage_pct'].min(),
                'max': df['grain_coverage_pct'].max(),
                'median': df['grain_coverage_pct'].median()
            },
            'timing_stats_ms': {
                'detect_mean': df['detect_ms'].mean(),
                'classify_mean': df['classify_ms'].mean(),
                'elapsed_mean': df['elapsed_ms'].mean()
            }
        }
        
        # Si hay ground truth, calcular accuracy
        if 'true_class' in df.columns and df['true_class'].notna().any():
            correct = (df['predicted_class'] == df['true_class']).sum()
            stats['accuracy'] = correct / len(df)
            stats['has_ground_truth'] = True
        else:
            stats['has_ground_truth'] = False
        
        # Confianza por clase
        stats['confidence_by_class'] = df.groupby('predicted_class')['confidence'].agg(['mean', 'std']).to_dict()
        
        return stats
    
    def _generate_figures(
        self,
        df: pd.DataFrame,
        stats: Dict,
        class_names: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Genera todos los gráficos del reporte"""
        figure_paths = {}
        
        # 1. Distribución de clases predichas
        fig1 = self._plot_class_distribution(df, stats)
        path1 = output_dir / "class_distribution.png"
        fig1.savefig(path1, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        figure_paths['class_distribution'] = path1
        
        # 2. Distribución de confianza
        fig2 = self._plot_confidence_distribution(df)
        path2 = output_dir / "confidence_distribution.png"
        fig2.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        figure_paths['confidence_distribution'] = path2
        
        # 3. Confianza por clase
        fig3 = self._plot_confidence_by_class(df)
        path3 = output_dir / "confidence_by_class.png"
        fig3.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        figure_paths['confidence_by_class'] = path3
        
        # 4. Distribución de distancias
        fig4 = self._plot_distance_distribution(df)
        path4 = output_dir / "distance_distribution.png"
        fig4.savefig(path4, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        figure_paths['distance_distribution'] = path4
        
        # 5. Matriz de confusión (si hay ground truth)
        if stats['has_ground_truth']:
            fig5 = self._plot_confusion_matrix(df, class_names)
            path5 = output_dir / "confusion_matrix.png"
            fig5.savefig(path5, dpi=150, bbox_inches='tight')
            plt.close(fig5)
            figure_paths['confusion_matrix'] = path5

        # 6. Distribución de granos por imagen
        fig6 = self._plot_grains_per_image(df)
        path6 = output_dir / "grains_per_image.png"
        fig6.savefig(path6, dpi=150, bbox_inches='tight')
        plt.close(fig6)
        figure_paths['grains_per_image'] = path6

        # 7. Cobertura de granos en imagen (%)
        fig7 = self._plot_grain_coverage_distribution(df)
        path7 = output_dir / "grain_coverage_distribution.png"
        fig7.savefig(path7, dpi=150, bbox_inches='tight')
        plt.close(fig7)
        figure_paths['grain_coverage_distribution'] = path7
        
        return figure_paths
    
    def _plot_class_distribution(self, df: pd.DataFrame, stats: Dict) -> Figure:
        """Gráfico de barras de distribución de clases"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        class_counts = df['predicted_class'].value_counts()
        
        bars = ax.bar(range(len(class_counts)), class_counts.values, color='#0e639c', alpha=0.7)
        ax.set_xlabel('Clase Predicha', fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Imágenes', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de Clases Predichas', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels(class_counts.index, rotation=45, ha='right')
        
        # Añadir valores sobre barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def _plot_confidence_distribution(self, df: pd.DataFrame) -> Figure:
        """Histograma de distribución de confianza"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(df['confidence'], bins=30, color='#4ec9b0', alpha=0.7, edgecolor='black')
        ax.axvline(df['confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["confidence"].mean():.3f}')
        ax.axvline(df['confidence'].median(), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {df["confidence"].median():.3f}')
        
        ax.set_xlabel('Confianza', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de Confianza de Predicciones', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def _plot_confidence_by_class(self, df: pd.DataFrame) -> Figure:
        """Box plot de confianza por clase"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        classes = df['predicted_class'].unique()
        data = [df[df['predicted_class'] == cls]['confidence'].values for cls in classes]
        
        bp = ax.boxplot(data, labels=classes, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#dcdcaa')
            patch.set_alpha(0.7)
        
        ax.set_xlabel('Clase Predicha', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confianza', fontsize=12, fontweight='bold')
        ax.set_title('Confianza por Clase', fontsize=14, fontweight='bold')
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def _plot_distance_distribution(self, df: pd.DataFrame) -> Figure:
        """Histograma de distribución de distancias"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(df['mean_distance'], bins=30, color='#f48771', alpha=0.7, edgecolor='black')
        ax.axvline(df['mean_distance'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["mean_distance"].mean():.4f}')
        ax.axvline(df['mean_distance'].median(), color='orange', linestyle='--', linewidth=2, label=f'Mediana: {df["mean_distance"].median():.4f}')
        
        ax.set_xlabel('Distancia Media', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de Distancias en Espacio de Embeddings', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def _plot_confusion_matrix(self, df: pd.DataFrame, class_names: List[str]) -> Figure:
        """Matriz de confusión"""
        from sklearn.metrics import confusion_matrix
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_true = df['true_class'].values
        y_pred = df['predicted_class'].values
        
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names,
               yticklabels=class_names,
               xlabel='Clase Predicha',
               ylabel='Clase Verdadera',
               title='Matriz de Confusión')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Añadir valores en celdas
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        return fig

    def _plot_grains_per_image(self, df: pd.DataFrame) -> Figure:
        """Histograma de detecciones (granos) por imagen."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['n_grains'], bins=30, color='#569cd6', alpha=0.8, edgecolor='black')
        ax.axvline(
            df['n_grains'].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Media: {df['n_grains'].mean():.2f}"
        )
        ax.axvline(
            df['n_grains'].median(),
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f"Mediana: {df['n_grains'].median():.2f}"
        )
        ax.set_xlabel('Número de Granos por Imagen', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title('Distribución de Detecciones por Imagen', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig

    def _plot_grain_coverage_distribution(self, df: pd.DataFrame) -> Figure:
        """Histograma de cobertura de granos (% del área de imagen)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['grain_coverage_pct'], bins=30, color='#c586c0', alpha=0.8, edgecolor='black')
        ax.axvline(
            df['grain_coverage_pct'].mean(),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Media: {df['grain_coverage_pct'].mean():.2f}%"
        )
        ax.axvline(
            df['grain_coverage_pct'].median(),
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f"Mediana: {df['grain_coverage_pct'].median():.2f}%"
        )
        ax.set_xlabel('Cobertura de Granos (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
        ax.set_title('Cobertura de Detecciones por Imagen', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        return fig
    
    def _generate_html_report(
        self,
        df: pd.DataFrame,
        stats: Dict,
        figure_paths: Dict[str, Path],
        folder_path: str,
        model_info: Dict,
        report_name: str,
        visual_samples: Optional[List[Dict[str, Any]]] = None
    ) -> Path:
        """Genera reporte HTML profesional"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Inferencia - {report_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #0e639c;
            border-bottom: 3px solid #0e639c;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #1177bb;
            margin-top: 30px;
            border-left: 4px solid #0e639c;
            padding-left: 10px;
        }}
        .metadata {{
            background-color: #f0f8ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #f9f9f9;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #0e639c;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #0e639c;
            font-size: 14px;
        }}
        .stat-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .stat-card .unit {{
            font-size: 14px;
            color: #666;
        }}
        .figure {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .figure-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #666;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #0e639c;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Reporte de Inferencia - MetricLearning</h1>
        
        <div class="metadata">
            <p><strong>Fecha de Generación:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Carpeta Analizada:</strong> {folder_path}</p>
            <p><strong>Modelo:</strong> {model_info.get('name', 'N/A')}</p>
            <p><strong>Checkpoint:</strong> {model_info.get('checkpoint', 'N/A')}</p>
        </div>
        
        <h2>📈 Resumen Ejecutivo</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total de Imágenes</h3>
                <div class="value">{stats['total_images']}</div>
            </div>
            <div class="stat-card">
                <h3>Clases Predichas</h3>
                <div class="value">{stats['num_classes_predicted']}</div>
            </div>
            <div class="stat-card">
                <h3>Total de Granos Detectados</h3>
                <div class="value">{stats['total_grains']}</div>
            </div>
            <div class="stat-card">
                <h3>Imágenes sin Granos</h3>
                <div class="value">{stats['images_without_grains']}</div>
            </div>
            <div class="stat-card">
                <h3>Confianza Media</h3>
                <div class="value">{stats['confidence_stats']['mean']:.3f}</div>
                <div class="unit">± {stats['confidence_stats']['std']:.3f}</div>
            </div>
            <div class="stat-card">
                <h3>Distancia Media</h3>
                <div class="value">{stats['distance_stats']['mean']:.4f}</div>
                <div class="unit">± {stats['distance_stats']['std']:.4f}</div>
            </div>
"""
        
        if stats['has_ground_truth']:
            html_content += f"""
            <div class="stat-card" style="border-left-color: #4ec9b0;">
                <h3>Accuracy</h3>
                <div class="value">{stats['accuracy']:.2%}</div>
            </div>
"""
        
        html_content += """
        </div>
        
        <h2>📊 Análisis Estadístico</h2>
"""
        html_content += f"""
        <div class="metadata">
            <p><strong>Granos por imagen (media ± std):</strong> {stats['grains_per_image_stats']['mean']:.2f} ± {0.0 if pd.isna(stats['grains_per_image_stats']['std']) else stats['grains_per_image_stats']['std']:.2f}</p>
            <p><strong>Cobertura de granos (media ± std):</strong> {stats['coverage_stats']['mean']:.2f}% ± {0.0 if pd.isna(stats['coverage_stats']['std']) else stats['coverage_stats']['std']:.2f}%</p>
            <p><strong>Tiempos medios:</strong> detección={stats['timing_stats_ms']['detect_mean']:.1f}ms, clasificación={stats['timing_stats_ms']['classify_mean']:.1f}ms, total={stats['timing_stats_ms']['elapsed_mean']:.1f}ms</p>
        </div>
"""
        
        # Añadir gráficos
        for fig_name, fig_path in figure_paths.items():
            caption = fig_name.replace('_', ' ').title()
            rel_path = fig_path.relative_to(self.output_dir)
            html_content += f"""
        <div class="figure">
            <img src="{rel_path}" alt="{caption}">
            <div class="figure-caption">{caption}</div>
        </div>
"""
        
        # Tabla de distribución de clases
        html_content += """
        <h2>📋 Distribución de Clases</h2>
        <table>
            <thead>
                <tr>
                    <th>Clase</th>
                    <th>Cantidad</th>
                    <th>Porcentaje</th>
                    <th>Confianza Media</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for class_name, count in stats['class_distribution'].items():
            percentage = (count / stats['total_images']) * 100
            conf_mean = stats['confidence_by_class']['mean'].get(class_name, 0)
            html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                    <td>{conf_mean:.3f}</td>
                </tr>
"""

        if visual_samples:
            html_content += """
            </tbody>
        </table>
        
        <h2>🖼️ Inspección Visual de Detecciones</h2>
        <table>
            <thead>
                <tr>
                    <th>Imagen</th>
                    <th>Motivo Muestra</th>
                    <th>Granos</th>
                    <th>Clase Dominante</th>
                    <th>Confianza</th>
                    <th>Cobertura (%)</th>
                    <th>Vista</th>
                </tr>
            </thead>
            <tbody>
"""
            for sample in visual_samples[:24]:
                ann_path = Path(sample.get('annotated_path', ''))
                img_ref = ""
                if ann_path.exists():
                    try:
                        img_ref = str(ann_path.relative_to(self.output_dir)).replace("\\", "/")
                    except ValueError:
                        img_ref = str(ann_path).replace("\\", "/")
                link_html = f'<a href="{img_ref}" target="_blank">Abrir</a>' if img_ref else 'N/A'
                html_content += f"""
                <tr>
                    <td>{sample.get('image_name', '')}</td>
                    <td>{sample.get('reason', '')}</td>
                    <td>{sample.get('n_grains', 0)}</td>
                    <td>{sample.get('dominant_class', '')}</td>
                    <td>{sample.get('confidence', 0.0):.3f}</td>
                    <td>{sample.get('grain_coverage_pct', 0.0):.2f}</td>
                    <td>{link_html}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <div class="footer">
            <p>Generado por MetricLearning v2.0 - Sistema de Clasificación con Metric Learning</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Guardar HTML
        html_path = self.output_dir / f"{report_name}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_path
