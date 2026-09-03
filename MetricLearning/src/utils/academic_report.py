"""
Generador de reportes académicos en formato IEEE/Elsevier.

Formatea métricas de Deep Metric Learning siguiendo estándares
de publicación en conferencias y revistas de alto impacto.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AcademicReportGenerator:
    """
    Genera reportes en formato académico estándar.
    
    Sigue convenciones de:
    - IEEE CVPR/ICCV/ECCV
    - Elsevier Pattern Recognition
    - Springer IJCV
    """
    
    def __init__(self, project_name: str = "Deep Metric Learning"):
        self.project_name = project_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate_latex_table(
        self,
        metrics: Dict[str, float],
        dataset_name: str = "Custom Dataset"
    ) -> str:
        """
        Genera tabla LaTeX lista para paper.
        
        Args:
            metrics: Dict con métricas
            dataset_name: Nombre del dataset
            
        Returns:
            String con código LaTeX
        """
        latex = []
        latex.append("% Tabla de resultados - Copiar a paper LaTeX")
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{Performance on " + dataset_name + " using Deep Metric Learning.}")
        latex.append("\\label{tab:results}")
        latex.append("\\begin{tabular}{lc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Metric} & \\textbf{Score} \\\\")
        latex.append("\\midrule")
        
        # Ordenar métricas en orden estándar
        metric_order = ['R@1', 'R@5', 'R@10', 'mAP@R', 'NMI', 'F1_macro']
        
        for metric_name in metric_order:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)):
                    latex.append(f"Recall@{metric_name[2:]} & {value:.2f}\\% \\\\" if 'R@' in metric_name 
                                else f"{metric_name.replace('_', '-')} & {value:.4f} \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_markdown_report(
        self,
        metrics: Dict[str, any],
        dataset_info: Optional[Dict] = None,
        model_info: Optional[Dict] = None,
        training_info: Optional[Dict] = None
    ) -> str:
        """
        Genera reporte completo en Markdown.
        
        Args:
            metrics: Métricas calculadas
            dataset_info: Información del dataset
            model_info: Información del modelo
            training_info: Información del entrenamiento
            
        Returns:
            String con reporte Markdown
        """
        lines = []
        
        # Header
        lines.append(f"# {self.project_name} - Evaluation Report")
        lines.append(f"**Generated:** {self.timestamp}")
        lines.append("")
        
        # Dataset Information
        if dataset_info:
            lines.append("## 📊 Dataset Information")
            lines.append("")
            lines.append(f"- **Name:** {dataset_info.get('name', 'N/A')}")
            lines.append(f"- **Total Samples:** {dataset_info.get('total_samples', 'N/A')}")
            lines.append(f"- **Number of Classes:** {dataset_info.get('num_classes', 'N/A')}")
            lines.append(f"- **Train/Val/Test Split:** {dataset_info.get('split', 'N/A')}")
            
            if 'class_distribution' in dataset_info:
                lines.append("")
                lines.append("### Class Distribution")
                lines.append("")
                lines.append("| Class | Samples | Percentage |")
                lines.append("|-------|---------|------------|")
                for class_name, count in dataset_info['class_distribution'].items():
                    pct = (count / dataset_info['total_samples'] * 100) if dataset_info.get('total_samples') else 0
                    lines.append(f"| {class_name} | {count} | {pct:.1f}% |")
            lines.append("")
        
        # Model Information
        if model_info:
            lines.append("## 🧠 Model Architecture")
            lines.append("")
            lines.append(f"- **Backbone:** {model_info.get('backbone', 'N/A')}")
            lines.append(f"- **Embedding Dimension:** {model_info.get('embedding_dim', 'N/A')}")
            lines.append(f"- **Loss Function:** {model_info.get('loss', 'N/A')}")
            lines.append(f"- **Projection Head:** {model_info.get('projection', 'N/A')}")
            lines.append("")
        
        # Training Information
        if training_info:
            lines.append("## 🎯 Training Configuration")
            lines.append("")
            lines.append(f"- **Optimizer:** {training_info.get('optimizer', 'N/A')}")
            lines.append(f"- **Learning Rate:** {training_info.get('learning_rate', 'N/A')}")
            lines.append(f"- **Batch Size:** {training_info.get('batch_size', 'N/A')}")
            lines.append(f"- **Epochs:** {training_info.get('epochs', 'N/A')}")
            lines.append(f"- **Scheduler:** {training_info.get('scheduler', 'N/A')}")
            lines.append("")
        
        # Main Metrics (SOTA)
        lines.append("## 📈 Performance Metrics (SOTA Protocol)")
        lines.append("")
        lines.append("### Primary Metrics")
        lines.append("")
        lines.append("| Metric | Score | Reference |")
        lines.append("|--------|-------|-----------|")
        
        metric_refs = {
            'R@1': 'Song et al. CVPR 2016',
            'R@5': 'Song et al. CVPR 2016',
            'R@10': 'Song et al. CVPR 2016',
            'mAP@R': 'Musgrave et al. ECCV 2020',
            'NMI': 'Song et al. CVPR 2016',
            'F1_macro': 'Barz & Denzler WACV 2020'
        }
        
        for metric_name, ref in metric_refs.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)):
                    display = f"{value*100:.2f}%" if metric_name.startswith('R@') else f"{value:.4f}"
                    lines.append(f"| {metric_name} | **{display}** | {ref} |")
        
        lines.append("")
        
        # Per-Class Metrics
        if 'per_class' in metrics and metrics['per_class']:
            lines.append("### Per-Class Performance")
            lines.append("")
            lines.append("| Class | Precision | Recall | F1-Score | Support |")
            lines.append("|-------|-----------|--------|----------|---------|")
            
            for class_name, class_metrics in metrics['per_class'].items():
                lines.append(f"| {class_name} | {class_metrics['precision']:.3f} | "
                           f"{class_metrics['recall']:.3f} | {class_metrics['f1_score']:.3f} | "
                           f"{class_metrics['support']} |")
            lines.append("")
        
        # Zero-Shot Results (if available)
        if metrics.get('evaluation_type') == 'zero_shot':
            lines.append("## 🎯 Zero-Shot Evaluation")
            lines.append("")
            lines.append("**Protocol:** Classes split into seen/unseen for generalization testing.")
            lines.append("")
            lines.append(f"- **Unseen Classes:** {metrics.get('num_unseen_classes', 'N/A')}")
            
            if 'protocol' in metrics:
                protocol = metrics['protocol']
                lines.append(f"- **Seen Classes:** {protocol.get('num_seen_classes', 'N/A')}")
                lines.append(f"- **Seed:** {protocol.get('seed', 'N/A')}")
            lines.append("")
        
        # Interpretation
        lines.append("## 💡 Interpretation")
        lines.append("")
        
        r1 = metrics.get('R@1', 0)
        map_r = metrics.get('mAP@R', 0)
        nmi = metrics.get('NMI', 0)
        
        if r1 > 0.90:
            quality_r1 = "EXCELLENT"
        elif r1 > 0.80:
            quality_r1 = "GOOD"
        elif r1 > 0.70:
            quality_r1 = "ACCEPTABLE"
        else:
            quality_r1 = "NEEDS IMPROVEMENT"
        
        lines.append(f"- **R@1 Quality:** {quality_r1} ({r1*100:.1f}%)")
        lines.append(f"- **mAP@R:** {map_r:.4f} - Measures ranking quality of retrieved samples")
        lines.append(f"- **NMI:** {nmi:.4f} - Measures cluster separation (1.0 = perfect)")
        lines.append("")
        
        # References
        lines.append("## 📚 References")
        lines.append("")
        lines.append("1. **Song et al.** (CVPR 2016). Deep Metric Learning via Lifted Structured Feature Embedding.")
        lines.append("2. **Musgrave et al.** (ECCV 2020). A Metric Learning Reality Check.")
        lines.append("3. **Barz & Denzler** (WACV 2020). Deep Learning on Small Datasets without Pre-Training.")
        lines.append("4. **Wah et al.** (2011). The Caltech-UCSD Birds-200-2011 Dataset.")
        lines.append("")
        
        return "\n".join(lines)
    
    def generate_json_export(
        self,
        metrics: Dict,
        output_path: Path
    ) -> None:
        """
        Exporta métricas a JSON para procesamiento posterior.
        
        Args:
            metrics: Dict con métricas
            output_path: Ruta de salida
        """
        # Convertir numpy types a Python natives
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        export_data = {
            'timestamp': self.timestamp,
            'project': self.project_name,
            'metrics': convert_numpy(metrics)
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metrics exported to {output_path}")
    
    def generate_full_report(
        self,
        metrics: Dict,
        output_dir: Path,
        dataset_info: Optional[Dict] = None,
        model_info: Optional[Dict] = None,
        training_info: Optional[Dict] = None
    ) -> Dict[str, Path]:
        """
        Genera reporte completo en múltiples formatos.
        
        Args:
            metrics: Métricas calculadas
            output_dir: Directorio de salida
            dataset_info: Información del dataset
            model_info: Información del modelo
            training_info: Información del entrenamiento
            
        Returns:
            Dict con rutas de archivos generados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # Markdown report
        md_content = self.generate_markdown_report(
            metrics, dataset_info, model_info, training_info
        )
        md_path = output_dir / 'evaluation_report.md'
        md_path.write_text(md_content, encoding='utf-8')
        generated_files['markdown'] = md_path
        logger.info(f"Generated Markdown report: {md_path}")
        
        # LaTeX table
        latex_content = self.generate_latex_table(metrics)
        latex_path = output_dir / 'results_table.tex'
        latex_path.write_text(latex_content, encoding='utf-8')
        generated_files['latex'] = latex_path
        logger.info(f"Generated LaTeX table: {latex_path}")
        
        # JSON export
        json_path = output_dir / 'metrics.json'
        self.generate_json_export(metrics, json_path)
        generated_files['json'] = json_path
        
        # Summary text file
        summary_path = output_dir / 'summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"{self.project_name} - EVALUATION SUMMARY\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            f.write(f"PRIMARY METRICS:\n")
            f.write(f"  R@1:      {metrics.get('R@1', 0):.4f}\n")
            f.write(f"  R@5:      {metrics.get('R@5', 0):.4f}\n")
            f.write(f"  mAP@R:    {metrics.get('mAP@R', 0):.4f}\n")
            f.write(f"  NMI:      {metrics.get('NMI', 0):.4f}\n")
            f.write(f"  F1-Macro: {metrics.get('F1_macro', 0):.4f}\n")
            f.write(f"\n{'='*70}\n")
        generated_files['summary'] = summary_path
        logger.info(f"Generated summary: {summary_path}")
        
        logger.info(f"✅ Full academic report generated in: {output_dir}")
        
        return generated_files
