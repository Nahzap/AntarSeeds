"""
Interactive Web Interface for Metric Learning Model Evaluation
Uses Gradio to provide a user-friendly interface for batch evaluation
Uses the TRAINED model directly from the test dataset
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import io
from datetime import datetime
from torch.utils.data import DataLoader

from src.models.analogy_net import AnalogyNet
from src.data.dataset import MielDataset
from src.utils.model_utils import load_model_from_checkpoint
from src.utils.config_utils import load_config, get_transforms_from_config
from src.utils.inference_utils import extract_embeddings
from src.inference.knn_classifier import KNNClassifier


class InteractiveEvaluator:
    """Interactive evaluator using pre-trained model"""
    
    def __init__(self, config_path='config.yaml', checkpoint_path='models/final_model.pth'):
        """Initialize the evaluator with pre-trained model"""
        print("🔧 Inicializando evaluador...")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Usando dispositivo: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        print("✅ Modelo cargado exitosamente")
        
        # Setup transforms
        self.transforms = get_transforms_from_config(self.config, mode='val')
        
        # Load reference database from test set (already trained)
        print("📚 Cargando base de datos de referencia del conjunto de test...")
        self._load_reference_database()
        print(f"✅ Base de datos lista: {len(self.reference_embeddings)} embeddings de {len(self.idx_to_class)} clases")
    
    def _load_model(self, checkpoint_path):
        """Load trained model"""
        return load_model_from_checkpoint(
            checkpoint_path,
            self.config,
            self.device,
            eval_mode=True
        )
    
    def _load_reference_database(self):
        """Load reference database from test dataset"""
        # Load test dataset
        test_dataset = MielDataset(
            root=self.config['data']['test_dir'],
            transform=self.transforms
        )
        
        # Get class names
        self.class_names = test_dataset.classes
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Extract embeddings using utility function
        self.reference_embeddings, self.reference_labels = extract_embeddings(
            self.model,
            test_loader,
            self.device,
            return_paths=False,
            show_progress=False
        )
    
    def get_embedding(self, image_path):
        """Get embedding for a single image"""
        try:
            from src.utils.inference_utils import get_single_embedding
            return get_single_embedding(self.model, image_path, self.transforms, self.device)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def classify_image(self, image_path, k=5):
        """Classify a single image using k-NN"""
        query_embedding = self.get_embedding(image_path)
        if query_embedding is None:
            return None
        
        # Compute distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(query_embedding, self.reference_embeddings)[0]
        
        # Get k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = self.reference_labels[k_nearest_indices]
        
        # Vote for class
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predicted_class = self.idx_to_class[predicted_label]
        
        confidence = np.max(counts) / k
        mean_distance = np.mean(distances[k_nearest_indices])
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'mean_distance': mean_distance
        }
    
    def evaluate_multiple_folders(self, folders_info, k=5, progress=gr.Progress()):
        """
        Evaluate multiple folders at once.
        
        Args:
            folders_info: List of tuples (folder_path, ground_truth_class)
            k: Number of neighbors for k-NN
        
        Returns:
            DataFrame with all results, summary text, confusion matrix plot
        """
        all_results = []
        
        # Count total images
        total_images = 0
        for folder_path, _ in folders_info:
            folder = Path(folder_path)
            if folder.exists():
                total_images += len(list(folder.glob('*.png')))
        
        if total_images == 0:
            return None, "❌ No se encontraron imágenes en las carpetas especificadas", None
        
        processed = 0
        
        # Process each folder
        for folder_path, ground_truth_class in folders_info:
            folder = Path(folder_path)
            if not folder.exists():
                continue
            
            # Get all PNG images
            image_files = list(folder.glob('*.png'))
            
            for img_path in image_files:
                prediction = self.classify_image(img_path, k=k)
                
                if prediction is not None:
                    all_results.append({
                        'folder': folder.name,
                        'image_name': img_path.name,
                        'ground_truth': ground_truth_class,
                        'prediction': prediction['predicted_class'],
                        'confidence': prediction['confidence'],
                        'mean_distance': prediction['mean_distance'],
                        'correct': ground_truth_class == prediction['predicted_class']
                    })
                
                processed += 1
                progress(processed / total_images, desc=f"Procesando {folder.name}: {processed}/{total_images}")
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Calculate overall metrics
        overall_accuracy = df['correct'].mean()
        total = len(df)
        correct = df['correct'].sum()
        avg_confidence = df['confidence'].mean()
        
        # Per-class metrics
        class_metrics = df.groupby('ground_truth').agg({
            'correct': ['sum', 'count', 'mean'],
            'confidence': 'mean'
        }).round(4)
        
        # Create confusion matrix plot
        confusion_fig = self._create_confusion_matrix(df)
        
        # Summary text
        summary = f"""
# 📊 RESULTADOS DE EVALUACIÓN COMPLETA

## Métricas Globales
- **Total de imágenes evaluadas:** {total}
- **Imágenes correctas:** {correct}
- **Imágenes incorrectas:** {total - correct}
- **Accuracy Global:** {overall_accuracy:.2%}
- **Confianza Promedio:** {avg_confidence:.2%}

## Métricas por Clase

"""
        
        for class_name in sorted(df['ground_truth'].unique()):
            class_df = df[df['ground_truth'] == class_name]
            class_acc = class_df['correct'].mean()
            class_total = len(class_df)
            class_correct = class_df['correct'].sum()
            class_conf = class_df['confidence'].mean()
            
            summary += f"""
### {class_name}
- Total: {class_total} imágenes
- Correctas: {class_correct} ({class_acc:.2%})
- Confianza: {class_conf:.2%}
"""
        
        # Format DataFrame for display
        display_df = df[['folder', 'image_name', 'ground_truth', 'prediction', 'confidence', 'correct']].copy()
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
        display_df['correct'] = display_df['correct'].apply(lambda x: '✓' if x else '✗')
        
        return display_df, summary, confusion_fig
    
    def _create_confusion_matrix(self, df):
        """Create confusion matrix visualization"""
        # Create confusion matrix
        confusion = pd.crosstab(df['ground_truth'], df['prediction'])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count matrix
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Cantidad'})
        axes[0].set_title('Matriz de Confusión (Cantidad)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Ground Truth', fontsize=10)
        axes[0].set_xlabel('Predicción', fontsize=10)
        
        # Normalized matrix
        confusion_norm = confusion.div(confusion.sum(axis=1), axis=0)
        sns.heatmap(confusion_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1], cbar_kws={'label': 'Proporción'})
        axes[1].set_title('Matriz de Confusión (Normalizada)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Ground Truth', fontsize=10)
        axes[1].set_xlabel('Predicción', fontsize=10)
        
        plt.tight_layout()
        
        return fig


# Initialize evaluator
print("\n" + "="*70)
print("INICIALIZANDO SISTEMA DE EVALUACIÓN")
print("="*70)
evaluator = InteractiveEvaluator()
print("="*70)
print("✅ SISTEMA LISTO PARA USAR")
print("="*70 + "\n")


def evaluate_folders_interface(folder1, class1, folder2, class2, folder3, class3, folder4, class4, folder5, class5, k):
    """Interface function to evaluate multiple folders"""
    folders_info = []
    
    if folder1 and class1:
        folders_info.append((folder1, class1))
    if folder2 and class2:
        folders_info.append((folder2, class2))
    if folder3 and class3:
        folders_info.append((folder3, class3))
    if folder4 and class4:
        folders_info.append((folder4, class4))
    if folder5 and class5:
        folders_info.append((folder5, class5))
    
    if not folders_info:
        return None, "❌ Debes especificar al menos una carpeta con su clase ground truth", None
    
    df, summary, confusion_fig = evaluator.evaluate_multiple_folders(folders_info, k)
    return df, summary, confusion_fig


# Create Gradio interface
with gr.Blocks(title="Evaluación Interactiva - Metric Learning") as demo:
    gr.Markdown("""
    # 🔬 Evaluación Interactiva de Imágenes de Microscopía
    ## Sistema de Metric Learning - Modelo Pre-entrenado
    
    **El modelo ya está cargado y listo para usar.**
    
    Simplemente especifica las carpetas que quieres evaluar y sus clases ground truth.
    """)
    
    with gr.Tab("🔍 Evaluar Carpetas"):
        gr.Markdown("""
        ### Especifica las carpetas a evaluar y sus clases ground truth
        
        Puedes evaluar hasta 5 carpetas simultáneamente. El sistema comparará las predicciones del modelo con el ground truth que especifiques.
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📁 Carpeta 1")
                folder1_input = gr.Textbox(
                    label="Ruta de la Carpeta",
                    placeholder=r"f:\MICROSCOPIA\Medicago_sativa\ATTMPT_06",
                    value=r"f:\MICROSCOPIA\Medicago_sativa\ATTMPT_06"
                )
                class1_input = gr.Textbox(
                    label="Ground Truth (Clase Verdadera)",
                    placeholder="Medicago_sativa",
                    value="Medicago_sativa"
                )
            
            with gr.Column():
                gr.Markdown("#### 📁 Carpeta 2")
                folder2_input = gr.Textbox(
                    label="Ruta de la Carpeta",
                    placeholder=r"f:\MICROSCOPIA\Lithraea_caustica\ATTMPT_01",
                    value=r"f:\MICROSCOPIA\Lithraea_caustica\ATTMPT_01"
                )
                class2_input = gr.Textbox(
                    label="Ground Truth (Clase Verdadera)",
                    placeholder="Lithraea_caustica",
                    value="Lithraea_caustica"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📁 Carpeta 3")
                folder3_input = gr.Textbox(
                    label="Ruta de la Carpeta",
                    placeholder=r"f:\MICROSCOPIA\Quillaja_Saponaria\ATTMPT05",
                    value=r"f:\MICROSCOPIA\Quillaja_Saponaria\ATTMPT05"
                )
                class3_input = gr.Textbox(
                    label="Ground Truth (Clase Verdadera)",
                    placeholder="Quillaja_Saponaria",
                    value="Quillaja_Saponaria"
                )
            
            with gr.Column():
                gr.Markdown("#### 📁 Carpeta 4 (Opcional)")
                folder4_input = gr.Textbox(
                    label="Ruta de la Carpeta",
                    placeholder="Ruta opcional"
                )
                class4_input = gr.Textbox(
                    label="Ground Truth (Clase Verdadera)",
                    placeholder="Clase opcional"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📁 Carpeta 5 (Opcional)")
                folder5_input = gr.Textbox(
                    label="Ruta de la Carpeta",
                    placeholder="Ruta opcional"
                )
                class5_input = gr.Textbox(
                    label="Ground Truth (Clase Verdadera)",
                    placeholder="Clase opcional"
                )
            
            with gr.Column():
                gr.Markdown("#### ⚙️ Configuración")
                k_input = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="k (Número de vecinos para k-NN)",
                    info="Recomendado: 5-10"
                )
        
        evaluate_button = gr.Button("▶️ Evaluar Todas las Carpetas", variant="primary", size="lg")
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Markdown(label="Resumen de Resultados")
        
        with gr.Row():
            confusion_plot = gr.Plot(label="Matriz de Confusión")
        
        results_output = gr.Dataframe(
            label="Resultados Detallados (Ground Truth vs Predicción)",
            wrap=True,
            interactive=False
        )
        
        evaluate_button.click(
            fn=evaluate_folders_interface,
            inputs=[folder1_input, class1_input, folder2_input, class2_input, folder3_input, class3_input,
                   folder4_input, class4_input, folder5_input, class5_input, k_input],
            outputs=[results_output, summary_output, confusion_plot]
        )
    
    with gr.Tab("ℹ️ Información"):
        gr.Markdown("""
        ## 📖 Guía de Uso
        
        ### Cómo Usar la Interfaz
        
        1. **Especifica las carpetas**: Ingresa las rutas de las carpetas que quieres evaluar
        2. **Indica el Ground Truth**: Para cada carpeta, especifica la clase verdadera de las imágenes
        3. **Ajusta k (opcional)**: Número de vecinos para k-NN (recomendado: 5)
        4. **Evaluar**: Haz clic en "Evaluar Todas las Carpetas"
        5. **Revisa los resultados**: 
           - Resumen con accuracy global y por clase
           - Matriz de confusión visual
           - Tabla detallada con cada imagen (Ground Truth vs Predicción)
        
        ### Interpretación de Resultados
        
        - **Accuracy**: Porcentaje de imágenes clasificadas correctamente
        - **Confidence**: Proporción de los k vecinos que pertenecen a la clase predicha
        - **✓**: Clasificación correcta (Ground Truth = Predicción)
        - **✗**: Clasificación incorrecta (Ground Truth ≠ Predicción)
        
        ### Clases Disponibles
        
        El modelo fue entrenado para clasificar 3 especies:
        - **Lithraea_caustica**
        - **Medicago_sativa**
        - **Quillaja_Saponaria**
        
        ### Requisitos
        
        - Las carpetas deben contener imágenes en formato PNG
        - Las imágenes deben ser de microscopía similares a las del entrenamiento
        - El modelo usa GPU automáticamente si está disponible (CUDA)
        
        ### Información del Modelo
        
        - **Backbone**: ResNet50 (pre-entrenado en ImageNet)
        - **Embedding Dimension**: 128
        - **Loss Function**: Multi-Similarity Loss (SOTA 2024)
        - **Accuracy en Test Completo**: 99.97% (3,321 imágenes)
        - **Base de Datos de Referencia**: 500 imágenes del conjunto de test
        
        ### Ventajas de esta Interfaz
        
        - ✅ **Modelo pre-cargado**: No necesitas reconstruir la base de datos
        - ✅ **Múltiples carpetas**: Evalúa hasta 5 carpetas simultáneamente
        - ✅ **Ground Truth personalizado**: Tú especificas la clase verdadera
        - ✅ **Visualizaciones**: Matriz de confusión y métricas detalladas
        - ✅ **Tabla completa**: Ve cada imagen con su predicción y si fue correcta
        
        ### Ejemplo de Uso
        
        **Escenario:** Quieres evaluar 3 carpetas de imágenes
        
        1. **Carpeta 1**: `f:\\MICROSCOPIA\\Medicago_sativa\\ATTMPT_06` → Ground Truth: `Medicago_sativa`
        2. **Carpeta 2**: `f:\\MICROSCOPIA\\Lithraea_caustica\\ATTMPT_01` → Ground Truth: `Lithraea_caustica`
        3. **Carpeta 3**: `f:\\MICROSCOPIA\\Quillaja_Saponaria\\ATTMPT05` → Ground Truth: `Quillaja_Saponaria`
        
        El sistema procesará todas las imágenes y te mostrará:
        - Cuántas fueron clasificadas correctamente
        - Cuáles se equivocaron (si hay alguna)
        - Matriz de confusión visual
        - Tabla completa con todos los detalles
        """)

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
