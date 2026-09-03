"""
Batch evaluation script for Metric Learning model.
Evaluates all images in specified folders and generates comparative results:
- Ground Truth (from folder name)
- Predictions (from model inference)
- Accuracy metrics per class and overall
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging
from datetime import datetime

from src.models.analogy_net import AnalogyNet
from src.utils.model_utils import load_model_from_checkpoint
from src.utils.config_utils import load_config, get_transforms_from_config
from src.utils.inference_utils import get_single_embedding
from src.inference.knn_classifier import KNNClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BatchEvaluator:
    """Batch evaluation system for microscopy images"""
    
    def __init__(self, config_path, checkpoint_path):
        """
        Initialize the batch evaluator.
        
        Args:
            config_path: Path to config.yaml
            checkpoint_path: Path to trained model checkpoint
        """
        logger.info("Initializing Batch Evaluator...")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Setup transforms
        self.transforms = get_transforms_from_config(self.config, mode='val')
        
        # Class mapping (will be populated during evaluation)
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        logger.info("Batch Evaluator initialized successfully!")
    
    def _load_model(self, checkpoint_path):
        """Load trained model"""
        return load_model_from_checkpoint(
            checkpoint_path,
            self.config,
            self.device,
            eval_mode=True
        )
    
    def get_embedding(self, image_path):
        """Get embedding for a single image"""
        try:
            return get_single_embedding(self.model, image_path, self.transforms, self.device)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None
    
    def build_reference_database(self, folders_dict):
        """
        Build reference database from training/validation data.
        
        Args:
            folders_dict: Dict mapping class names to folder paths
        """
        logger.info("Building reference database from provided folders...")
        
        self.reference_embeddings = []
        self.reference_labels = []
        self.reference_class_names = []
        
        # Create class mapping
        class_names = sorted(folders_dict.keys())
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        logger.info(f"Classes: {class_names}")
        
        # Process each class folder
        for class_name, folder_path in folders_dict.items():
            folder = Path(folder_path)
            if not folder.exists():
                logger.warning(f"Folder not found: {folder_path}")
                continue
            
            # Get all PNG images
            image_files = list(folder.glob('*.png'))
            logger.info(f"Processing {class_name}: {len(image_files)} images")
            
            # Sample subset for reference (to avoid memory issues)
            # Use up to 200 images per class for reference
            if len(image_files) > 200:
                import random
                random.seed(42)
                image_files = random.sample(image_files, 200)
                logger.info(f"  Sampled 200 images for reference database")
            
            for img_path in tqdm(image_files, desc=f"Building reference for {class_name}"):
                embedding = self.get_embedding(img_path)
                if embedding is not None:
                    self.reference_embeddings.append(embedding)
                    self.reference_labels.append(self.class_to_idx[class_name])
                    self.reference_class_names.append(class_name)
        
        self.reference_embeddings = np.vstack(self.reference_embeddings)
        self.reference_labels = np.array(self.reference_labels)
        
        logger.info(f"Reference database built: {len(self.reference_embeddings)} embeddings")
    
    def classify_image(self, image_path, k=5):
        """
        Classify a single image using k-NN.
        
        Args:
            image_path: Path to image
            k: Number of neighbors
            
        Returns:
            dict: Classification results
        """
        # Get embedding
        query_embedding = self.get_embedding(image_path)
        if query_embedding is None:
            return None
        
        # Compute distances to reference database
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(query_embedding, self.reference_embeddings)[0]
        
        # Get k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]
        k_nearest_labels = self.reference_labels[k_nearest_indices]
        k_nearest_distances = distances[k_nearest_indices]
        
        # Vote for class
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predicted_class = self.idx_to_class[predicted_label]
        
        # Confidence
        confidence = np.max(counts) / k
        
        # Mean distance to k nearest neighbors
        mean_distance = np.mean(k_nearest_distances)
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'mean_distance': mean_distance,
            'k_nearest_distances': k_nearest_distances
        }
    
    def evaluate_folders(self, folders_dict, k=5, output_dir='batch_evaluation_results'):
        """
        Evaluate all images in the provided folders.
        
        Args:
            folders_dict: Dict mapping class names (ground truth) to folder paths
            k: Number of neighbors for k-NN
            output_dir: Directory to save results
            
        Returns:
            pd.DataFrame: Results dataframe
        """
        logger.info("="*70)
        logger.info("BATCH EVALUATION - GROUND TRUTH vs PREDICTIONS")
        logger.info("="*70)
        
        # Build reference database first
        self.build_reference_database(folders_dict)
        
        # Results storage
        results = []
        
        # Evaluate each folder
        for ground_truth_class, folder_path in folders_dict.items():
            folder = Path(folder_path)
            if not folder.exists():
                logger.warning(f"Folder not found: {folder_path}")
                continue
            
            # Get all PNG images
            image_files = list(folder.glob('*.png'))
            logger.info(f"\nEvaluating {ground_truth_class}: {len(image_files)} images")
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"Evaluating {ground_truth_class}"):
                # Classify
                prediction = self.classify_image(img_path, k=k)
                
                if prediction is not None:
                    results.append({
                        'image_path': str(img_path),
                        'image_name': img_path.name,
                        'ground_truth': ground_truth_class,
                        'prediction': prediction['predicted_class'],
                        'confidence': prediction['confidence'],
                        'mean_distance': prediction['mean_distance'],
                        'correct': ground_truth_class == prediction['predicted_class']
                    })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = output_path / f'detailed_results_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"\nDetailed results saved to: {csv_path}")
        
        # Generate summary
        self._generate_summary(df, output_path, timestamp)
        
        return df
    
    def _generate_summary(self, df, output_path, timestamp):
        """Generate summary statistics and reports"""
        logger.info("\n" + "="*70)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*70)
        
        # Overall accuracy
        overall_accuracy = df['correct'].mean()
        logger.info(f"\nOverall Accuracy: {overall_accuracy:.2%}")
        
        # Per-class accuracy
        logger.info("\nPer-Class Accuracy:")
        class_accuracy = df.groupby('ground_truth')['correct'].agg(['mean', 'sum', 'count'])
        class_accuracy.columns = ['Accuracy', 'Correct', 'Total']
        logger.info(class_accuracy)
        
        # Confusion matrix
        logger.info("\nConfusion Matrix:")
        confusion = pd.crosstab(
            df['ground_truth'], 
            df['prediction'], 
            rownames=['Ground Truth'], 
            colnames=['Prediction'],
            margins=True
        )
        logger.info(confusion)
        
        # Average confidence
        logger.info(f"\nAverage Confidence: {df['confidence'].mean():.2%}")
        logger.info(f"Average Confidence (Correct): {df[df['correct']]['confidence'].mean():.2%}")
        logger.info(f"Average Confidence (Incorrect): {df[~df['correct']]['confidence'].mean():.2%}")
        
        # Save summary report
        report_path = output_path / f'summary_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BATCH EVALUATION SUMMARY REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images Evaluated: {len(df)}\n")
            f.write(f"Number of Classes: {df['ground_truth'].nunique()}\n\n")
            
            f.write(f"Overall Accuracy: {overall_accuracy:.2%}\n\n")
            
            f.write("Per-Class Results:\n")
            f.write("-"*70 + "\n")
            for class_name in sorted(df['ground_truth'].unique()):
                class_df = df[df['ground_truth'] == class_name]
                acc = class_df['correct'].mean()
                total = len(class_df)
                correct = class_df['correct'].sum()
                conf = class_df['confidence'].mean()
                
                f.write(f"\n{class_name}:\n")
                f.write(f"  Accuracy: {acc:.2%} ({correct}/{total})\n")
                f.write(f"  Average Confidence: {conf:.2%}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("Confusion Matrix:\n")
            f.write("="*70 + "\n")
            f.write(confusion.to_string())
            f.write("\n\n")
            
            # Misclassified examples
            if (~df['correct']).any():
                f.write("="*70 + "\n")
                f.write("MISCLASSIFIED EXAMPLES:\n")
                f.write("="*70 + "\n\n")
                
                misclassified = df[~df['correct']][['image_name', 'ground_truth', 'prediction', 'confidence', 'mean_distance']]
                f.write(misclassified.to_string(index=False))
                f.write("\n")
        
        logger.info(f"\nSummary report saved to: {report_path}")
        
        # Save confusion matrix as CSV
        confusion_path = output_path / f'confusion_matrix_{timestamp}.csv'
        confusion.to_csv(confusion_path)
        logger.info(f"Confusion matrix saved to: {confusion_path}")
        
        # Save per-class accuracy
        accuracy_path = output_path / f'per_class_accuracy_{timestamp}.csv'
        class_accuracy.to_csv(accuracy_path)
        logger.info(f"Per-class accuracy saved to: {accuracy_path}")
        
        logger.info("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description='Batch Evaluation of Microscopy Images')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='models/final_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for k-NN classification')
    parser.add_argument('--output_dir', type=str, default='batch_evaluation_results',
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Define folders to evaluate (ground truth from folder names)
    folders_to_evaluate = {
        'Medicago_sativa': r'f:\MICROSCOPIA\Medicago_sativa\ATTMPT_06',
        'Lithraea_caustica': r'f:\MICROSCOPIA\Lithraea_caustica\ATTMPT_01',
        'Quillaja_Saponaria': r'f:\MICROSCOPIA\Quillaja_Saponaria\ATTMPT05'
    }
    
    # Initialize evaluator
    evaluator = BatchEvaluator(
        config_path=args.config,
        checkpoint_path=args.checkpoint
    )
    
    # Run evaluation
    results_df = evaluator.evaluate_folders(
        folders_dict=folders_to_evaluate,
        k=args.k,
        output_dir=args.output_dir
    )
    
    logger.info("\n" + "="*70)
    logger.info("BATCH EVALUATION COMPLETED SUCCESSFULLY!")
    logger.info("="*70)
    logger.info(f"\nResults saved to: {args.output_dir}/")
    logger.info("\nFiles generated:")
    logger.info("  - detailed_results_*.csv: Complete results for all images")
    logger.info("  - summary_report_*.txt: Summary statistics and analysis")
    logger.info("  - confusion_matrix_*.csv: Confusion matrix")
    logger.info("  - per_class_accuracy_*.csv: Accuracy per class")


if __name__ == '__main__':
    main()
