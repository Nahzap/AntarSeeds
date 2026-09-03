"""
Inference script for Metric Learning model.
Allows you to:
1. Classify a single image
2. Find similar images from the database
3. Build an index for fast retrieval
"""

import torch
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import logging
from sklearn.metrics.pairwise import euclidean_distances

from src.models.analogy_net import AnalogyNet
from src.data.dataset import MielDataset
from src.utils.model_utils import load_model_from_checkpoint
from src.utils.config_utils import load_config, get_transforms_from_config
from src.utils.inference_utils import extract_embeddings, get_single_embedding
from src.inference.knn_classifier import KNNClassifier
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MicroscopyImageRetrieval:
    """
    Image retrieval system for microscopy images (legacy MielDataset path).

    .. deprecated::
        Para clasificación alineada con entrenamiento ``sliced_ms``, use
        :class:`src.grain_detection.full_image_classifier.FullImageClassifier`.
    """
    
    def __init__(self, config_path, checkpoint_path, database_path=None):
        """
        Initialize the retrieval system.
        
        Args:
            config_path: Path to config.yaml
            checkpoint_path: Path to trained model checkpoint
            database_path: Optional path to database images (if None, uses checkpoint embeddings)
        """
        logger.info("Initializing Microscopy Image Retrieval System...")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and checkpoint
        self.checkpoint_path = checkpoint_path
        self.model, self.checkpoint = self._load_model_and_checkpoint(checkpoint_path)
        
        # Load reference embeddings
        self.database_embeddings = None
        self.database_labels = None
        self.database_paths = None
        self.class_names = None
        
        if database_path:
            # Modo legacy: construir desde carpeta
            logger.info("Using database from directory (legacy mode)")
            self.database_path = Path(database_path)
            self._build_database()
        else:
            # Modo nuevo: usar embeddings del checkpoint
            logger.info("Using reference embeddings from checkpoint")
            self._load_reference_from_checkpoint()
        
        logger.info("System initialized successfully!")
    
    def _load_model_and_checkpoint(self, checkpoint_path):
        """Load trained model and checkpoint"""
        model = load_model_from_checkpoint(
            checkpoint_path,
            self.config,
            self.device,
            eval_mode=True
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        return model, checkpoint
    
    def _load_reference_from_checkpoint(self):
        """Load reference embeddings from checkpoint or sibling files."""
        if 'reference_embeddings' in self.checkpoint:
            self.database_embeddings = self.checkpoint['reference_embeddings']
            self.database_labels = self.checkpoint['reference_labels']
            self.class_names = self.checkpoint['class_names']
            logger.info(f"Loaded reference from checkpoint: {len(self.database_embeddings)} embeddings, "
                       f"{len(self.class_names)} classes")
            logger.info(f"Classes: {', '.join(self.class_names)}")
            return

        # Fallback: look for separate files in the run directory
        run_dir = Path(self.checkpoint_path).parent.parent
        ref_path = run_dir / 'reference_embeddings.pt'
        cent_path = run_dir / 'class_centroids.pt'

        if ref_path.exists() and cent_path.exists():
            logger.info("Checkpoint lacks embedded references — loading from sibling files...")
            ref_data = torch.load(str(ref_path), map_location=self.device, weights_only=False)
            cent_data = torch.load(str(cent_path), map_location=self.device, weights_only=False)

            emb = ref_data['embeddings']
            self.database_embeddings = emb.numpy() if torch.is_tensor(emb) else emb
            lbl = ref_data['labels']
            self.database_labels = lbl.numpy() if torch.is_tensor(lbl) else lbl
            self.class_names = cent_data['class_names']

            logger.info(f"Loaded reference from files: {len(self.database_embeddings)} embeddings, "
                       f"{len(self.class_names)} classes")
            logger.info(f"Classes: {', '.join(self.class_names)}")
            return

        raise ValueError(
            "Checkpoint does not contain reference embeddings and no sibling files found. "
            "Please retrain the model or provide a database_path."
        )
    
    def _build_database(self):
        """Build embedding database from images"""
        logger.info("Building image database...")
        
        # Setup transforms
        transforms = get_transforms_from_config(self.config, mode='val')
        
        # Load dataset
        dataset = MielDataset(
            root=str(self.database_path),
            transform=transforms,
            return_paths=True
        )
        
        self.class_names = dataset.classes
        logger.info(f"Database: {len(dataset)} images, {len(self.class_names)} classes")
        logger.info(f"Classes: {', '.join(self.class_names)}")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Extract embeddings
        embeddings_list = []
        labels_list = []
        paths_list = []
        
        with torch.no_grad():
            for images, labels, paths in dataloader:
                images = images.to(self.device)
                embeddings = self.model(images)
                
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(labels.numpy())
                paths_list.extend(paths)
        
        self.database_embeddings = np.vstack(embeddings_list)
        self.database_labels = np.concatenate(labels_list)
        self.database_paths = paths_list
        
        logger.info(f"Database built: {len(self.database_embeddings)} embeddings")
    
    def get_embedding(self, image_path):
        """
        Get embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            numpy array: Embedding vector
        """
        transforms = get_transforms_from_config(self.config, mode='val')
        return get_single_embedding(self.model, image_path, transforms, self.device)
    
    def find_similar(self, query_image_path, k=5, return_distances=True):
        """
        Find k most similar images to query image.
        
        Args:
            query_image_path: Path to query image
            k: Number of similar images to return
            return_distances: Whether to return distances
            
        Returns:
            dict: Results with paths, labels, and optionally distances
        """
        logger.info(f"Finding {k} similar images to: {query_image_path}")
        
        # Get query embedding
        query_embedding = self.get_embedding(query_image_path)
        
        # Compute distances
        distances = euclidean_distances(query_embedding, self.database_embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(distances)[:k]
        
        # database_paths may be None when loaded from checkpoint (no file paths)
        if self.database_paths is not None:
            paths = [self.database_paths[i] for i in top_k_indices]
        else:
            paths = [f"ref_{i}" for i in top_k_indices]
        
        results = {
            'paths': paths,
            'labels': [self.database_labels[i] for i in top_k_indices],
            'class_names': [self.class_names[self.database_labels[i]] for i in top_k_indices]
        }
        
        if return_distances:
            results['distances'] = distances[top_k_indices]
        
        return results
    
    def classify_image(self, image_path, k=5):
        """
        Classify image using k-NN on embeddings.

        .. deprecated::
            Usa geometría 128D global (k-NN euclidiano). Para pipeline
            slice-aware coherente con ``sliced_ms``, use
            ``FullImageClassifier.from_checkpoint()`` + ``classify_full_image()``.

        Args:
            image_path: Path to image to classify
            k: Number of neighbors to consider

        Returns:
            dict: Classification results
        """
        import warnings
        warnings.warn(
            "MicroscopyImageRetrieval.classify_image usa k-NN legacy (128D). "
            "Use FullImageClassifier para clasificación slice-aware.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(f"Classifying image: {image_path}")
        
        # Find k similar images
        results = self.find_similar(image_path, k=k, return_distances=True)
        
        # Vote for class
        labels = results['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]
        predicted_class = self.class_names[predicted_label]
        
        # Compute confidence (proportion of k neighbors with same class)
        confidence = np.max(counts) / k
        
        # Get class distribution
        class_distribution = {}
        for label, count in zip(unique_labels, counts):
            class_distribution[self.class_names[label]] = count / k
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'distance': results['distances'][0],  # Distance to nearest neighbor
            'class_distribution': class_distribution,
            'nearest_neighbors': results
        }
    
    def visualize_retrieval(self, query_image_path, k=5, save_path=None):
        """
        Visualize retrieval results.
        
        Args:
            query_image_path: Path to query image
            k: Number of results to show
            save_path: Path to save visualization (optional)
        """
        results = self.find_similar(query_image_path, k=k, return_distances=True)
        
        # Create figure
        fig, axes = plt.subplots(2, k + 1, figsize=(3 * (k + 1), 6))
        
        # Show query image
        query_img = Image.open(query_image_path)
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Show retrieved images
        for i in range(k):
            img = Image.open(results['paths'][i])
            axes[0, i + 1].imshow(img)
            axes[0, i + 1].set_title(
                f"#{i+1}\n{results['class_names'][i]}",
                fontsize=10
            )
            axes[0, i + 1].axis('off')
            
            # Show distance
            axes[1, i + 1].text(
                0.5, 0.5,
                f"Distance:\n{results['distances'][i]:.4f}",
                ha='center', va='center',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
            axes[1, i + 1].axis('off')
        
        plt.suptitle(f'Image Retrieval Results (Top {k})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Microscopy Image Retrieval and Classification')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='models/final_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--database', type=str, default='data/processed/test',
                        help='Path to database images')
    parser.add_argument('--query', type=str, required=True,
                        help='Path to query image')
    parser.add_argument('--mode', type=str, choices=['classify', 'retrieve'], default='classify',
                        help='Mode: classify or retrieve')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors/results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for visualization')
    args = parser.parse_args()
    
    # Initialize retrieval system
    retrieval_system = MicroscopyImageRetrieval(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        database_path=args.database
    )
    
    print("\n" + "=" * 70)
    print("MICROSCOPY IMAGE ANALYSIS")
    print("=" * 70)
    
    if args.mode == 'classify':
        # Classify image
        results = retrieval_system.classify_image(args.query, k=args.k)
        
        print(f"\nQuery Image: {args.query}")
        print(f"\nPredicted Class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print("\nClass Distribution:")
        for class_name, prob in results['class_distribution'].items():
            print(f"  {class_name}: {prob:.2%}")
        
        print(f"\nNearest Neighbors (k={args.k}):")
        for i, (path, class_name, dist) in enumerate(zip(
            results['nearest_neighbors']['paths'],
            results['nearest_neighbors']['class_names'],
            results['nearest_neighbors']['distances']
        )):
            print(f"  {i+1}. {class_name} (distance: {dist:.4f})")
            print(f"     Path: {path}")
    
    elif args.mode == 'retrieve':
        # Retrieve similar images
        results = retrieval_system.find_similar(args.query, k=args.k, return_distances=True)
        
        print(f"\nQuery Image: {args.query}")
        print(f"\nTop {args.k} Similar Images:")
        for i, (path, class_name, dist) in enumerate(zip(
            results['paths'],
            results['class_names'],
            results['distances']
        )):
            print(f"  {i+1}. {class_name} (distance: {dist:.4f})")
            print(f"     Path: {path}")
    
    # Visualize if requested
    if args.visualize:
        output_path = args.output or f'retrieval_result_{Path(args.query).stem}.png'
        retrieval_system.visualize_retrieval(args.query, k=args.k, save_path=output_path)
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
