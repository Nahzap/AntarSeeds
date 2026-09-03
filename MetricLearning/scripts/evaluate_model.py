"""
Evaluation script for trained Metric Learning model.
Computes metrics, generates visualizations, and tests retrieval performance.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import logging

from src.models.analogy_net import AnalogyNet
from src.data.dataset import MielDataset
from src.utils.model_utils import load_model_from_checkpoint
from src.utils.config_utils import load_config, get_transforms_from_config, get_backbone_name
from src.utils.inference_utils import extract_embeddings
from src.utils.metrics_utils import compute_confusion_matrix, compute_all_metrics
from src.utils.academic_report import AcademicReportGenerator
from pytorch_metric_learning.utils import accuracy_calculator
from pytorch_metric_learning import testers

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(config, checkpoint_path, device):
    """Load trained model from checkpoint"""
    return load_model_from_checkpoint(checkpoint_path, config, device, eval_mode=True)


def get_distances(X, Y, config=None):
    """Dynamically get distances based on config metric space."""
    from src.inference.knn_classifier import euclidean_distances, cosine_distances, poincare_distances_np
    if config is None:
        return euclidean_distances(X, Y)
    
    metric_type = 'euclidean'
    space = config.get('sota', {}).get('embedding_space', 'euclidean')
    if space == 'hyperbolic':
        metric_type = 'hyperbolic'
    else:
        loss_type = config.get('training', {}).get('loss', {}).get('type', 'arcface')
        if loss_type in ['arcface', 'sg_softmax', 'proxy_anchor']:
            metric_type = 'cosine'
            
    if metric_type == 'hyperbolic':
        return poincare_distances_np(X, Y)
    elif metric_type == 'cosine':
        return cosine_distances(X, Y)
    return euclidean_distances(X, Y)


def compute_metrics(embeddings, labels, class_names=None, k_values=[1, 5, 10]):
    """Compute ALL SOTA metrics for Deep Metric Learning"""
    logger.info("Computing SOTA metrics...")
    
    # Use new comprehensive metrics function
    metrics = compute_all_metrics(
        embeddings=embeddings,
        labels=labels,
        k_list=k_values,
        class_names=class_names
    )
    
    return metrics


def plot_tsne(embeddings, labels, class_names, save_path):
    """Plot t-SNE visualization of embeddings"""
    logger.info("Generating t-SNE visualization...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    for label_idx, class_name in enumerate(class_names):
        mask = labels == label_idx
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            label=class_name,
            alpha=0.6,
            s=50
        )
    
    plt.legend(fontsize=12)
    plt.title('t-SNE Visualization of Learned Embeddings', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"t-SNE plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(embeddings, labels, class_names, save_path, config=None):
    """Plot confusion matrix based on nearest neighbor classification"""
    logger.info("Generating confusion matrix...")
    
    # Compute pairwise distances dynamically
    distances = get_distances(embeddings, embeddings, config)
    
    # For each sample, find nearest neighbor (excluding itself)
    np.fill_diagonal(distances, np.inf)
    nearest_neighbors = np.argmin(distances, axis=1)
    predicted_labels = labels[nearest_neighbors]
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predicted_labels)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    plt.title('Confusion Matrix (Nearest Neighbor Classification)', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()
    
    # Compute per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    logger.info("Per-class accuracy:")
    for class_name, acc in zip(class_names, per_class_acc):
        logger.info(f"  {class_name}: {acc:.4f}")


def plot_distance_distribution(embeddings, labels, class_names, save_path, config=None):
    """Plot distribution of intra-class vs inter-class distances"""
    logger.info("Analyzing distance distributions...")
    
    distances = get_distances(embeddings, embeddings, config)
    
    intra_class_distances = []
    inter_class_distances = []
    
    n_samples = len(labels)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = distances[i, j]
            if labels[i] == labels[j]:
                intra_class_distances.append(dist)
            else:
                inter_class_distances.append(dist)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.hist(intra_class_distances, bins=50, alpha=0.6, label='Intra-class', color='green')
    plt.hist(inter_class_distances, bins=50, alpha=0.6, label='Inter-class', color='red')
    
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Intra-class vs Inter-class Distances', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Distance distribution plot saved to {save_path}")
    plt.close()
    
    logger.info(f"Mean intra-class distance: {np.mean(intra_class_distances):.4f}")
    logger.info(f"Mean inter-class distance: {np.mean(inter_class_distances):.4f}")
    logger.info(f"Distance ratio (inter/intra): {np.mean(inter_class_distances) / np.mean(intra_class_distances):.4f}")


def test_retrieval(model, dataset, device, n_queries=5, k=5, config=None):
    """Test image retrieval with sample queries"""
    logger.info(f"Testing retrieval with {n_queries} query images...")
    
    # Extract all embeddings
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    embeddings, labels = extract_embeddings(model, dataloader, device)
    
    # Select random query images
    np.random.seed(42)
    query_indices = np.random.choice(len(dataset), n_queries, replace=False)
    
    results = []
    for query_idx in query_indices:
        query_embedding = embeddings[query_idx:query_idx+1]
        query_label = labels[query_idx]
        
        # Compute distances dynamically
        distances = get_distances(query_embedding, embeddings, config)[0]
        
        # Get top-k nearest neighbors (excluding query itself)
        nearest_indices = np.argsort(distances)[1:k+1]
        nearest_labels = labels[nearest_indices]
        nearest_distances = distances[nearest_indices]
        
        # Check how many are correct
        correct = np.sum(nearest_labels == query_label)
        
        results.append({
            'query_idx': query_idx,
            'query_label': query_label,
            'retrieved_labels': nearest_labels,
            'distances': nearest_distances,
            'precision': correct / k
        })
        
        logger.info(f"Query {query_idx} (class {query_label}): {correct}/{k} correct retrievals")
    
    avg_precision = np.mean([r['precision'] for r in results])
    logger.info(f"Average retrieval precision@{k}: {avg_precision:.4f}")
    
    return results


def generate_report(metrics, class_names, output_dir, dataset_info=None, model_info=None, training_info=None):
    """Generate academic evaluation report using SOTA format"""
    logger.info("Generating academic evaluation report...")
    
    # Create academic report generator
    report_gen = AcademicReportGenerator(project_name="Pollen Metric Learning")
    
    # Generate full report (Markdown, LaTeX, JSON, Summary)
    generated_files = report_gen.generate_full_report(
        metrics=metrics,
        output_dir=output_dir,
        dataset_info=dataset_info,
        model_info=model_info,
        training_info=training_info
    )
    
    logger.info("Academic report generated:")
    for format_name, file_path in generated_files.items():
        logger.info(f"  {format_name}: {file_path}")
    
    return generated_files


def main():
    parser = argparse.ArgumentParser(description='Evaluate Metric Learning Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='models/final_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(config, args.checkpoint, device)
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_transforms = get_transforms_from_config(config, mode='val')
    
    test_dataset = MielDataset(
        root=config['data']['test_dir'],
        transform=test_transforms
    )
    
    class_names = test_dataset.classes
    logger.info(f"Test dataset: {len(test_dataset)} images, {len(class_names)} classes")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(model, test_loader, device)
    
    # Compute SOTA metrics
    metrics = compute_metrics(embeddings, labels, class_names=class_names, k_values=[1, 5, 10])
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    plot_tsne(embeddings, labels, class_names, output_dir / 'tsne_visualization.png')
    plot_confusion_matrix(embeddings, labels, class_names, output_dir / 'confusion_matrix.png', config=config)
    plot_distance_distribution(embeddings, labels, class_names, output_dir / 'distance_distribution.png', config=config)
    
    # Test retrieval
    retrieval_results = test_retrieval(model, test_dataset, device, n_queries=10, k=5, config=config)
    
    # Prepare metadata for academic report
    dataset_info = {
        'name': 'Pollen Grain Classification Dataset',
        'total_samples': len(test_dataset),
        'num_classes': len(class_names),
        'split': 'Test set (15%)',
        'class_distribution': {class_name: int(np.sum(labels == idx)) 
                              for idx, class_name in enumerate(class_names)}
    }
    
    model_info = {
        'backbone': get_backbone_name(config),
        'embedding_dim': config['model']['embedding_dim'],
        'loss': config['training'].get('loss', {}).get('type', 'arcface'),
        'projection': config['model'].get('projection_head', 'MLP')
    }
    
    training_info = {
        'optimizer': config['training']['optimizer'],
        'learning_rate': config['training']['learning_rate'],
        'batch_size': config['training']['batch_size'],
        'epochs': config['training']['epochs'],
        'scheduler': config['training'].get('scheduler', 'CosineAnnealingLR')
    }
    
    # Generate academic report
    generate_report(metrics, class_names, output_dir, dataset_info, model_info, training_info)
    
    logger.info("=" * 70)
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
