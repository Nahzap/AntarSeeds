"""
Visual Analytics Module for Metric Learning System
Provides functions for input inspection, latent space visualization, and YOLO-style inference.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import cv2
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torchvision.utils as vutils

from .visualization_utils import (
    denormalize_image,
    tensor_to_numpy,
    get_distance_color,
    add_text_with_background,
    create_color_palette
)
from .save_utils import save_plot_with_data


def visualize_input_batch(dataloader: torch.utils.data.DataLoader,
                          output_path: str,
                          num_images: int = 64,
                          nrow: int = 8,
                          device: torch.device = None) -> None:
    """
    Visualize a batch of augmented input images to verify transformations.
    Samples images from ALL classes for balanced representation.
    
    Args:
        dataloader: DataLoader with augmented images
        output_path: Path to save the visualization
        num_images: Number of images to display (default: 64 for 8x8 grid)
        nrow: Number of images per row in the grid
        device: Device to use for GPU acceleration
    """
    print(f"📸 Generating input batch visualization (GPU-accelerated)...")
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Collect images from multiple batches to ensure class diversity
    all_images = []
    all_labels = []
    
    for batch_data in dataloader:
        images, labels = batch_data[0], batch_data[1]
        all_images.append(images)
        all_labels.append(labels)
        
        if sum(len(img) for img in all_images) >= num_images * 2:
            break
    
    # Concatenate all collected images
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Sample balanced images from each class
    unique_labels = torch.unique(all_labels)
    images_per_class = num_images // len(unique_labels)
    
    selected_images = []
    for label in unique_labels:
        mask = all_labels == label
        class_images = all_images[mask]
        
        # Sample images_per_class from this class
        if len(class_images) >= images_per_class:
            indices = torch.randperm(len(class_images))[:images_per_class]
            selected_images.append(class_images[indices])
    
    # Concatenate and limit to num_images
    images = torch.cat(selected_images, dim=0)[:num_images]
    
    # Move to GPU for denormalization
    images = images.to(device)
    
    # Denormalize images (on GPU)
    images_denorm = denormalize_image(images)
    
    # Create grid (on GPU)
    grid = vutils.make_grid(images_denorm, nrow=nrow, padding=4, normalize=False)
    
    # Move to CPU only for final conversion
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 24))
    ax.imshow(grid_np)
    ax.axis('off')
    ax.set_title(f'Input Batch Profile - {num_images} Augmented Images ({nrow}x{nrow})', 
                 fontsize=18, fontweight='bold', pad=20)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Not tabular data, just creating a simple manifest
    df_batch = pd.DataFrame({'image_index': range(num_images)})
    save_plot_with_data(output_path, df_batch, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Input batch visualization saved to: {output_path}")
    print(f"   Grid size: {nrow}x{num_images//nrow}, Total images: {num_images}")


def compute_class_centroids(model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           device: torch.device,
                           num_classes: int) -> torch.Tensor:
    """
    Compute class centroids (mean 128D embeddings per class).

    Para centroides **por slice** (inferencia slice-aware), usar
    ``src.inference.slice_classifier.compute_slice_centroids_from_embeddings``.
    El camino canónico post-entrenamiento es ``scripts/post_training.compute_and_save_centroids``.

    Returns:
        Tensor of shape (num_classes, embedding_dim) with centroids
    """
    print(f"🎯 Computing class centroids...")
    
    model.eval()
    
    # Accumulate embeddings per class
    class_embeddings = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Processing batches"):
            images, labels = batch_data[0], batch_data[1]
            images = images.to(device)
            embeddings = model(images)
            
            # Group by class
            for emb, label in zip(embeddings, labels):
                class_embeddings[label.item()].append(emb.cpu().numpy())
    
    # Compute centroids
    centroids = []
    for class_id in range(num_classes):
        if len(class_embeddings[class_id]) > 0:
            class_embs = np.stack(class_embeddings[class_id])
            centroid = np.mean(class_embs, axis=0)
            centroids.append(centroid)
        else:
            # If no samples, use zero vector
            centroids.append(np.zeros(embeddings.shape[1]))
    
    centroids = torch.tensor(np.stack(centroids), dtype=torch.float32)
    
    print(f"✅ Centroids computed for {num_classes} classes")
    
    return centroids


def plot_embeddings_tsne(model: torch.nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         output_path: str,
                         class_names: List[str],
                         perplexity: int = 30,
                         n_iter: int = 1000) -> None:
    """
    Visualize embeddings in 2D using t-SNE (GPU-accelerated extraction).
    
    Args:
        model: Trained model
        dataloader: DataLoader with labeled data
        device: Device to run on (GPU recommended)
        output_path: Path to save the visualization
        class_names: List of class names
        perplexity: t-SNE perplexity parameter
        n_iter: Number of t-SNE iterations
    """
    print(f"🗺️  Generating t-SNE visualization (GPU-accelerated)...")
    
    model.eval()
    model = model.to(device)
    
    # Extract embeddings and labels (keep on GPU as long as possible)
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Extracting embeddings on GPU"):
            images, labels = batch_data[0], batch_data[1]
            images = images.to(device)
            embeddings = model(images)
            
            # Keep on GPU, only move to CPU at the end
            all_embeddings.append(embeddings)
            all_labels.append(labels.to(device))
    
    # Concatenate on GPU
    embeddings_tensor = torch.cat(all_embeddings, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    
    print(f"📊 Embeddings shape: {embeddings_tensor.shape} (on {device})")
    
    # Move to CPU only for t-SNE (scikit-learn doesn't support GPU)
    embeddings_matrix = embeddings_tensor.cpu().numpy()
    labels_array = labels_tensor.cpu().numpy()
    
    # Apply t-SNE
    print(f"🔄 Running t-SNE (perplexity={perplexity}, max_iter={n_iter})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter, random_state=42, n_jobs=-1)
    embeddings_2d = tsne.fit_transform(embeddings_matrix)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Colored by class
    colors = create_color_palette(len(class_names))
    
    for class_id, class_name in enumerate(class_names):
        mask = labels_array == class_id
        color_rgb = tuple(c / 255.0 for c in colors[class_id])
        axes[0].scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color_rgb],
            label=class_name,
            alpha=0.6,
            s=20
        )
    
    axes[0].set_title('t-SNE Visualization (Colored by Class)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Density plot
    from scipy.stats import gaussian_kde
    
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]
    
    # Calculate point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    
    scatter = axes[1].scatter(x, y, c=z, s=20, cmap='viridis', alpha=0.6)
    axes[1].set_title('t-SNE Density Map', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=axes[1], label='Density')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_tsne = pd.DataFrame({
        'tsne_dim1': embeddings_2d[:, 0],
        'tsne_dim2': embeddings_2d[:, 1],
        'label_id': labels_array,
        'label_name': [class_names[l] for l in labels_array],
        'density': z
    })
    save_plot_with_data(output_path, df_tsne, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ t-SNE visualization saved to: {output_path}")


def visualize_inference_batch(model: torch.nn.Module,
                              dataloader: torch.utils.data.DataLoader,
                              centroids: torch.Tensor,
                              device: torch.device,
                              output_path: str,
                              class_names: List[str],
                              num_images: int = 64,
                              distance_threshold: float = 0.8,
                              nrow: int = 8) -> None:
    """
    Create YOLO-style visualization with predictions and distance-based coloring.
    GPU-accelerated with balanced sampling from all classes.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test images
        centroids: Class centroids tensor (num_classes, embedding_dim)
        device: Device to run on (GPU recommended)
        output_path: Path to save the visualization
        class_names: List of class names
        num_images: Number of images to display (default: 64 for 8x8 grid)
        distance_threshold: Threshold for safe/unsafe classification
        nrow: Number of images per row
    """
    print(f"🎨 Generating YOLO-style inference visualization (GPU-accelerated)...")
    
    model.eval()
    model = model.to(device)
    centroids = centroids.to(device)
    
    # Collect images from multiple batches to ensure class diversity
    all_images = []
    all_labels = []
    
    for batch_data in dataloader:
        images, labels = batch_data[0], batch_data[1]
        all_images.append(images)
        all_labels.append(labels)
        
        if sum(len(img) for img in all_images) >= num_images * 2:
            break
    
    # Concatenate all collected images
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Sample balanced images from each class
    unique_labels = torch.unique(all_labels)
    images_per_class = num_images // len(unique_labels)
    
    selected_images = []
    selected_labels = []
    for label in unique_labels:
        mask = all_labels == label
        class_images = all_images[mask]
        class_labels = all_labels[mask]
        
        # Sample images_per_class from this class
        if len(class_images) >= images_per_class:
            indices = torch.randperm(len(class_images))[:images_per_class]
            selected_images.append(class_images[indices])
            selected_labels.append(class_labels[indices])
    
    # Concatenate and limit to num_images
    images = torch.cat(selected_images, dim=0)[:num_images]
    true_labels = torch.cat(selected_labels, dim=0)[:num_images]
    
    # Get predictions (all on GPU)
    with torch.no_grad():
        images_gpu = images.to(device)
        embeddings = model(images_gpu)
        
        # Compute distances to centroids (on GPU)
        distances = torch.cdist(embeddings, centroids)  # (batch_size, num_classes)
        predicted_classes = torch.argmin(distances, dim=1)
        min_distances = torch.min(distances, dim=1)[0]
    
    # Denormalize images for visualization (on GPU)
    images_denorm = denormalize_image(images_gpu)
    
    # Process each image
    annotated_images = []
    
    for idx in range(len(images)):
        # Convert to numpy (BGR for OpenCV)
        img_np = tensor_to_numpy(images_denorm[idx])
        
        # Get prediction info
        pred_class = predicted_classes[idx].item()
        true_class = true_labels[idx].item()
        distance = min_distances[idx].item()
        
        pred_name = class_names[pred_class]
        true_name = class_names[true_class]
        
        # Determine border color based on distance
        border_color = get_distance_color(distance, distance_threshold)
        
        # Add border
        border_thickness = 8
        img_np = cv2.copyMakeBorder(
            img_np,
            border_thickness, border_thickness,
            border_thickness, border_thickness,
            cv2.BORDER_CONSTANT,
            value=border_color
        )
        
        # Add prediction label at top
        label_text = f"Pred: {pred_name} | Dist: {distance:.3f}"
        img_np = add_text_with_background(
            img_np,
            label_text,
            position=(10, 25),
            font_scale=0.5,
            thickness=1,
            text_color=(255, 255, 255),
            bg_color=(0, 0, 0),
            padding=3
        )
        
        # Add ground truth label at bottom
        gt_text = f"GT: {true_name}"
        gt_color = (0, 255, 0) if pred_class == true_class else (0, 0, 255)
        img_np = add_text_with_background(
            img_np,
            gt_text,
            position=(10, img_np.shape[0] - 10),
            font_scale=0.5,
            thickness=1,
            text_color=(255, 255, 255),
            bg_color=gt_color,
            padding=3
        )
        
        # Convert back to RGB tensor
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        
        annotated_images.append(img_tensor)
    
    # Create grid (on GPU)
    grid = vutils.make_grid(torch.stack(annotated_images), nrow=nrow, padding=6, normalize=False)
    
    # Convert to numpy for saving
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    
    # Create figure with legend
    fig = plt.figure(figsize=(28, 28 + 2))
    
    # Main image
    ax_main = plt.subplot2grid((11, 1), (0, 0), rowspan=10)
    ax_main.imshow(grid_np)
    ax_main.axis('off')
    ax_main.set_title(f'YOLO-Style Inference - {num_images} Images ({nrow}x{nrow}) - Balanced Sampling', 
                     fontsize=20, fontweight='bold', pad=20)
    
    # Legend
    ax_legend = plt.subplot2grid((11, 1), (10, 0))
    ax_legend.axis('off')
    
    legend_text = (
        f"Border Colors (Distance-based Confidence):\n"
        f"🟢 Green: Distance < {distance_threshold * 0.5:.2f} (Very Confident) | "
        f"🟡 Yellow: {distance_threshold * 0.5:.2f}-{distance_threshold * 1.5:.2f} (Warning) | "
        f"🔴 Red: > {distance_threshold * 1.5:.2f} (Anomaly)\n"
        f"Ground Truth Box: Green = Correct | Red = Incorrect"
    )
    
    ax_legend.text(0.5, 0.5, legend_text, ha='center', va='center', 
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_inference = pd.DataFrame({
        'image_idx': range(len(images)),
        'pred_class': predicted_classes.cpu().numpy(),
        'true_class': true_labels.cpu().numpy(),
        'distance': min_distances.cpu().numpy()
    })
    save_plot_with_data(output_path, df_inference, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"✅ YOLO-style visualization saved to: {output_path}")
    print(f"   Grid size: {nrow}x{num_images//nrow}, Total images: {num_images}")
    print(f"   Images per class: ~{images_per_class} (balanced sampling)")
