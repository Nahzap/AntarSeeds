"""
Master Script for Visual Analytics Pipeline
Executes complete visual inspection workflow: Input → Latent Space → YOLO-style Output
"""

import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.analogy_net import AnalogyNet
from src.data.dataset import MielDataset
from src.utils.model_utils import load_model_from_checkpoint
from src.utils.config_utils import load_config, get_transforms_from_config
from src.visualization.analytics import (
    visualize_input_batch,
    plot_embeddings_tsne,
    visualize_inference_batch,
    compute_class_centroids
)


def load_model(config, checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"📦 Loading model from {checkpoint_path}...")
    model = load_model_from_checkpoint(checkpoint_path, config, device, eval_mode=True)
    print(f"✅ Model loaded successfully")
    return model


def create_dataloaders(config):
    """Create dataloaders for visualization"""
    print(f"📂 Creating dataloaders...")
    
    # Get transforms from config
    train_transforms = get_transforms_from_config(config, mode='train')
    val_transforms = get_transforms_from_config(config, mode='val')
    
    # Create datasets
    train_dataset = MielDataset(
        root=config['data']['train_dir'],
        transform=train_transforms
    )
    
    val_dataset = MielDataset(
        root=config['data']['val_dir'],
        transform=val_transforms
    )
    
    test_dataset = MielDataset(
        root=config['data']['test_dir'],
        transform=val_transforms
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✅ Dataloaders created")
    print(f"   Train: {len(train_dataset)} images")
    print(f"   Val: {len(val_dataset)} images")
    print(f"   Test: {len(test_dataset)} images")
    
    return train_loader, val_loader, test_loader, train_dataset.classes


def main():
    parser = argparse.ArgumentParser(description='Visual Analytics Pipeline for Metric Learning')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='models/final_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results/viz',
                       help='Output directory for visualizations')
    parser.add_argument('--num_images', type=int, default=64,
                       help='Number of images to visualize (default: 64 for 8x8 grid)')
    parser.add_argument('--distance_threshold', type=float, default=0.8,
                       help='Distance threshold for YOLO-style coloring')
    parser.add_argument('--skip_input', action='store_true',
                       help='Skip input batch visualization')
    parser.add_argument('--skip_tsne', action='store_true',
                       help='Skip t-SNE visualization')
    parser.add_argument('--skip_inference', action='store_true',
                       help='Skip inference visualization')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("VISUAL ANALYTICS PIPELINE - METRIC LEARNING SYSTEM")
    print("="*70 + "\n")
    
    # Load configuration
    print(f"📋 Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load model
    model = load_model(config, args.checkpoint, device)
    
    # Create dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(config)
    
    print(f"\n📊 Classes: {class_names}\n")
    
    # ========================================================================
    # STEP 1: Input Batch Visualization
    # ========================================================================
    if not args.skip_input:
        print("\n" + "="*70)
        print("STEP 1: INPUT BATCH VISUALIZATION")
        print("="*70 + "\n")
        
        visualize_input_batch(
            dataloader=train_loader,
            output_path=output_dir / 'input_batch_augmented.png',
            num_images=args.num_images,
            nrow=8,
            device=device
        )
    
    # ========================================================================
    # STEP 2: Latent Space Visualization (t-SNE)
    # ========================================================================
    if not args.skip_tsne:
        print("\n" + "="*70)
        print("STEP 2: LATENT SPACE VISUALIZATION (t-SNE)")
        print("="*70 + "\n")
        
        plot_embeddings_tsne(
            model=model,
            dataloader=val_loader,
            device=device,
            output_path=output_dir / 'tsne_clusters.png',
            class_names=class_names,
            perplexity=30,
            n_iter=1000
        )
    
    # ========================================================================
    # STEP 3: YOLO-Style Inference Visualization
    # ========================================================================
    if not args.skip_inference:
        print("\n" + "="*70)
        print("STEP 3: YOLO-STYLE INFERENCE VISUALIZATION")
        print("="*70 + "\n")
        
        # Compute class centroids
        centroids = compute_class_centroids(
            model=model,
            dataloader=train_loader,
            device=device,
            num_classes=len(class_names)
        )
        
        # Create inference visualization
        visualize_inference_batch(
            model=model,
            dataloader=test_loader,
            centroids=centroids,
            device=device,
            output_path=output_dir / 'inference_grid_yolo_style.png',
            class_names=class_names,
            num_images=args.num_images,
            distance_threshold=args.distance_threshold,
            nrow=8
        )
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("✅ VISUAL ANALYTICS PIPELINE COMPLETED")
    print("="*70 + "\n")
    
    print("📁 Output files:")
    if not args.skip_input:
        print(f"   • Input Batch: {output_dir / 'input_batch_augmented.png'}")
    if not args.skip_tsne:
        print(f"   • t-SNE Clusters: {output_dir / 'tsne_clusters.png'}")
    if not args.skip_inference:
        print(f"   • YOLO-Style Inference: {output_dir / 'inference_grid_yolo_style.png'}")
    
    print("\n🎉 All visualizations generated successfully!")
    print(f"📂 Check the output directory: {output_dir.absolute()}\n")


if __name__ == "__main__":
    main()
