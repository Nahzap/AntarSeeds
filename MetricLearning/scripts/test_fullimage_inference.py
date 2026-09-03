"""
Test Standalone para FullImage Inference
Ejecuta fullimage_inference independientemente para debugging

Uso:
    python scripts/test_fullimage_inference.py --run_dir runs/2026-03-07_12-55-37
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import logging
from src.utils.config_utils import load_config
from src.utils.model_utils import load_model_from_checkpoint
from scripts.post_training import generate_fullimage_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='Run directory (e.g., runs/2026-03-07_12-55-37)')
    parser.add_argument('--n_images', type=int, default=3, help='Number of images per class')
    parser.add_argument('--with_gradcam', action='store_true', help='Generate GradCAM overlays')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return
    
    # Load config
    config_path = run_dir / 'config' / 'config.yaml'
    if not config_path.exists():
        config_path = Path('config.yaml')
    
    config = load_config(str(config_path))
    logger.info(f"Config loaded from: {config_path}")
    
    # Load model
    checkpoint_path = run_dir / 'checkpoints' / 'best_model.pth'
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loading model from: {checkpoint_path}")
    model = load_model_from_checkpoint(str(checkpoint_path), config, device, eval_mode=True)
    
    # Load centroids
    centroids_path = run_dir / 'class_centroids.pt'
    if not centroids_path.exists():
        logger.error(f"Centroids not found: {centroids_path}")
        return
    
    cent_data = torch.load(centroids_path, map_location=device, weights_only=False)
    centroids = cent_data['centroids'].to(device)
    class_names = cent_data['class_names']
    logger.info(f"Loaded centroids for {len(class_names)} classes: {class_names}")
    
    # Run fullimage inference
    post_dir = run_dir / 'images' / 'post_training'
    post_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("RUNNING FULLIMAGE INFERENCE")
    logger.info("="*70)
    
    try:
        results = generate_fullimage_inference(
            model=model,
            centroids=centroids,
            device=device,
            config=config,
            output_dir=str(post_dir),
            class_names=class_names,
            n_images_per_class=args.n_images,
            with_gradcam=args.with_gradcam
        )
        
        if results:
            logger.info("="*70)
            logger.info("RESULTS:")
            for cls_name, img_results in results.items():
                logger.info(f"  {cls_name}: {len(img_results)} images processed")
                for img_res in img_results:
                    logger.info(f"    - {img_res['image']}: {img_res['correct']}/{img_res['grains']} ({img_res['accuracy']:.1%})")
            logger.info("="*70)
        else:
            logger.warning("No results generated")
            
    except Exception as e:
        logger.error(f"Error during fullimage inference: {e}", exc_info=True)
        return
    
    logger.info(f"Fullimage inference complete! Check: {post_dir / 'fullimage_inference'}")


if __name__ == '__main__':
    main()
