"""
Zero-Shot Evaluation Script for Deep Metric Learning.

Evalúa el modelo en clases nunca vistas durante el entrenamiento,
siguiendo el protocolo estándar de CUB-200 y Stanford Online Products.
"""

import torch
import argparse
from pathlib import Path
import logging
import numpy as np

from src.models.analogy_net import AnalogyNet
from src.data.dataset import MielDataset
from src.utils.model_utils import load_model_from_checkpoint
from src.utils.config_utils import load_config, get_backbone_name, get_transforms_from_config
from src.utils.zero_shot_eval import evaluate_zero_shot_protocol, compare_standard_vs_zeroshot
from src.utils.academic_report import AcademicReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Zero-Shot Evaluation for Metric Learning')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='zeroshot_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_seen_classes', type=int, default=5,
                        help='Number of classes to treat as "seen" (rest are unseen)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for class split')
    parser.add_argument('--mode', type=str, default='protocol', choices=['protocol', 'compare'],
                        help='Evaluation mode: protocol (Zero-Shot only) or compare (Standard vs Zero-Shot)')
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
    logger.info(f"Loading model from {args.checkpoint}...")
    model = load_model_from_checkpoint(args.checkpoint, config, device, eval_mode=True)
    
    # Load full dataset (typically train+val for Zero-Shot protocol)
    logger.info("Loading dataset...")
    transforms = get_transforms_from_config(config, mode='val')
    
    # For Zero-Shot, we use the complete dataset to split by classes
    full_dataset = MielDataset(
        root=config['data']['train_dir'],  # Use train_dir to access all classes
        transform=transforms
    )
    
    logger.info(f"Dataset: {len(full_dataset)} images, {len(full_dataset.classes)} classes")
    
    if args.mode == 'protocol':
        # Standard Zero-Shot protocol
        logger.info("="*70)
        logger.info("ZERO-SHOT EVALUATION PROTOCOL")
        logger.info("="*70)
        
        metrics = evaluate_zero_shot_protocol(
            model=model,
            full_dataset=full_dataset,
            num_train_classes=args.num_seen_classes,
            device=device,
            batch_size=config['training']['batch_size'],
            seed=args.seed
        )
        
        # Generate academic report
        report_gen = AcademicReportGenerator(project_name="Pollen Metric Learning - Zero-Shot")
        
        dataset_info = {
            'name': 'Pollen Grain Dataset (Zero-Shot Protocol)',
            'total_samples': len(full_dataset),
            'num_classes': len(full_dataset.classes),
            'num_seen_classes': args.num_seen_classes,
            'num_unseen_classes': metrics['protocol']['num_unseen_classes'],
            'split': f'Zero-Shot: {args.num_seen_classes} seen, {metrics["protocol"]["num_unseen_classes"]} unseen'
        }
        
        model_info = {
            'backbone': get_backbone_name(config),
            'embedding_dim': config['model']['embedding_dim'],
            'loss': config['training']['loss_function'],
            'projection': config['model'].get('projection_head', 'MLP')
        }
        
        report_gen.generate_full_report(
            metrics=metrics,
            output_dir=output_dir,
            dataset_info=dataset_info,
            model_info=model_info
        )
        
    elif args.mode == 'compare':
        # Compare Standard vs Zero-Shot
        logger.info("="*70)
        logger.info("STANDARD vs ZERO-SHOT COMPARISON")
        logger.info("="*70)
        
        # Load test dataset for standard evaluation
        test_dataset = MielDataset(
            root=config['data']['test_dir'],
            transform=transforms
        )
        
        # Create zero-shot dataset using protocol
        from src.utils.zero_shot_eval import split_dataset_by_classes
        
        np.random.seed(args.seed)
        all_classes = np.arange(len(full_dataset.classes))
        np.random.shuffle(all_classes)
        
        train_classes = all_classes[:args.num_seen_classes].tolist()
        test_classes = all_classes[args.num_seen_classes:].tolist()
        
        _, zeroshot_dataset = split_dataset_by_classes(
            full_dataset, train_classes, test_classes
        )
        
        # Compare
        comparison_results = compare_standard_vs_zeroshot(
            model=model,
            standard_dataset=test_dataset,
            zeroshot_dataset=zeroshot_dataset,
            device=device,
            batch_size=config['training']['batch_size'],
            class_names=full_dataset.classes
        )
        
        # Generate separate reports for each
        report_gen = AcademicReportGenerator(project_name="Pollen Metric Learning - Comparison")
        
        # Standard report
        standard_dir = output_dir / 'standard'
        standard_dir.mkdir(exist_ok=True)
        report_gen.generate_full_report(
            metrics=comparison_results['standard'],
            output_dir=standard_dir,
            dataset_info={'name': 'Standard Evaluation (Seen Classes)'}
        )
        
        # Zero-Shot report
        zeroshot_dir = output_dir / 'zeroshot'
        zeroshot_dir.mkdir(exist_ok=True)
        report_gen.generate_full_report(
            metrics=comparison_results['zero_shot'],
            output_dir=zeroshot_dir,
            dataset_info={'name': 'Zero-Shot Evaluation (Unseen Classes)'}
        )
        
        # Save comparison
        comparison_file = output_dir / 'comparison.txt'
        with open(comparison_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("STANDARD vs ZERO-SHOT COMPARISON\n")
            f.write("="*70 + "\n\n")
            
            f.write("STANDARD (Seen Classes):\n")
            f.write(f"  R@1:      {comparison_results['standard']['R@1']:.4f}\n")
            f.write(f"  R@5:      {comparison_results['standard']['R@5']:.4f}\n")
            f.write(f"  mAP@R:    {comparison_results['standard']['mAP@R']:.4f}\n")
            f.write(f"  NMI:      {comparison_results['standard']['NMI']:.4f}\n")
            f.write(f"  F1-Macro: {comparison_results['standard']['F1_macro']:.4f}\n\n")
            
            f.write("ZERO-SHOT (Unseen Classes):\n")
            f.write(f"  R@1:      {comparison_results['zero_shot']['R@1']:.4f}\n")
            f.write(f"  R@5:      {comparison_results['zero_shot']['R@5']:.4f}\n")
            f.write(f"  mAP@R:    {comparison_results['zero_shot']['mAP@R']:.4f}\n")
            f.write(f"  NMI:      {comparison_results['zero_shot']['NMI']:.4f}\n")
            f.write(f"  F1-Macro: {comparison_results['zero_shot']['F1_macro']:.4f}\n\n")
            
            f.write("DELTA (Standard - Zero-Shot):\n")
            for key, value in comparison_results['comparison'].items():
                f.write(f"  {key}: {value:+.4f}\n")
            f.write("\n" + "="*70 + "\n")
            f.write("Note: Small delta indicates good generalization to unseen classes\n")
            f.write("="*70 + "\n")
        
        logger.info(f"Comparison saved to {comparison_file}")
    
    logger.info("="*70)
    logger.info("Zero-Shot evaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
