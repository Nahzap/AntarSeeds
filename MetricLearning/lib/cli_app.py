"""
MetricLearning - Aplicación Principal
Sistema de Metric Learning para clasificación de imágenes de microscopía.

Punto de entrada único para todas las funcionalidades del sistema.
"""

import sys
import argparse
from pathlib import Path

from src.utils.logging_utils import setup_logger, cleanup_old_logs


def print_banner():
    """Muestra banner de la aplicación"""
    print("\n" + "="*70)
    print("  MetricLearning - Sistema de Clasificación de Imágenes")
    print("  Metric Learning — Sliced MS + clasificacion slice-aware")
    print("="*70 + "\n")


def train_model(args):
    """Ejecuta entrenamiento del modelo"""
    from scripts.train import run_training
    
    logger = setup_logger("training")
    logger.info("Iniciando entrenamiento del modelo...")
    
    train_args = argparse.Namespace(
        config=args.config,
        resume=args.resume,
        checkpoint=args.checkpoint,
        gpu=None
    )
    
    return run_training(train_args)


def run_inference(args):
    """Ejecuta inferencia sobre imágenes"""
    from scripts.inference import MicroscopyImageRetrieval
    from src.utils.config_utils import load_config
    
    logger = setup_logger("inference")
    logger.info("Iniciando inferencia...")
    
    config = load_config(args.config)
    
    # Base de datos opcional
    database = args.database_dir if args.database_dir else None
    retrieval = MicroscopyImageRetrieval(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        database_path=database
    )
    
    if args.mode == 'classify':
        result = retrieval.classify_image(args.query, k=args.k)
        print(f"\n📊 Resultado de Clasificación:")
        print(f"  Clase predicha: {result['predicted_class']}")
        print(f"  Confianza: {result['confidence']:.2%}")
        print(f"  Distancia: {result['distance']:.4f}")
    
    elif args.mode == 'search':
        results = retrieval.find_similar(args.query, k=args.k)
        print(f"\n🔍 Top-{args.k} Imágenes Similares:")
        for i, (path, dist) in enumerate(zip(results['paths'], results['distances']), 1):
            print(f"  {i}. {Path(path).name} (distancia: {dist:.4f})")
    
    logger.info("Inferencia completada")


def evaluate_model(args):
    """Evalúa el modelo — ejecuta el mismo pipeline post-entrenamiento que train.py"""
    from scripts.post_training import run_evaluation_from_checkpoint

    logger = setup_logger("evaluation")
    if getattr(args, "skip_test", False):
        logger.info("Iniciando evaluacion post-entrenamiento (val only, test omitido)...")
    else:
        logger.info("Iniciando evaluacion post-entrenamiento (regeneracion completa val + test)...")

    return run_evaluation_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        skip_test=getattr(args, "skip_test", False),
    )


def batch_evaluation(args):
    """Evaluación en batch de carpetas"""
    from scripts.batch_evaluation import BatchEvaluator
    
    logger = setup_logger("batch_evaluation")
    logger.info("Iniciando evaluación en batch...")
    
    evaluator = BatchEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config
    )
    
    results = evaluator.evaluate_folders(
        folder_paths=args.folders,
        ground_truth_labels=args.labels,
        k=args.k
    )
    
    print(f"\n📊 Resultados de Evaluación en Batch:")
    print(f"  Accuracy general: {results['overall_accuracy']:.2%}")
    print(f"  Total de imágenes: {results['total_images']}")
    
    logger.info("Evaluación en batch completada")


def interactive_evaluation(args):
    """Lanza interfaz web interactiva"""
    from scripts.interactive_evaluation import launch_interface
    
    logger = setup_logger("interactive")
    logger.info("Lanzando interfaz web interactiva...")
    
    launch_interface(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        port=args.port
    )


def visual_analytics(args):
    """Genera visualizaciones analíticas"""
    from scripts.run_visual_analytics import main as analytics_main
    
    logger = setup_logger("analytics")
    logger.info("Generando visualizaciones analíticas...")
    
    analytics_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        config=args.config,
        output_dir=args.output_dir
    )
    
    return analytics_main(analytics_args)


def prepare_dataset(args):
    """Prepara dataset dividiendo en train/val/test"""
    from scripts.prepare_dataset import main as prepare_main
    
    logger = setup_logger("dataset_preparation")
    logger.info("Preparando dataset...")
    
    prepare_args = argparse.Namespace(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    return prepare_main(prepare_args)


def main():
    """Función principal con CLI"""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="MetricLearning - Sistema de Clasificación de Imágenes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar modelo
  python main.py train --config config.yaml

  # Inferencia - Clasificar imagen
  python main.py inference --checkpoint models/final_model.pth --query image.jpg --mode classify

  # Inferencia - Buscar similares
  python main.py inference --checkpoint models/final_model.pth --query image.jpg --mode search --k 10

  # Evaluar modelo
  python main.py evaluate --checkpoint models/final_model.pth --output-dir results/

  # Evaluación en batch
  python main.py batch --checkpoint models/final_model.pth --folders folder1 folder2 --labels class1 class2

  # Interfaz web interactiva
  python main.py interactive --checkpoint models/final_model.pth --port 7860

  # Visualizaciones analíticas
  python main.py analytics --checkpoint models/final_model.pth --output-dir visualizations/

  # Preparar dataset
  python main.py prepare --source-dir data/raw --output-dir data/processed
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Ruta al archivo de configuración')
    
    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')
    
    # === TRAIN ===
    train_parser = subparsers.add_parser('train', help='Entrenar modelo')
    train_parser.add_argument('--resume', action='store_true',
                             help='Reanudar entrenamiento desde checkpoint')
    train_parser.add_argument('--checkpoint', type=str,
                             help='Ruta al checkpoint para reanudar')
    
    # === INFERENCE ===
    inference_parser = subparsers.add_parser('inference', help='Inferencia sobre imágenes')
    inference_parser.add_argument('--checkpoint', type=str, required=True,
                                 help='Ruta al checkpoint del modelo')
    inference_parser.add_argument('--query', type=str, required=True,
                                 help='Ruta a la imagen query')
    inference_parser.add_argument('--database-dir', type=str, required=True,
                                 help='Directorio con imágenes de referencia')
    inference_parser.add_argument('--mode', type=str, choices=['classify', 'search'],
                                 default='classify', help='Modo de inferencia')
    inference_parser.add_argument('--k', type=int, default=5,
                                 help='Número de vecinos k-NN')
    
    # === EVALUATE ===
    eval_parser = subparsers.add_parser('evaluate', help='Evaluar modelo')
    eval_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Ruta al checkpoint del modelo')
    eval_parser.add_argument('--output-dir', type=str, default='results',
                            help='Directorio de salida para resultados')
    eval_parser.add_argument('--skip-test', action='store_true',
                            help='Omitir test (solo val); por defecto se ejecuta val + test')
    
    # === BATCH ===
    batch_parser = subparsers.add_parser('batch', help='Evaluación en batch')
    batch_parser.add_argument('--checkpoint', type=str, required=True,
                             help='Ruta al checkpoint del modelo')
    batch_parser.add_argument('--folders', type=str, nargs='+', required=True,
                             help='Carpetas a evaluar')
    batch_parser.add_argument('--labels', type=str, nargs='+', required=True,
                             help='Labels ground truth para cada carpeta')
    batch_parser.add_argument('--k', type=int, default=5,
                             help='Número de vecinos k-NN')
    
    # === INTERACTIVE ===
    interactive_parser = subparsers.add_parser('interactive', help='Interfaz web interactiva')
    interactive_parser.add_argument('--checkpoint', type=str, required=True,
                                   help='Ruta al checkpoint del modelo')
    interactive_parser.add_argument('--port', type=int, default=7860,
                                   help='Puerto para servidor web')
    
    # === ANALYTICS ===
    analytics_parser = subparsers.add_parser('analytics', help='Visualizaciones analíticas')
    analytics_parser.add_argument('--checkpoint', type=str, required=True,
                                 help='Ruta al checkpoint del modelo')
    analytics_parser.add_argument('--output-dir', type=str, default='visualizations',
                                 help='Directorio de salida')
    
    # === PREPARE ===
    prepare_parser = subparsers.add_parser('prepare', help='Preparar dataset')
    prepare_parser.add_argument('--source-dir', type=str, required=True,
                               help='Directorio fuente con imágenes')
    prepare_parser.add_argument('--output-dir', type=str, default='data/processed',
                               help='Directorio de salida')
    prepare_parser.add_argument('--train-ratio', type=float, default=0.7,
                               help='Proporción para entrenamiento')
    prepare_parser.add_argument('--val-ratio', type=float, default=0.15,
                               help='Proporción para validación')
    prepare_parser.add_argument('--test-ratio', type=float, default=0.15,
                               help='Proporción para test')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    cleanup_old_logs(days_to_keep=7)
    
    try:
        if args.command == 'train':
            return train_model(args)
        elif args.command == 'inference':
            return run_inference(args)
        elif args.command == 'evaluate':
            return evaluate_model(args)
        elif args.command == 'batch':
            return batch_evaluation(args)
        elif args.command == 'interactive':
            return interactive_evaluation(args)
        elif args.command == 'analytics':
            return visual_analytics(args)
        elif args.command == 'prepare':
            return prepare_dataset(args)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operación cancelada por el usuario")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
