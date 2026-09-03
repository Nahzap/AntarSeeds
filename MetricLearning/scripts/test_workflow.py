"""
Script de prueba para verificar el flujo de trabajo completo con logging.
Prueba todos los módulos refactorizados y el sistema de logging.
"""

import torch
import numpy as np
from pathlib import Path
import sys

from src.utils.logging_utils import (
    setup_logger,
    log_system_info,
    log_config,
    log_model_summary,
    cleanup_old_logs
)
from src.utils.model_utils import load_model_from_checkpoint, get_model_info
from src.utils.config_utils import load_config, get_transforms_from_config, get_device_from_config
from src.utils.inference_utils import get_single_embedding
from src.inference.knn_classifier import KNNClassifier
from src.models.analogy_net import AnalogyNet
from src.utils.config_utils import get_backbone_name


def test_logging_system():
    """Prueba el sistema de logging"""
    print("\n" + "="*70)
    print("TEST 1: SISTEMA DE LOGGING")
    print("="*70 + "\n")
    
    cleanup_old_logs(days_to_keep=7)
    
    logger = setup_logger(
        name="test_workflow",
        log_dir="logs",
        log_level=10,
        console_output=True,
        file_output=True
    )
    
    logger.debug("Debug message - lowest level")
    logger.info("Info message - normal operations")
    logger.warning("Warning message - something to watch")
    logger.error("Error message - something went wrong")
    
    log_system_info(logger)
    
    print("✅ Sistema de logging funcionando correctamente")
    print(f"✅ Logs guardados en: logs/test_workflow_{Path('logs').absolute()}")
    
    return logger


def test_config_loading(logger):
    """Prueba carga de configuración"""
    print("\n" + "="*70)
    print("TEST 2: CARGA DE CONFIGURACIÓN")
    print("="*70 + "\n")
    
    try:
        config = load_config("config.yaml")
        logger.info("✅ Configuración cargada exitosamente")
        
        log_config(logger, config)
        
        device = get_device_from_config(config)
        logger.info(f"✅ Device obtenido: {device}")
        
        train_transforms = get_transforms_from_config(config, mode='train')
        val_transforms = get_transforms_from_config(config, mode='val')
        logger.info("✅ Transforms obtenidos correctamente")
        
        print("✅ Carga de configuración exitosa")
        return config, device, train_transforms, val_transforms
        
    except Exception as e:
        logger.error(f"❌ Error cargando configuración: {e}", exc_info=True)
        raise


def test_model_creation(logger, config, device):
    """Prueba creación de modelo"""
    print("\n" + "="*70)
    print("TEST 3: CREACIÓN DE MODELO")
    print("="*70 + "\n")
    
    try:
        model = AnalogyNet(
            backbone=get_backbone_name(config),
            embedding_dim=config['model']['embedding_dim'],
            pretrained=False,
            projection_head_config=config['model'].get('projection_head'),
            use_attention=config['model'].get('use_attention', False),
            freeze_backbone=False
        )
        model = model.to(device)
        model.eval()
        
        logger.info("✅ Modelo creado exitosamente")
        
        log_model_summary(logger, model)
        
        model_info = get_model_info(model)
        logger.info(f"Total parameters: {model_info['total_params']:,}")
        logger.info(f"Trainable parameters: {model_info['trainable_params']:,}")
        
        print("✅ Creación de modelo exitosa")
        return model
        
    except Exception as e:
        logger.error(f"❌ Error creando modelo: {e}", exc_info=True)
        raise


def test_model_inference(logger, model, transforms, device):
    """Prueba inferencia del modelo"""
    print("\n" + "="*70)
    print("TEST 4: INFERENCIA DEL MODELO")
    print("="*70 + "\n")
    
    try:
        dummy_image = torch.randn(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            embedding = model(dummy_image)
        
        logger.info(f"✅ Embedding shape: {embedding.shape}")
        logger.info(f"✅ Embedding norm: {torch.norm(embedding).item():.4f}")
        
        embedding_np = embedding.cpu().numpy()
        logger.info(f"✅ Embedding convertido a numpy: {embedding_np.shape}")
        
        print("✅ Inferencia del modelo exitosa")
        return embedding_np
        
    except Exception as e:
        logger.error(f"❌ Error en inferencia: {e}", exc_info=True)
        raise


def test_knn_classifier(logger):
    """Prueba clasificador k-NN"""
    print("\n" + "="*70)
    print("TEST 5: CLASIFICADOR k-NN")
    print("="*70 + "\n")
    
    try:
        np.random.seed(42)
        n_samples = 100
        n_classes = 5
        embedding_dim = 128
        
        reference_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
        reference_embeddings = reference_embeddings / np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
        
        reference_labels = np.random.randint(0, n_classes, n_samples)
        
        class_names = [f"Class_{i}" for i in range(n_classes)]
        
        classifier = KNNClassifier(
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            class_names=class_names,
            k=5
        )
        
        logger.info("✅ Clasificador k-NN creado exitosamente")
        
        query_embedding = np.random.randn(1, embedding_dim).astype(np.float32)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        result = classifier.predict(query_embedding, return_distances=True, return_neighbors=True)
        
        logger.info(f"✅ Predicción: {result['predicted_class']}")
        logger.info(f"✅ Confianza: {result['confidence']:.2%}")
        logger.info(f"✅ Distancia media: {result['mean_distance']:.4f}")
        
        distribution = classifier.get_class_distribution(query_embedding)
        logger.info("✅ Distribución de clases en k vecinos:")
        for class_name, prob in distribution.items():
            logger.info(f"   {class_name}: {prob:.2%}")
        
        similar = classifier.find_similar(query_embedding, k=3)
        logger.info(f"✅ Top-3 similares: {similar['class_names']}")
        
        print("✅ Clasificador k-NN funcionando correctamente")
        return classifier
        
    except Exception as e:
        logger.error(f"❌ Error en clasificador k-NN: {e}", exc_info=True)
        raise


def test_batch_prediction(logger, classifier):
    """Prueba predicción en batch"""
    print("\n" + "="*70)
    print("TEST 6: PREDICCIÓN EN BATCH")
    print("="*70 + "\n")
    
    try:
        n_queries = 10
        embedding_dim = 128
        
        query_embeddings = np.random.randn(n_queries, embedding_dim).astype(np.float32)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        results = classifier.predict_batch(query_embeddings, show_progress=False)
        
        logger.info(f"✅ Predicciones en batch: {len(results)} resultados")
        
        confidences = [r['confidence'] for r in results]
        logger.info(f"✅ Confianza promedio: {np.mean(confidences):.2%}")
        logger.info(f"✅ Confianza mínima: {np.min(confidences):.2%}")
        logger.info(f"✅ Confianza máxima: {np.max(confidences):.2%}")
        
        print("✅ Predicción en batch exitosa")
        
    except Exception as e:
        logger.error(f"❌ Error en predicción batch: {e}", exc_info=True)
        raise


def test_utils_modules(logger):
    """Prueba módulos de utilidades"""
    print("\n" + "="*70)
    print("TEST 7: MÓDULOS DE UTILIDADES")
    print("="*70 + "\n")
    
    try:
        from src.utils.model_utils import load_model_from_checkpoint
        from src.utils.inference_utils import extract_embeddings, build_embedding_database
        from src.utils.config_utils import validate_config, merge_configs
        from src.utils.metrics_utils import compute_accuracy, compute_confusion_matrix
        
        logger.info("✅ Todos los módulos de utilidades importados correctamente")
        
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])
        
        accuracy = compute_accuracy(y_true, y_pred)
        logger.info(f"✅ Accuracy calculado: {accuracy:.2%}")
        
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=3)
        logger.info(f"✅ Matriz de confusión shape: {cm.shape}")
        
        print("✅ Módulos de utilidades funcionando correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error en módulos de utilidades: {e}", exc_info=True)
        raise


def generate_test_report(logger):
    """Genera reporte final de pruebas"""
    print("\n" + "="*70)
    print("REPORTE FINAL DE PRUEBAS")
    print("="*70 + "\n")
    
    logger.info("=" * 70)
    logger.info("RESUMEN DE PRUEBAS")
    logger.info("=" * 70)
    logger.info("✅ Test 1: Sistema de logging - PASSED")
    logger.info("✅ Test 2: Carga de configuración - PASSED")
    logger.info("✅ Test 3: Creación de modelo - PASSED")
    logger.info("✅ Test 4: Inferencia del modelo - PASSED")
    logger.info("✅ Test 5: Clasificador k-NN - PASSED")
    logger.info("✅ Test 6: Predicción en batch - PASSED")
    logger.info("✅ Test 7: Módulos de utilidades - PASSED")
    logger.info("=" * 70)
    logger.info("🎉 TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
    logger.info("=" * 70)
    
    print("\n✅ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
    print(f"✅ Logs guardados en: logs/test_workflow_*.log")
    print("\n" + "="*70 + "\n")


def main():
    """Función principal de prueba"""
    print("\n" + "="*70)
    print("PRUEBA DE FLUJO DE TRABAJO COMPLETO - MetricLearning")
    print("="*70 + "\n")
    
    try:
        logger = test_logging_system()
        
        config, device, train_transforms, val_transforms = test_config_loading(logger)
        
        model = test_model_creation(logger, config, device)
        
        embedding = test_model_inference(logger, model, val_transforms, device)
        
        classifier = test_knn_classifier(logger)
        
        test_batch_prediction(logger, classifier)
        
        test_utils_modules(logger)
        
        generate_test_report(logger)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR EN PRUEBAS: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
