"""
Exportación del modelo a formato ONNX para inferencia optimizada.
Permite deployment con ONNX Runtime (2-3x speedup vs PyTorch).

Usage:
    python scripts/export_onnx.py --checkpoint models/best_model.pth --output models/model.onnx
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_utils import load_config
from src.utils.model_utils import load_model_from_checkpoint

logger = logging.getLogger(__name__)


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config_path: str = 'config.yaml',
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 17,
    validate: bool = True
):
    """
    Exporta modelo PyTorch a ONNX.
    
    Args:
        checkpoint_path: Ruta al checkpoint .pth
        output_path: Ruta de salida .onnx
        config_path: Ruta al config.yaml
        input_size: Tamaño de input (B, C, H, W)
        opset_version: Versión de ONNX opset
        validate: Si True, valida el modelo exportado
    """
    config = load_config(config_path)
    device = torch.device('cpu')
    
    logger.info(f"Loading model from {checkpoint_path}...")
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    model.eval()
    
    dummy_input = torch.randn(*input_size, device=device)
    
    logger.info(f"Exporting to ONNX: {output_path} (opset={opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"ONNX model saved: {output_path} ({file_size:.1f} MB)")
    
    if validate:
        _validate_onnx(output_path, model, dummy_input)
    
    return output_path


def _validate_onnx(onnx_path: str, pytorch_model, dummy_input):
    """Valida que el modelo ONNX produce outputs equivalentes al PyTorch."""
    import onnx
    import onnxruntime as ort
    
    logger.info("Validating ONNX model...")
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("  ONNX model structure: OK")
    
    session = ort.InferenceSession(onnx_path)
    
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()
    
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = session.run(None, ort_inputs)[0]
    
    max_diff = np.abs(pytorch_output - ort_output).max()
    logger.info(f"  Max output difference: {max_diff:.2e}")
    
    if max_diff < 1e-4:
        logger.info("  Validation: PASSED (outputs match)")
    else:
        logger.warning(f"  Validation: WARNING (max diff={max_diff:.2e} > 1e-4)")
    
    _benchmark(session, dummy_input.numpy(), pytorch_model, dummy_input)


def _benchmark(ort_session, numpy_input, pytorch_model, torch_input, n_runs=50):
    """Compara latencia PyTorch vs ONNX Runtime."""
    logger.info(f"Benchmarking ({n_runs} runs)...")
    
    pytorch_model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(n_runs):
            _ = pytorch_model(torch_input)
        pytorch_time = (time.perf_counter() - start) / n_runs * 1000
    
    ort_inputs = {ort_session.get_inputs()[0].name: numpy_input}
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = ort_session.run(None, ort_inputs)
    ort_time = (time.perf_counter() - start) / n_runs * 1000
    
    speedup = pytorch_time / ort_time if ort_time > 0 else 0
    logger.info(f"  PyTorch:      {pytorch_time:.2f} ms/inference")
    logger.info(f"  ONNX Runtime: {ort_time:.2f} ms/inference")
    logger.info(f"  Speedup:      {speedup:.1f}x")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--output', type=str, default='models/model.onnx', help='Output .onnx path')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--no-validate', action='store_true', help='Skip validation')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        opset_version=args.opset,
        validate=not args.no_validate
    )


if __name__ == '__main__':
    main()
