"""
Genera .seg para val y test datasets usando PollenGrainDetector.
Ejecutar: python scripts/generate_seg_batch.py
"""
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    from src.grain_detection.annotator import SegmentationAnnotator

    config = {
        "model_type": "u2netp",
        "saliency_threshold": 0.54,
        "adaptive_k": 0.33,
        "min_area": 10002,
        "max_area": 500000,
        "morph_kernel_size": 3,
        "min_circularity": 0.48,
    }

    annotator = SegmentationAnnotator(config=config)

    for split in ["train", "val", "test"]:
        root = f"data/processed/{split}"
        print(f"\n{'='*60}")
        print(f"Generando .seg para: {root}")
        print(f"{'='*60}")

        t0 = time.perf_counter()

        def progress(current, total, msg):
            if current % 50 == 0 or current == total:
                elapsed = time.perf_counter() - t0
                rate = current / max(elapsed, 0.01)
                print(f"  [{current}/{total}] {msg}  ({rate:.1f} img/s)")

        stats = annotator.annotate_directory(
            root_dir=root,
            annotation_root="data/annotations",  # Sistema centralizado
            overwrite=False,
            progress_callback=progress
        )

        elapsed_s = (time.perf_counter() - t0)
        print(f"\nResultado {split}:")
        print(f"  Anotados: {stats['annotated']}")
        print(f"  Omitidos: {stats['skipped']}")
        print(f"  Errores:  {stats['errors']}")
        print(f"  Granos:   {stats['total_grains']}")
        print(f"  Tiempo:   {elapsed_s:.1f}s")
        print(f"  Por clase: {stats.get('grains_per_class', {})}")

        # Guardar registro
        from src.grain_detection.contour_registry import ContourRegistry
        reg_path = ContourRegistry.save(root, detection_params=config, annotation_stats=stats)
        print(f"  Registro: {reg_path}")

    print("\n✅ Generación y registros completos.")


if __name__ == "__main__":
    main()
