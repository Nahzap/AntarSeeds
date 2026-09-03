"""
Test de integración: U²-Net saliency + PollenGrainDetector + seg_format round-trip.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path

from src.grain_detection.salient_detector import SalientObjectDetector
from src.grain_detection.grain_detector import PollenGrainDetector
from src.grain_detection.seg_format import SegFileWriter, SegFileReader, get_seg_path
from src.grain_detection.annotator import SegmentationAnnotator


def find_test_image():
    """Busca una imagen válida en el dataset."""
    for split in ["val", "train"]:
        root = Path(f"data/processed/{split}")
        if not root.exists():
            continue
        for img in root.rglob("*.png"):
            if img.is_file() and os.path.getsize(str(img)) > 0:
                return str(img)
    return None


def test_saliency():
    """Test 1: SalientObjectDetector genera mapa de saliencia."""
    print("=" * 60)
    print("TEST 1: SalientObjectDetector")
    print("=" * 60)

    img_path = find_test_image()
    if img_path is None:
        print("SKIP: No se encontraron imágenes de test")
        return False

    print(f"  Imagen: {img_path}")
    image = cv2.imread(img_path)
    print(f"  Shape: {image.shape}")

    sod = SalientObjectDetector(model_type="u2netp")
    saliency = sod.get_saliency_map(image)

    print(f"  Saliency shape: {saliency.shape}")
    print(f"  Saliency range: [{saliency.min():.3f}, {saliency.max():.3f}]")
    print(f"  Saliency mean: {saliency.mean():.3f}")

    assert saliency.shape == image.shape[:2], "Saliency debe tener mismo H,W que imagen"
    assert saliency.min() >= 0.0, "Min debe ser >= 0"
    assert saliency.max() <= 1.0, "Max debe ser <= 1"
    print("  PASS")
    return True


def test_grain_detector():
    """Test 2: PollenGrainDetector detecta granos."""
    print("\n" + "=" * 60)
    print("TEST 2: PollenGrainDetector")
    print("=" * 60)

    img_path = find_test_image()
    if img_path is None:
        print("SKIP: No se encontraron imágenes de test")
        return False

    image = cv2.imread(img_path)
    print(f"  Imagen: {img_path} ({image.shape})")

    detector = PollenGrainDetector()
    saliency, grains = detector.detect(image)

    print(f"  Granos detectados: {len(grains)}")
    for g in grains:
        print(f"    #{g.index}: bbox={g.bbox}, area={g.area:.0f}, sal={g.saliency_prob:.3f}, "
              f"contour={g.contour.shape if g.contour is not None else None}")

    assert isinstance(grains, list), "Debe retornar lista"
    print("  PASS")
    return True


def test_seg_roundtrip_with_detector():
    """Test 3: Detector → .seg → Reader → verificar integridad."""
    print("\n" + "=" * 60)
    print("TEST 3: Detector → .seg round-trip")
    print("=" * 60)

    img_path = find_test_image()
    if img_path is None:
        print("SKIP")
        return False

    image = cv2.imread(img_path)
    detector = PollenGrainDetector()
    _, grains = detector.detect(image)

    # Escribir .seg temporal
    import tempfile
    seg_path = os.path.join(tempfile.gettempdir(), "test_roundtrip.seg")

    detections = []
    for g in grains:
        detections.append({
            "index": g.index,
            "class_index": 0,
            "bbox": g.bbox,
            "saliency": g.saliency_prob,
            "contour": g.contour,
        })

    h, w = image.shape[:2]
    SegFileWriter.write(seg_path, Path(img_path).name, (h, w), detections,
                        detector.get_parameters())

    # Leer y verificar
    result = SegFileReader.read(seg_path)
    read_grains = result["grains"]

    print(f"  Escritos: {len(detections)} granos")
    print(f"  Leídos: {len(read_grains)} granos")

    assert len(read_grains) == len(detections), "Cantidad debe coincidir"

    for orig, read in zip(detections, read_grains):
        assert orig["bbox"] == read["bbox"], f"BBox mismatch: {orig['bbox']} vs {read['bbox']}"
        assert abs(orig["saliency"] - read["saliency"]) < 0.01, "Saliency mismatch"

    os.remove(seg_path)
    print("  PASS")
    return True


def test_crop_extraction():
    """Test 4: Extracción de crops."""
    print("\n" + "=" * 60)
    print("TEST 4: Crop extraction")
    print("=" * 60)

    img_path = find_test_image()
    if img_path is None:
        print("SKIP")
        return False

    image = cv2.imread(img_path)
    detector = PollenGrainDetector()
    _, grains = detector.detect(image)

    if len(grains) == 0:
        print("  No grains detected — SKIP")
        return True

    crops = detector.extract_all_crops(image, grains)
    print(f"  {len(crops)} crops extraídos")

    for i, crop in enumerate(crops):
        print(f"    Crop #{i}: shape={crop.shape}")
        assert crop.shape == (256, 256, 3), f"Crop debe ser 256x256x3, got {crop.shape}"

    print("  PASS")
    return True


if __name__ == "__main__":
    results = []
    results.append(("Saliency", test_saliency()))
    results.append(("GrainDetector", test_grain_detector()))
    results.append(("SegRoundtrip", test_seg_roundtrip_with_detector()))
    results.append(("CropExtraction", test_crop_extraction()))

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL/SKIP"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    if all_pass:
        print("\nTODOS LOS TESTS PASARON")
    else:
        print("\nALGUNOS TESTS FALLARON O FUERON OMITIDOS")
