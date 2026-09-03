"""
Contratos del tab de contornos que no necesitan GUI viva.

Cubren dos fallos reales: las estadísticas buscaban el .seg junto a la imagen
(siempre 0 con anotaciones centralizadas) y el caché del detector se indexaba
por la config completa, recargando U²-Net en cada movimiento de slider.
"""

from pathlib import Path

import pytest

from lib.contour_analysis_tab import (
    ANNOTATION_ROOT,
    IMAGE_SUFFIXES,
    ManualContourWorker,
    class_images,
    seg_path_for,
)


def test_seg_path_is_centralized_by_class():
    seg = seg_path_for(Path("data/processed/train/C_Quitensis/plate_0001_f1.png"))
    assert seg == Path(ANNOTATION_ROOT) / "C_Quitensis" / "plate_0001_f1.seg"


def test_seg_path_never_sits_next_to_image():
    img = Path("data/processed/train/D_Antartica/img.png")
    assert seg_path_for(img) != img.with_suffix(".seg")


def test_class_images_filters_and_sorts(tmp_path):
    cls_dir = tmp_path / "P_Annua"
    cls_dir.mkdir()
    for name in ("b.png", "a.PNG", "c.tif", "notes.txt", "old.seg"):
        (cls_dir / name).write_bytes(b"")

    found = [p.name for p in class_images(cls_dir)]
    assert found == ["a.PNG", "b.png", "c.tif"]
    assert all(Path(n).suffix.lower() in IMAGE_SUFFIXES for n in found)


def test_class_images_missing_dir_is_empty(tmp_path):
    assert class_images(tmp_path / "no_existe") == []


# ---------------------------------------------------------------------------
# Caché del detector: los filtros no pueden invalidarlo
# ---------------------------------------------------------------------------

def test_detector_cache_ignores_filter_changes():
    base = {"model_type": "u2netp", "input_size": 320, "min_area": 800}
    moved = {**base, "min_area": 40000, "saliency_threshold": 0.61, "crop_radius": 900}
    assert ManualContourWorker.detector_cache_key(base) == \
        ManualContourWorker.detector_cache_key(moved)


def test_detector_cache_separates_models():
    a = ManualContourWorker.detector_cache_key({"model_type": "u2netp"})
    b = ManualContourWorker.detector_cache_key({"model_type": "u2net"})
    c = ManualContourWorker.detector_cache_key({"model_type": "u2netp", "input_size": 512})
    assert a != b and a != c


def test_detector_cache_key_defaults():
    assert ManualContourWorker.detector_cache_key(None) == ("u2netp", 320, "auto")


# ---------------------------------------------------------------------------
# Resumen de anotación: una sola función para uno o varios splits
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tab_class():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5.QtWidgets")
    from lib.contour_analysis_tab import ContourAnalysisTab

    return ContourAnalysisTab


def test_summary_single_split_reports_totals(tab_class):
    text = tab_class._format_annotation_summary(
        {"annotated": 3, "skipped": 1, "total_grains": 9, "elapsed_ms": 120.0,
         "per_split": {"train": {}}}
    )
    assert "3 anotados" in text and "9 granos" in text


def test_summary_multi_split_reports_each_split(tab_class):
    text = tab_class._format_annotation_summary({
        "per_split": {
            "train": {"total_grains": 10, "annotated": 5, "skipped": 0},
            "val": {"total_grains": 4, "annotated": 2, "skipped": 1},
        }
    })
    assert "train:" in text and "val:" in text
