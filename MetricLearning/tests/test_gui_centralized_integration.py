"""
Tests de integración entre GUI y sistema de anotaciones centralizadas.

Valida que:
1. Tab de Contornos usa annotation_root="data/annotations"
2. Anotaciones se preservan al recrear dataset
3. Flujo completo funciona correctamente
"""
import sys
from pathlib import Path
import tempfile
import shutil
import pytest
import numpy as np
from PIL import Image

# Añadir proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestContourTabCentralizedAnnotations:
    """Tests para validar que ContourAnalysisTab usa sistema centralizado"""
    
    @pytest.fixture
    def temp_dataset(self):
        """Crea dataset temporal con estructura train/val/test"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Crear estructura
        for split in ["train", "val", "test"]:
            for class_name in ["ClassA", "ClassB"]:
                class_dir = temp_dir / "processed" / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # Crear 3 imágenes por clase/split
                for i in range(3):
                    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    img_path = class_dir / f"img_{split}_{class_name}_{i:03d}.png"
                    Image.fromarray(img).save(img_path)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.parametrize("n_splits", [1, 3])
    def test_annotation_worker_uses_centralized_root(self, temp_dataset, n_splits):
        """Un solo worker para cualquier alcance; todos los splits van a ANNOTATION_ROOT."""
        from lib.contour_analysis_tab import ANNOTATION_ROOT, AnnotationWorker
        from unittest.mock import MagicMock, patch

        # El annotator se importa DENTRO de run(), por eso se parchea el módulo.
        with patch('src.grain_detection.annotator.SegmentationAnnotator') as MockAnnotator:
            mock_annotator = MagicMock()
            mock_annotator.annotate_directory.return_value = {
                "total_images": 3,
                "annotated": 3,
                "skipped": 0,
                "errors": 0,
                "total_grains": 10,
                "elapsed_ms": 5.0,
                "grains_per_class": {"ClassA": 5, "ClassB": 5},
            }
            MockAnnotator.return_value = mock_annotator

            names = ["train", "val", "test"][:n_splits]
            splits = [str(temp_dataset / "processed" / s) for s in names]

            worker = AnnotationWorker(splits, overwrite=False, config={}, class_filter=None)
            worker.run()

            assert mock_annotator.annotate_directory.call_count == n_splits
            for call in mock_annotator.annotate_directory.call_args_list:
                assert call[1]["annotation_root"] == ANNOTATION_ROOT

    def test_worker_accumulates_stats_across_splits(self):
        """La acumulación es el núcleo del worker: se verifica sin Qt ni disco."""
        from lib.contour_analysis_tab import AnnotationWorker

        split_stats = {
            "total_images": 4, "annotated": 3, "skipped": 1, "errors": 0,
            "total_grains": 12, "elapsed_ms": 10.0,
            "grains_per_class": {"ClassA": 7, "ClassB": 5},
        }

        combined = AnnotationWorker.empty_stats()
        AnnotationWorker.accumulate(combined, "train", split_stats)
        AnnotationWorker.accumulate(combined, "val", split_stats)

        assert combined["total_images"] == 8
        assert combined["annotated"] == 6
        assert combined["total_grains"] == 24
        assert combined["grains_per_class"] == {"ClassA": 14, "ClassB": 10}
        assert combined["grains_per_image"] == pytest.approx(4.0)
        assert set(combined["per_split"]) == {"train", "val"}

    def test_single_split_summary_matches_totals(self):
        """Un split produce la misma forma de stats que tres."""
        from lib.contour_analysis_tab import AnnotationWorker

        combined = AnnotationWorker.empty_stats()
        AnnotationWorker.accumulate(combined, "train", {
            "total_images": 2, "annotated": 2, "skipped": 0, "errors": 0,
            "total_grains": 6, "elapsed_ms": 3.0, "grains_per_class": {"ClassA": 6},
        })

        assert combined["annotated"] == 2
        assert combined["grains_per_image"] == pytest.approx(3.0)
        assert list(combined["per_split"]) == ["train"]


class TestDatasetManagerPreservesAnnotations:
    """Tests para validar que DatasetManager preserva anotaciones centralizadas"""
    
    @pytest.fixture
    def temp_dataset_with_annotations(self):
        """Dataset temporal con anotaciones en data/annotations/"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Crear processed/ con imágenes
        for split in ["train", "val"]:
            for class_name in ["ClassA", "ClassB"]:
                class_dir = temp_dir / "processed" / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                for i in range(2):
                    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                    img_path = class_dir / f"img_{i:03d}.png"
                    Image.fromarray(img).save(img_path)
        
        # Crear data/annotations/ con .seg files
        for class_name in ["ClassA", "ClassB"]:
            annot_dir = temp_dir / "data" / "annotations" / class_name
            annot_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(2):
                seg_path = annot_dir / f"img_{i:03d}.seg"
                seg_path.write_text(
                    "# Config: test\n"
                    "0 | 0 | 10,10,20,20 | 0.95 | 10,10,30,10,30,30,10,30\n"
                )
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_prepare_processed_preserves_annotations(self, temp_dataset_with_annotations):
        """Verifica que prepare_processed_directory NO borra anotaciones centralizadas"""
        from lib.dataset_manager import DatasetMapManager
        
        temp_dir = temp_dataset_with_annotations
        annot_root = temp_dir / "data" / "annotations"
        
        # Verificar que existen anotaciones ANTES
        seg_files_before = list(annot_root.rglob("*.seg"))
        assert len(seg_files_before) == 4, "Deben existir 4 archivos .seg antes"
        
        # Crear manager con source_dir dummy
        manager = DatasetMapManager(source_dir=str(temp_dir / "data" / "raw"))
        
        # Simular splits directamente (API interna)
        manager.splits = {
            "train": {
                "ClassA": [temp_dir / "processed" / "train" / "ClassA" / "img_000.png"],
                "ClassB": [temp_dir / "processed" / "train" / "ClassB" / "img_000.png"]
            },
            "val": {
                "ClassA": [temp_dir / "processed" / "val" / "ClassA" / "img_001.png"],
                "ClassB": [temp_dir / "processed" / "val" / "ClassB" / "img_001.png"]
            }
        }
        
        # Ejecutar prepare_processed_directory (esto hace rmtree de processed/)
        processed_dir = str(temp_dir / "processed_new")
        counts = manager.prepare_processed_directory(
            processed_dir=processed_dir,
            strategy="copy"
        )
        
        assert counts["train"] == 2
        assert counts["val"] == 2
        
        # Verificar que anotaciones SIGUEN existiendo
        seg_files_after = list(annot_root.rglob("*.seg"))
        assert len(seg_files_after) == 4, \
            f"Anotaciones deben preservarse. Antes: {len(seg_files_before)}, Después: {len(seg_files_after)}"
        
        # Verificar contenido idéntico
        for seg_before in seg_files_before:
            seg_after = annot_root / seg_before.relative_to(annot_root)
            assert seg_after.exists(), f"{seg_after} debe existir"
            assert seg_before.read_text() == seg_after.read_text(), \
                "Contenido de .seg debe ser idéntico"
    
    def test_rmtree_only_affects_processed_not_annotations(self, temp_dataset_with_annotations):
        """Verifica que rmtree de processed/ NO afecta data/annotations/"""
        import shutil
        
        temp_dir = temp_dataset_with_annotations
        processed_dir = temp_dir / "processed"
        annot_root = temp_dir / "data" / "annotations"
        
        # Contar .seg antes
        seg_files_before = list(annot_root.rglob("*.seg"))
        assert len(seg_files_before) == 4
        
        # Borrar processed/ (simula lo que hace DatasetManager)
        if processed_dir.exists():
            shutil.rmtree(processed_dir)
        
        # Verificar que processed/ fue borrado
        assert not processed_dir.exists()
        
        # Verificar que anotaciones NO fueron tocadas
        seg_files_after = list(annot_root.rglob("*.seg"))
        assert len(seg_files_after) == 4, \
            "rmtree de processed/ NO debe afectar data/annotations/"


class TestEndToEndGUIWorkflow:
    """Tests de flujo completo: anotar → recrear dataset → verificar"""
    
    @pytest.fixture
    def temp_complete_dataset(self):
        """Dataset completo con estructura raw/ y processed/"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Crear data/raw/ (source images)
        for class_name in ["ClassA", "ClassB"]:
            class_dir = temp_dir / "data" / "raw" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(5):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img_path = class_dir / f"img_{i:03d}.png"
                Image.fromarray(img).save(img_path)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_workflow_annotations_preserved(self, temp_complete_dataset):
        """Workflow completo: preparar → anotar → re-preparar → verificar"""
        from lib.dataset_manager import DatasetMapManager
        
        temp_dir = temp_complete_dataset
        processed_dir = temp_dir / "processed"
        annot_root = temp_dir / "data" / "annotations"
        
        # Paso 1: Preparar splits iniciales (simular manager.splits)
        raw_dir = temp_dir / "data" / "raw"
        manager = DatasetMapManager(source_dir=str(raw_dir))
        raw_images = list(raw_dir.rglob("*.png"))
        
        # Dividir manualmente en splits
        manager.splits = {
            "train": {"ClassA": raw_images[0:3], "ClassB": raw_images[5:8]},
            "val": {"ClassA": [raw_images[3]], "ClassB": [raw_images[8]]},
            "test": {"ClassA": [raw_images[4]], "ClassB": [raw_images[9]]}
        }
        
        # Paso 2: Preparar processed/
        counts = manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        assert sum(counts.values()) == 10  # 5 imágenes × 2 clases
        
        # Paso 3: Simular anotación (crear .seg en data/annotations/)
        for class_name in ["ClassA", "ClassB"]:
            class_annot_dir = annot_root / class_name
            class_annot_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(5):
                seg_path = class_annot_dir / f"img_{i:03d}.seg"
                seg_path.write_text(
                    f"# Generated for {class_name}\n"
                    f"0 | 0 | 10,10,20,20 | 0.95 | 10,10,30,10,30,30,10,30\n"
                )
        
        seg_files_after_annotation = list(annot_root.rglob("*.seg"))
        assert len(seg_files_after_annotation) == 10
        
        # Paso 4: Regenerar splits (simula "Preparar Datos" desde GUI)
        # Cambiar distribución de splits
        manager.splits = {
            "train": {"ClassA": raw_images[0:4], "ClassB": raw_images[5:9]},
            "val": {"ClassA": [raw_images[4]], "ClassB": [raw_images[9]]},
        }
        
        counts2 = manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        assert sum(counts2.values()) == 10
        
        # Paso 5: Verificar que anotaciones SIGUEN existiendo
        seg_files_final = list(annot_root.rglob("*.seg"))
        assert len(seg_files_final) == 10, \
            f"Anotaciones deben preservarse tras regenerar splits. " \
            f"Esperado: 10, Got: {len(seg_files_final)}"
        
        # Verificar que los archivos son los mismos
        for seg_after in seg_files_after_annotation:
            seg_final = annot_root / seg_after.relative_to(annot_root)
            assert seg_final.exists(), f"{seg_final} debe existir"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
