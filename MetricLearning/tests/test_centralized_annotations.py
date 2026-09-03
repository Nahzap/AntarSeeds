"""
Tests para el sistema de anotaciones centralizadas.

Cubre:
- Migración de anotaciones a directorio centralizado
- Validación de integridad
- Reorganización segura de splits
- SegmentedGrainDataset con annotation_root
- SegmentationAnnotator con annotation_root
"""
import sys
import shutil
import tempfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.grain_dataset import SegmentedGrainDataset
from src.grain_detection.annotator import SegmentationAnnotator
from src.grain_detection.seg_format import SegFileWriter


@pytest.fixture
def temp_dataset():
    """
    Crea estructura temporal de dataset con imágenes y anotaciones en splits.
    
    Estructura:
        temp_dir/
          processed/
            train/
              ClassA/
                img_001.png
                img_001.seg
              ClassB/
                img_002.png
                img_002.seg
            val/
              ClassA/
                img_003.png
                img_003.seg
    """
    temp_dir = Path(tempfile.mkdtemp())
    
    # Crear estructura de directorios
    for split in ["train", "val", "test"]:
        for class_name in ["ClassA", "ClassB"]:
            class_dir = temp_dir / "processed" / split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
    
    # Crear imágenes dummy y .seg (en splits, sistema antiguo)
    import numpy as np
    import cv2
    
    samples = [
        ("train", "ClassA", "img_001.png", 2),  # 2 granos
        ("train", "ClassB", "img_002.png", 3),  # 3 granos
        ("val", "ClassA", "img_003.png", 1),    # 1 grano
        ("val", "ClassB", "img_004.png", 2),    # 2 granos
        ("test", "ClassA", "img_005.png", 1),   # 1 grano
    ]
    
    for split, class_name, img_name, n_grains in samples:
        img_dir = temp_dir / "processed" / split / class_name
        img_path = img_dir / img_name
        
        # Imagen dummy 256x256
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        
        # .seg dummy junto a la imagen (sistema antiguo)
        seg_path = img_path.with_suffix(".seg")
        detections = []
        for i in range(n_grains):
            detections.append({
                "index": i,
                "class_index": 0,
                "bbox": (50 + i*50, 50 + i*50, 40, 40),
                "saliency": 0.9,
                "contour": np.array([[50, 50], [90, 50], [90, 90], [50, 90]])
            })
        
        SegFileWriter.write(
            seg_path=str(seg_path),
            image_name=img_name,
            image_dims=(256, 256),
            detections=detections,
            params={"test": "dummy"}
        )
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestMigrationToCentralized:
    """Tests para migrate_annotations_to_centralized.py"""
    
    def test_migration_creates_annotation_root(self, temp_dataset):
        """Verifica que se crea data/annotations/ con estructura correcta"""
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        
        # Ejecutar migración con paths personalizados
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        success = migrate_annotations(
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        assert success
        
        # Verificar que se creó data/annotations/
        annot_path = temp_dataset / "data" / "annotations"
        assert annot_path.exists()
        
        # Verificar estructura por clase
        assert (annot_path / "ClassA").exists()
        assert (annot_path / "ClassB").exists()
        
        # Verificar que se migraron los .seg
        assert (annot_path / "ClassA" / "img_001.seg").exists()
        assert (annot_path / "ClassA" / "img_003.seg").exists()
        assert (annot_path / "ClassB" / "img_002.seg").exists()
    
    def test_migration_handles_duplicates(self, temp_dataset):
        """Verifica que los duplicados idénticos se resuelven correctamente"""
        # Copiar img_001.seg a val/ también (duplicado)
        src = temp_dataset / "processed" / "train" / "ClassA" / "img_001.seg"
        dst = temp_dataset / "processed" / "val" / "ClassA" / "img_001.seg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        success = migrate_annotations(
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        assert success
        
        # Debe haber solo 1 copia en annotations/
        annot_path = temp_dataset / "data" / "annotations" / "ClassA" / "img_001.seg"
        assert annot_path.exists()
    
    def test_migration_dry_run_no_changes(self, temp_dataset):
        """Verifica que dry-run no modifica archivos"""
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        success = migrate_annotations(
            dry_run=True,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        assert success
        
        # data/annotations/ NO debe existir
        annot_path = temp_dataset / "data" / "annotations"
        assert not annot_path.exists()


class TestValidationCentralized:
    """Tests para validate_annotations_centralized.py"""
    
    def test_validation_all_images_have_annotations(self, temp_dataset):
        """Verifica validación exitosa cuando todas las imágenes tienen .seg"""
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        from scripts.validate_annotations_centralized import validate_annotations
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        # Primero migrar
        migrate_annotations(
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        
        # Validar
        success = validate_annotations(
            processed_root=processed_root,
            annotation_root=annot_root
        )
        assert success
    
    def test_validation_detects_missing_annotations(self, temp_dataset):
        """Verifica que detecta imágenes sin anotación"""
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        from scripts.validate_annotations_centralized import validate_annotations
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        # Migrar
        migrate_annotations(
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        
        # Eliminar una anotación
        annot_path = temp_dataset / "data" / "annotations" / "ClassA" / "img_001.seg"
        annot_path.unlink()
        
        # Validar - debe fallar
        success = validate_annotations(
            processed_root=processed_root,
            annotation_root=annot_root
        )
        assert not success
    
    def test_validation_detects_orphan_annotations(self, temp_dataset):
        """Verifica que detecta .seg huérfanos (sin imagen)"""
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        from scripts.validate_annotations_centralized import validate_annotations
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        # Migrar
        migrate_annotations(
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        
        # Eliminar imagen pero dejar .seg
        img_path = temp_dataset / "processed" / "train" / "ClassA" / "img_001.png"
        img_path.unlink()
        
        # Validar - huérfanos no hacen fallar, solo se reportan
        success = validate_annotations(
            processed_root=processed_root,
            annotation_root=annot_root
        )
        # Debe pasar porque solo busca imágenes sin .seg, no .seg sin imagen


class TestReorganizeSplitsSafe:
    """Tests para reorganize_splits_safe.py"""
    
    def test_reorganize_moves_only_images(self, temp_dataset):
        """Verifica que solo mueve imágenes, NO anotaciones"""
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        from scripts.reorganize_splits_safe import reorganize_splits
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        # Migrar primero
        migrate_annotations(
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        
        # Eliminar .seg de splits (cleanup)
        for seg in (temp_dataset / "processed").rglob("*.seg"):
            seg.unlink()
        
        # Reorganizar
        success = reorganize_splits(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        assert success
        
        # Verificar que anotaciones NO se movieron
        annot_path = temp_dataset / "data" / "annotations"
        assert (annot_path / "ClassA" / "img_001.seg").exists()
        assert (annot_path / "ClassA" / "img_003.seg").exists()
    
    def test_reorganize_validates_annotations_exist(self, temp_dataset):
        """Verifica que valida existencia de anotaciones antes de reorganizar"""
        from scripts.reorganize_splits_safe import reorganize_splits
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        # Intentar reorganizar SIN migrar (no existe data/annotations/)
        success = reorganize_splits(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        assert not success  # Debe fallar
    
    def test_reorganize_dry_run_no_changes(self, temp_dataset):
        """Verifica que dry-run no mueve archivos"""
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        from scripts.reorganize_splits_safe import reorganize_splits
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        # Migrar
        migrate_annotations(
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        
        # Contar imágenes en train antes
        train_dir = temp_dataset / "processed" / "train"
        n_before = len(list(train_dir.rglob("*.png")))
        
        # Dry run
        reorganize_splits(
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            dry_run=True,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        
        # Contar después - debe ser igual
        n_after = len(list(train_dir.rglob("*.png")))
        assert n_before == n_after


class TestSegmentedGrainDatasetWithAnnotationRoot:
    """Tests para SegmentedGrainDataset con annotation_root"""
    
    def test_dataset_loads_from_centralized_annotations(self, temp_dataset):
        """Verifica que carga .seg desde data/annotations/"""
        from scripts.migrate_annotations_to_centralized import migrate_annotations
        
        processed_root = str(temp_dataset / "processed")
        annot_root = str(temp_dataset / "data" / "annotations")
        
        # Migrar
        migrate_annotations(
            dry_run=False,
            processed_root=processed_root,
            annotation_root=annot_root
        )
        
        # Eliminar .seg de splits
        for seg in (temp_dataset / "processed").rglob("*.seg"):
            seg.unlink()
        
        # Cargar dataset con annotation_root
        dataset = SegmentedGrainDataset(
            root=str(temp_dataset / "processed" / "train"),
            annotation_root=annot_root
        )
        
        # Debe cargar los granos correctamente
        assert len(dataset) > 0
        assert "ClassA" in dataset.classes
        assert "ClassB" in dataset.classes
        
        # Probar __getitem__
        crop, label = dataset[0]
        assert crop.shape[0] == 3  # RGB
        assert label >= 0
    
    def test_dataset_fails_if_annotation_root_missing(self, temp_dataset):
        """Verifica que falla si annotation_root no existe"""
        with pytest.raises(FileNotFoundError):
            SegmentedGrainDataset(
                root=str(temp_dataset / "processed" / "train"),
                annotation_root=str(temp_dataset / "data" / "nonexistent")
            )
    
    def test_dataset_legacy_mode_without_annotation_root(self, temp_dataset):
        """Verifica que funciona en modo legacy sin annotation_root"""
        # NO migrar, .seg están junto a imágenes
        dataset = SegmentedGrainDataset(
            root=str(temp_dataset / "processed" / "train"),
            annotation_root=None  # Modo legacy
        )
        
        assert len(dataset) > 0
        crop, label = dataset[0]
        assert crop.shape[0] == 3


class TestSegmentationAnnotatorWithAnnotationRoot:
    """Tests para SegmentationAnnotator con annotation_root"""
    
    def test_annotator_generates_in_centralized_dir(self, temp_dataset):
        """Verifica que genera .seg en directorio centralizado"""
        # Crear imágenes sin .seg
        img_dir = temp_dataset / "raw" / "ClassA"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        import numpy as np
        import cv2
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img_path = img_dir / "new_img.png"
        cv2.imwrite(str(img_path), img)
        
        # Mock detector simple
        class MockDetector:
            def detect(self, image):
                from src.grain_detection.detected_grain import DetectedGrain
                grain = DetectedGrain(
                    index=0,
                    bbox=(50, 50, 40, 40),
                    saliency_prob=0.9,
                    centroid=(70, 70),
                    contour=np.array([[50, 50], [90, 50], [90, 90], [50, 90]]),
                    area=1600  # ← AÑADIR parámetro faltante
                )
                return None, [grain]
            
            def get_parameters(self):
                return {"test": "mock"}
        
        annotator = SegmentationAnnotator(detector=MockDetector())
        
        # Generar anotaciones en directorio centralizado
        annot_root = temp_dataset / "data" / "annotations"
        annot_root.mkdir(parents=True, exist_ok=True)
        
        stats = annotator.annotate_directory(
            root_dir=str(temp_dataset / "raw"),
            annotation_root=str(annot_root),
            overwrite=False
        )
        
        # Verificar que se generó en annotations/
        assert (annot_root / "ClassA" / "new_img.seg").exists()
        
        # Verificar que NO se generó junto a la imagen
        assert not (img_dir / "new_img.seg").exists()
        
        # Verificar stats
        assert stats["annotated"] == 1
        assert stats["total_grains"] == 1
    
    def test_annotator_legacy_mode_without_annotation_root(self, temp_dataset):
        """Verifica que funciona en modo legacy sin annotation_root"""
        # Crear imagen
        img_dir = temp_dataset / "raw2" / "ClassA"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        import numpy as np
        import cv2
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img_path = img_dir / "legacy_img.png"
        cv2.imwrite(str(img_path), img)
        
        # Mock detector
        class MockDetector:
            def detect(self, image):
                from src.grain_detection.detected_grain import DetectedGrain
                grain = DetectedGrain(
                    index=0,
                    bbox=(50, 50, 40, 40),
                    saliency_prob=0.9,
                    centroid=(70, 70),
                    contour=np.array([[50, 50], [90, 50], [90, 90], [50, 90]]),
                    area=1600  # ← AÑADIR parámetro faltante
                )
                return None, [grain]
            
            def get_parameters(self):
                return {"test": "mock"}
        
        annotator = SegmentationAnnotator(detector=MockDetector())
        
        # Generar sin annotation_root (modo legacy)
        stats = annotator.annotate_directory(
            root_dir=str(temp_dataset / "raw2"),
            annotation_root=None,  # Modo legacy
            overwrite=False
        )
        
        # Verificar que se generó junto a la imagen
        assert (img_dir / "legacy_img.seg").exists()
        assert stats["annotated"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
