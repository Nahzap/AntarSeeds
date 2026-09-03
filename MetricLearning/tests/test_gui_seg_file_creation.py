"""
Test de escritura REAL de archivos .seg desde GUI.
Valida que los archivos se crean físicamente en data/annotations/.
"""
import sys
from pathlib import Path
import tempfile
import shutil
import pytest
import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestSegFileCreationReal:
    """Tests que verifican creación FÍSICA de archivos .seg"""
    
    @pytest.fixture
    def temp_dataset_with_images(self):
        """Dataset temporal con imágenes reales (no mocks)"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Crear data/processed/train con imágenes
        train_dir = temp_dir / "data" / "processed" / "train"
        for class_name in ["ClassA", "ClassB"]:
            class_dir = train_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(3):
                # Crear imagen con contenido (círculo blanco sobre fondo negro)
                img = np.zeros((200, 200, 3), dtype=np.uint8)
                cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
                img_path = class_dir / f"img_{i:03d}.png"
                cv2.imwrite(str(img_path), img)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_annotator_creates_real_seg_files_centralized(self, temp_dataset_with_images):
        """Verifica que SegmentationAnnotator crea archivos .seg físicos en data/annotations/"""
        from src.grain_detection.annotator import SegmentationAnnotator
        
        temp_dir = temp_dataset_with_images
        train_dir = temp_dir / "data" / "processed" / "train"
        annot_root = temp_dir / "data" / "annotations"
        
        # Crear annotator con config mínimo
        config = {
            "grain_detection": {
                "model_type": "u2netp",
                "input_size": 320,
                "saliency_threshold": 0.3,
                "adaptive_k": 0.3,
                "min_area": 100,
                "max_area": 50000,
                "morph_kernel_size": 3,
                "min_circularity": 0.0,
                "crop_padding": 0.2,
                "crop_size": 256
            }
        }
        
        annotator = SegmentationAnnotator(config=config)
        
        # Ejecutar anotación con annotation_root
        stats = annotator.annotate_directory(
            root_dir=str(train_dir),
            overwrite=False,
            annotation_root=str(annot_root)
        )
        
        # Verificar stats
        assert stats["total_images"] == 6  # 3 imgs × 2 clases
        assert stats["annotated"] == 6
        assert stats["skipped"] == 0
        
        # Verificar que archivos .seg EXISTEN físicamente
        seg_files = list(annot_root.rglob("*.seg"))
        assert len(seg_files) == 6, \
            f"Deben existir 6 archivos .seg, encontrados: {len(seg_files)}"
        
        # Verificar estructura de directorios
        assert (annot_root / "ClassA").exists()
        assert (annot_root / "ClassB").exists()
        
        # Verificar contenido de un archivo
        seg_a = annot_root / "ClassA" / "img_000.seg"
        assert seg_a.exists(), f"{seg_a} debe existir"
        
        content = seg_a.read_text()
        assert "# Generado por:" in content
        assert "# Imagen:" in content
        assert "# Dimensiones:" in content
        
        # Verificar que NO se crearon en processed/
        seg_in_processed = list(train_dir.rglob("*.seg"))
        assert len(seg_in_processed) == 0, \
            "NO deben existir .seg en data/processed/"
    
    def test_annotator_overwrites_existing_seg(self, temp_dataset_with_images):
        """Verifica que overwrite=True recrea archivos .seg"""
        from src.grain_detection.annotator import SegmentationAnnotator
        
        temp_dir = temp_dataset_with_images
        train_dir = temp_dir / "data" / "processed" / "train"
        annot_root = temp_dir / "data" / "annotations"
        
        config = {
            "grain_detection": {
                "model_type": "u2netp",
                "input_size": 320,
                "saliency_threshold": 0.3,
                "adaptive_k": 0.3,
                "min_area": 100,
                "max_area": 50000,
                "morph_kernel_size": 3,
                "min_circularity": 0.0,
                "crop_padding": 0.2,
                "crop_size": 256
            }
        }
        
        annotator = SegmentationAnnotator(config=config)
        
        # Primera ejecución
        stats1 = annotator.annotate_directory(
            root_dir=str(train_dir),
            overwrite=False,
            annotation_root=str(annot_root)
        )
        assert stats1["annotated"] == 6
        
        # Segunda ejecución sin overwrite (debe skip todos)
        stats2 = annotator.annotate_directory(
            root_dir=str(train_dir),
            overwrite=False,
            annotation_root=str(annot_root)
        )
        assert stats2["annotated"] == 0
        assert stats2["skipped"] == 6
        
        # Tercera ejecución CON overwrite (debe anotar todos)
        stats3 = annotator.annotate_directory(
            root_dir=str(train_dir),
            overwrite=True,
            annotation_root=str(annot_root)
        )
        assert stats3["annotated"] == 6
        assert stats3["skipped"] == 0
    
    def test_seg_files_persist_after_processed_deletion(self, temp_dataset_with_images):
        """Verifica que .seg sobreviven a eliminación de data/processed/"""
        from src.grain_detection.annotator import SegmentationAnnotator
        
        temp_dir = temp_dataset_with_images
        train_dir = temp_dir / "data" / "processed" / "train"
        annot_root = temp_dir / "data" / "annotations"
        
        config = {
            "grain_detection": {
                "model_type": "u2netp",
                "input_size": 320,
                "saliency_threshold": 0.3,
                "adaptive_k": 0.3,
                "min_area": 100,
                "max_area": 50000,
                "morph_kernel_size": 3,
                "min_circularity": 0.0,
                "crop_padding": 0.2,
                "crop_size": 256
            }
        }
        
        annotator = SegmentationAnnotator(config=config)
        
        # Anotar
        stats = annotator.annotate_directory(
            root_dir=str(train_dir),
            overwrite=False,
            annotation_root=str(annot_root)
        )
        assert stats["annotated"] == 6
        
        # Contar .seg ANTES
        seg_before = list(annot_root.rglob("*.seg"))
        assert len(seg_before) == 6
        
        # Borrar TODO data/processed/
        shutil.rmtree(temp_dir / "data" / "processed")
        
        # Verificar que .seg SIGUEN existiendo
        seg_after = list(annot_root.rglob("*.seg"))
        assert len(seg_after) == 6, \
            f"Anotaciones deben sobrevivir a borrado de processed/. " \
            f"Antes: {len(seg_before)}, Después: {len(seg_after)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
