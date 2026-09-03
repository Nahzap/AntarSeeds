"""
Test de carga de .seg existentes desde GUI.
Valida que "Cargar Existentes" encuentra .seg en data/annotations/.
"""
import sys
from pathlib import Path
import tempfile
import shutil
import pytest
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestLoadExistingSegFromCentralized:
    """Tests para validar carga de .seg desde data/annotations/"""
    
    @pytest.fixture
    def temp_dataset_with_centralized_segs(self):
        """Dataset con .seg en data/annotations/"""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Crear data/processed/train con imágenes
        for split in ["train", "val"]:
            for class_name in ["ClassA", "ClassB"]:
                class_dir = temp_dir / "data" / "processed" / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                for i in range(3):
                    img = np.zeros((200, 200, 3), dtype=np.uint8)
                    cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
                    img_path = class_dir / f"img_{i:03d}.png"
                    cv2.imwrite(str(img_path), img)
        
        # Crear data/annotations/ con .seg
        for class_name in ["ClassA", "ClassB"]:
            annot_dir = temp_dir / "data" / "annotations" / class_name
            annot_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(3):
                seg_path = annot_dir / f"img_{i:03d}.seg"
                seg_path.write_text(
                    f"# Generated for {class_name}\n"
                    f"# Imagen: img_{i:03d}.png\n"
                    f"# Dimensiones: 200x200\n"
                    f"0 | 0 | 80,80,40,40 | 0.95 | 80,80,120,80,120,120,80,120\n"
                )
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_registry_finds_centralized_segs(self, temp_dataset_with_centralized_segs):
        """Verifica que ContourRegistry encuentra .seg en data/annotations/"""
        from src.grain_detection.contour_registry import ContourRegistry
        
        temp_dir = temp_dataset_with_centralized_segs
        train_dir = temp_dir / "data" / "processed" / "train"
        annot_root = temp_dir / "data" / "annotations"
        
        # Guardar registry con annotation_root
        reg_path = ContourRegistry.save(
            str(train_dir),
            detection_params={"test": 1},
            annotation_root=str(annot_root)
        )
        
        assert reg_path.exists()
        
        # Cargar registry
        registry = ContourRegistry.load(str(train_dir))
        assert registry is not None
        
        # Verificar contenido
        summary = registry["summary"]
        assert summary["total_images"] == 6  # 3×2 clases
        assert summary["images_with_seg"] == 6  # Todos tienen .seg
        assert summary["total_grains"] == 6  # 1 grano por imagen
        
        # Verificar que annotation_root está guardado
        assert registry.get("annotation_root") == str(annot_root)
    
    def test_validator_finds_centralized_segs(self, temp_dataset_with_centralized_segs):
        """Verifica que validate_annotations encuentra .seg en data/annotations/"""
        from src.grain_detection.annotator import SegmentationAnnotator
        
        temp_dir = temp_dataset_with_centralized_segs
        train_dir = temp_dir / "data" / "processed" / "train"
        annot_root = temp_dir / "data" / "annotations"
        
        annotator = SegmentationAnnotator()
        result = annotator.validate_annotations(
            str(train_dir),
            annotation_root=str(annot_root)
        )
        
        # Verificar resultados
        assert result["total_images"] == 6
        assert result["valid"] == 6
        assert len(result["missing_seg"]) == 0
        assert len(result["corrupt_seg"]) == 0
        assert len(result["orphan_seg"]) == 0
    
    def test_refresh_explorer_finds_centralized_segs(self, temp_dataset_with_centralized_segs):
        """Simula _refresh_explorer buscando .seg en data/annotations/"""
        temp_dir = temp_dataset_with_centralized_segs
        train_dir = temp_dir / "data" / "processed" / "train"
        annot_root = temp_dir / "data" / "annotations"
        
        # Simular lógica de _refresh_explorer
        classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        assert classes == ["ClassA", "ClassB"]
        
        seg_counts = {}
        img_counts = {}
        
        for cls in classes:
            cls_dir = train_dir / cls
            imgs = [f for f in cls_dir.iterdir()
                   if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}]
            
            # Buscar .seg en data/annotations/clase/
            cls_annot_dir = annot_root / cls
            segs = []
            if cls_annot_dir.exists():
                for img_file in imgs:
                    seg_file = cls_annot_dir / f"{img_file.stem}.seg"
                    if seg_file.exists():
                        segs.append(img_file)
            
            img_counts[cls] = len(imgs)
            seg_counts[cls] = len(segs)
        
        # Verificar conteos
        assert img_counts["ClassA"] == 3
        assert img_counts["ClassB"] == 3
        assert seg_counts["ClassA"] == 3
        assert seg_counts["ClassB"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
