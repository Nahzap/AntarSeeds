"""
Tests unitarios para DatasetMapManager — prepare_processed_directory.
Cubre: creación de estructura ImageFolder, symlink/copy fallback, progreso.
"""

import pytest
import json
import shutil
from pathlib import Path
from collections import defaultdict
from unittest.mock import MagicMock

from lib.dataset_manager import DatasetMapManager


@pytest.fixture
def tmp_source(tmp_path):
    """Crea estructura de directorios fuente con imágenes fake."""
    classes = {"ClassA": 10, "ClassB": 8, "ClassC": 5}
    folder_class_map = {}
    
    for cls_name, count in classes.items():
        cls_dir = tmp_path / "source" / cls_name
        cls_dir.mkdir(parents=True)
        for i in range(count):
            img = cls_dir / f"{cls_name}_{i:04d}.png"
            img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)  # fake PNG header
        folder_class_map[str(cls_dir)] = cls_name
    
    return folder_class_map, tmp_path


@pytest.fixture
def manager_with_splits(tmp_source):
    """Manager con splits ya generados."""
    folder_class_map, tmp_path = tmp_source
    output_dir = tmp_path / "maps"
    
    manager = DatasetMapManager(folder_class_map, str(output_dir))
    manager.scan_source_directory()
    manager.set_split_ratios(0.7, 0.15, 0.15)
    manager.generate_splits(seed=42)
    
    return manager, tmp_path


class TestPrepareProcessedDirectory:
    """Tests para prepare_processed_directory."""
    
    def test_creates_split_directories(self, manager_with_splits):
        """Verifica que se crean los directorios train/val/test."""
        manager, tmp_path = manager_with_splits
        processed_dir = tmp_path / "processed"
        
        manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        
        assert (processed_dir / "train").exists()
        assert (processed_dir / "val").exists()
        assert (processed_dir / "test").exists()
    
    def test_creates_class_subdirectories(self, manager_with_splits):
        """Verifica que se crean subdirectorios por clase."""
        manager, tmp_path = manager_with_splits
        processed_dir = tmp_path / "processed"
        
        manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        
        for split_name in ["train", "val", "test"]:
            split_dir = processed_dir / split_name
            class_dirs = [d.name for d in split_dir.iterdir() if d.is_dir()]
            assert len(class_dirs) > 0
            for cls in class_dirs:
                assert cls in ["ClassA", "ClassB", "ClassC"]
    
    def test_copies_correct_number_of_files(self, manager_with_splits):
        """Verifica que el número total de archivos copiados es correcto."""
        manager, tmp_path = manager_with_splits
        processed_dir = tmp_path / "processed"
        
        counts = manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        
        total_expected = sum(
            len(imgs) for split_data in manager.splits.values()
            for imgs in split_data.values()
        )
        total_actual = sum(counts.values())
        assert total_actual == total_expected
    
    def test_returns_counts_dict(self, manager_with_splits):
        """Verifica que retorna diccionario con conteos por split."""
        manager, tmp_path = manager_with_splits
        processed_dir = tmp_path / "processed"
        
        counts = manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        
        assert "train" in counts
        assert "val" in counts
        assert "test" in counts
        assert all(isinstance(v, int) for v in counts.values())
        assert counts["train"] > counts["val"]
        assert counts["train"] > counts["test"]
    
    def test_copy_strategy_creates_real_files(self, manager_with_splits):
        """Verifica que strategy='copy' crea archivos reales (no symlinks)."""
        manager, tmp_path = manager_with_splits
        processed_dir = tmp_path / "processed"
        
        manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        
        # Check a file in train
        train_dir = processed_dir / "train"
        some_class_dir = next(train_dir.iterdir())
        some_file = next(some_class_dir.iterdir())
        
        assert some_file.is_file()
        assert not some_file.is_symlink()
        assert some_file.stat().st_size > 0
    
    def test_cleans_stale_data(self, manager_with_splits):
        """Verifica que limpia datos previos antes de preparar."""
        manager, tmp_path = manager_with_splits
        processed_dir = tmp_path / "processed"
        
        # Create stale data
        stale_dir = processed_dir / "train" / "StaleClass"
        stale_dir.mkdir(parents=True)
        (stale_dir / "stale.png").write_bytes(b"stale")
        
        manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        
        # StaleClass should be gone
        assert not (processed_dir / "train" / "StaleClass").exists()
    
    def test_progress_callback_called(self, manager_with_splits):
        """Verifica que el callback de progreso se llama."""
        manager, tmp_path = manager_with_splits
        processed_dir = tmp_path / "processed"
        
        callback = MagicMock()
        
        manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy",
            progress_callback=callback
        )
        
        # Callback should be called at least once (for "Completado")
        assert callback.call_count >= 1
        # Last call should be the "Completado" call
        last_call_args = callback.call_args_list[-1][0]
        assert last_call_args[0] == last_call_args[1]  # current == total
    
    def test_raises_on_empty_splits(self, tmp_source):
        """Verifica que lanza error si no hay splits generados."""
        folder_class_map, tmp_path = tmp_source
        manager = DatasetMapManager(folder_class_map, str(tmp_path / "maps"))
        manager.scan_source_directory()
        # No llamamos generate_splits()
        
        with pytest.raises(ValueError, match="No hay splits generados"):
            manager.prepare_processed_directory(
                processed_dir=str(tmp_path / "processed"),
                strategy="copy"
            )
    
    def test_handles_missing_source_file(self, manager_with_splits):
        """Verifica que maneja archivos fuente faltantes sin fallar."""
        manager, tmp_path = manager_with_splits
        processed_dir = tmp_path / "processed"
        
        # Borrar un archivo fuente
        for cls_images in manager.splits['train'].values():
            if cls_images:
                Path(cls_images[0]).unlink()
                break
        
        # No debe fallar
        counts = manager.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        
        assert sum(counts.values()) > 0


class TestSaveAndLoadMaps:
    """Tests para save_split_maps y load_split_map."""
    
    def test_save_json_maps(self, manager_with_splits):
        """Verifica que se guardan mapas JSON correctamente."""
        manager, tmp_path = manager_with_splits
        
        files = manager.save_split_maps(format='json')
        
        assert 'train' in files
        assert files['train'].exists()
        
        with open(files['train'], 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
        assert len(data) > 0
    
    def test_save_txt_maps(self, manager_with_splits):
        """Verifica que se guardan mapas TXT correctamente."""
        manager, tmp_path = manager_with_splits
        
        files = manager.save_split_maps(format='txt')
        
        assert 'train' in files
        assert files['train'].exists()
        
        content = files['train'].read_text(encoding='utf-8')
        assert "# Dataset Map" in content
    
    def test_round_trip_json(self, manager_with_splits):
        """Verifica que guardar y cargar JSON preserva los datos."""
        manager, tmp_path = manager_with_splits
        
        files = manager.save_split_maps(format='json')
        
        loaded = manager.load_split_map(files['train'])
        
        original_count = sum(len(imgs) for imgs in manager.splits['train'].values())
        loaded_count = sum(len(imgs) for imgs in loaded.values())
        
        assert loaded_count == original_count


class TestLoadFromMapsDirectory:
    """Tests para load_from_maps_directory (recarga de mapas existentes)."""
    
    def test_restores_metadata(self, manager_with_splits):
        """Verifica que restaura metadata correctamente."""
        manager, tmp_path = manager_with_splits
        maps_dir = tmp_path / "maps"
        
        # Guardar mapas
        manager.save_split_maps(format='json')
        
        # Recargar
        restored = DatasetMapManager.load_from_maps_directory(str(maps_dir))
        
        assert restored.metadata['total_images'] == manager.metadata['total_images']
        assert set(restored.metadata['classes']) == set(manager.metadata['classes'])
    
    def test_restores_splits(self, manager_with_splits):
        """Verifica que restaura splits con conteos correctos."""
        manager, tmp_path = manager_with_splits
        maps_dir = tmp_path / "maps"
        
        manager.save_split_maps(format='json')
        
        restored = DatasetMapManager.load_from_maps_directory(str(maps_dir))
        
        for split_name in ['train', 'val', 'test']:
            original_count = sum(len(imgs) for imgs in manager.splits[split_name].values())
            restored_count = sum(len(imgs) for imgs in restored.splits[split_name].values())
            assert restored_count == original_count, f"Mismatch in {split_name}"
    
    def test_restores_class_distribution(self, manager_with_splits):
        """Verifica que restaura distribución de clases."""
        manager, tmp_path = manager_with_splits
        maps_dir = tmp_path / "maps"
        
        manager.save_split_maps(format='json')
        
        restored = DatasetMapManager.load_from_maps_directory(str(maps_dir))
        
        for cls in manager.metadata['classes']:
            assert cls in restored.metadata['class_distribution']
            assert restored.metadata['class_distribution'][cls] > 0
    
    def test_restores_folder_class_map(self, manager_with_splits):
        """Verifica que restaura el mapeo carpeta->clase."""
        manager, tmp_path = manager_with_splits
        maps_dir = tmp_path / "maps"
        
        manager.save_split_maps(format='json')
        
        restored = DatasetMapManager.load_from_maps_directory(str(maps_dir))
        
        assert restored.metadata.get('folder_class_map') is not None
        assert len(restored.metadata['folder_class_map']) > 0
    
    def test_restores_ratios(self, manager_with_splits):
        """Verifica que restaura los ratios de splits."""
        manager, tmp_path = manager_with_splits
        maps_dir = tmp_path / "maps"
        
        manager.save_split_maps(format='json')
        
        restored = DatasetMapManager.load_from_maps_directory(str(maps_dir))
        
        assert abs(restored.ratios['train'] - 0.7) < 0.01
        assert abs(restored.ratios['val'] - 0.15) < 0.01
    
    def test_raises_if_no_metadata(self, tmp_path):
        """Verifica que lanza error si no hay metadata."""
        empty_dir = tmp_path / "empty_maps"
        empty_dir.mkdir()
        
        with pytest.raises(FileNotFoundError):
            DatasetMapManager.load_from_maps_directory(str(empty_dir))
    
    def test_enables_prepare_after_load(self, manager_with_splits):
        """Verifica que el manager cargado puede preparar directorio procesado."""
        manager, tmp_path = manager_with_splits
        maps_dir = tmp_path / "maps"
        
        manager.save_split_maps(format='json')
        
        restored = DatasetMapManager.load_from_maps_directory(str(maps_dir))
        
        processed_dir = tmp_path / "processed"
        counts = restored.prepare_processed_directory(
            processed_dir=str(processed_dir),
            strategy="copy"
        )
        
        assert sum(counts.values()) > 0
        assert (processed_dir / "train").exists()
