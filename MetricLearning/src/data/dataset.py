from torch.utils.data import Dataset
from torchvision import datasets
from pathlib import Path
import os
import numpy as np
import logging


class MielDataset(datasets.ImageFolder):
    """
    Dataset personalizado para imágenes de miel con metadata adicional.
    
    Hereda de ImageFolder para mantener compatibilidad con estructura estándar:
    root/
        clase_1/
            img_001.jpg
            img_002.jpg
        clase_2/
            ...
    
    Args:
        root (str): Directorio raíz del dataset
        transform (callable, optional): Transformaciones a aplicar
        return_paths (bool): Si True, retorna también la ruta de la imagen
        cache_in_memory (bool): Si True, pre-carga todas las imágenes en RAM
    """
    
    def __init__(self, root, transform=None, return_paths=False, cache_in_memory=False):
        super(MielDataset, self).__init__(root, transform=transform)
        self.return_paths = return_paths
        self._image_cache = None
        
        self._filter_broken_samples()
        
        if cache_in_memory:
            self._preload_images()
    
    def _filter_broken_samples(self):
        """
        Remove broken symlinks and inaccessible files from self.samples and self.targets.
        This prevents FileNotFoundError when the dataset source files are modified,
        moved, or stored on a disconnected drive.
        """
        logger = logging.getLogger(__name__)
        original_count = len(self.samples)
        
        valid_samples = []
        valid_targets = []
        broken_by_class = {}
        
        for path, target in self.samples:
            if os.path.isfile(path):
                valid_samples.append((path, target))
                valid_targets.append(target)
            else:
                class_name = self.classes[target] if target < len(self.classes) else f"class_{target}"
                broken_by_class[class_name] = broken_by_class.get(class_name, 0) + 1
        
        removed = original_count - len(valid_samples)
        
        if removed > 0:
            self.samples = valid_samples
            self.targets = valid_targets
            self.imgs = self.samples
            
            logger.warning(
                f"Filtered {removed}/{original_count} inaccessible files "
                f"(broken symlinks or missing files)"
            )
            for cls_name, count in sorted(broken_by_class.items()):
                logger.warning(f"  {cls_name}: {count} files skipped")
            logger.info(f"Dataset usable: {len(self.samples)} valid samples remain")
    
    def _preload_images(self):
        """Pre-load all images into RAM to eliminate disk I/O during training."""
        logger = logging.getLogger(__name__)
        n = len(self.samples)
        logger.info(f"Caching {n} images in RAM (eliminates disk I/O)...")
        
        self._image_cache = [None] * n
        for idx in range(n):
            path = self.samples[idx][0]
            self._image_cache[idx] = self.loader(path)
            if (idx + 1) % 2000 == 0:
                logger.info(f"  Cached {idx+1}/{n} images")
        
        logger.info(f"Caching complete: {n} images in RAM")
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Índice
        
        Returns:
            tuple: (image, label) o (image, label, path) si return_paths=True
        """
        path, target = self.samples[index]
        
        if self._image_cache is not None:
            sample = self._image_cache[index]
        else:
            sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.return_paths:
            return sample, target, path
        else:
            return sample, target
