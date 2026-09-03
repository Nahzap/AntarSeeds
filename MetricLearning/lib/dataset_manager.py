"""
Dataset Manager - Sistema de Gestión de Datasets con Mapas Vectoriales
Implementa IEEE 829 (Documentación de Pruebas) y mejores prácticas de gestión de datos.

Características:
- Mapas vectoriales de referencias (archivos .txt)
- Balanceo dinámico train/val/test con sliders
- Sin duplicación de archivos (solo referencias)
- Validación de integridad
- Estadísticas en tiempo real
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Callable
from collections import defaultdict
import random
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DatasetMapManager:
    """
    Gestor de mapas vectoriales de datasets.
    Implementa IEEE 829 para documentación y trazabilidad.
    """
    
    def __init__(self, source_dir: Union[str, List[str], Dict[str, str]], output_dir: str = "data/maps"):
        """
        Inicializa gestor de datasets.
        
        Args:
            source_dir: Puede ser:
                - Un directorio: "ruta/carpeta"
                - Lista de directorios: ["ruta/carpeta1", "ruta/carpeta2"]
                - Diccionario carpeta->clase: {"ruta/carpeta1": "Clase_A", "ruta/carpeta2": "Clase_B"}
            output_dir: Directorio para guardar mapas vectoriales
        """
        # Soportar múltiples formatos de entrada
        if isinstance(source_dir, dict):
            # Mapeo carpeta -> clase (NUEVO)
            self.folder_class_map = {Path(folder): class_name for folder, class_name in source_dir.items()}
            self.source_dirs = list(self.folder_class_map.keys())
            self.source_dir = self.source_dirs[0] if self.source_dirs else Path(".")
        elif isinstance(source_dir, (list, tuple)):
            # Lista de directorios (sin mapeo explícito)
            self.source_dirs = [Path(d) for d in source_dir]
            self.source_dir = self.source_dirs[0]
            self.folder_class_map = None
        else:
            # Un solo directorio
            self.source_dir = Path(source_dir)
            self.source_dirs = [self.source_dir]
            self.folder_class_map = None
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapas vectoriales: {clase: [rutas]}
        self.image_map: Dict[str, List[Path]] = defaultdict(list)
        
        # Splits: {split_name: {clase: [rutas]}}
        self.splits: Dict[str, Dict[str, List[Path]]] = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        
        # Proporciones (suman 1.0)
        self.ratios = {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
        
        # Metadata
        self.metadata = {
            'created': datetime.now().isoformat(),
            'source_dirs': [str(d) for d in self.source_dirs],
            'folder_class_map': {str(k): v for k, v in self.folder_class_map.items()} if self.folder_class_map else None,
            'total_images': 0,
            'classes': [],
            'class_distribution': {}
        }
    
    def scan_source_directory(self) -> Dict[str, int]:
        """
        Escanea directorio(s) fuente y construye mapa vectorial unificado.
        
        Soporta dos modos:
        1. Carpetas con subcarpetas por clase (modo tradicional)
        2. Carpetas con mapeo explícito carpeta->clase (NUEVO)
        
        Returns:
            Dict con conteo por clase
        """
        self.image_map.clear()
        
        # Extensiones válidas
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        if self.folder_class_map:
            # MODO NUEVO: Mapeo explícito carpeta -> clase
            logger.info("Escaneando con mapeo carpeta->clase")
            
            for folder, class_name in self.folder_class_map.items():
                if not folder.exists():
                    logger.warning(f"Carpeta no existe: {folder}")
                    continue
                
                logger.info(f"Escaneando {folder} como clase '{class_name}'")
                
                # Todas las imágenes en esta carpeta pertenecen a class_name
                for img_path in folder.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                        self.image_map[class_name].append(img_path)
                        logger.debug(f"  Añadida: {img_path.name} -> {class_name}")
        else:
            # MODO TRADICIONAL: Carpetas con subcarpetas por clase
            logger.info("Escaneando con estructura tradicional (subcarpetas=clases)")
            
            for source_dir in self.source_dirs:
                if not source_dir.exists():
                    logger.warning(f"Directorio fuente no existe: {source_dir}")
                    continue
                
                logger.info(f"Escaneando directorio: {source_dir}")
                
                # Escanear por clase
                for class_dir in source_dir.iterdir():
                    if not class_dir.is_dir():
                        continue
                    
                    class_name = class_dir.name
                    
                    # Buscar imágenes en clase
                    for img_path in class_dir.iterdir():
                        if img_path.suffix.lower() in valid_extensions:
                            self.image_map[class_name].append(img_path)
        
        # Actualizar metadata
        self.metadata['classes'] = sorted(self.image_map.keys())
        self.metadata['class_distribution'] = {
            cls: len(imgs) for cls, imgs in self.image_map.items()
        }
        self.metadata['total_images'] = sum(self.metadata['class_distribution'].values())
        
        logger.info(f"Total escaneado: {self.metadata['total_images']} imágenes en {len(self.metadata['classes'])} clases")
        
        return self.metadata['class_distribution']
    
    def set_split_ratios(self, train: float, val: float, test: float):
        """
        Establece proporciones de splits.
        
        Args:
            train: Proporción de entrenamiento (0.0-1.0)
            val: Proporción de validación (0.0-1.0)
            test: Proporción de test (0.0-1.0)
        
        Raises:
            ValueError: Si las proporciones no suman 1.0
        """
        total = train + val + test
        if abs(total - 1.0) > 0.001:  # Tolerancia para errores de punto flotante
            raise ValueError(f"Las proporciones deben sumar 1.0 (actual: {total})")
        
        self.ratios = {
            'train': train,
            'val': val,
            'test': test
        }
    
    def generate_splits(self, seed: int = 42, stratified: bool = True):
        """
        Genera splits balanceados según proporciones.
        
        Args:
            seed: Semilla para reproducibilidad
            stratified: Si True, mantiene proporción de clases en cada split
        """
        random.seed(seed)
        
        # Limpiar splits anteriores
        for split in self.splits.values():
            split.clear()
        
        # Generar splits por clase (stratified)
        for class_name, images in self.image_map.items():
            # Mezclar imágenes
            shuffled = images.copy()
            random.shuffle(shuffled)
            
            total = len(shuffled)
            
            # Calcular índices de corte
            train_end = int(total * self.ratios['train'])
            val_end = train_end + int(total * self.ratios['val'])
            
            # Asignar a splits
            self.splits['train'][class_name] = shuffled[:train_end]
            self.splits['val'][class_name] = shuffled[train_end:val_end]
            self.splits['test'][class_name] = shuffled[val_end:]
    
    def save_split_maps(self, format: str = 'txt') -> Dict[str, Path]:
        """
        Guarda mapas vectoriales de splits en archivos.
        
        Args:
            format: Formato de salida ('txt' o 'json')
        
        Returns:
            Dict con rutas de archivos generados
        """
        output_files = {}
        
        for split_name, split_data in self.splits.items():
            if format == 'txt':
                output_file = self.output_dir / f"{split_name}_map.txt"
                self._save_txt_map(split_data, output_file)
            elif format == 'json':
                output_file = self.output_dir / f"{split_name}_map.json"
                self._save_json_map(split_data, output_file)
            else:
                raise ValueError(f"Formato no soportado: {format}")
            
            output_files[split_name] = output_file
        
        # Guardar metadata
        metadata_file = self.output_dir / "dataset_metadata.json"
        self._save_metadata(metadata_file)
        output_files['metadata'] = metadata_file
        
        return output_files
    
    def _save_txt_map(self, split_data: Dict[str, List[Path]], output_file: Path):
        """
        Guarda mapa en formato texto.
        
        Formato:
        /ruta/absoluta/imagen.jpg clase_id
        /ruta/absoluta/imagen2.jpg clase_id
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            # Header con metadata
            f.write(f"# Dataset Map - {output_file.stem}\n")
            f.write(f"# Generated: {datetime.now().isoformat()}\n")
            f.write(f"# Source directories: {len(self.source_dirs)}\n")
            f.write(f"# Format: image_path class_name\n")
            f.write("#\n")
            
            # Escribir referencias
            for class_name, images in sorted(split_data.items()):
                for img_path in images:
                    f.write(f"{img_path.absolute()}\t{class_name}\n")
    
    def _save_json_map(self, split_data: Dict[str, List[Path]], output_file: Path):
        """
        Guarda mapa en formato JSON.
        
        Formato:
        {
            "clase_1": ["/ruta/img1.jpg", "/ruta/img2.jpg"],
            "clase_2": ["/ruta/img3.jpg"]
        }
        """
        json_data = {
            class_name: [str(img.absolute()) for img in images]
            for class_name, images in split_data.items()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _save_metadata(self, output_file: Path):
        """Guarda metadata del dataset"""
        # Agregar estadísticas de splits
        split_stats = {}
        for split_name, split_data in self.splits.items():
            total = sum(len(imgs) for imgs in split_data.values())
            split_stats[split_name] = {
                'total': total,
                'percentage': total / self.metadata['total_images'] * 100 if self.metadata['total_images'] > 0 else 0,
                'per_class': {cls: len(imgs) for cls, imgs in split_data.items()}
            }
        
        self.metadata['splits'] = split_stats
        self.metadata['ratios'] = self.ratios
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load_split_map(self, map_file: Path) -> Dict[str, List[Path]]:
        """
        Carga mapa vectorial desde archivo.
        
        Args:
            map_file: Archivo de mapa (.txt o .json)
        
        Returns:
            Dict {clase: [rutas]}
        """
        if map_file.suffix == '.txt':
            return self._load_txt_map(map_file)
        elif map_file.suffix == '.json':
            return self._load_json_map(map_file)
        else:
            raise ValueError(f"Formato no soportado: {map_file.suffix}")
    
    def _load_txt_map(self, map_file: Path) -> Dict[str, List[Path]]:
        """Carga mapa desde archivo texto"""
        split_data = defaultdict(list)
        
        with open(map_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) == 2:
                    img_path, class_name = parts
                    split_data[class_name].append(Path(img_path))
        
        return dict(split_data)
    
    def _load_json_map(self, map_file: Path) -> Dict[str, List[Path]]:
        """Carga mapa desde archivo JSON"""
        with open(map_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        return {
            class_name: [Path(img) for img in images]
            for class_name, images in json_data.items()
        }
    
    @classmethod
    def load_from_maps_directory(cls, maps_dir: Union[str, Path] = "data/maps") -> 'DatasetMapManager':
        """
        Carga un DatasetMapManager completo desde un directorio de mapas existente.
        Restaura metadata, splits, image_map, ratios y folder_class_map.
        
        Args:
            maps_dir: Directorio con train_map.json, val_map.json, test_map.json, dataset_metadata.json
            
        Returns:
            DatasetMapManager completamente restaurado
            
        Raises:
            FileNotFoundError: Si no existe dataset_metadata.json
        """
        maps_dir = Path(maps_dir)
        metadata_file = maps_dir / "dataset_metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"No se encontró metadata en: {metadata_file}")
        
        # Cargar metadata
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Reconstruir folder_class_map desde metadata
        folder_class_map = metadata.get('folder_class_map')
        if folder_class_map:
            manager = cls(folder_class_map, str(maps_dir))
        else:
            source_dirs = metadata.get('source_dirs', [''])
            manager = cls(source_dirs[0] if len(source_dirs) == 1 else source_dirs, str(maps_dir))
        
        # Restaurar metadata
        manager.metadata = metadata
        
        # Restaurar ratios
        if 'ratios' in metadata:
            manager.ratios = metadata['ratios']
        
        # Restaurar image_map desde todos los splits combinados
        manager.image_map = defaultdict(list)
        
        # Cargar cada split desde JSON
        for split_name in ['train', 'val', 'test']:
            json_file = maps_dir / f"{split_name}_map.json"
            if json_file.exists():
                split_data = manager.load_split_map(json_file)
                manager.splits[split_name] = split_data
                
                # Reconstruir image_map (unión de todos los splits)
                for cls_name, imgs in split_data.items():
                    manager.image_map[cls_name].extend(imgs)
                
                logger.info(f"Split '{split_name}' cargado: {sum(len(v) for v in split_data.values())} imágenes")
            else:
                logger.warning(f"Mapa no encontrado: {json_file}")
        
        # Recalcular class_distribution desde image_map
        manager.metadata['class_distribution'] = {
            cls: len(imgs) for cls, imgs in manager.image_map.items()
        }
        manager.metadata['total_images'] = sum(manager.metadata['class_distribution'].values())
        manager.metadata['classes'] = sorted(manager.image_map.keys())
        
        logger.info(
            f"Manager restaurado: {manager.metadata['total_images']} imágenes, "
            f"{len(manager.metadata['classes'])} clases"
        )
        
        return manager

    def validate_maps(self) -> Dict[str, any]:
        """
        Valida integridad de mapas vectoriales.
        
        Returns:
            Dict con resultados de validación
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Validar que todos los archivos existen
        for split_name, split_data in self.splits.items():
            missing = []
            for class_name, images in split_data.items():
                for img_path in images:
                    if not img_path.exists():
                        missing.append(str(img_path))
            
            if missing:
                validation['valid'] = False
                validation['errors'].append({
                    'split': split_name,
                    'type': 'missing_files',
                    'count': len(missing),
                    'files': missing[:10]  # Primeros 10
                })
        
        # Validar proporciones
        total = self.metadata['total_images']
        if total > 0:
            for split_name, split_data in self.splits.items():
                split_total = sum(len(imgs) for imgs in split_data.values())
                actual_ratio = split_total / total
                expected_ratio = self.ratios[split_name]
                
                if abs(actual_ratio - expected_ratio) > 0.05:  # 5% tolerancia
                    validation['warnings'].append({
                        'split': split_name,
                        'type': 'ratio_mismatch',
                        'expected': expected_ratio,
                        'actual': actual_ratio
                    })
        
        # Estadísticas
        validation['stats'] = {
            'total_images': total,
            'splits': {
                split_name: sum(len(imgs) for imgs in split_data.values())
                for split_name, split_data in self.splits.items()
            }
        }
        
        return validation
    
    def get_statistics(self) -> Dict:
        """
        Obtiene estadísticas detalladas del dataset.
        
        Returns:
            Dict con estadísticas completas
        """
        stats = {
            'total_images': self.metadata['total_images'],
            'num_classes': len(self.metadata['classes']),
            'classes': self.metadata['classes'],
            'class_distribution': self.metadata['class_distribution'],
            'splits': {}
        }
        
        for split_name, split_data in self.splits.items():
            split_total = sum(len(imgs) for imgs in split_data.values())
            stats['splits'][split_name] = {
                'total': split_total,
                'percentage': split_total / self.metadata['total_images'] * 100 if self.metadata['total_images'] > 0 else 0,
                'per_class': {cls: len(imgs) for cls, imgs in split_data.items()}
            }
        
        # Balance de clases
        if self.metadata['class_distribution']:
            counts = list(self.metadata['class_distribution'].values())
            stats['class_balance'] = {
                'min': min(counts),
                'max': max(counts),
                'mean': sum(counts) / len(counts),
                'std': (sum((x - sum(counts)/len(counts))**2 for x in counts) / len(counts)) ** 0.5
            }
        
        return stats


    def prepare_processed_directory(
        self,
        processed_dir: Union[str, Path] = "data/processed",
        strategy: str = "symlink",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, int]:
        """
        Prepara directorio ImageFolder desde los splits generados.
        Crea estructura data/processed/{train,val,test}/ClassName/ con
        symlinks o copias de las imágenes originales.
        
        Args:
            processed_dir: Directorio destino (default: data/processed)
            strategy: 'symlink' (default, sin duplicar) o 'copy' (copia física)
            progress_callback: Callable(current, total, message) para reportar progreso
            
        Returns:
            Dict con conteo de archivos procesados por split
        """
        processed_dir = Path(processed_dir)
        counts = {}
        
        # Calcular total de archivos para progress
        total_files = sum(
            len(imgs) for split_data in self.splits.values()
            for imgs in split_data.values()
        )
        
        if total_files == 0:
            raise ValueError("No hay splits generados. Ejecute generate_splits() primero.")
        
        current = 0
        use_symlink = (strategy == "symlink")
        symlink_failed = False
        
        for split_name, split_data in self.splits.items():
            split_dir = processed_dir / split_name
            
            # Limpiar directorio existente para evitar datos stale
            # IMPORTANTE: Solo limpia imágenes. Las anotaciones .seg están en
            # data/annotations/ (sistema centralizado) y NO se tocan aquí.
            if split_dir.exists():
                logger.info(f"Limpiando directorio existente: {split_dir}")
                shutil.rmtree(split_dir)
            
            split_count = 0
            
            for class_name, images in split_data.items():
                class_dir = split_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                for img_path in images:
                    img_path = Path(img_path)
                    dest = class_dir / img_path.name
                    
                    # Evitar conflicto de nombres
                    if dest.exists():
                        stem = img_path.stem
                        suffix = img_path.suffix
                        counter = 1
                        while dest.exists():
                            dest = class_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                    
                    try:
                        if use_symlink and not symlink_failed:
                            try:
                                os.symlink(str(img_path.absolute()), str(dest))
                            except (OSError, NotImplementedError):
                                # Symlink failed (Windows sin permisos admin)
                                # Fallback a copia para el resto
                                symlink_failed = True
                                logger.warning(
                                    "Symlinks no disponibles, usando copia física. "
                                    "Esto puede tomar más tiempo y espacio en disco."
                                )
                                shutil.copy2(str(img_path), str(dest))
                        else:
                            shutil.copy2(str(img_path), str(dest))
                    except FileNotFoundError:
                        logger.warning(f"Archivo no encontrado, omitido: {img_path}")
                        continue
                    except Exception as e:
                        logger.error(f"Error procesando {img_path}: {e}")
                        continue
                    
                    split_count += 1
                    current += 1
                    
                    if progress_callback and current % 50 == 0:
                        progress_callback(
                            current, total_files,
                            f"{split_name}/{class_name}: {img_path.name}"
                        )
            
            counts[split_name] = split_count
            logger.info(f"Split '{split_name}': {split_count} archivos preparados en {split_dir}")
        
        # Reportar resultado final
        if progress_callback:
            progress_callback(total_files, total_files, "Completado")
        
        method = "symlinks" if (use_symlink and not symlink_failed) else "copias"
        logger.info(f"Directorio procesado listo ({method}): {processed_dir}")
        logger.info(f"Total: {sum(counts.values())} archivos")
        
        return counts
    
    @staticmethod
    def prepare_from_maps(
        maps_dir: Union[str, Path] = "data/maps",
        processed_dir: Union[str, Path] = "data/processed",
        strategy: str = "symlink",
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, int]:
        """
        Método estático para preparar directorio procesado directamente
        desde archivos de mapas guardados (sin necesidad de re-escanear).
        
        Args:
            maps_dir: Directorio con los mapas (train_map.json, val_map.json, test_map.json)
            processed_dir: Directorio destino
            strategy: 'symlink' o 'copy'
            progress_callback: Callable para reportar progreso
            
        Returns:
            Dict con conteo de archivos procesados por split
        """
        maps_dir = Path(maps_dir)
        processed_dir = Path(processed_dir)
        
        # Crear un manager temporal para usar prepare_processed_directory
        manager = DatasetMapManager("", str(maps_dir))
        
        # Cargar cada split desde JSON
        for split_name in ['train', 'val', 'test']:
            json_file = maps_dir / f"{split_name}_map.json"
            if json_file.exists():
                split_data = manager.load_split_map(json_file)
                manager.splits[split_name] = {
                    cls: imgs for cls, imgs in split_data.items()
                }
                logger.info(f"Mapa cargado: {json_file.name} ({sum(len(v) for v in split_data.values())} imgs)")
            else:
                logger.warning(f"Mapa no encontrado: {json_file}")
        
        # Calcular metadata para total_images
        manager.metadata['total_images'] = sum(
            len(imgs) for split_data in manager.splits.values()
            for imgs in split_data.values()
        )
        
        return manager.prepare_processed_directory(
            processed_dir=processed_dir,
            strategy=strategy,
            progress_callback=progress_callback
        )


class DatasetMapLoader:
    """
    Cargador de datasets desde mapas vectoriales.
    Compatible con PyTorch DataLoader.
    """
    
    def __init__(self, map_file: Path, transform=None):
        """
        Inicializa cargador.
        
        Args:
            map_file: Archivo de mapa vectorial
            transform: Transformaciones a aplicar
        """
        self.map_file = Path(map_file)
        self.transform = transform
        
        # Cargar mapa
        manager = DatasetMapManager("", "")
        self.data = manager.load_split_map(self.map_file)
        
        # Crear índice lineal
        self.samples = []
        self.class_to_idx = {}
        
        for idx, class_name in enumerate(sorted(self.data.keys())):
            self.class_to_idx[class_name] = idx
            for img_path in self.data[class_name]:
                self.samples.append((img_path, idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Obtiene item del dataset.
        
        Args:
            idx: Índice
        
        Returns:
            (image, label) o (image, label, path)
        """
        img_path, label = self.samples[idx]
        
        # Cargar imagen
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        # Aplicar transformaciones
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @property
    def classes(self):
        """Lista de clases"""
        return sorted(self.class_to_idx.keys())
