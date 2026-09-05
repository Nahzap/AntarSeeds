"""
SegmentedGrainDataset — Dataset que lee anotaciones .seg y genera crops on-the-fly.

A diferencia de MielDataset (1 sample = 1 imagen completa),
aquí 1 sample = 1 grano individual recortado de la imagen original.

No genera imágenes adicionales en disco. Los crops se computan
en runtime a partir de la imagen original + las coordenadas del .seg.

Uso:
    dataset = SegmentedGrainDataset("data/processed/train", transform=val_transform)
    crop_tensor, class_idx = dataset[0]
"""

import collections
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from src.grain_detection.annotation_paths import find_existing_seg_path, seg_path_for
from src.grain_detection.seg_format import SegFileReader, has_seg_file

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class SegmentedGrainDataset(Dataset):
    """
    Dataset que genera crops de granos individuales on-the-fly.

    Escanea un directorio ImageFolder buscando pares imagen/.seg.
    Cada grano anotado en un .seg se convierte en un sample independiente.

    Args:
        root: Directorio raíz (estructura ImageFolder: root/class/img.ext)
        transform: Transforms de torchvision a aplicar sobre el crop PIL
        crop_padding: Padding proporcional alrededor del bbox del grano
        crop_size: Tamaño final del crop cuadrado (para AnalogyNet)
        use_mask: Si True, aplica la máscara del grano (fondo fuera del contorno)
        mask_bg_mode: Modo de fondo fuera del contorno cuando use_mask=True.
            'imagenet_neutral' = RGB por canal = mean*255 → ~0 tras Normalize ImageNet
                                 (zero activation del autor: fondo no activa atención)
            'random' = color aleatorio sólido (training)
            'gray'   = gris 128 uniforme (legacy)
            'black'  = negro puro (legacy, NO recomendado)
            'none'   = fondo original del crop
        min_saliency: Umbral mínimo de saliency para incluir un grano
        return_metadata: Si True, retorna (tensor, class_idx, metadata_dict)
        return_mask: Si True, retorna máscara binaria del grano para masked pooling
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        crop_padding: float = 0.2,
        crop_size: int = 256,
        use_mask: bool = False,
        mask_bg_mode: str = 'gray',
        min_saliency: float = 0.0,
        return_metadata: bool = False,
        return_mask: bool = False,
        annotation_root: Optional[str] = None,
        image_cache_size: int = 8,
        image_cache_max_mb: int = 96,
        max_cached_image_mb: int = 12,
    ):
        self.root = Path(root)
        self.transform = transform
        self.crop_padding = crop_padding
        self.crop_size = crop_size
        self.use_mask = use_mask
        self.mask_bg_mode = mask_bg_mode
        self.min_saliency = min_saliency
        self.return_metadata = return_metadata
        self.return_mask = return_mask
        
        # Annotation root: directorio centralizado para .seg files
        # Si es None, busca .seg junto a las imágenes (legacy behavior)
        if annotation_root is None:
            self.annotation_root = None
            logger.info("[SegmentedGrainDataset] Usando anotaciones junto a imágenes (legacy mode)")
        else:
            self.annotation_root = Path(annotation_root)
            if not self.annotation_root.exists():
                raise FileNotFoundError(f"Annotation root no existe: {self.annotation_root}")
            logger.info(f"[SegmentedGrainDataset] Usando anotaciones centralizadas en: {self.annotation_root}")

        # _grain_samples: cached grain data — avoids re-reading .seg in __getitem__
        # Each entry: (image_path, class_idx, bbox, saliency, contour_or_None)
        self._grain_samples: List[Tuple[str, int, Tuple[int,int,int,int], float, Optional[np.ndarray]]] = []
        # samples: 2-tuples (image_path, class_idx) — compatible con MielDataset / run_setup
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.targets: List[int] = []

        self._logged_first_getitem = False
        self.h5_path: Optional[str] = None
        self._h5_file = None

        self._build_index()

        logger.info(
            f"[SegmentedGrainDataset] {len(self.samples)} granos de "
            f"{len(self.classes)} clases en {self.root}"
        )
        logger.info(
            f"  config: crop_size={self.crop_size}, padding={self.crop_padding}, "
            f"use_mask={self.use_mask}, mask_bg={self.mask_bg_mode}, return_mask={self.return_mask}"
        )
        stats = self.get_stats()
        for cls_name, count in stats.get('class_distribution', {}).items():
            logger.info(f"  {cls_name}: {count} granos")

    def close(self) -> None:
        """Close HDF5 handle opened in this process (main or worker)."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None

    def _build_index(self):
        """
        Escanea el directorio buscando pares imagen/.seg.
        Cada grano en un .seg se convierte en un sample independiente.
        """
        if not self.root.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {self.root}")

        class_dirs = sorted(
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        skipped_no_seg = 0
        skipped_saliency = 0

        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Buscar .seg en annotation_root si está configurado, sino junto a la imagen
                if self.annotation_root is not None:
                    existing = find_existing_seg_path(img_path, self.annotation_root)
                    seg_path = existing or seg_path_for(img_path, self.annotation_root)
                else:
                    seg_path = img_path.with_suffix(".seg")
                
                if not seg_path.exists():
                    skipped_no_seg += 1
                    continue

                try:
                    grains = SegFileReader.read_grains(str(seg_path))
                except Exception as e:
                    logger.warning(
                        f"[SegGrainDataset] Error leyendo {seg_path.name}: {e}"
                    )
                    continue

                img_str = str(img_path)
                for grain in grains:
                    if grain["saliency"] < self.min_saliency:
                        skipped_saliency += 1
                        continue

                    # Cache grain data directly — no .seg re-reads at runtime
                    contour = grain.get("contour")  # numpy array or None
                    
                    # Adjust bbox if image is downscaled later
                    bbox = list(grain["bbox"])
                    
                    self._grain_samples.append((
                        img_str,
                        class_idx,
                        tuple(bbox),
                        grain["saliency"],
                        contour,
                    ))
                    self.samples.append((img_str, class_idx))
                    self.targets.append(class_idx)

        if skipped_no_seg > 0:
            logger.info(
                f"[SegGrainDataset] {skipped_no_seg} imágenes sin .seg (omitidas)"
            )
        if skipped_saliency > 0:
            logger.debug(
                f"[SegGrainDataset] {skipped_saliency} granos bajo min_saliency"
            )

    def __len__(self) -> int:
        return len(self._grain_samples)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5_file'] = None
        return state

    def _load_image(self, image_path: str) -> Tuple[np.ndarray, float]:
        """
        Load BGR frame without downscaling to extract high-resolution crops.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"No se pudo leer: {image_path}")
        
        return image, 1.0

    @staticmethod
    def _scale_geometry(
        bbox: Tuple[int, int, int, int],
        contour: Optional[np.ndarray],
        scale: float,
    ):
        if scale == 1.0:
            return bbox, contour
        x, y, w, h = bbox
        bbox = (
            int(x * scale),
            int(y * scale),
            max(1, int(w * scale)),
            max(1, int(h * scale)),
        )
        if contour is not None:
            contour = (np.asarray(contour, dtype=np.float32) * scale).astype(np.int32)
        return bbox, contour

    def __getitem__(self, idx: int):
        image_path, class_idx, bbox, saliency, contour = self._grain_samples[idx]

        if self.h5_path is not None:
            if self._h5_file is None:
                import h5py
                # Parallel reads are safe without SWMR in 'r' mode
                self._h5_file = h5py.File(self.h5_path, 'r')
            
            data = self._h5_file['crops'][idx]
            crop = data[:, :, :3].copy()
            mask_binary = data[:, :, 3].copy() if self.use_mask else None
            
            # Aplicar mask_bg_mode on-the-fly si usamos máscara
            if mask_binary is not None:
                crop = self._apply_mask_bg(crop, mask_binary)
        else:
            # Fallback a lectura on-the-fly
            image, scale = self._load_image(image_path)
            bbox, contour = self._scale_geometry(bbox, contour, scale)
            grain = {"bbox": bbox, "saliency": saliency, "contour": contour}
            crop = self._extract_crop(image, grain)
            if crop is None:
                crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

            mask_binary = None
            if (self.use_mask or self.return_mask) and contour is not None:
                mask_binary = self._generate_mask(grain, image.shape[:2])
                if self.use_mask:
                    crop = self._apply_mask_bg(crop, mask_binary)

        # BGR → RGB numpy (skip PIL — Albumentations accepts ndarray directly)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        mask_tensor = None
        if self.transform:
            from src.data.grain_transform import apply_transform
            if self.return_mask and mask_binary is not None:
                out = apply_transform(self.transform, crop_rgb, mask_binary)
                crop_tensor = out[0].contiguous()
                mask_tensor = out[1].contiguous()
            else:
                crop_tensor = apply_transform(self.transform, crop_rgb).contiguous()
                if self.return_mask:
                    mask_tensor = torch.ones(crop_tensor.shape[1:], dtype=torch.float32).contiguous()
        else:
            from torchvision.transforms.functional import to_tensor
            crop_tensor = to_tensor(crop_rgb)
            if self.return_mask:
                mask_tensor = torch.from_numpy(mask_binary.astype(np.float32)/255.0) if mask_binary is not None else torch.ones(crop_tensor.shape[1:], dtype=torch.float32)

        if self.return_metadata:
            meta = {
                "image_path": image_path,
                "grain_index": idx,
                "bbox": bbox,
                "saliency": saliency,
                "class_name": self.classes[class_idx],
            }
            return crop_tensor, class_idx, meta

        # First-call diagnostic (once, in worker 0)
        if not self._logged_first_getitem:
            self._logged_first_getitem = True
            logger.info(
                f"  [DIAG] First crop: tensor={tuple(crop_tensor.shape)}, "
                f"range=[{crop_tensor.min():.3f}, {crop_tensor.max():.3f}], "
                f"class={self.classes[class_idx]}, mask_applied={mask_binary is not None}"
            )

        if self.return_mask:
            return crop_tensor, class_idx, mask_tensor

        return crop_tensor, class_idx

    def _extract_crop(self, image: np.ndarray, grain: Dict) -> np.ndarray:
        """Extrae crop cuadrado centrado en el grano con padding."""
        from src.grain_detection.grain_crop_utils import extract_grain_crop

        return extract_grain_crop(image, grain, crop_padding=self.crop_padding)

    def get_raw_crop_and_mask(self, idx: int) -> np.ndarray:
        """Extract raw crop and mask (without background coloring) for HDF5 cache."""
        image_path, _, bbox, _, contour = self._grain_samples[idx]
        image, scale = self._load_image(image_path)
        bbox, contour = self._scale_geometry(bbox, contour, scale)
        grain = {"bbox": bbox, "contour": contour}
        
        crop = self._extract_crop(image, grain)
        
        if self.use_mask and contour is not None:
            mask_crop = self._generate_mask(grain, image.shape[:2])
        else:
            mask_crop = np.full((self.crop_size, self.crop_size), 255, dtype=np.uint8)
            
        out = np.empty((self.crop_size, self.crop_size, 4), dtype=np.uint8)
        out[:,:,:3] = crop
        out[:,:,3] = mask_crop
        return out

    def _generate_mask(self, grain: Dict, original_shape: Tuple[int, int]) -> np.ndarray:
        """Genera la máscara binaria pura desde el contorno."""
        from src.grain_detection.grain_crop_utils import generate_grain_mask

        mask = generate_grain_mask(grain, original_shape, crop_padding=self.crop_padding)
        if grain.get("contour") is None or len(grain.get("contour", [])) < 3:
            return np.full((self.crop_size, self.crop_size), 255, dtype=np.uint8)
        return mask

    def _apply_mask_bg(self, crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
        """Aplica el color de fondo a la imagen basándose en la máscara (BGR uint8)."""
        from src.grain_detection.grain_crop_utils import apply_mask_bg

        return apply_mask_bg(crop, mask_crop, self.mask_bg_mode)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del dataset."""
        class_counts = {}
        for _, class_idx, _, _, _ in self._grain_samples:
            name = self.classes[class_idx]
            class_counts[name] = class_counts.get(name, 0) + 1

        return {
            "total_grains": len(self.samples),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "grains_per_class": class_counts,
        }
