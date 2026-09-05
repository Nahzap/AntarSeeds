"""
PollenGrainDetector — Detector multi-grano de polen usando U²-Net.

Uso previsto: **solo preprocesamiento** (generar .seg / recortes antes del entrenamiento).
No requiere fine-tune en el pipeline de metric learning.

Adaptado del U2NetDetector validado en XYZ_Ctrl_L206_GUI.
Pipeline: Saliency map → Umbral adaptativo → Morfología → Contornos → Filtrado

Ref:
- Qin et al. (2020) U²-Net: Going Deeper with Nested U-Structure for SOD
- Expert Systems with Applications (2024): Saliency-guided cell counting
- XYZ_Ctrl_L206_GUI/src/core/detection/u2net_detector.py (implementación validada)
"""

import logging
import time
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .detected_grain import DetectedGrain
from .salient_detector import SalientObjectDetector
from .detection_profiles import resolve_grain_detection_config
from .saliency_grain_extraction import (
    bbox_iou,
    extract_grains_from_saliency,
    nms_merge_grains,
)

logger = logging.getLogger(__name__)


class PollenGrainDetector:
    """
    Detector de granos de polen individuales usando U²-Net como SOD.

    Genera lista de DetectedGrain con bbox, contorno, centroide y saliency.
    Los parámetros están preconfigurados para microscopía de polen (modo SENSITIVE
    de XYZ) pero son completamente configurables.

    Uso:
        detector = PollenGrainDetector()
        saliency, grains = detector.detect(image)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el detector.

        Args:
            config: Dict con parámetros. Keys opcionales:
                model_type, input_size, saliency_threshold, adaptive_k,
                min_area, max_area, morph_kernel_size, min_circularity,
                crop_padding, crop_size, device
        """
        config = resolve_grain_detection_config(config)

        # Parámetros de detección (defaults = modo SENSITIVE para polen)
        self.saliency_threshold = config.get("saliency_threshold", 0.30)
        self.adaptive_k = config.get("adaptive_k", 0.30)
        self.min_area = config.get("min_area", 1000)
        self.max_area = config.get("max_area", 80000)
        self.morph_kernel_size = config.get("morph_kernel_size", 3)
        self.min_circularity = config.get("min_circularity", 0.45)
        self.max_bbox_side_frac = float(config.get("max_bbox_side_frac", 0.35))
        self.max_aspect_ratio = float(config.get("max_aspect_ratio", 2.5))
        max_area_frac = config.get("max_area_frac")
        self.max_area_frac = float(max_area_frac) if max_area_frac is not None else None
        self.detection_profile = str(config.get("detection_profile", "balanced"))
        self.object_finder = str(config.get("object_finder", "roi_seed")).strip().lower()
        self.crop_radius = int(config.get("crop_radius", 300))
        self.split_touching = bool(config.get("split_touching", True))
        self.seed_core_frac = float(config.get("seed_core_frac", 0.45))
        self.waist_frac = float(config.get("waist_frac", 0.80))
        self.drop_border_objects = bool(config.get("drop_border_objects", False))

        # Parámetros de crop DML (no definen qué es un objeto)
        self.crop_padding = config.get("crop_padding", 0.20)
        self.crop_size = config.get("crop_size", 256)

        # Modelo U²-Net
        model_type = config.get("model_type", "u2netp")
        input_size = config.get("input_size", 320)
        device = config.get("device", None)

        self._sod = SalientObjectDetector(
            model_type=model_type,
            device=device,
            input_size=input_size,
            auto_download=True,
        )

        logger.info(
            f"[PollenGrainDetector] Inicializado — "
            f"threshold={self.saliency_threshold}, k={self.adaptive_k}, "
            f"min_area={self.min_area}, kernel={self.morph_kernel_size}"
        )

    def detect(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[DetectedGrain]]:
        """
        Detecta granos de polen individuales en la imagen.

        Args:
            image: Imagen BGR (numpy array), cualquier resolución.

        Returns:
            saliency_map: Mapa de probabilidad [0-1] del tamaño original.
            grains: Lista de DetectedGrain ordenados por área descendente.
        """
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.float32), []

        t0 = time.perf_counter()

        # Paso 1: Saliency map
        saliency = self.saliency(image)

        if self.object_finder == "roi_seed":
            grains = self._extract_grains_roi_seed(saliency, image)
        else:
            grains = self._extract_grains(saliency, image)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[PollenGrainDetector] {len(grains)} granos detectados en {elapsed:.0f}ms"
        )

        return saliency, grains

    def detect_exhaustive(
        self, image: np.ndarray, *, iou_threshold: float = 0.35
    ) -> Tuple[np.ndarray, List[DetectedGrain]]:
        """
        Detección agresiva para full-image inference: múltiples umbrales y fusión NMS.

        Encuentra granos no anotados en .seg (descubrimiento más allá de etiquetas previas).
        """
        if image is None or image.size == 0:
            return np.zeros((100, 100), dtype=np.float32), []

        t0 = time.perf_counter()
        saliency = self.saliency(image)

        merged: List[DetectedGrain] = []
        k_scales = (1.0, 0.65, 0.4)
        circ_scales = (1.0, 0.75, 0.5)
        if self.object_finder == "roi_seed":
            # Mismo buscador, sembrado más permisivo: no se cambia de algoritmo
            # a mitad del pipeline solo para descubrir más cuerpos.
            merged.extend(self._extract_grains_roi_seed_exhaustive(saliency, image, k_scales))
        else:
            for k_scale in k_scales:
                for circ_scale in circ_scales:
                    merged.extend(
                        self._extract_grains(
                            saliency,
                            image,
                            adaptive_k_override=max(0.08, self.adaptive_k * k_scale),
                            min_circularity_override=max(0.12, self.min_circularity * circ_scale),
                        )
                    )

        grains = nms_merge_grains(merged, iou_threshold=iou_threshold)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            f"[PollenGrainDetector] exhaustive: {len(grains)} granos "
            f"(candidatos={len(merged)}) en {elapsed:.0f}ms"
        )
        return saliency, grains

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        return bbox_iou(a, b)

    def _nms_merge_grains(
        self, grains: List[DetectedGrain], *, iou_threshold: float = 0.35
    ) -> List[DetectedGrain]:
        return nms_merge_grains(grains, iou_threshold=iou_threshold)

    def _seeded_params(self):
        """Misma fábrica que click/preview: SeededParams.from_config."""
        from .seeded_contour import SeededParams

        return SeededParams.from_config(
            {
                "min_area": self.min_area,
                "max_area": self.max_area,
                "max_area_frac": (
                    self.max_area_frac if self.max_area_frac is not None else 0.50
                ),
                "min_circularity": self.min_circularity,
                "max_bbox_side_frac": self.max_bbox_side_frac,
                "max_aspect_ratio": self.max_aspect_ratio,
                "morph_kernel_size": self.morph_kernel_size,
                "saliency_threshold": self.saliency_threshold,
                "adaptive_k": self.adaptive_k,
                "crop_radius": self.crop_radius,
                "split_touching": self.split_touching,
                "seed_core_frac": self.seed_core_frac,
                "waist_frac": self.waist_frac,
                "drop_border_objects": self.drop_border_objects,
            }
        )

    @staticmethod
    def _dicts_to_detected_grains(grain_dicts: List[Dict]) -> List[DetectedGrain]:
        grains: List[DetectedGrain] = []
        for i, g in enumerate(grain_dicts):
            contour = np.asarray(g["contour"], dtype=np.int32)
            bbox = tuple(int(v) for v in g["bbox"])
            area = abs(cv2.contourArea(contour.reshape(-1, 1, 2)))
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2
            grains.append(
                DetectedGrain(
                    index=i,
                    bbox=bbox,
                    area=float(area),
                    saliency_prob=float(g.get("saliency", 0.0)),
                    centroid=(cx, cy),
                    contour=contour,
                )
            )
        return grains

    def saliency(self, image: np.ndarray) -> np.ndarray:
        """Mapa de saliencia U²-Net [0-1]. Punto de acceso público al SOD."""
        return self._sod.get_saliency_map(image)

    def _crop_saliency_provider(self):
        """La red se ejecuta por recorte: misma resolución efectiva que el click."""

        def provider(image_bgr, box):
            x1, y1, x2, y2 = box
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            return self.saliency(crop)

        return provider

    def _extract_grains_roi_seed(
        self, saliency: np.ndarray, image: np.ndarray
    ) -> List[DetectedGrain]:
        """Mismo algoritmo que el click de ROI: picos = semillas automáticas."""
        from .seeded_contour import propose_objects

        return self._dicts_to_detected_grains(
            propose_objects(
                saliency,
                image,
                self._seeded_params(),
                saliency_provider=self._crop_saliency_provider(),
            )
        )

    def _extract_grains_roi_seed_exhaustive(
        self, saliency: np.ndarray, image: np.ndarray, k_scales: Tuple[float, ...]
    ) -> List[DetectedGrain]:
        """Descubrimiento: mismas semillas del buscador ROI con k progresivamente menor."""
        from .seeded_contour import peak_seeds, propose_objects

        params = self._seeded_params()
        provider = self._crop_saliency_provider()
        seeds: List[Tuple[int, int]] = []
        for k_scale in k_scales:
            relaxed = replace(params, adaptive_k=max(0.05, params.adaptive_k * k_scale))
            seeds.extend(peak_seeds(saliency, relaxed))

        return self._dicts_to_detected_grains(
            propose_objects(
                saliency, image, params, extra_seeds=seeds, saliency_provider=provider
            )
        )

    def _extract_grains(
        self,
        saliency: np.ndarray,
        image: np.ndarray,
        *,
        adaptive_k_override: Optional[float] = None,
        min_circularity_override: Optional[float] = None,
    ) -> List[DetectedGrain]:
        """Delega extracción morfológica a saliency_grain_extraction."""
        return extract_grains_from_saliency(
            saliency,
            image,
            adaptive_k=self.adaptive_k,
            min_area=self.min_area,
            max_area=self.max_area,
            max_area_frac=self.max_area_frac,
            morph_kernel_size=self.morph_kernel_size,
            min_circularity=self.min_circularity,
            max_bbox_side_frac=self.max_bbox_side_frac,
            max_aspect_ratio=self.max_aspect_ratio,
            adaptive_k_override=adaptive_k_override,
            min_circularity_override=min_circularity_override,
        )


    def set_parameters(self, **kwargs):
        """Actualiza parámetros de detección."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"[PollenGrainDetector] {key} = {value}")

    def get_parameters(self) -> Dict:
        """Retorna parámetros actuales."""
        return {
            "detection_profile": self.detection_profile,
            "object_finder": self.object_finder,
            "saliency_threshold": self.saliency_threshold,
            "adaptive_k": self.adaptive_k,
            "min_area": self.min_area,
            "max_area": self.max_area,
            "max_area_frac": self.max_area_frac,
            "morph_kernel_size": self.morph_kernel_size,
            "min_circularity": self.min_circularity,
            "max_bbox_side_frac": self.max_bbox_side_frac,
            "max_aspect_ratio": self.max_aspect_ratio,
            "crop_radius": self.crop_radius,
            "split_touching": self.split_touching,
            "seed_core_frac": self.seed_core_frac,
            "waist_frac": self.waist_frac,
            "drop_border_objects": self.drop_border_objects,
            "crop_padding": self.crop_padding,
            "crop_size": self.crop_size,
        }

    def is_ready(self) -> bool:
        return self._sod.is_ready()
