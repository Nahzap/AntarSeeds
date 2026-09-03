"""
DetectedGrain — Modelos de datos para granos de polen detectados.

Dataclasses que representan detecciones individuales y resultados de conteo.
Análogo a DetectedObject de XYZ_Ctrl_L206_GUI pero extendido con campos
de clasificación por Metric Learning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class DetectedGrain:
    """
    Grano de polen detectado y opcionalmente clasificado.
    
    Campos de localización (Etapa 1 — U²-Net):
        index, bbox, area, saliency_prob, centroid, contour
    
    Campos de clasificación (Etapa 2 — Metric Learning):
        predicted_class, class_index, embedding, distance_to_centroid,
        confidence, is_unknown
    
    Campos de explicabilidad (Grad-CAM):
        gradcam_heatmap
    """
    # --- Localización (U²-Net) ---
    index: int
    bbox: Tuple[int, int, int, int]        # (x, y, w, h) en imagen original
    area: float                             # Área del contorno en px²
    saliency_prob: float                    # Probabilidad U²-Net [0-1]
    centroid: Tuple[int, int]               # (cx, cy) en imagen original
    contour: Optional[np.ndarray] = None    # Puntos del contorno Nx2
    classification_bbox: Optional[Tuple[int, int, int, int]] = None

    # --- Clasificación (Metric Learning) ---
    predicted_class: Optional[str] = None
    class_index: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    distance_to_centroid: Optional[float] = None
    confidence: Optional[float] = None
    is_unknown: bool = False

    # --- Explicabilidad ---
    gradcam_heatmap: Optional[np.ndarray] = None

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    def crop_bbox(self) -> Tuple[int, int, int, int]:
        """Tight bbox for classifier crops (matches .seg training); never expanded."""
        return self.classification_bbox or self.bbox


@dataclass
class GrainCountResult:
    """
    Resultado completo del conteo de granos en una imagen.
    """
    image_path: str
    total_grains: int
    counts_per_class: Dict[str, int] = field(default_factory=dict)
    unknown_count: int = 0
    detections: List[DetectedGrain] = field(default_factory=list)
    annotated_image: Optional[np.ndarray] = None
    saliency_map: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
