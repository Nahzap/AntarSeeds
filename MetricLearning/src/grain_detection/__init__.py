"""
grain_detection — Detección y segmentación de granos de polen.

Preprocesamiento e inferencia full-image (localización): PollenGrainDetector (U²-Net SOD).
Clasificación: AnalogyNet + SliceAwareClassifier (best model entrenado).

Componentes:
- salient_detector / grain_detector: U²-Net (localización + .seg)
- model_grain_detector: backend legacy (atención ViT) — opcional
- seg_format, annotator, full_image_classifier
"""

from .detected_grain import DetectedGrain, GrainCountResult
from .seg_format import SegFileWriter, SegFileReader
from .full_image_classifier import FullImageClassifier, resolve_localization_detector
from .grain_detector import PollenGrainDetector
from .model_grain_detector import ModelGrainDetector
