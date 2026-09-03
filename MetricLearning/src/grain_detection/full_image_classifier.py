"""
FullImageClassifier — Pipeline completo para clasificar granos en imagen de cámara.

Flujo:
    1. Localiza granos con PollenGrainDetector (U²-Net SOD + morfología de polen)
    2. Recorta cada grano detectado
    3. Extrae embedding con AnalogyNet (best model entrenado con metric learning)
    4. Clasifica vía slice-aware classifier o k-NN de referencia
    5. Genera visualización con bboxes, labels y heatmap de confianza

El ViT entrenado gobierna identidad taxonómica; U²-Net gobierna localización (tarea distinta).
ModelGrainDetector (atención ViT) queda como backend legacy opcional.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

LOCALIZATION_DETECTORS = ("u2net", "vit", "vit_dense")


def resolve_localization_detector(grain_config: Optional[Dict]) -> str:
    """Return active full-image localization backend."""
    from src.grain_detection.detector_registry import get_active_detector_config, normalize_backend

    cfg = grain_config or {}
    active = get_active_detector_config({"grain_detection": cfg})
    backend = normalize_backend(active.get("backend", "u2net"))
    if backend not in LOCALIZATION_DETECTORS:
        raise ValueError(
            f"active_detector.backend must be one of {LOCALIZATION_DETECTORS}, got {backend!r}"
        )
    return backend


class FullImageClassifier:
    """
    Pipeline end-to-end: imagen completa de cámara → clasificación por grano.

    Combina:
    - PollenGrainDetector (U²-Net + filtros morfológicos de polen) para localizar
    - AnalogyNet (metric learning) para extraer embeddings y clasificar
    """

    @staticmethod
    def _create_localization_detector(
        grain_config: Dict,
        *,
        model: nn.Module,
        transforms,
        device: torch.device,
        slice_classifier=None,
    ):
        backend = resolve_localization_detector(grain_config)
        if backend == "vit":
            from src.grain_detection.model_grain_detector import ModelGrainDetector

            logger.info(
                "[FullImageClassifier] Localización: ModelGrainDetector (ViT attention, legacy)"
            )
            return ModelGrainDetector(
                model=model,
                transforms=transforms,
                device=device,
                config=grain_config,
                slice_classifier=slice_classifier,
            )

        if backend == "vit_dense":
            from src.grain_detection.detector_registry import get_active_detector_config
            from src.grain_detection.vit_dense_detector import PollenViTDetector

            active = get_active_detector_config({"grain_detection": grain_config})
            ckpt = active.get("checkpoint")
            logger.info(f"[FullImageClassifier] Localización: PollenViTDetector @ {ckpt}")
            det_config = dict(grain_config)
            det_config["device"] = str(device)
            return PollenViTDetector(str(ckpt), config=det_config, device=str(device))

        from src.grain_detection.grain_detector import PollenGrainDetector

        det_config = dict(grain_config)
        det_config["device"] = str(device)
        logger.info(
            "[FullImageClassifier] Localización: PollenGrainDetector (U²-Net SOD + morfología polen) "
            f"[adaptive_k={det_config.get('adaptive_k', 0.3)}, "
            f"min_area={det_config.get('min_area', 1000)}, "
            f"min_circularity={det_config.get('min_circularity', 0.3)}]"
        )
        return PollenGrainDetector(config=det_config)

    def __init__(
        self,
        model: nn.Module,
        transforms,
        reference_embeddings: np.ndarray,
        reference_labels: np.ndarray,
        class_names: List[str],
        device: torch.device,
        grain_config: Optional[Dict] = None,
        k: int = 5,
        metric_type: str = "cosine",
        slice_classifier=None,
    ):
        """
        Args:
            model: AnalogyNet entrenado (en eval mode)
            transforms: Transforms de validación (torchvision)
            reference_embeddings: Base de referencia (N, D)
            reference_labels: Labels de referencia (N,)
            class_names: Nombres de clases
            device: torch device
            grain_config: Config de detección (parámetros morfológicos; sin U²-Net)
            k: Vecinos para k-NN
        """
        self.model = model
        self.model.eval()
        self.transforms = transforms
        self.device = device
        self.k = k

        # Clasificador k-NN con threshold de rechazo OOD
        from src.inference.knn_classifier import KNNClassifier
        dist_threshold = (grain_config or {}).get("distance_threshold", 0.0)
        self.knn = KNNClassifier(
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            class_names=class_names,
            k=k,
            distance_threshold=dist_threshold,
            metric_type=metric_type,
        )
        self.slice_classifier = slice_classifier
        self.class_names = class_names

        # Detector de granos (lazy-loaded)
        self._grain_config = grain_config or {}
        self.max_load_side = int(self._grain_config.get("max_load_side", 1536))
        self._localization_backend = resolve_localization_detector(self._grain_config)

        self._detector = self._create_localization_detector(
            self._grain_config,
            model=self.model,
            transforms=self.transforms,
            device=self.device,
            slice_classifier=slice_classifier,
        )

        # Parámetros de crop (deben coincidir con el entrenamiento)
        self.crop_padding = self._grain_config.get("crop_padding", 0.2)
        self.crop_size = self._grain_config.get("crop_size", 256)
        self.use_mask = self._grain_config.get("use_mask", True)
        self.masked_pooling = bool(self._grain_config.get("masked_pooling", True))
        self.mask_bg_mode = str(
            self._grain_config.get("val_mask_bg_mode")
            or self._grain_config.get("mask_bg_mode")
            or "imagenet_neutral"
        )
        self.nms_iou_threshold = float(self._grain_config.get("nms_iou_threshold", 0.5))
        self.drop_low_confidence_detections = bool(
            self._grain_config.get("drop_low_confidence_detections", False)
        )

        from src.grain_detection.detection_quality import resolve_quality_config

        self._quality_cfg = resolve_quality_config(self._grain_config)

        mode = "slice-aware" if slice_classifier is not None else f"kNN (k={k})"
        logger.info(
            f"[FullImageClassifier] Inicializado — "
            f"{len(class_names)} clases, {mode}, "
            f"localización={self._localization_backend}, "
            f"{len(reference_embeddings)} embeddings de referencia"
        )

    @property
    def detector(self):
        """Detector de localización (U²-Net por defecto)."""
        return self._detector

    @property
    def localization_backend(self) -> str:
        return self._localization_backend

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = "config.yaml",
        reference_path: Optional[str] = None,
        k: int = 5,
    ) -> "FullImageClassifier":
        """
        Factory method: carga modelo + referencia desde archivos.

        Args:
            checkpoint_path: Ruta al checkpoint del modelo entrenado
            config_path: Ruta al config.yaml
            reference_path: Ruta a reference_embeddings.pt (opcional,
                           busca en checkpoint o en el mismo directorio)
            k: Vecinos para k-NN

        Returns:
            FullImageClassifier listo para usar
        """
        from src.utils.config_utils import (
            load_config,
            get_transforms_from_config,
            merge_global_active_detector,
            resolve_run_config_path,
        )
        from src.utils.model_utils import load_model_from_checkpoint
        from src.inference.slice_classifier import load_slice_classifier_from_run

        config_path = resolve_run_config_path(checkpoint_path, config_path)
        config = load_config(config_path)
        config = merge_global_active_detector(config, "config.yaml")
        device = torch.device(
            "cuda" if torch.cuda.is_available() and config.get("use_gpu", True)
            else "cpu"
        )

        # Cargar modelo
        model = load_model_from_checkpoint(
            checkpoint_path, config, device, eval_mode=True
        )

        # Transforms de validación
        transforms = get_transforms_from_config(config, mode="val")

        # Cargar embeddings de referencia
        ref_data = cls._load_reference(checkpoint_path, reference_path, device)
        reference_embeddings = ref_data["embeddings"]
        reference_labels = ref_data["labels"]
        class_names = ref_data["class_names"]

        grain_config = config.get("grain_detection", {})

        run_dir = Path(checkpoint_path).resolve().parent.parent
        embed_dim = config.get("model", {}).get("embedding_dim", 128)
        num_slices = config.get("training", {}).get("loss", {}).get("params", {}).get("num_slices", 4)
        slice_classifier = load_slice_classifier_from_run(run_dir, device, embed_dim, num_slices)

        return cls(
            model=model,
            transforms=transforms,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            class_names=class_names,
            device=device,
            grain_config=grain_config,
            k=k,
            metric_type=config.get("loss", {}).get("metric", "cosine"),
            slice_classifier=slice_classifier,
        )

    @staticmethod
    def _load_reference(
        checkpoint_path: str,
        reference_path: Optional[str],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Carga embeddings de referencia desde checkpoint o archivo separado."""
        # 1. Intentar archivo separado
        if reference_path and Path(reference_path).exists():
            data = torch.load(reference_path, map_location="cpu", weights_only=False)
            embs = data["embeddings"].numpy() if torch.is_tensor(data["embeddings"]) else data["embeddings"]
            labels = data["labels"].numpy() if torch.is_tensor(data["labels"]) else data["labels"]
            return {
                "embeddings": embs,
                "labels": labels,
                "class_names": data["class_names"],
            }

        # 2. Buscar en el mismo directorio del checkpoint y en el directorio padre
        #    (estructura: runs/XXX/checkpoints/best_model.pth, refs en runs/XXX/)
        ckpt_dir = Path(checkpoint_path).parent
        search_dirs = [ckpt_dir, ckpt_dir.parent]
        for search_dir in search_dirs:
            ref_file = search_dir / "reference_embeddings.pt"
            cent_file = search_dir / "class_centroids.pt"
            slice_file = search_dir / "slice_representatives.pt"
            if ref_file.exists() and (cent_file.exists() or slice_file.exists()):
                ref_data = torch.load(ref_file, map_location="cpu", weights_only=False)
                if cent_file.exists():
                    meta = torch.load(cent_file, map_location="cpu", weights_only=False)
                    class_names = meta["class_names"]
                else:
                    slice_data = torch.load(slice_file, map_location="cpu", weights_only=False)
                    class_names = slice_data["class_names"]
                embs = ref_data["embeddings"].numpy() if torch.is_tensor(ref_data["embeddings"]) else ref_data["embeddings"]
                labels = ref_data["labels"].numpy() if torch.is_tensor(ref_data["labels"]) else ref_data["labels"]
                return {
                    "embeddings": embs,
                    "labels": labels,
                    "class_names": class_names,
                }

        # 3. Intentar desde dentro del checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "reference_embeddings" in ckpt:
            embs = ckpt["reference_embeddings"]
            if torch.is_tensor(embs):
                embs = embs.numpy()
            labels = ckpt["reference_labels"]
            if torch.is_tensor(labels):
                labels = labels.numpy()
            return {
                "embeddings": embs,
                "labels": labels,
                "class_names": ckpt["class_names"],
            }

        raise FileNotFoundError(
            f"No se encontraron embeddings de referencia.\n"
            f"Buscado en: {reference_path}, {ckpt_dir}, y dentro del checkpoint.\n"
            f"Ejecuta post_training.py primero para generar reference_embeddings.pt"
        )

    def _classify_embedding(self, emb_1d: np.ndarray) -> Dict[str, Any]:
        """Classify one embedding: slice-aware if available, else kNN."""
        if self.slice_classifier is not None:
            emb_t = torch.tensor(emb_1d, dtype=torch.float32).unsqueeze(0)
            preds, scores, class_scores = self.slice_classifier.predict(emb_t)
            pred_idx = int(preds[0].item())
            confidence = float(scores[0].item())
            cs = class_scores[0].detach().cpu().numpy()
            distribution = {
                self.class_names[i]: float(cs[i]) for i in range(len(self.class_names))
            }
            dist_thr = float(self._grain_config.get("distance_threshold", 0.0))
            min_conf = float(self._grain_config.get("min_classifier_confidence", 0.35))
            if dist_thr > 0:
                min_conf = max(min_conf, 1.0 - dist_thr)
            rejected = confidence < min_conf
            return {
                "predicted_class": self.class_names[pred_idx],
                "predicted_label": pred_idx,
                "confidence": confidence,
                "mean_distance": 1.0 - confidence,
                "class_distribution": distribution,
                "rejected": rejected,
                "display_class": "UNKNOWN" if rejected else self.class_names[pred_idx],
            }

        emb_2d = np.expand_dims(emb_1d, axis=0)
        prediction = self.knn.predict(
            emb_2d, return_distances=True, return_neighbors=False
        )
        return {
            "predicted_class": prediction["predicted_class"],
            "predicted_label": prediction["predicted_label"],
            "confidence": prediction["confidence"],
            "mean_distance": prediction["mean_distance"],
            "class_distribution": self.knn.get_class_distribution(emb_2d),
            "rejected": bool(prediction.get("rejected", False)),
            "display_class": (
                "UNKNOWN"
                if prediction.get("rejected")
                else prediction["predicted_class"]
            ),
        }

    # =========================================================================
    # Pipeline principal
    # =========================================================================

    def classify_full_image(
        self,
        image_or_path,
        return_crops: bool = False,
        discovery_mode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Pipeline completo: detectar granos → clasificar cada uno.

        Args:
            image_or_path: np.ndarray BGR o ruta a imagen
            return_crops: Si True, incluye los crops en los resultados
            discovery_mode: Multi-umbral + NMS (más recall, más FP). Si None, usa
                            ``grain_detection.discovery_mode`` (default False).

        Returns:
            Dict con:
                - grains: lista de dicts por grano detectado
                    {bbox, contour, saliency, predicted_class, confidence,
                     class_distribution, embedding, crop?}
                - summary: {total_grains, per_class_counts, elapsed_ms}
                - image_shape: (H, W)
        """
        t0 = time.perf_counter()

        if discovery_mode is None:
            discovery_mode = bool(self._grain_config.get("discovery_mode", False))

        # Cargar imagen
        if isinstance(image_or_path, (str, Path)):
            image = cv2.imread(str(image_or_path))
            if image is None:
                raise IOError(f"No se pudo leer: {image_or_path}")
        else:
            image = image_or_path

        # 3.1 Explicit Grayscale detection
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            logger.info("[FullImage] Converted single-channel grayscale to BGR")
        elif image.shape[2] == 3:
            b, g, r = cv2.split(image)
            if np.allclose(b, g, atol=3) and np.allclose(g, r, atol=3):
                logger.info("[FullImage] Detected 3-channel grayscale input")
                
        # Reducir resolución SÓLO para detección (ahorra RAM/cálculo)
        # Los crops se extraerán de la imagen en resolución original
        orig_image = image
        scale = 1.0
        h, w = image.shape[:2]
        longest = max(h, w)
        if self.max_load_side > 0 and longest > self.max_load_side:
            scale = self.max_load_side / float(longest)
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
            logger.info(f"[FullImage] Resized for detection from {h}x{w} to {image.shape[0]}x{image.shape[1]}")

        # 1. Localizar granos (U²-Net por defecto; discovery_mode = multi-umbral opcional)
        t1 = time.perf_counter()
        if discovery_mode and hasattr(self.detector, "detect_exhaustive"):
            saliency_map, detected_grains = self.detector.detect_exhaustive(image)
        else:
            saliency_map, detected_grains = self.detector.detect(image)
        t_detect = (time.perf_counter() - t1) * 1000

        # Escalar coordenadas detectadas de vuelta a la resolución original
        if scale != 1.0:
            orig_h, orig_w = orig_image.shape[:2]
            inv = 1.0 / scale
            for grain in detected_grains:
                bx, by, bw, bh = grain.bbox
                grain.bbox = (
                    int(bx * inv),
                    int(by * inv),
                    max(1, int(bw * inv)),
                    max(1, int(bh * inv)),
                )
                if grain.classification_bbox is not None:
                    cx, cy, cw, ch = grain.classification_bbox
                    grain.classification_bbox = (
                        int(cx * inv),
                        int(cy * inv),
                        max(1, int(cw * inv)),
                        max(1, int(ch * inv)),
                    )
                if grain.contour is not None:
                    grain.contour = (
                        np.asarray(grain.contour, dtype=np.float32) * inv
                    ).astype(np.int32)
                gcx, gcy = grain.centroid
                grain.centroid = (int(gcx * inv), int(gcy * inv))
            saliency_map = cv2.resize(saliency_map, (orig_w, orig_h))
            image = orig_image  # Volver a usar la imagen original para extraer los crops
            h, w = orig_h, orig_w

        logger.info(
            f"[FullImage] Detección: {len(detected_grains)} granos en {t_detect:.0f}ms"
        )

        from src.grain_detection.grain_crop_utils import nms_detected_grains
        from src.grain_detection.detection_quality import (
            filter_detections,
            split_oversized_detections,
        )

        rejected_detections: List[Dict[str, Any]] = []
        if saliency_map is not None and self._quality_cfg.get("split_multi_peak_enabled"):
            before = len(detected_grains)
            detected_grains = split_oversized_detections(
                detected_grains, saliency_map, self._quality_cfg
            )
            if len(detected_grains) != before:
                logger.info(
                    "[FullImage] Multi-peak split: %d → %d",
                    before,
                    len(detected_grains),
                )

        if self.nms_iou_threshold > 0 and len(detected_grains) > 1:
            before = len(detected_grains)
            detected_grains = nms_detected_grains(
                detected_grains, iou_threshold=self.nms_iou_threshold
            )
            if len(detected_grains) < before:
                logger.info(
                    f"[FullImage] NMS: {before} → {len(detected_grains)} "
                    f"(iou<{self.nms_iou_threshold})"
                )

        detected_grains, quality_rejected = filter_detections(
            detected_grains,
            (h, w),
            self._quality_cfg,
            saliency=saliency_map,
            crop_padding=self.crop_padding,
        )
        for grain, reason in quality_rejected:
            rejected_detections.append(
                {
                    "bbox": grain.bbox,
                    "classification_bbox": grain.crop_bbox(),
                    "contour": grain.contour,
                    "saliency": grain.saliency_prob,
                    "rejection_reason": reason,
                }
            )
        if quality_rejected:
            logger.info(
                "[FullImage] Quality filter: %d rejected (%s)",
                len(quality_rejected),
                ", ".join(sorted({r for _, r in quality_rejected})),
            )

        if not detected_grains:
            return {
                "grains": [],
                "rejected_detections": rejected_detections,
                "summary": {
                    "total_grains": 0,
                    "rejected_detections": len(rejected_detections),
                    "per_class_counts": {},
                    "elapsed_ms": (time.perf_counter() - t0) * 1000,
                },
                "image_shape": (h, w),
                "saliency_map": saliency_map,
            }

        # 2. Recortar, pre-procesar e inferir en batch (O(1) GPU)
        t2 = time.perf_counter()
        grain_results = []
        tensor_list = []
        mask_tensor_list = []
        crop_bgr_list = []
        valid_grains = []

        # 2.1. Preparar lista de tensores (solo granos con crop válido)
        for grain in detected_grains:
            from src.grain_detection.grain_crop_utils import prepare_grain_crop_and_mask

            crop_bgr, mask_crop = prepare_grain_crop_and_mask(
                image,
                grain,
                crop_padding=self.crop_padding,
                use_mask=self.use_mask,
                mask_bg_mode=self.mask_bg_mode,
            )
            if crop_bgr is None:
                continue
            valid_grains.append(grain)
            if mask_crop is not None:
                mask_float = torch.from_numpy(mask_crop.astype(np.float32) / 255.0)
            else:
                mask_float = torch.ones(self.crop_size, self.crop_size, dtype=torch.float32)
                
            mask_tensor_list.append(mask_float)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

            # Ruta unificada: numpy directo a Albumentations (igual que entrenamiento)
            if self.transforms:
                mask_uint8 = (mask_float.numpy() * 255).astype(np.uint8) if self.use_mask else None
                if mask_uint8 is not None and hasattr(self.transforms, 'apply_with_mask'):
                    tensor, mask_t = self.transforms.apply_with_mask(crop_rgb, mask_uint8)
                    mask_tensor_list[-1] = mask_t
                elif mask_uint8 is not None and hasattr(self.transforms, '__call__'):
                    tensor = self.transforms(crop_rgb, mask=mask_uint8)
                    if isinstance(tensor, tuple):
                        tensor, mask_t = tensor
                        mask_tensor_list[-1] = mask_t
                    # Si devuelve solo tensor, la máscara ya está en mask_tensor_list
                else:
                    tensor = self.transforms(crop_rgb)
            else:
                from torchvision import transforms as T
                tensor = T.ToTensor()(crop_rgb)
            
            tensor_list.append(tensor)
            if return_crops:
                crop_bgr_list.append(crop_bgr)

        # 2.2. Inferencia paralelizada en batches seguros
        batch_size = 64
        embeddings_list = []
        for i in range(0, len(tensor_list), batch_size):
            batch_chunk = torch.stack(tensor_list[i:i+batch_size]).to(self.device)
            mask_chunk = torch.stack(mask_tensor_list[i:i+batch_size]).to(self.device)
            with torch.no_grad():
                emb_chunk = self.model(
                    batch_chunk,
                    mask=mask_chunk if self.masked_pooling else None,
                ).cpu().numpy()
                embeddings_list.append(emb_chunk)
        
        batch_embeddings = np.vstack(embeddings_list)

        if len(valid_grains) != batch_embeddings.shape[0]:
            logger.warning(
                "[FullImage] Desalineación crops/embeddings: %d vs %d",
                len(valid_grains),
                batch_embeddings.shape[0],
            )

        # 2.3. Clasificación e interconexión diccionarios (O(N) CPU)
        for i, grain in enumerate(valid_grains):
            emb_1d = batch_embeddings[i]
            emb_2d = np.expand_dims(emb_1d, axis=0)

            pred_info = self._classify_embedding(emb_1d)

            display_cls = pred_info.get("display_class", pred_info["predicted_class"])
            rejected_cls = bool(pred_info.get("rejected", False))
            if rejected_cls and self.drop_low_confidence_detections:
                logger.debug(
                    "[FullImage] Grano descartado (baja confianza/OOD): "
                    f"dist={pred_info.get('mean_distance', 0):.3f}"
                )
                continue

            result = {
                "index": grain.index,
                "bbox": grain.bbox,
                "classification_bbox": grain.crop_bbox(),
                "contour": grain.contour,
                "saliency": grain.saliency_prob,
                "predicted_class": display_cls,
                "raw_predicted_class": pred_info["predicted_class"],
                "predicted_label": pred_info["predicted_label"],
                "confidence": pred_info["confidence"],
                "mean_distance": pred_info["mean_distance"],
                "class_distribution": pred_info["class_distribution"],
                "embedding": emb_1d,
                "rejected_by_classifier": rejected_cls,
            }
            if return_crops:
                result["crop"] = crop_bgr_list[i]
                result["tensor"] = tensor_list[i]

            grain_results.append(result)

        t_classify = (time.perf_counter() - t2) * 1000

        # 3. Resumen
        per_class_counts = {}
        for gr in grain_results:
            cls = gr["predicted_class"]
            per_class_counts[cls] = per_class_counts.get(cls, 0) + 1

        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(
            f"[FullImage] Clasificación: {len(grain_results)} granos en {t_classify:.0f}ms "
            f"(total {elapsed:.0f}ms)"
        )

        return {
            "grains": grain_results,
            "rejected_detections": rejected_detections,
            "summary": {
                "total_grains": len(grain_results),
                "rejected_detections": len(rejected_detections),
                "per_class_counts": per_class_counts,
                "elapsed_ms": elapsed,
                "detect_ms": t_detect,
                "classify_ms": t_classify,
            },
            "image_shape": (h, w),
            "saliency_map": saliency_map,
        }

    def _extract_crop(self, image: np.ndarray, grain) -> np.ndarray:
        """Legacy wrapper — prefer grain_crop_utils.prepare_grain_crop_and_mask."""
        from src.grain_detection.grain_crop_utils import extract_grain_crop

        return extract_grain_crop(image, grain, crop_padding=self.crop_padding)

    def _apply_contour_mask(self, crop: np.ndarray, image: np.ndarray, grain) -> tuple:
        """Legacy wrapper — prefer grain_crop_utils.prepare_grain_crop_and_mask."""
        from src.grain_detection.grain_crop_utils import apply_mask_bg, generate_grain_mask

        mask_crop = generate_grain_mask(grain, image.shape[:2], crop_padding=self.crop_padding)
        crop = apply_mask_bg(crop, mask_crop, self.mask_bg_mode)
        return crop, mask_crop

    # =========================================================================
    # Visualización
    # =========================================================================

    # Paleta de colores por clase (cíclica)
    CLASS_COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128),
    ]

    def draw_results(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        draw_contours: bool = True,
        draw_labels: bool = True,
        draw_confidence: bool = True,
        font_scale: float = 0.6,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Dibuja resultados de clasificación sobre la imagen.

        Args:
            image: Imagen BGR original
            results: Output de classify_full_image
            draw_contours: Dibujar contornos de granos
            draw_labels: Dibujar etiquetas de clase
            draw_confidence: Incluir confianza en la etiqueta

        Returns:
            Imagen anotada (copia)
        """
        annotated = image.copy()
        grains = results.get("grains", [])

        # Mapear clases a colores
        class_color_map = {}
        for i, name in enumerate(self.class_names):
            class_color_map[name] = self.CLASS_COLORS[i % len(self.CLASS_COLORS)]

        for grain in grains:
            cls = grain["predicted_class"]
            conf = grain["confidence"]
            color = class_color_map.get(cls, (255, 255, 255))
            bx, by, bw, bh = grain["bbox"]

            # Contorno
            if draw_contours and grain.get("contour") is not None:
                contour = grain["contour"]
                if len(contour) >= 3:
                    pts = contour.reshape(-1, 1, 2).astype(np.int32)
                    cv2.drawContours(annotated, [pts], -1, color, thickness)

            # BBox
            cv2.rectangle(
                annotated, (bx, by), (bx + bw, by + bh), color, thickness
            )

            # Label
            if draw_labels:
                label = cls
                if draw_confidence:
                    label = f"{cls} ({conf:.0%})"

                label_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                lw, lh = label_size

                # Fondo del label
                cv2.rectangle(
                    annotated,
                    (bx, by - lh - baseline - 4),
                    (bx + lw + 4, by),
                    color, -1,
                )
                cv2.putText(
                    annotated, label,
                    (bx + 2, by - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA,
                )

        return annotated

    def generate_heatmap(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Genera heatmap de confianza sobre la imagen.

        Cada grano se colorea según la confianza de clasificación:
        - Verde: alta confianza (>80%)
        - Amarillo: media (50-80%)
        - Rojo: baja (<50%)

        Args:
            image: Imagen BGR original
            results: Output de classify_full_image
            alpha: Transparencia del heatmap (0-1)

        Returns:
            Imagen con heatmap overlay
        """
        overlay = image.copy()
        grains = results.get("grains", [])

        for grain in grains:
            conf = grain["confidence"]
            contour = grain.get("contour")
            bx, by, bw, bh = grain["bbox"]

            # Color según confianza: verde(alto) → amarillo(medio) → rojo(bajo)
            if conf >= 0.8:
                heat_color = (0, 200, 0)     # Verde
            elif conf >= 0.5:
                heat_color = (0, 200, 200)   # Amarillo
            else:
                heat_color = (0, 0, 200)     # Rojo

            if contour is not None and len(contour) >= 3:
                pts = contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], heat_color)
            else:
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), heat_color, -1)

        # Blend
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return result

    def generate_class_heatmap(
        self,
        image: np.ndarray,
        results: Dict[str, Any],
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Genera heatmap coloreado por clase predicha.

        Cada grano se rellena con el color de su clase.

        Args:
            image: Imagen BGR original
            results: Output de classify_full_image
            alpha: Transparencia

        Returns:
            Imagen con class-heatmap overlay
        """
        overlay = image.copy()
        grains = results.get("grains", [])

        class_color_map = {}
        for i, name in enumerate(self.class_names):
            class_color_map[name] = self.CLASS_COLORS[i % len(self.CLASS_COLORS)]

        for grain in grains:
            cls = grain["predicted_class"]
            color = class_color_map.get(cls, (128, 128, 128))
            contour = grain.get("contour")
            bx, by, bw, bh = grain["bbox"]

            if contour is not None and len(contour) >= 3:
                pts = contour.reshape(-1, 1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], color)
            else:
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, -1)

        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return result
