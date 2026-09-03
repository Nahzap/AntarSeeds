"""
InferenceTab - Tab de inferencia con soporte para carpetas y reportes de ingeniería.
Permite inferencia en imágenes individuales o carpetas completas con generación de reportes.
"""

from pathlib import Path
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QTextEdit, QFileDialog, QMessageBox,
    QComboBox, QSpinBox, QFormLayout, QProgressBar, QCheckBox, QDoubleSpinBox,
    QScrollArea
)
from PyQt5.QtCore import QThread, pyqtSignal
import logging
from typing import Dict, Any, List

from lib.styles import Styles, Colors
from lib.active_detector_widget import ActiveDetectorWidget
from src.utils.logging_utils import setup_logger
from src.utils.run_artifact_utils import check_slice_inference_artifacts

logger = logging.getLogger(__name__)


class InferenceWorker(QThread):
    """Worker thread para inferencia en carpeta usando detección de granos"""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, int)
    detailed_progress = pyqtSignal(object)
    
    def __init__(self, checkpoint, folder, database, k, noise_config: Dict[str, Any]):
        super().__init__()
        self.checkpoint = checkpoint
        self.folder = folder
        self.database = database
        self.k = k
        self.noise_config = noise_config or {}

    def _background_correct_image(self, image_bgr, blur_kernel: int):
        """
        Flat-field-like correction using heavy blur background estimate.
        Kept local to inference tab to avoid changing core detector.
        """
        import cv2
        import numpy as np

        kernel = max(3, int(blur_kernel))
        if kernel % 2 == 0:
            kernel += 1

        img = image_bgr.astype(np.float32)
        background = cv2.GaussianBlur(img, (kernel, kernel), 0)
        # Fix B&N / Domain Shift: Promedio por canales para no desaturar las texturas
        corrected = (img / (background + 1.0)) * np.mean(background, axis=(0, 1))
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return corrected

    def _postfilter_grains(
        self,
        grains: List[Dict[str, Any]],
        image_shape,
        confidence_min: float
    ):
        """
        Adaptive post-filtering: Removes static-like noise but dynamically relaxes 
        thresholds based on image statistics to prevent NO_GRAINS_DETECTED on valid external samples.
        """
        import cv2

        if not grains:
            return [], []

        h, w = image_shape
        image_area = max(1.0, float(h * w))
        
        # Base (strict) thresholds from UI
        base_min_w = int(self.noise_config.get("min_bbox_w", 20))
        base_min_h = int(self.noise_config.get("min_bbox_h", 20))
        base_min_area = float(self.noise_config.get("min_contour_area", 1500.0))
        base_min_ratio = float(self.noise_config.get("min_area_ratio", 0.0002))
        base_min_sal = float(self.noise_config.get("min_saliency", 0.35))
        exclude_unknown = bool(self.noise_config.get("exclude_unknown", True))

        # Adaptive logic: calculate max stats from current raw grains
        max_sal = max((float(g.get('saliency', 0.0)) for g in grains), default=0.0)
        
        # Relax thresholds to a dynamic minimum (e.g. 50% of the max saliency found)
        # with an absolute floor to avoid complete noise.
        min_saliency = max(0.15, min(base_min_sal, max_sal * 0.50))
        min_confidence = max(0.20, confidence_min * 0.50)
        min_contour_area = max(200.0, base_min_area * 0.30)
        min_bbox_w = max(10, int(base_min_w * 0.50))
        min_bbox_h = max(10, int(base_min_h * 0.50))
        min_area_ratio = max(0.00005, base_min_ratio * 0.25)

        kept = []
        dropped = []
        for grain in grains:
            bx, by, bw, bh = grain.get('bbox', (0, 0, 0, 0))
            contour = grain.get('contour')
            contour_area = float(max(0, bw) * max(0, bh))
            if contour is not None and len(contour) >= 3:
                contour_area = float(cv2.contourArea(contour.astype('int32')))
            area_ratio = contour_area / image_area
            sal = float(grain.get('saliency', 0.0))
            conf = float(grain.get('confidence', 0.0))
            pred_class = str(grain.get('predicted_class', ''))

            reason = None
            if bw < min_bbox_w or bh < min_bbox_h:
                reason = f"small_bbox (w={bw},h={bh} < {min_bbox_w})"
            elif contour_area < min_contour_area:
                reason = f"small_area ({contour_area:.1f} < {min_contour_area:.1f})"
            elif area_ratio < min_area_ratio:
                reason = f"low_area_ratio ({area_ratio:.5f} < {min_area_ratio:.5f})"
            elif sal < min_saliency:
                reason = f"low_saliency ({sal:.3f} < {min_saliency:.3f})"
            elif conf < min_confidence:
                reason = f"low_confidence ({conf:.3f} < {min_confidence:.3f})"
            elif exclude_unknown and ("unknown" in pred_class.lower()):
                reason = "unknown_class"

            if reason is not None:
                dropped.append({
                    "grain_index": grain.get("index", -1),
                    "reason": reason,
                    "bbox": (bx, by, bw, bh),
                    "contour_area_px": contour_area,
                    "area_ratio": area_ratio,
                    "saliency": sal,
                    "confidence": conf,
                    "predicted_class": pred_class,
                })
            else:
                kept.append(grain)

        return kept, dropped
    
    def run(self):
        """Ejecuta inferencia con detección de granos por imagen"""
        try:
            from src.grain_detection.full_image_classifier import FullImageClassifier
            from src.utils.report_generator import InferenceReportGenerator
            import numpy as np
            import cv2
            import csv

            from src.utils.config_utils import resolve_run_config_path

            config_path = resolve_run_config_path(self.checkpoint, "config.yaml")
            logger.info("Inicializando FullImageClassifier (pipeline E2E: detectar + clasificar)...")
            classifier = FullImageClassifier.from_checkpoint(
                checkpoint_path=self.checkpoint,
                config_path=config_path,
                k=self.k,
            )
            cls_mode = (
                "slice-aware"
                if classifier.slice_classifier is not None
                else f"kNN (k={self.k})"
            )
            det_backend = getattr(classifier, "localization_backend", "?")
            det_label = type(classifier.detector).__name__
            strategy = classifier._grain_config.get("detection_strategy", "single_pass")
            logger.info(
                f"[E2E] Detector={det_backend} ({det_label}) | "
                f"strategy={strategy} | Clasificador={cls_mode}"
            )
            
            # Buscar todas las imágenes en la carpeta
            folder_path = Path(self.folder)
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_files = []
            
            for ext in valid_extensions:
                image_files.extend(folder_path.glob(f"*{ext}"))
                image_files.extend(folder_path.glob(f"*{ext.upper()}"))
            
            # Deduplicar (glob mayúsculas puede repetir en Windows)
            image_files = sorted(set(image_files))
            
            if not image_files:
                raise ValueError("No se encontraron imágenes en la carpeta")
            
            # Crear carpeta de salida UNICA con timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = folder_path / f"inference_{timestamp}"
            output_dir.mkdir(exist_ok=True)
            report_output_dir = output_dir / "report"
            report_output_dir.mkdir(exist_ok=True)
            samples_dir = output_dir / "samples"
            samples_dir.mkdir(exist_ok=True)
            annotated_dir = output_dir / "annotated"
            annotated_dir.mkdir(exist_ok=True)

            runtime_logger = setup_logger(
                name="inference_runtime",
                log_level=logging.INFO,
                console_output=True,
                file_output=True
            )
            runtime_logger.info(
                f"[RUN] Inicio inferencia carpeta={folder_path} total_imagenes={len(image_files)} "
                f"checkpoint={self.checkpoint} k={self.k}"
            )
            
            # Procesar cada imagen con detección de granos
            predictions = []
            all_grain_details = []
            dropped_grain_details = []
            visual_samples = []
            total = len(image_files)
            running_class_counts = {}  # conteo acumulado por clase
            running_total_grains = 0
            running_no_grains = 0
            
            # Milestones cada 5% para guardar imágenes de muestra
            sample_pcts = set(range(0, 101, 5))  # {0, 5, 10, ..., 100}
            saved_samples = set()
            max_visual_samples = 80
            save_all_annotated = total <= 300
            
            for idx, img_path in enumerate(image_files, 1):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    
                    if len(img.shape) == 2:
                        image_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.shape[2] == 4:
                        image_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    else:
                        image_bgr = img

                    # Preprocesamiento local SOLO para inferencia en tab:
                    # corrige fondo/viñeteo y reduce ruido óptico de baja escala.
                    if self.noise_config.get("enable_background_correction", True):
                        blur_kernel = int(self.noise_config.get("background_blur_kernel", 61))
                        image_for_detection = self._background_correct_image(image_bgr, blur_kernel)
                    else:
                        image_for_detection = image_bgr

                    # Pipeline: detectar granos → recortar → clasificar cada grano
                    results = classifier.classify_full_image(image_for_detection)
                    grains_raw = results.get("grains", [])
                    summary = results.get("summary", {})

                    confidence_min = float(self.noise_config.get("min_confidence", 0.45))
                    if self.noise_config.get("enable_postfilter", True):
                        grains, dropped = self._postfilter_grains(
                            grains_raw,
                            image_for_detection.shape[:2],
                            confidence_min=confidence_min
                        )
                    else:
                        grains = grains_raw
                        dropped = []

                    filtered_results = dict(results)
                    filtered_results["grains"] = grains
                    per_class_counts = {}
                    for g in grains:
                        cls_name = g.get("predicted_class", "UNKNOWN")
                        per_class_counts[cls_name] = per_class_counts.get(cls_name, 0) + 1
                    filtered_results["summary"] = dict(summary)
                    filtered_results["summary"]["total_grains"] = len(grains)
                    filtered_results["summary"]["per_class_counts"] = per_class_counts

                    image_h, image_w = image_bgr.shape[:2]
                    image_area = float(image_w * image_h)
                    
                    if not grains:
                        logger.warning(f"No se detectaron granos en {img_path.name}")
                        running_no_grains += 1
                        coverage_pct = 0.0
                        predictions.append({
                            'image_path': str(img_path),
                            'predicted_class': 'NO_GRAINS_DETECTED',
                            'confidence': 0.0,
                            'mean_distance': 0.0,
                            'n_grains': 0,
                            'class_distribution': {},
                            'image_width': image_w,
                            'image_height': image_h,
                            'image_area_px': image_area,
                            'grain_area_sum_px': 0.0,
                            'grain_coverage_pct': coverage_pct,
                            'detect_ms': float(summary.get('detect_ms', 0.0)),
                            'classify_ms': float(summary.get('classify_ms', 0.0)),
                            'elapsed_ms': float(summary.get('elapsed_ms', 0.0)),
                            'n_grains_raw': len(grains_raw),
                            'n_grains_dropped': len(dropped),
                        })
                        runtime_logger.info(
                            f"[{idx}/{total}] image={img_path.name} grains_kept=0 grains_raw={len(grains_raw)} "
                            f"dropped={len(dropped)} dominant=NO_GRAINS_DETECTED "
                            f"coverage_pct=0.00 detect_ms={summary.get('detect_ms', 0.0):.1f} "
                            f"classify_ms={summary.get('classify_ms', 0.0):.1f}"
                        )

                        if len(visual_samples) < max_visual_samples:
                            try:
                                annotated = classifier.draw_results(image_bgr, filtered_results)
                                sample_name = f"sample_{idx:04d}_NO_GRAINS_{img_path.stem}.png"
                                sample_path = samples_dir / sample_name
                                cv2.imwrite(str(sample_path), annotated)
                                visual_samples.append({
                                    'image_name': img_path.name,
                                    'image_path': str(img_path),
                                    'annotated_path': str(sample_path),
                                    'reason': 'NO_GRAINS',
                                    'n_grains': 0,
                                    'dominant_class': 'NO_GRAINS_DETECTED',
                                    'confidence': 0.0,
                                    'grain_coverage_pct': 0.0
                                })
                            except Exception as e_draw:
                                logger.warning(f"Error guardando muestra NO_GRAINS {img_path.name}: {e_draw}")

                        for d in dropped:
                            dropped_grain_details.append({
                                'image_name': img_path.name,
                                'image_path': str(img_path),
                                **d
                            })

                        self.progress.emit(idx, total)
                        self.detailed_progress.emit({
                            'idx': idx, 'total': total,
                            'image_name': img_path.name,
                            'grains_this': 0,
                            'total_grains': running_total_grains,
                            'class_counts': dict(running_class_counts),
                        })
                        continue
                    
                    # Agregar detalles por grano
                    grain_area_sum = 0.0
                    for grain in grains:
                        bx, by, bw, bh = grain['bbox']
                        bbox_area = float(max(0, bw) * max(0, bh))
                        contour = grain.get('contour')
                        contour_area = bbox_area
                        if contour is not None and len(contour) >= 3:
                            contour_area = float(cv2.contourArea(contour.astype(np.int32)))
                        area_ratio = float(contour_area / image_area) if image_area > 0 else 0.0
                        grain_area_sum += contour_area

                        all_grain_details.append({
                            'image_path': str(img_path),
                            'image_name': img_path.name,
                            'image_width': image_w,
                            'image_height': image_h,
                            'image_area_px': image_area,
                            'grain_index': grain['index'],
                            'predicted_class': grain['predicted_class'],
                            'confidence': grain['confidence'],
                            'mean_distance': grain['mean_distance'],
                            'bbox': grain['bbox'],
                            'bbox_x': bx,
                            'bbox_y': by,
                            'bbox_w': bw,
                            'bbox_h': bh,
                            'bbox_area_px': bbox_area,
                            'contour_area_px': contour_area,
                            'area_ratio_vs_image': area_ratio,
                            'saliency': float(grain.get('saliency', 0.0)),
                        })
                    
                    # Predicción por imagen: voto mayoritario de granos detectados
                    grain_classes = [g['predicted_class'] for g in grains]
                    unique, counts = np.unique(grain_classes, return_counts=True)
                    dominant_class = unique[np.argmax(counts)]
                    dominant_conf = np.max(counts) / len(grains)
                    
                    # Distancia media de todos los granos
                    mean_dist = np.mean([g['mean_distance'] for g in grains])
                    coverage_pct = float((grain_area_sum / image_area) * 100.0) if image_area > 0 else 0.0
                    
                    # Distribución de clases en la imagen
                    class_dist = {}
                    for cls, cnt in zip(unique, counts):
                        class_dist[cls] = float(cnt / len(grains))
                    
                    predictions.append({
                        'image_path': str(img_path),
                        'predicted_class': dominant_class,
                        'confidence': float(dominant_conf),
                        'mean_distance': float(mean_dist),
                        'n_grains': len(grains),
                        'class_distribution': class_dist,
                        'image_width': image_w,
                        'image_height': image_h,
                        'image_area_px': image_area,
                        'grain_area_sum_px': float(grain_area_sum),
                        'grain_coverage_pct': coverage_pct,
                        'detect_ms': float(summary.get('detect_ms', 0.0)),
                        'classify_ms': float(summary.get('classify_ms', 0.0)),
                        'elapsed_ms': float(summary.get('elapsed_ms', 0.0)),
                    })
                    
                    # Actualizar conteos acumulados
                    running_total_grains += len(grains)
                    for cls in grain_classes:
                        running_class_counts[cls] = running_class_counts.get(cls, 0) + 1

                    runtime_logger.info(
                        f"[{idx}/{total}] image={img_path.name} grains_kept={len(grains)} "
                        f"grains_raw={len(grains_raw)} dropped={len(dropped)} "
                        f"dominant={dominant_class} conf={dominant_conf:.3f} "
                        f"coverage_pct={coverage_pct:.2f} detect_ms={summary.get('detect_ms', 0.0):.1f} "
                        f"classify_ms={summary.get('classify_ms', 0.0):.1f}"
                    )
                    
                    # Guardar imagen anotada para trazabilidad completa (datasets pequeños/medianos)
                    if save_all_annotated:
                        try:
                            annotated_full = classifier.draw_results(image_bgr, filtered_results)
                            full_out_path = annotated_dir / f"annotated_{idx:04d}_{img_path.stem}.png"
                            cv2.imwrite(str(full_out_path), annotated_full)
                        except Exception as e_draw:
                            logger.warning(f"Error guardando anotada completa en {img_path.name}: {e_draw}")

                    # Guardar muestra anotada por hitos y casos críticos
                    pct = int((idx / total) * 100)
                    save_this = False
                    reason = ""
                    for p in sample_pcts:
                        if p not in saved_samples and pct >= p:
                            saved_samples.add(p)
                            save_this = True
                            reason = f"MILESTONE_{p}PCT"

                    if dominant_conf < 0.45:
                        save_this = True
                        reason = "LOW_CONFIDENCE"
                    if len(grains) >= 12:
                        save_this = True
                        reason = "HIGH_DENSITY"
                    
                    if save_this and len(visual_samples) < max_visual_samples:
                        try:
                            annotated = classifier.draw_results(image_bgr, filtered_results)
                            out_path = samples_dir / f"sample_{idx:04d}_{reason}_{img_path.stem}.png"
                            cv2.imwrite(str(out_path), annotated)
                            visual_samples.append({
                                'image_name': img_path.name,
                                'image_path': str(img_path),
                                'annotated_path': str(out_path),
                                'reason': reason,
                                'n_grains': int(len(grains)),
                                'dominant_class': dominant_class,
                                'confidence': float(dominant_conf),
                                'grain_coverage_pct': float(coverage_pct),
                            })
                        except Exception as e_draw:
                            logger.warning(f"Error dibujando resultados en {img_path.name}: {e_draw}")
                    
                    self.progress.emit(idx, total)
                    self.detailed_progress.emit({
                        'idx': idx, 'total': total,
                        'image_name': img_path.name,
                        'grains_this': len(grains),
                        'total_grains': running_total_grains,
                        'class_counts': dict(running_class_counts),
                    })
                    
                except Exception as e:
                    logger.warning(f"Error procesando {img_path}: {e}")
                    predictions.append({
                        'image_path': str(img_path),
                        'predicted_class': f'ERROR: {e}',
                        'confidence': 0.0,
                        'mean_distance': 0.0,
                        'n_grains': 0,
                        'class_distribution': {},
                    })
                    self.progress.emit(idx, total)
                    self.detailed_progress.emit({
                        'idx': idx, 'total': total,
                        'image_name': img_path.name,
                        'grains_this': 0,
                        'total_grains': running_total_grains,
                        'class_counts': dict(running_class_counts),
                    })
            
            # Generar reporte
            report_gen = InferenceReportGenerator(output_dir=str(report_output_dir))
            
            model_info = {
                'name': 'MeliVision E2E (ViT-dense + AnalogyNet)',
                'checkpoint': self.checkpoint,
                'detector_backend': getattr(classifier, 'localization_backend', ''),
                'classifier_mode': cls_mode,
                'detection_strategy': classifier._grain_config.get('detection_strategy', ''),
            }
            
            report_path = report_gen.generate_report(
                predictions=predictions,
                folder_path=str(folder_path),
                model_info=model_info,
                class_names=classifier.class_names,
                visual_samples=visual_samples
            )
            
            # Guardar detalle por grano como CSV extendido
            if all_grain_details:
                grain_csv = output_dir / "grain_details.csv"
                with open(grain_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'image_name', 'image_path', 'image_width', 'image_height', 'image_area_px',
                        'grain_index', 'predicted_class', 'confidence', 'mean_distance',
                        'saliency', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h',
                        'bbox_area_px', 'contour_area_px', 'area_ratio_vs_image'
                    ])
                    writer.writeheader()
                    for g in all_grain_details:
                        writer.writerow({
                            'image_name': g['image_name'],
                            'image_path': g['image_path'],
                            'image_width': g['image_width'],
                            'image_height': g['image_height'],
                            'image_area_px': f"{g['image_area_px']:.0f}",
                            'grain_index': g['grain_index'],
                            'predicted_class': g['predicted_class'],
                            'confidence': g['confidence'],
                            'mean_distance': f"{g['mean_distance']:.4f}",
                            'saliency': f"{g['saliency']:.4f}",
                            'bbox_x': g['bbox_x'],
                            'bbox_y': g['bbox_y'],
                            'bbox_w': g['bbox_w'],
                            'bbox_h': g['bbox_h'],
                            'bbox_area_px': f"{g['bbox_area_px']:.1f}",
                            'contour_area_px': f"{g['contour_area_px']:.1f}",
                            'area_ratio_vs_image': f"{g['area_ratio_vs_image']:.6f}",
                        })
                logger.info(f"Detalle por grano guardado en {grain_csv}")

            if dropped_grain_details:
                dropped_csv = output_dir / "dropped_grains.csv"
                with open(dropped_csv, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'image_name', 'image_path', 'grain_index', 'reason', 'bbox',
                        'contour_area_px', 'area_ratio', 'saliency', 'confidence',
                        'predicted_class'
                    ])
                    writer.writeheader()
                    for row in dropped_grain_details:
                        writer.writerow(row)
                logger.info(f"Detalle de granos descartados guardado en {dropped_csv}")

            # Guardar resumen por imagen con relación tamaño/detecciones
            image_summary_csv = output_dir / "image_summary.csv"
            with open(image_summary_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'image_path', 'predicted_class', 'confidence', 'mean_distance',
                    'n_grains', 'image_width', 'image_height', 'image_area_px',
                    'grain_area_sum_px', 'grain_coverage_pct', 'detect_ms',
                    'classify_ms', 'elapsed_ms'
                ])
                writer.writeheader()
                for p in predictions:
                    writer.writerow({
                        'image_path': p.get('image_path', ''),
                        'predicted_class': p.get('predicted_class', ''),
                        'confidence': f"{float(p.get('confidence', 0.0)):.4f}",
                        'mean_distance': f"{float(p.get('mean_distance', 0.0)):.4f}",
                        'n_grains': int(p.get('n_grains', 0)),
                        'image_width': int(p.get('image_width', 0)),
                        'image_height': int(p.get('image_height', 0)),
                        'image_area_px': f"{float(p.get('image_area_px', 0.0)):.0f}",
                        'grain_area_sum_px': f"{float(p.get('grain_area_sum_px', 0.0)):.1f}",
                        'grain_coverage_pct': f"{float(p.get('grain_coverage_pct', 0.0)):.4f}",
                        'detect_ms': f"{float(p.get('detect_ms', 0.0)):.2f}",
                        'classify_ms': f"{float(p.get('classify_ms', 0.0)):.2f}",
                        'elapsed_ms': f"{float(p.get('elapsed_ms', 0.0)):.2f}",
                    })
            logger.info(f"Resumen por imagen guardado en {image_summary_csv}")

            # Generar inspección rápida de casos dudosos
            suspicious = sorted(
                predictions,
                key=lambda x: (
                    0 if x.get('n_grains', 0) == 0 else 1,
                    float(x.get('confidence', 0.0)),
                    -float(x.get('mean_distance', 0.0))
                )
            )[:20]
            quick_md = output_dir / "quick_inspection.md"
            with open(quick_md, 'w', encoding='utf-8') as f:
                f.write("# Quick Inspection\n\n")
                f.write(f"- Total imagenes: {len(predictions)}\n")
                f.write(f"- Total granos: {len(all_grain_details)}\n")
                f.write(f"- Imagenes sin granos: {running_no_grains}\n\n")
                f.write("## Top casos a revisar\n\n")
                for i, item in enumerate(suspicious, 1):
                    f.write(
                        f"{i}. `{Path(item.get('image_path', '')).name}` | "
                        f"clase={item.get('predicted_class', '')} | "
                        f"conf={float(item.get('confidence', 0.0)):.3f} | "
                        f"dist={float(item.get('mean_distance', 0.0)):.4f} | "
                        f"granos={int(item.get('n_grains', 0))} | "
                        f"coverage={float(item.get('grain_coverage_pct', 0.0)):.2f}%\n"
                    )

            runtime_logger.info(
                f"[RUN] Fin inferencia total_imagenes={len(predictions)} "
                f"total_granos={len(all_grain_details)} imagenes_sin_granos={running_no_grains} "
                f"granos_descartados={len(dropped_grain_details)} "
                f"report={report_path} quick_inspection={quick_md}"
            )
            
            self.finished.emit({
                'report_path': report_path,
                'total_images': len(predictions),
                'total_grains': len(all_grain_details),
                'predictions': predictions,
                'output_dir': str(output_dir),
                'quick_inspection_path': str(quick_md),
            })
            
        except Exception as e:
            import traceback
            logger.error(f"Error en inferencia: {traceback.format_exc()}")
            self.error.emit(str(e))


class SingleImageInferenceWorker(QThread):
    """Worker thread para inferencia slice-aware en una imagen (FullImageClassifier)."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, checkpoint: str, image_path: str, k: int):
        super().__init__()
        self.checkpoint = checkpoint
        self.image_path = image_path
        self.k = k

    def run(self):
        try:
            from src.grain_detection.full_image_classifier import FullImageClassifier
            from src.utils.config_utils import resolve_run_config_path

            config_path = resolve_run_config_path(self.checkpoint, "config.yaml")
            logger.info("Single-image E2E: FullImageClassifier (detectar + clasificar)...")
            classifier = FullImageClassifier.from_checkpoint(
                checkpoint_path=self.checkpoint,
                config_path=config_path,
                k=self.k,
            )
            mode = (
                "slice-aware"
                if classifier.slice_classifier is not None
                else f"kNN (k={self.k})"
            )
            det_backend = getattr(classifier, "localization_backend", "?")
            strategy = classifier._grain_config.get("detection_strategy", "single_pass")
            logger.info(
                f"[E2E] Detector={det_backend} | strategy={strategy} | Clasificador={mode}"
            )
            results = classifier.classify_full_image(self.image_path)
            self.finished.emit({
                "classifier_mode": mode,
                "detector_backend": det_backend,
                "detection_strategy": strategy,
                "results": results,
                "image_path": self.image_path,
            })
        except Exception as e:
            import traceback
            logger.error("Error en inferencia single-image: %s", traceback.format_exc())
            self.error.emit(str(e))


class InferenceTab(QWidget):
    """Tab de inferencia con soporte para carpetas y reportes de ingeniería"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        # Wrap tab content in a scroll area so all controls stay editable
        # on smaller windows / high DPI setups.
        root_layout = QVBoxLayout()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content_widget = QWidget()
        layout = QVBoxLayout()
        
        # Clasificador
        model_group = QGroupBox("Clasificador (AnalogyNet)")
        model_layout = QFormLayout()
        
        checkpoint_layout = QHBoxLayout()
        self.checkpoint_input = QLineEdit()
        self.checkpoint_input.setPlaceholderText(
            "Checkpoint del clasificador (ej. runs/.../checkpoints/best_model.pth)"
        )
        checkpoint_btn = QPushButton("📁 Buscar")
        checkpoint_btn.clicked.connect(self.browse_checkpoint)
        checkpoint_layout.addWidget(self.checkpoint_input)
        checkpoint_layout.addWidget(checkpoint_btn)
        model_layout.addRow("Checkpoint:", checkpoint_layout)
        
        self.artifact_status_label = QLabel("")
        self.artifact_status_label.setWordWrap(True)
        self.artifact_status_label.setStyleSheet("color: #666; font-size: 9pt; padding: 2px 0;")
        model_layout.addRow("", self.artifact_status_label)
        self.checkpoint_input.textChanged.connect(self._update_checkpoint_artifact_status)
        self.checkpoint_input.textChanged.connect(self._update_pipeline_status)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Pipeline E2E: detector + clasificador
        pipeline_group = QGroupBox("Pipeline E2E (detección + clasificación)")
        pipeline_layout = QVBoxLayout()

        pipeline_note = QLabel(
            "Cada imagen de campo completo se procesa en dos fases: "
            "<b>1)</b> localización de granos (ViT-denso / U²-Net) y "
            "<b>2)</b> clasificación taxonómica (AnalogyNet slice-aware). "
            "Configure el detector activo y pulse <i>Cargar detector</i>."
        )
        pipeline_note.setWordWrap(True)
        pipeline_note.setStyleSheet("color: #555; font-size: 11px; margin-bottom: 4px;")
        pipeline_layout.addWidget(pipeline_note)

        self.detector_widget = ActiveDetectorWidget()
        self.detector_widget.detectorLoaded.connect(lambda _: self._update_pipeline_status())
        pipeline_layout.addWidget(self.detector_widget)

        self.pipeline_status_label = QLabel("Configure clasificador y detector para ver el pipeline.")
        self.pipeline_status_label.setWordWrap(True)
        self.pipeline_status_label.setStyleSheet(
            "color: #666; font-size: 9pt; padding: 4px 0; font-weight: bold;"
        )
        pipeline_layout.addWidget(self.pipeline_status_label)

        pipeline_group.setLayout(pipeline_layout)
        layout.addWidget(pipeline_group)
        
        # Modo de inferencia
        mode_group = QGroupBox("Modo de Inferencia")
        mode_layout = QVBoxLayout()
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Carpeta E2E (detectar + clasificar + reporte)",
            "Imagen individual E2E",
        ])
        self.mode_combo.currentIndexChanged.connect(self.toggle_inference_mode)
        mode_layout.addWidget(self.mode_combo)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Carpeta de inferencia
        self.folder_group = QGroupBox("Carpeta a Analizar")
        folder_layout = QFormLayout()
        
        folder_dir_layout = QHBoxLayout()
        self.folder_path = QLineEdit()
        self.folder_path.setPlaceholderText(
            "Directorio con imágenes de microscopía (campo completo, sin recortar)"
        )
        folder_browse = QPushButton("📁 Buscar")
        folder_browse.clicked.connect(self.browse_folder)
        folder_dir_layout.addWidget(self.folder_path)
        folder_dir_layout.addWidget(folder_browse)
        folder_layout.addRow("Carpeta:", folder_dir_layout)
        
        self.folder_group.setLayout(folder_layout)
        layout.addWidget(self.folder_group)
        
        # Imagen Individual
        self.image_group = QGroupBox("Imagen E2E (campo completo)")
        image_layout = QFormLayout()
        
        query_file_layout = QHBoxLayout()
        self.query_path = QLineEdit()
        self.query_path.setPlaceholderText("Ruta a imagen de microscopía (se detectan y clasifican granos)")
        query_browse = QPushButton("📁 Buscar")
        query_browse.clicked.connect(self.browse_query)
        query_file_layout.addWidget(self.query_path)
        query_file_layout.addWidget(query_browse)
        image_layout.addRow("Imagen:", query_file_layout)
        
        self.image_group.setLayout(image_layout)
        self.image_group.setVisible(False)
        layout.addWidget(self.image_group)
        
        # Base de Datos (Opcional)
        db_group = QGroupBox("Base de Datos de Referencia (Opcional)")
        db_layout = QFormLayout()
        
        db_dir_layout = QHBoxLayout()
        self.database_path = QLineEdit("")
        self.database_path.setPlaceholderText("Opcional: usa embeddings del checkpoint si vacío")
        db_browse = QPushButton("📁 Buscar")
        db_browse.clicked.connect(self.browse_database)
        db_dir_layout.addWidget(self.database_path)
        db_dir_layout.addWidget(db_browse)
        db_layout.addRow("Directorio:", db_dir_layout)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)
        
        # Opciones
        options_group = QGroupBox("Opciones")
        options_layout = QFormLayout()
        
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 50)
        self.k_spin.setValue(5)
        options_layout.addRow("k vecinos:", self.k_spin)
        self.k_spin.valueChanged.connect(self._update_pipeline_status)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Filtro de ruido óptico (SOLO inferencia tab)
        noise_group = QGroupBox("Filtro de Ruido Óptico Adaptativo")
        noise_layout = QFormLayout()

        self.enable_bg_correction = QCheckBox("Activar corrección de background")
        self.enable_bg_correction.setChecked(False)
        noise_layout.addRow("", self.enable_bg_correction)

        self.bg_blur_kernel = QSpinBox()
        self.bg_blur_kernel.setRange(15, 301)
        self.bg_blur_kernel.setSingleStep(2)
        self.bg_blur_kernel.setValue(61)
        noise_layout.addRow("Kernel blur background:", self.bg_blur_kernel)

        self.enable_postfilter = QCheckBox("Activar post-filtro adaptativo (dinámico por imagen)")
        self.enable_postfilter.setChecked(True)
        noise_layout.addRow("", self.enable_postfilter)

        self.min_bbox_w = QSpinBox()
        self.min_bbox_w.setRange(1, 500)
        self.min_bbox_w.setValue(8)
        noise_layout.addRow("Base Min bbox width (px):", self.min_bbox_w)

        self.min_bbox_h = QSpinBox()
        self.min_bbox_h.setRange(1, 500)
        self.min_bbox_h.setValue(8)
        noise_layout.addRow("Base Min bbox height (px):", self.min_bbox_h)

        self.min_contour_area = QSpinBox()
        self.min_contour_area.setRange(10, 200000)
        self.min_contour_area.setValue(100)
        noise_layout.addRow("Base Min contour area (px²):", self.min_contour_area)

        self.min_area_ratio = QDoubleSpinBox()
        self.min_area_ratio.setDecimals(5)
        self.min_area_ratio.setRange(0.0, 0.1)
        self.min_area_ratio.setSingleStep(0.0001)
        self.min_area_ratio.setValue(0.00020)
        noise_layout.addRow("Base Min area ratio:", self.min_area_ratio)

        self.min_saliency = QDoubleSpinBox()
        self.min_saliency.setDecimals(3)
        self.min_saliency.setRange(0.0, 1.0)
        self.min_saliency.setSingleStep(0.01)
        self.min_saliency.setValue(0.20)
        noise_layout.addRow("Base Min saliency:", self.min_saliency)

        self.min_confidence = QDoubleSpinBox()
        self.min_confidence.setDecimals(3)
        self.min_confidence.setRange(0.0, 1.0)
        self.min_confidence.setSingleStep(0.01)
        self.min_confidence.setValue(0.30)
        noise_layout.addRow("Base Min confidence:", self.min_confidence)

        self.exclude_unknown = QCheckBox("Excluir UNKNOWN del conteo final")
        self.exclude_unknown.setChecked(True)
        noise_layout.addRow("", self.exclude_unknown)

        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)
        
        # Botón Ejecutar
        self.infer_btn = QPushButton("Ejecutar pipeline E2E")
        self.infer_btn.setStyleSheet(Styles.BUTTON_PRIMARY)
        self.infer_btn.clicked.connect(self.run_inference)
        layout.addWidget(self.infer_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Resultados
        results_group = QGroupBox("Resultados")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(150)
        results_layout.addWidget(self.results_text)
        
        results_group.setLayout(results_layout)
        # Darle el factor de stretch (1) al grupo de resultados para que ocupe todo el espacio libre
        layout.addWidget(results_group, 1)
        
        content_widget.setLayout(layout)
        scroll.setWidget(content_widget)
        root_layout.addWidget(scroll)
        self.setLayout(root_layout)
        self._update_pipeline_status()
    
    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def _get_pipeline_summary(self) -> Dict[str, str]:
        """Resumen legible del pipeline E2E (detector global + clasificador del run)."""
        import yaml
        from src.grain_detection.detector_registry import get_active_detector_config

        summary: Dict[str, str] = {}
        global_cfg_path = self._project_root() / "config.yaml"
        if global_cfg_path.exists():
            with open(global_cfg_path, encoding="utf-8") as f:
                global_cfg = yaml.safe_load(f) or {}
            active = get_active_detector_config(global_cfg)
            backend = active.get("backend", "u2net")
            label = active.get("label") or backend
            ckpt = active.get("checkpoint")
            if backend == "vit_dense" and ckpt:
                ckpt_name = Path(str(ckpt)).name
                summary["detector"] = f"{label} ({ckpt_name})"
            else:
                summary["detector"] = str(label)
            gd = global_cfg.get("grain_detection") or {}
            summary["strategy"] = str(gd.get("detection_strategy", "single_pass"))

        checkpoint = self.checkpoint_input.text().strip()
        if checkpoint and Path(checkpoint).exists():
            info = check_slice_inference_artifacts(checkpoint)
            if info["has_slice_representatives"]:
                summary["classifier"] = "slice-aware (slice_representatives.pt)"
            elif info["has_class_proxies"]:
                summary["classifier"] = "slice-aware (class_proxies.pt)"
            else:
                summary["classifier"] = f"k-NN fallback (k={self.k_spin.value()})"
            summary["classifier_ckpt"] = Path(checkpoint).name

        return summary

    def _update_pipeline_status(self):
        """Actualiza la barra de estado del pipeline E2E en la UI."""
        summary = self._get_pipeline_summary()
        parts = []
        if summary.get("classifier"):
            ckpt = summary.get("classifier_ckpt", "")
            parts.append(f"Clasificador: {summary['classifier']}" + (f" [{ckpt}]" if ckpt else ""))
        if summary.get("detector"):
            parts.append(f"Detector: {summary['detector']}")
        if summary.get("strategy"):
            parts.append(f"Estrategia: {summary['strategy']}")

        if parts:
            self.pipeline_status_label.setText(" | ".join(parts))
            self.pipeline_status_label.setStyleSheet(
                "color: #4ec9b0; font-size: 9pt; padding: 4px 0; font-weight: bold;"
            )
        else:
            self.pipeline_status_label.setText(
                "Configure checkpoint del clasificador y detector activo (pulse Cargar detector)."
            )
            self.pipeline_status_label.setStyleSheet(
                "color: #ffa94d; font-size: 9pt; padding: 4px 0; font-weight: bold;"
            )

    def _validate_e2e_pipeline(self) -> bool:
        """Verifica que el detector activo esté listo antes de ejecutar E2E."""
        import yaml
        from src.grain_detection.detector_registry import get_active_detector_config

        global_cfg_path = self._project_root() / "config.yaml"
        if not global_cfg_path.exists():
            QMessageBox.warning(
                self,
                "Configuración no encontrada",
                "No se encontró config.yaml en la raíz del proyecto.",
            )
            return False

        with open(global_cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        active = get_active_detector_config(cfg)
        backend = active.get("backend", "u2net")
        if backend == "vit_dense":
            ckpt = active.get("checkpoint")
            if not ckpt or not Path(str(ckpt)).exists():
                QMessageBox.warning(
                    self,
                    "Detector no configurado",
                    "ViT-denso requiere un checkpoint válido.\n\n"
                    "Seleccione el detector en la sección Pipeline E2E y pulse "
                    "'Cargar detector' antes de ejecutar.",
                )
                return False
        return True
    
    def toggle_inference_mode(self):
        """Alterna entre modo carpeta e imagen individual"""
        is_folder_mode = self.mode_combo.currentIndex() == 0
        self.folder_group.setVisible(is_folder_mode)
        self.image_group.setVisible(not is_folder_mode)
    
    def browse_checkpoint(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Checkpoint", "", "PyTorch Files (*.pth *.pt)"
        )
        if file_path:
            self.checkpoint_input.setText(file_path)
    
    def _update_checkpoint_artifact_status(self):
        """Actualiza indicador de modo de inferencia según artefactos del run."""
        checkpoint = self.checkpoint_input.text().strip()
        if not checkpoint or not Path(checkpoint).exists():
            self.artifact_status_label.setText("")
            return

        info = check_slice_inference_artifacts(checkpoint)
        if info["has_slice_representatives"]:
            self.artifact_status_label.setText(
                "Modo inferencia: slice-aware (slice_representatives.pt)"
            )
            self.artifact_status_label.setStyleSheet(
                "color: #4ec9b0; font-size: 9pt; padding: 2px 0; font-weight: bold;"
            )
        elif info["has_class_proxies"]:
            self.artifact_status_label.setText(
                "Modo inferencia: slice-aware (class_proxies.pt — run legacy)"
            )
            self.artifact_status_label.setStyleSheet(
                "color: #4ec9b0; font-size: 9pt; padding: 2px 0; font-weight: bold;"
            )
        else:
            self.artifact_status_label.setText(
                "Modo inferencia: k-NN fallback — no se encontró slice_representatives.pt"
            )
            self.artifact_status_label.setStyleSheet(
                "color: #ffa94d; font-size: 9pt; padding: 2px 0; font-weight: bold;"
            )

    def _confirm_inference_artifacts(self, checkpoint: str) -> bool:
        """
        Confirma inferencia cuando faltan artefactos slice-aware.
        Returns True si el usuario acepta continuar (o si los artefactos existen).
        """
        info = check_slice_inference_artifacts(checkpoint)
        if info["has_slice_representatives"] or info["has_class_proxies"]:
            return True

        run_dir = info["run_dir"]
        reply = QMessageBox.warning(
            self,
            "Artefactos slice-aware no encontrados",
            f"No se encontró slice_representatives.pt en:\n{run_dir}\n\n"
            "La inferencia usará k-NN sobre embeddings 128D (geometría distinta al "
            "entrenamiento con sliced_ms).\n\n"
            "Recomendación: ejecute post-training o scripts/migrate_slice_artifacts.py "
            "antes de inferir.\n\n"
            "¿Desea continuar con fallback k-NN?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes
    
    def browse_folder(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Carpeta para Inferencia"
        )
        if dir_path:
            self.folder_path.setText(dir_path)
    
    def browse_query(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar Imagen", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            self.query_path.setText(file_path)
    
    def browse_database(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Seleccionar Directorio de Base de Datos"
        )
        if dir_path:
            self.database_path.setText(dir_path)
    
    def run_inference(self):
        """Ejecuta inferencia según el modo seleccionado"""
        is_folder_mode = self.mode_combo.currentIndex() == 0
        
        if is_folder_mode:
            self.run_folder_inference()
        else:
            self.run_single_image_inference()
    
    def run_single_image_inference(self):
        """Ejecuta inferencia slice-aware en imagen individual (FullImageClassifier)."""
        checkpoint = self.checkpoint_input.text()
        query = self.query_path.text()
        
        if not checkpoint or not Path(checkpoint).exists():
            QMessageBox.warning(self, "Error", "Checkpoint no válido")
            return
        
        if not query or not Path(query).exists():
            QMessageBox.warning(self, "Error", "Imagen query no válida")
            return
        
        if not self._confirm_inference_artifacts(checkpoint):
            return

        if not self._validate_e2e_pipeline():
            return
        
        self.infer_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.results_text.clear()
        pipeline = self._get_pipeline_summary()
        self.results_text.append("Procesando imagen E2E (detectar granos → clasificar)...")
        self.results_text.append(f"Detector: {pipeline.get('detector', '?')}")
        self.results_text.append(f"Clasificador: {pipeline.get('classifier', '?')}")
        
        if self.parent_window:
            self.parent_window.log_widget.append_log(
                "Iniciando inferencia E2E single-image (FullImageClassifier)...", "INFO"
            )
        
        k = self.k_spin.value()
        self.worker = SingleImageInferenceWorker(checkpoint, query, k)
        self.worker.finished.connect(self._single_image_inference_finished)
        self.worker.error.connect(self._single_image_inference_error)
        self.worker.start()
    
    def _single_image_inference_finished(self, payload: Dict[str, Any]):
        """Callback cuando termina inferencia de imagen individual."""
        self.infer_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        mode = payload["classifier_mode"]
        det_backend = payload.get("detector_backend", "?")
        strategy = payload.get("detection_strategy", "?")
        results = payload["results"]
        grains = results.get("grains", [])
        summary = results.get("summary", {})
        per_class = summary.get("per_class_counts", {})
        
        self.results_text.clear()
        self.results_text.append("=" * 50)
        self.results_text.append("RESULTADO PIPELINE E2E (IMAGEN INDIVIDUAL)")
        self.results_text.append("=" * 50)
        self.results_text.append(f"\nDetector: {det_backend} | Estrategia: {strategy}")
        self.results_text.append(f"Modo clasificación: {mode}")
        self.results_text.append(f"Granos detectados: {summary.get('total_grains', len(grains))}")
        self.results_text.append(f"Tiempo: {summary.get('elapsed_ms', 0):.0f} ms")
        
        if not grains:
            self.results_text.append("\nNo se detectaron granos en la imagen.")
        else:
            self.results_text.append("\n--- Resumen por clase ---")
            for cls_name, count in sorted(per_class.items(), key=lambda x: -x[1]):
                self.results_text.append(f"  {cls_name}: {count}")
            
            self.results_text.append("\n--- Detalle por grano ---")
            for i, grain in enumerate(grains, 1):
                cls = grain.get("predicted_class", "?")
                conf = grain.get("confidence", 0.0)
                self.results_text.append(f"  Grano {i}: {cls} ({conf:.1%})")
        
        if self.parent_window:
            top_class = max(per_class, key=per_class.get) if per_class else "N/A"
            self.parent_window.log_widget.append_log(
                f"Inferencia [{mode}]: {len(grains)} granos, principal={top_class}",
                "SUCCESS",
            )
    
    def _single_image_inference_error(self, error_msg: str):
        """Callback de error en inferencia de imagen individual."""
        self.infer_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.results_text.clear()
        self.results_text.append(f"Error: {error_msg}")
        if self.parent_window:
            self.parent_window.log_widget.append_log(f"Error en inferencia: {error_msg}", "ERROR")
        QMessageBox.critical(self, "Error", f"Error durante inferencia:\n{error_msg}")
    
    def run_folder_inference(self):
        """Ejecuta inferencia en carpeta completa y genera reporte"""
        checkpoint = self.checkpoint_input.text()
        folder = self.folder_path.text()
        database = self.database_path.text()
        
        # Validación
        if not checkpoint or not Path(checkpoint).exists():
            QMessageBox.warning(self, "Error", "Checkpoint no válido")
            return
        
        if not folder or not Path(folder).exists():
            QMessageBox.warning(self, "Error", "Carpeta no válida")
            return
        
        # Base de datos es opcional
        if database and not Path(database).exists():
            QMessageBox.warning(self, "Error", "Directorio de base de datos no válido")
            return
        
        if not self._confirm_inference_artifacts(checkpoint):
            return

        if not self._validate_e2e_pipeline():
            return
        
        self.infer_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.results_text.clear()
        pipeline = self._get_pipeline_summary()
        self.results_text.append("Procesando carpeta E2E (detectar + clasificar + reporte)...")
        self.results_text.append(f"Detector: {pipeline.get('detector', '?')}")
        self.results_text.append(f"Clasificador: {pipeline.get('classifier', '?')}")
        if pipeline.get("strategy"):
            self.results_text.append(f"Estrategia: {pipeline['strategy']}")
        
        if self.parent_window:
            self.parent_window.log_widget.append_log(
                f"Iniciando pipeline E2E en carpeta | det={pipeline.get('detector', '?')} "
                f"| cls={pipeline.get('classifier', '?')}",
                "INFO",
            )
        
        # Ejecutar en thread
        k = self.k_spin.value()
        noise_config = {
            "enable_background_correction": self.enable_bg_correction.isChecked(),
            "background_blur_kernel": self.bg_blur_kernel.value(),
            "enable_postfilter": self.enable_postfilter.isChecked(),
            "min_bbox_w": self.min_bbox_w.value(),
            "min_bbox_h": self.min_bbox_h.value(),
            "min_contour_area": self.min_contour_area.value(),
            "min_area_ratio": self.min_area_ratio.value(),
            "min_saliency": self.min_saliency.value(),
            "min_confidence": self.min_confidence.value(),
            "exclude_unknown": self.exclude_unknown.isChecked(),
        }
        self.worker = InferenceWorker(checkpoint, folder, database, k, noise_config)
        self.worker.finished.connect(self.folder_inference_finished)
        self.worker.error.connect(self.folder_inference_error)
        self.worker.progress.connect(self.update_progress)
        self.worker.detailed_progress.connect(self.update_detailed_progress)
        self.worker.start()
    
    def update_progress(self, current, total):
        """Actualiza barra de progreso"""
        percentage = int((current / total) * 100)
        self.progress_bar.setValue(percentage)
    
    def update_detailed_progress(self, info):
        """Muestra conteo en vivo de granos y clases identificadas"""
        idx = info['idx']
        total = info['total']
        img_name = info['image_name']
        grains_this = info['grains_this']
        total_grains = info['total_grains']
        class_counts = info['class_counts']
        pct = int((idx / total) * 100)
        
        self.results_text.clear()
        self.results_text.append(f"Procesando {idx}/{total} ({pct}%)")
        self.results_text.append(f"Ultima: {img_name} -> {grains_this} granos")
        self.results_text.append(f"")
        self.results_text.append(f"=== ACUMULADO ===")
        self.results_text.append(f"Total granos detectados: {total_grains}")
        
        if class_counts:
            self.results_text.append(f"")
            self.results_text.append(f"Clases identificadas:")
            for cls in sorted(class_counts, key=class_counts.get, reverse=True):
                cnt = class_counts[cls]
                bar = '#' * min(cnt, 40)
                self.results_text.append(f"  {cls}: {cnt} granos  {bar}")
    
    def folder_inference_finished(self, result):
        """Callback cuando termina inferencia de carpeta"""
        self.infer_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        report_path = result['report_path']
        total = result['total_images']
        total_grains = result.get('total_grains', 0)
        output_dir = result.get('output_dir', '')
        quick_inspection = result.get('quick_inspection_path', '')
        
        self.results_text.clear()
        self.results_text.append("=" * 50)
        self.results_text.append("PIPELINE E2E COMPLETADO")
        self.results_text.append("=" * 50)
        self.results_text.append(f"\nTotal imagenes procesadas: {total}")
        self.results_text.append(f"Total granos detectados: {total_grains}")
        
        # Resumen por clase
        predictions = result.get('predictions', [])
        class_counts = {}
        for pred in predictions:
            cls = pred.get('predicted_class', 'unknown')
            n = pred.get('n_grains', 0)
            if cls not in class_counts:
                class_counts[cls] = {'images': 0, 'grains': 0}
            class_counts[cls]['images'] += 1
            class_counts[cls]['grains'] += n
        
        if class_counts:
            self.results_text.append(f"\nDistribucion por clase:")
            for cls, info in sorted(class_counts.items()):
                self.results_text.append(
                    f"  {cls}: {info['images']} imgs, {info['grains']} granos"
                )
        
        self.results_text.append(f"\nReporte: {report_path}")
        if output_dir:
            self.results_text.append(f"Imagenes anotadas: {output_dir}")
        if quick_inspection:
            self.results_text.append(f"Inspeccion rapida: {quick_inspection}")
        
        if self.parent_window:
            self.parent_window.log_widget.append_log(
                f"Inferencia completada: {total} imagenes, {total_grains} granos",
                "SUCCESS"
            )
        
        # Abrir reporte automáticamente
        reply = QMessageBox.question(
            self,
            "Reporte E2E Generado",
            f"Pipeline E2E completado.\n\n"
            f"Imagenes procesadas: {total}\n"
            f"Granos detectados y clasificados: {total_grains}\n"
            f"Reporte: {report_path}\n\n"
            f"Desea abrir el reporte ahora?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            import webbrowser
            webbrowser.open(str(report_path))
    
    def folder_inference_error(self, error_msg):
        """Callback cuando hay error en inferencia de carpeta"""
        self.infer_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        self.results_text.clear()
        self.results_text.append(f"❌ Error: {error_msg}")
        
        if self.parent_window:
            self.parent_window.log_widget.append_log(f"Error en inferencia: {error_msg}", "ERROR")
        
        QMessageBox.critical(self, "Error", f"Error durante inferencia:\n{error_msg}")
