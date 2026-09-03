"""
Terminal-only post-hoc morphology extractor using model evidence (Grad-CAM).

What it does:
1) Asks you (in terminal) for the checkpoint to evaluate.
2) Asks you for images to characterize. If empty, uses training data defaults.
3) Runs full-image classification + per-grain Grad-CAM.
4) Extracts morphology from Grad-CAM masks (circularity, eccentricity, solidity, etc.).
5) Writes a rich report in reports/<timestamp_run>/ with:
   - detailed morphology bank CSV,
   - class tables,
   - 15-image gallery per class.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Ensure `src` imports work when launching as `python scripts/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.grain_detection.full_image_classifier import FullImageClassifier
from src.utils.inference_utils import generate_gradcam

LOG = logging.getLogger("posthoc_morphology_bank")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MORPH_FEATURES = [
    "cam_area_px",
    "cam_perimeter_px",
    "cam_circularity",
    "cam_eccentricity",
    "cam_solidity",
    "cam_extent",
    "cam_major_axis_px",
    "cam_minor_axis_px",
]


@dataclass
class MorphologyResult:
    area_px: float
    perimeter_px: float
    circularity: float
    eccentricity: float
    solidity: float
    extent: float
    major_axis_px: float
    minor_axis_px: float
    centroid_x: float
    centroid_y: float
    contour_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive post-hoc morphology extraction (terminal only)."
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=5,
        help="k neighbors for FullImageClassifier k-NN.",
    )
    parser.add_argument(
        "--heatmap-threshold",
        type=float,
        default=0.65,
        help="Base threshold [0-1] for Grad-CAM binarization.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit images (0 = all).",
    )
    parser.add_argument(
        "--bank-csv",
        default="",
        help="Optional bank CSV for morphology distance comparison.",
    )
    parser.add_argument(
        "--materialize-grain-artifacts",
        action="store_true",
        help="If set, saves crop/overlay files for every grain (slower).",
    )
    parser.add_argument(
        "--export-top15-images",
        action="store_true",
        help="If set, renders top-15 image galleries per class (JPG). Default: only text manifests.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def find_images(images_dir: Path, max_images: int = 0) -> List[Path]:
    paths = [
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    paths = sorted(paths)
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def choose_checkpoint_interactively(repo_root: Path) -> Path:
    runs_dir = repo_root / "runs"
    candidates = sorted(
        runs_dir.glob("*/checkpoints/*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    print("\n=== Seleccion de modelo a evaluar ===")
    if candidates:
        print("Checkpoints encontrados (recientes):")
        for idx, pth in enumerate(candidates[:12], start=1):
            print(f"  [{idx}] {pth}")
    else:
        print("No se detectaron checkpoints automaticamente.")

    while True:
        raw = input(
            "\nIngresa numero de lista o ruta completa del checkpoint (.pth): "
        ).strip()
        if raw.isdigit() and candidates:
            idx = int(raw)
            if 1 <= idx <= min(12, len(candidates)):
                return candidates[idx - 1].resolve()
        p = Path(raw)
        if p.exists() and p.is_file() and p.suffix.lower() == ".pth":
            return p.resolve()
        print("Entrada invalida. Intenta nuevamente.")


def load_run_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No existe config en {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_original_paths_from_seg(seg_path: Path) -> Optional[str]:
    try:
        with open(seg_path, "r", encoding="utf-8") as f:
            for _ in range(15):
                line = f.readline()
                if not line:
                    break
                if line.startswith("# Imagen:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        return None
    return None


def resolve_default_images_from_training(repo_root: Path, config: Dict) -> List[Path]:
    data_cfg = config.get("data", {})
    annotation_root = data_cfg.get("annotation_root", "")
    candidates: List[Path] = []

    # 1) Preferred: original names from .seg headers.
    if annotation_root:
        ann_dir = (repo_root / annotation_root).resolve()
        if ann_dir.exists():
            seg_files = sorted(ann_dir.rglob("*.seg"))
            for seg in seg_files:
                original_name = extract_original_paths_from_seg(seg)
                if not original_name:
                    continue
                name = Path(original_name).name
                class_name = seg.parent.name
                search_paths = [
                    seg.parent / name,
                    (repo_root / data_cfg.get("train_dir", "")) / class_name / name,
                    (repo_root / data_cfg.get("val_dir", "")) / class_name / name,
                    (repo_root / data_cfg.get("test_dir", "")) / class_name / name,
                ]
                for sp in search_paths:
                    if sp.exists() and sp.suffix.lower() in IMAGE_EXTENSIONS:
                        candidates.append(sp.resolve())
                        break
    if candidates:
        unique = sorted(set(candidates))
        return unique

    # 2) Fallback: train_dir images directly.
    train_dir = data_cfg.get("train_dir", "")
    if train_dir:
        tdir = (repo_root / train_dir).resolve()
        if tdir.exists():
            return find_images(tdir)
    return []


def choose_images_interactively(repo_root: Path, config: Dict) -> List[Path]:
    defaults = resolve_default_images_from_training(repo_root, config)
    print("\n=== Seleccion de datos a caracterizar ===")
    if defaults:
        print(
            f"Default disponible: {len(defaults)} imagenes desde data de entrenamiento original/configurada."
        )
    else:
        print("No se pudo resolver automaticamente el dataset por defecto.")

    raw = input(
        "Ingresa carpeta de imagenes (Enter para usar default de entrenamiento): "
    ).strip()
    if raw:
        custom_dir = Path(raw).resolve()
        if not custom_dir.exists():
            raise FileNotFoundError(f"No existe carpeta: {custom_dir}")
        images = find_images(custom_dir)
        if not images:
            raise RuntimeError(f"No se encontraron imagenes en: {custom_dir}")
        return images
    if not defaults:
        raise RuntimeError(
            "No hay dataset por defecto disponible. Debes indicar una carpeta de imagenes."
        )
    return defaults


def create_report_run_dir(repo_root: Path) -> Path:
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = reports_dir / f"posthoc_morphology_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["tables", "galleries", "manifests", "artifacts", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def load_centroids(run_dir: Path, device: torch.device) -> Optional[Dict]:
    centroids_path = run_dir / "class_centroids.pt"
    if not centroids_path.exists():
        LOG.warning("No class_centroids.pt found. CAM will run without centroid target.")
        return None
    return torch.load(centroids_path, map_location=device, weights_only=False)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    hm = heatmap.astype(np.float32)
    hm -= hm.min()
    hm /= (hm.max() + 1e-8)
    return hm


def heatmap_to_mask(heatmap: np.ndarray, base_threshold: float) -> np.ndarray:
    hm = normalize_heatmap(heatmap)
    hm_u8 = (hm * 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(hm_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(int(otsu_thr), int(np.clip(base_threshold, 0.0, 1.0) * 255))
    mask = (hm_u8 >= thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def contour_to_morphology(contour: np.ndarray) -> Optional[MorphologyResult]:
    if contour is None or len(contour) < 3:
        return None
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 0 or perimeter <= 0:
        return None

    circularity = float(np.clip((4.0 * np.pi * area) / (perimeter * perimeter), 0.0, 1.0))
    x, y, w, h = cv2.boundingRect(contour)
    extent = float(area / max(float(w * h), 1.0))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = float(area / hull_area) if hull_area > 1e-8 else 0.0

    major_axis = 0.0
    minor_axis = 0.0
    eccentricity = 0.0
    if len(contour) >= 5:
        (_, _), (a, b), _ = cv2.fitEllipse(contour)
        major_axis = float(max(a, b))
        minor_axis = float(min(a, b))
        if major_axis > 1e-8:
            eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2)))

    m = cv2.moments(contour)
    if m["m00"] > 1e-8:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
    else:
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)

    return MorphologyResult(
        area_px=area,
        perimeter_px=perimeter,
        circularity=circularity,
        eccentricity=eccentricity,
        solidity=float(np.clip(solidity, 0.0, 1.5)),
        extent=float(np.clip(extent, 0.0, 1.0)),
        major_axis_px=major_axis,
        minor_axis_px=minor_axis,
        centroid_x=cx,
        centroid_y=cy,
        contour_points=int(len(contour)),
    )


def compare_with_bank(row: Dict, bank_df: pd.DataFrame) -> Tuple[float, str]:
    if bank_df.empty:
        return float("nan"), "empty_bank"
    cls_df = bank_df[bank_df["predicted_class"] == row.get("predicted_class", "")]
    mode = "class_specific"
    if cls_df.empty:
        cls_df = bank_df
        mode = "global_fallback"
    valid = [f for f in MORPH_FEATURES if f in row and f in cls_df.columns]
    if not valid:
        return float("nan"), "missing_features"
    x = np.array([row[f] for f in valid], dtype=np.float64)
    mu = cls_df[valid].mean().to_numpy(dtype=np.float64)
    sigma = cls_df[valid].std().to_numpy(dtype=np.float64)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    z = (x - mu) / sigma
    return float(np.sqrt(np.sum(z ** 2))), mode


def save_grain_artifacts(
    out_dir: Path,
    image_stem: str,
    grain_idx: int,
    crop_bgr: np.ndarray,
    heatmap: np.ndarray,
    mask: np.ndarray,
) -> Tuple[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_stem}_g{grain_idx:03d}"
    crop_path = out_dir / f"{stem}_crop.jpg"
    overlay_path = out_dir / f"{stem}_overlay.jpg"

    hm_u8 = (normalize_heatmap(heatmap) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    if hm_color.shape[:2] != crop_rgb.shape[:2]:
        hm_color = cv2.resize(
            hm_color,
            (crop_rgb.shape[1], crop_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    overlay_rgb = cv2.addWeighted(crop_rgb, 0.6, hm_color, 0.4, 0).astype(np.uint8)

    mask_draw = mask
    if mask_draw.shape[:2] != crop_rgb.shape[:2]:
        mask_draw = cv2.resize(
            mask_draw,
            (crop_rgb.shape[1], crop_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    contours, _ = cv2.findContours(mask_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(overlay_rgb, contours, -1, (255, 255, 255), 1)

    cv2.imwrite(str(crop_path), crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return str(crop_path), str(overlay_path)


def ensure_link(src: Path, links_dir: Path) -> str:
    links_dir.mkdir(parents=True, exist_ok=True)
    dst = links_dir / src.name
    if not dst.exists():
        try:
            os.symlink(str(src), str(dst))
        except Exception:
            # Fallback for environments where symlink privileges are restricted.
            try:
                os.link(str(src), str(dst))
            except Exception:
                # Last resort: keep original path only.
                return str(src)
    return str(dst)


def build_class_tables(df: pd.DataFrame, out_tables: Path) -> None:
    out_tables.mkdir(parents=True, exist_ok=True)

    agg = df.groupby("predicted_class").agg(
        grains=("predicted_class", "count"),
        images=("image_path", "nunique"),
        confidence_mean=("knn_confidence", "mean"),
        confidence_std=("knn_confidence", "std"),
        distance_mean=("knn_mean_distance", "mean"),
        circularity_mean=("cam_circularity", "mean"),
        eccentricity_mean=("cam_eccentricity", "mean"),
        solidity_mean=("cam_solidity", "mean"),
        extent_mean=("cam_extent", "mean"),
        area_mean=("cam_area_px", "mean"),
        area_std=("cam_area_px", "std"),
    ).reset_index().sort_values("grains", ascending=False)

    agg.to_csv(out_tables / "class_summary.csv", index=False)
    try:
        agg.to_markdown(out_tables / "class_summary.md", index=False)
    except Exception:
        # Fallback if optional dependency for markdown rendering is missing.
        with open(out_tables / "class_summary.md", "w", encoding="utf-8") as f:
            f.write(agg.to_string(index=False))

    for cls in sorted(df["predicted_class"].dropna().unique()):
        cdf = df[df["predicted_class"] == cls].copy()
        safe = cls.replace(" ", "_").replace("/", "_")
        cdf.to_csv(out_tables / f"class_{safe}_details.csv", index=False)


def create_gallery_for_class(class_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    selected = class_df.sort_values("knn_confidence", ascending=False).head(top_n)
    if selected.empty:
        return

    thumb_w = 260
    thumb_h = 260
    text_h = 70
    cell_h = thumb_h + text_h
    cols = 5
    rows = int(np.ceil(len(selected) / cols))

    canvas = np.full((rows * cell_h, cols * thumb_w, 3), 20, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, (_, row) in enumerate(selected.iterrows()):
        r = idx // cols
        c = idx % cols
        x0 = c * thumb_w
        y0 = r * cell_h

        overlay = None
        overlay_path = str(row.get("overlay_path", "")).strip()
        if overlay_path.lower() == "nan":
            overlay_path = ""
        if overlay_path:
            overlay = cv2.imread(overlay_path)

        if overlay is None:
            source_path = str(row.get("source_link_path", row.get("image_path", ""))).strip()
            if source_path.lower() == "nan":
                source_path = str(row.get("image_path", "")).strip()
            source_img = cv2.imread(source_path)
            if source_img is None:
                continue
            bx = int(row.get("bbox_x", 0))
            by = int(row.get("bbox_y", 0))
            bw = int(row.get("bbox_w", source_img.shape[1]))
            bh = int(row.get("bbox_h", source_img.shape[0]))
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(source_img.shape[1], bx + max(1, bw))
            y2 = min(source_img.shape[0], by + max(1, bh))
            crop = source_img[y1:y2, x1:x2]
            if crop.size == 0:
                crop = source_img
            overlay = crop

        overlay = cv2.resize(overlay, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        canvas[y0:y0 + thumb_h, x0:x0 + thumb_w] = overlay

        label1 = f"{row['predicted_class']} | conf={row['knn_confidence']:.2f}"
        label2 = f"circ={row['cam_circularity']:.2f} ecc={row['cam_eccentricity']:.2f} sol={row['cam_solidity']:.2f}"
        cv2.putText(canvas, label1, (x0 + 6, y0 + thumb_h + 24), font, 0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, label2, (x0 + 6, y0 + thumb_h + 50), font, 0.45, (170, 240, 170), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def build_class_galleries(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls in sorted(df["predicted_class"].dropna().unique()):
        safe = cls.replace(" ", "_").replace("/", "_")
        cdf = df[df["predicted_class"] == cls]
        create_gallery_for_class(cdf, out_dir / f"{safe}_top15.jpg", top_n=15)


def build_top15_manifests(df: pd.DataFrame, out_dir: Path, top_n: int = 15) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(df["predicted_class"].dropna().unique())
    summary_lines = ["# Top-15 manifest per class", ""]

    for cls in classes:
        safe = cls.replace(" ", "_").replace("/", "_")
        cdf = df[df["predicted_class"] == cls].sort_values("knn_confidence", ascending=False).head(top_n)
        if cdf.empty:
            continue

        header = (
            "rank|predicted_class|confidence|circularity|eccentricity|solidity|"
            "area_px|source_link_path|image_path|bbox_x|bbox_y|bbox_w|bbox_h"
        )
        lines = [header]
        for rank, (_, row) in enumerate(cdf.iterrows(), start=1):
            source_link = str(row.get("source_link_path", "")).strip()
            if source_link.lower() == "nan":
                source_link = str(row.get("image_path", "")).strip()
            lines.append(
                "|".join(
                    [
                        str(rank),
                        str(row["predicted_class"]),
                        f"{float(row['knn_confidence']):.6f}",
                        f"{float(row['cam_circularity']):.6f}",
                        f"{float(row['cam_eccentricity']):.6f}",
                        f"{float(row['cam_solidity']):.6f}",
                        f"{float(row['cam_area_px']):.6f}",
                        source_link,
                        str(row["image_path"]),
                        str(int(row["bbox_x"])),
                        str(int(row["bbox_y"])),
                        str(int(row["bbox_w"])),
                        str(int(row["bbox_h"])),
                    ]
                )
            )

        txt_path = out_dir / f"{safe}_top{top_n}.txt"
        srg_path = out_dir / f"{safe}_top{top_n}.srg"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(srg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        summary_lines.append(f"- {cls}: {txt_path.name}, {srg_path.name} ({len(cdf)} rows)")

    with open(out_dir / "top15_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


def main() -> None:
    setup_logging()
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    print("\n==============================================================")
    print(" POST-HOC MORPHOLOGY BANK (terminal-only)")
    print("==============================================================")
    print("Este modulo NO altera entrenamiento. Solo analiza post-entrenamiento.\n")

    checkpoint_path = choose_checkpoint_interactively(repo_root)
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / "config" / "config.yaml"
    reference_path = run_dir / "reference_embeddings.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"No se encontro config en run: {config_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"No se encontro reference_embeddings.pt: {reference_path}")

    config = load_run_config(run_dir)
    image_paths = choose_images_interactively(repo_root, config)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    report_dir = create_report_run_dir(repo_root)
    output_csv = report_dir / "tables" / "morphology_bank.csv"
    log_json = report_dir / "run_context.json"

    print("\n=== Configuracion final ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Run dir:    {run_dir}")
    print(f"Imagenes:   {len(image_paths)}")
    print(f"Salida:     {report_dir}")
    print("--------------------------------------------------------------")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    centroids_data = load_centroids(run_dir, device)
    centroids = centroids_data["centroids"].to(device) if centroids_data is not None else None

    classifier = FullImageClassifier.from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        config_path=str(config_path),
        reference_path=str(reference_path),
        k=args.k_neighbors,
    )

    bank_df = pd.DataFrame()
    if args.bank_csv:
        bank_path = Path(args.bank_csv).resolve()
        if bank_path.exists():
            bank_df = pd.read_csv(bank_path)
            LOG.info("Loaded external bank: %s (%d rows)", bank_path, len(bank_df))
        else:
            LOG.warning("External bank not found: %s", bank_path)

    rows: List[Dict] = []
    artifacts_dir = report_dir / "artifacts"
    links_dir = report_dir / "input_links"

    for img_path in tqdm(image_paths, desc="Procesando imagenes"):
        image = cv2.imread(str(img_path))
        if image is None:
            LOG.warning("No se pudo leer: %s", img_path)
            continue

        result = classifier.classify_full_image(image, return_crops=True)
        grains = result.get("grains", [])
        if not grains:
            continue

        for grain in grains:
            crop_bgr = grain.get("crop")
            if crop_bgr is None:
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_tensor = classifier.transforms(Image.fromarray(crop_rgb)).unsqueeze(0)
            pred_label = int(grain.get("predicted_label", -1))
            centroid = None
            if centroids is not None and 0 <= pred_label < centroids.shape[0]:
                centroid = centroids[pred_label]

            heatmap = generate_gradcam(
                model=classifier.model,
                image_tensor=crop_tensor,
                device=classifier.device,
                centroid=centroid,
            )
            mask = heatmap_to_mask(heatmap, args.heatmap_threshold)
            contour = largest_contour(mask)
            morph = contour_to_morphology(contour)
            if morph is None:
                continue

            grain_idx = int(grain.get("index", -1))
            source_link_path = ensure_link(img_path, links_dir)
            crop_path = ""
            overlay_path = ""
            if args.materialize_grain_artifacts:
                crop_path, overlay_path = save_grain_artifacts(
                    artifacts_dir, img_path.stem, grain_idx, crop_bgr, heatmap, mask
                )

            row = {
                "image_path": str(img_path),
                "grain_index": grain_idx,
                "predicted_class": grain.get("predicted_class", "unknown"),
                "predicted_label": pred_label,
                "knn_confidence": float(grain.get("confidence", np.nan)),
                "knn_mean_distance": float(grain.get("mean_distance", np.nan)),
                "saliency": float(grain.get("saliency", np.nan)),
                "bbox_x": int(grain["bbox"][0]),
                "bbox_y": int(grain["bbox"][1]),
                "bbox_w": int(grain["bbox"][2]),
                "bbox_h": int(grain["bbox"][3]),
                "cam_area_px": morph.area_px,
                "cam_perimeter_px": morph.perimeter_px,
                "cam_circularity": morph.circularity,
                "cam_eccentricity": morph.eccentricity,
                "cam_solidity": morph.solidity,
                "cam_extent": morph.extent,
                "cam_major_axis_px": morph.major_axis_px,
                "cam_minor_axis_px": morph.minor_axis_px,
                "cam_centroid_x": morph.centroid_x,
                "cam_centroid_y": morph.centroid_y,
                "cam_contour_points": morph.contour_points,
                "cam_heatmap_mean": float(np.mean(heatmap)),
                "cam_heatmap_std": float(np.std(heatmap)),
                "cam_heatmap_p95": float(np.percentile(heatmap, 95)),
                "source_link_path": source_link_path,
                "crop_path": crop_path,
                "overlay_path": overlay_path,
            }

            dist, mode = compare_with_bank(row, bank_df)
            row["bank_morph_distance"] = dist
            row["bank_compare_mode"] = mode
            rows.append(row)

    if not rows:
        raise RuntimeError("No se extrajeron granos con morfologia valida.")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    build_class_tables(df, report_dir / "tables")
    build_top15_manifests(df, report_dir / "manifests", top_n=15)
    if args.export_top15_images:
        build_class_galleries(df, report_dir / "galleries")

    context = {
        "timestamp": datetime.now().isoformat(),
        "repo_root": str(repo_root),
        "checkpoint_path": str(checkpoint_path),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "images_count": int(len(image_paths)),
        "rows_extracted": int(len(df)),
        "classes_detected": sorted(df["predicted_class"].dropna().unique().tolist()),
        "k_neighbors": int(args.k_neighbors),
        "heatmap_threshold": float(args.heatmap_threshold),
        "materialize_grain_artifacts": bool(args.materialize_grain_artifacts),
        "export_top15_images": bool(args.export_top15_images),
        "output_csv": str(output_csv),
        "report_dir": str(report_dir),
    }
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2, ensure_ascii=True)

    print("\n==============================================================")
    print(" PROCESO COMPLETADO")
    print("==============================================================")
    print(f"Filas extraidas:    {len(df)}")
    print(f"Clases detectadas:  {len(context['classes_detected'])}")
    print(f"Reporte completo:   {report_dir}")
    print(f"Banco principal:    {output_csv}")
    if args.export_top15_images:
        print("Galerias (15/clase): reports/.../galleries/")
    else:
        print("Galerias:            desactivadas (usa --export-top15-images para habilitar)")
    print("Top15 manifiestos:   reports/.../manifests/*.txt y *.srg")
    print("Tablas por clase:    reports/.../tables/")
    print("==============================================================\n")


if __name__ == "__main__":
    main()

