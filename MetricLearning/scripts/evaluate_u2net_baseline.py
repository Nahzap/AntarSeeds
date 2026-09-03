"""Evaluate U²-Net baseline with the same IEEE A+B metric suite as ViT-dense."""



from __future__ import annotations



import sys

from pathlib import Path



_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:

    sys.path.insert(0, str(_ROOT))



import argparse

import json

import logging



import cv2

import yaml



from src.data.detection_dataset import FullImageDetectionDataset

from src.data.detection_transforms import letterbox_to_square

from src.grain_detection.grain_detector import PollenGrainDetector

from src.training.detector_eval_utils import run_detector_evaluation

from src.utils.detector_resolution_config import resolve_detector_input_size



logger = logging.getLogger(__name__)





def evaluate_u2net_baseline(config_path: str, output_dir: str | None = None) -> dict:

    with open(config_path, encoding="utf-8") as f:

        config = yaml.safe_load(f)



    data_cfg = config["data"]

    grain_cfg = config.get("grain_detection", {}) or {}

    input_size = resolve_detector_input_size(config)

    test_dir = data_cfg.get("test_dir") or data_cfg.get("val_dir")

    ds = FullImageDetectionDataset(

        test_dir,

        annotation_root=data_cfg.get("annotation_root"),

        input_size=input_size,

        augment=False,

    )



    det_params = {

        "model_type": grain_cfg.get("model_type", "u2netp"),

        "input_size": grain_cfg.get("input_size", 320),

        "adaptive_k": grain_cfg.get("adaptive_k", 0.30),

        "min_area": grain_cfg.get("min_area", 1000),

        "max_area": grain_cfg.get("max_area", 120000),

        "morph_kernel_size": grain_cfg.get("morph_kernel_size", 3),

        "min_circularity": grain_cfg.get("min_circularity", 0.3),

        "device": grain_cfg.get("device"),

    }

    detector = PollenGrainDetector(det_params)



    if output_dir is None:

        output_dir = "evaluation/u2net_baseline"



    det_cfg = config.get("detection_training", {}) or {}

    vis_every = int(det_cfg.get("vis_every_n_images", 50) or 0)



    def predict_fn(image_lb):

        saliency, _ = detector.detect(image_lb)

        return saliency



    report = run_detector_evaluation(

        ds,

        predict_fn,

        config=config,

        output_dir=output_dir,

        checkpoint_label="u2net_pretrained",

        vis_every_n=vis_every,

    )

    report["detector"] = "u2net"

    out = Path(output_dir)

    with open(out / "u2net_baseline_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, indent=2)

    return report





def main():

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config.yaml")

    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    report = evaluate_u2net_baseline(args.config, args.output_dir)

    print(json.dumps(report, indent=2))





if __name__ == "__main__":

    main()


