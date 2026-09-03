"""Post-training detector evaluation on test split (IEEE A+B metrics)."""



from __future__ import annotations



import sys

from pathlib import Path



_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:

    sys.path.insert(0, str(_ROOT))



import argparse

import json

import logging



import torch

import yaml



from src.data.detection_dataset import FullImageDetectionDataset

from src.data.detection_transforms import letterbox_to_square, to_model_tensor

from src.models.vit_dense_segmentation import ViTDenseSegmentationModel

from src.training.detector_eval_utils import run_detector_evaluation

from src.utils.detector_resolution_config import resolve_detector_input_size



logger = logging.getLogger(__name__)





def _load_model(checkpoint: str, device: torch.device) -> ViTDenseSegmentationModel:

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    cfg = ckpt.get("config", {})

    model = ViTDenseSegmentationModel.from_config(cfg if isinstance(cfg, dict) else {})

    state = ckpt.get("model_state_dict", ckpt)

    model.load_state_dict(state, strict=False)

    model.to(device)

    model.eval()

    return model





@torch.inference_mode()

def evaluate_detector(config_path: str, checkpoint: str, output_dir: str | None = None) -> dict:

    with open(config_path, encoding="utf-8") as f:

        config = yaml.safe_load(f)



    data_cfg = config["data"]

    input_size = resolve_detector_input_size(config)

    test_dir = data_cfg.get("test_dir") or data_cfg.get("val_dir")

    ds = FullImageDetectionDataset(

        test_dir,

        annotation_root=data_cfg.get("annotation_root"),

        input_size=input_size,

        augment=False,

    )



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _load_model(checkpoint, device)



    if output_dir is None:

        output_dir = str(Path(checkpoint).parent / "evaluation")



    det_cfg = config.get("detection_training", {}) or {}

    vis_every = int(det_cfg.get("vis_every_n_images", 50) or 0)



    def predict_fn(image_lb):

        tensor = to_model_tensor(image_lb).unsqueeze(0).to(device)

        logits = model(tensor)

        return torch.sigmoid(logits)[0, 0].cpu().numpy()



    report = run_detector_evaluation(

        ds,

        predict_fn,

        config=config,

        output_dir=output_dir,

        checkpoint_label=checkpoint,

        vis_every_n=vis_every,

    )

    return report





def main():

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config.yaml")

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    report = evaluate_detector(args.config, args.checkpoint, args.output_dir)

    print(json.dumps(report, indent=2))





if __name__ == "__main__":

    main()


