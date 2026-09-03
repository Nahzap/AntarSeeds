"""

DetectorTrainer — ViT-dense saliency training loop.



Validation metrics (IEEE A+B): Fm, sod_mae, mask_iou, det_precision, det_recall, det_f1, grains_mae.

"""



from __future__ import annotations



import csv

import json

import logging

import time

from pathlib import Path

from typing import Dict, Optional



import numpy as np

import torch

import torch.nn.functional as F

from torch.amp import GradScaler, autocast

from tqdm import tqdm



from src.training.detector_loss_factory import build_detector_loss
from src.training.detector_metrics import (
    batch_sod_metrics_torch,
    evaluate_prediction_map,
)
from src.training.detector_viz import plot_detector_curves, save_val_vis_batch



logger = logging.getLogger(__name__)



DETECTOR_VALIDATION_METRICS = ("det_f1", "Fm", "mask_iou")

METRIC_CSV_FIELDS = [

    "epoch", "train_loss", "val_loss", "mask_iou", "Fm", "sod_mae",

    "det_precision", "det_recall", "det_f1", "grains_mae", "lr",

]





def normalize_detector_validation_metric(name: str) -> str:

    key = str(name).strip()

    if key.lower() == "fm":

        return "Fm"

    return key





class DetectorTrainer:

    def __init__(

        self,

        model: torch.nn.Module,

        train_loader,

        val_loader,

        config: dict,

        *,

        run_dir: str,

        logger_inst: Optional[logging.Logger] = None,

    ):

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.config = config

        self.run_dir = Path(run_dir)

        self.det_cfg = config.get("detection_training", {}) or {}

        self.grain_cfg = config.get("grain_detection", {}) or {}

        self.logger = logger_inst or logger



        hw = config.get("hardware", {}) or {}

        self.device = torch.device(

            "cuda" if torch.cuda.is_available() and hw.get("use_gpu", True) else "cpu"

        )

        self.model.to(self.device)

        self.use_amp = bool(hw.get("use_amp", True)) and self.device.type == "cuda"

        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        self.criterion = build_detector_loss(self.det_cfg)



        lr = float(self.det_cfg.get("learning_rate", 1e-4))

        bb_factor = float(self.det_cfg.get("backbone_lr_factor", 0.1))

        backbone_params = []

        head_params = []

        for name, p in self.model.named_parameters():

            if not p.requires_grad:

                continue

            if "raw_vit" in name:

                backbone_params.append(p)

            else:

                head_params.append(p)

        param_groups = []

        if head_params:

            param_groups.append({"params": head_params, "lr": lr})

        if backbone_params:

            param_groups.append({"params": backbone_params, "lr": lr * bb_factor})

        if not param_groups:

            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": lr}]



        opt_name = str(self.det_cfg.get("optimizer", "adamw")).lower()

        wd = float(self.det_cfg.get("weight_decay", 0.01))

        if opt_name == "adamw":

            self.optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

        else:

            self.optimizer = torch.optim.Adam(param_groups, weight_decay=wd)



        epochs = int(self.det_cfg.get("epochs", 30))

        sched_cfg = self.det_cfg.get("scheduler_params", {}) or {}

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

            self.optimizer,

            T_max=int(sched_cfg.get("T_max", epochs)),

            eta_min=float(sched_cfg.get("eta_min", 1e-6)),

        )



        self.epochs = epochs

        self.val_fast = bool(self.det_cfg.get("val_fast", True))
        self.val_det_f1_max = int(self.det_cfg.get("val_det_f1_max_images", 100) or 0)
        if not self.val_fast and self.val_det_f1_max <= 0:
            self.val_det_f1_max = 100
            self.logger.info(
                "[Detector] val_fast=false sin val_det_f1_max_images: "
                "métricas instancia limitadas a 100 imgs/época (eval completa en scripts/evaluate_detector.py)"
            )

        self.val_every = max(1, int(self.det_cfg.get("val_every_n_epochs", 1)))



        eff_metric = normalize_detector_validation_metric(

            self.det_cfg.get("validation_metric", "det_f1")

        )

        if eff_metric not in DETECTOR_VALIDATION_METRICS:

            raise ValueError(

                f"validation_metric={eff_metric!r} inválida. "

                f"Use: {', '.join(DETECTOR_VALIDATION_METRICS)}"

            )

        self.validation_metric = eff_metric

        if self.val_fast and eff_metric in ("det_f1", "Fm"):

            self.logger.warning(

                f"[Detector] val_fast=true: {eff_metric} calculado en submuestra "

                f"(≤{self.val_det_f1_max} imgs/época); checkpoint por {eff_metric} "

                "(no se sustituye por mask_iou)"

            )

        self.best_metric = -1.0

        self.history = []



        self.detector_dir = self.run_dir / "detector"

        self.detector_dir.mkdir(parents=True, exist_ok=True)

        (self.detector_dir / "curves").mkdir(exist_ok=True)
        (self.detector_dir / "logs").mkdir(exist_ok=True)
        (self.detector_dir / "evaluation").mkdir(exist_ok=True)
        self.vis_every_n_epochs = max(1, int(self.det_cfg.get("visualize_every_n_epochs", 5)))



    def _save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:

        torch.save(

            {

                "epoch": epoch,

                "model_state_dict": self.model.state_dict(),

                "optimizer_state_dict": self.optimizer.state_dict(),

                "metrics": metrics,

                "config": self.config,

                "validation_metric": self.validation_metric,

                "best_metric": self.best_metric,

            },

            path,

        )



    def train_epoch(self, epoch: int) -> float:

        self.model.train()

        total_loss = 0.0

        n = 0

        log_interval = int(self.det_cfg.get("log_interval", 20))

        clip = float(self.config.get("training", {}).get("gradient_clip_norm", 5.0))



        pbar = tqdm(self.train_loader, desc=f"Det Train {epoch+1}/{self.epochs}", leave=False)

        for step, batch in enumerate(pbar):

            images = batch["image"].to(self.device, non_blocking=True)

            targets = batch["saliency_gt"].to(self.device, non_blocking=True)



            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp):

                logits = self.model(images)

            loss = self.criterion(logits.float(), targets.float())



            self.scaler.scale(loss).backward()

            if clip > 0:

                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.scaler.step(self.optimizer)

            self.scaler.update()



            total_loss += loss.item()

            n += 1

            if step % log_interval == 0:

                pbar.set_postfix(loss=f"{loss.item():.4f}")



        return total_loss / max(n, 1)



    @torch.inference_mode()
    def validate(self, epoch: int = 0) -> Dict[str, float]:
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        sod_iou: list[float] = []
        sod_fm: list[float] = []
        sod_mae_vals: list[float] = []
        prec_sum = rec_sum = f1_sum = mae_sum = 0.0
        n_inst_samples = 0
        f1_budget = self.val_det_f1_max
        need_instance = f1_budget != 0

        for batch in tqdm(self.val_loader, desc="Det Val", leave=False):
            images = batch["image"].to(self.device, non_blocking=True)
            targets = batch["saliency_gt"].to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(images)
            val_loss += self.criterion(logits.float(), targets.float()).item()
            n_batches += 1

            probs = torch.sigmoid(logits).float()
            sod = batch_sod_metrics_torch(probs, targets.float())
            sod_iou.extend(sod["mask_iou"].detach().cpu().tolist())
            sod_fm.extend(sod["Fm"].detach().cpu().tolist())
            sod_mae_vals.extend(sod["sod_mae"].detach().cpu().tolist())

            if not need_instance or n_inst_samples >= f1_budget:
                continue

            probs_np = probs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            for i in range(probs_np.shape[0]):
                if n_inst_samples >= f1_budget:
                    break
                prf = evaluate_prediction_map(
                    probs_np[i, 0],
                    targets_np[i, 0],
                    grain_cfg=self.grain_cfg,
                )
                prec_sum += prf["det_precision"]
                rec_sum += prf["det_recall"]
                f1_sum += prf["det_f1"]
                mae_sum += prf["grains_mae"]
                n_inst_samples += 1

        inst_denom = max(n_inst_samples, 1)
        sod_denom = max(len(sod_iou), 1)
        return {
            "val_loss": val_loss / max(n_batches, 1),
            "mask_iou": float(np.mean(sod_iou)) if sod_iou else 0.0,
            "Fm": float(np.mean(sod_fm)) if sod_fm else 0.0,
            "sod_mae": float(np.mean(sod_mae_vals)) if sod_mae_vals else 0.0,
            "det_precision": prec_sum / inst_denom if n_inst_samples > 0 else 0.0,
            "det_recall": rec_sum / inst_denom if n_inst_samples > 0 else 0.0,
            "det_f1": f1_sum / inst_denom if n_inst_samples > 0 else 0.0,
            "grains_mae": mae_sum / inst_denom if n_inst_samples > 0 else 0.0,
            "val_inst_samples": n_inst_samples,
        }



    def train(self, resume_path: Optional[str] = None) -> Dict:

        start_epoch = 0

        if resume_path and Path(resume_path).exists():

            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)

            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)

            if "optimizer_state_dict" in ckpt:

                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            start_epoch = int(ckpt.get("epoch", 0)) + 1

            self.best_metric = float(ckpt.get("best_metric", -1.0))

            self.logger.info(f"Resumed detector from epoch {start_epoch}")



        es_cfg = self.det_cfg.get("early_stopping", {}) or {}

        patience = int(es_cfg.get("patience", 10))

        min_delta = float(es_cfg.get("min_delta", 0.005))

        stale = 0



        csv_path = self.detector_dir / "metrics.csv"

        write_header = not csv_path.exists()

        with open(csv_path, "a", newline="", encoding="utf-8") as fcsv:

            writer = csv.DictWriter(fcsv, fieldnames=METRIC_CSV_FIELDS)

            if write_header:

                writer.writeheader()



            for epoch in range(start_epoch, self.epochs):

                t0 = time.time()

                train_loss = self.train_epoch(epoch)

                if (epoch + 1) % self.val_every == 0 or epoch == self.epochs - 1:

                    metrics = self.validate(epoch)

                else:

                    metrics = {k: 0.0 for k in METRIC_CSV_FIELDS if k not in ("epoch", "lr")}

                    self.logger.debug(

                        f"[Detector] Epoch {epoch+1}: validación omitida (val_every={self.val_every})"

                    )

                metrics["train_loss"] = train_loss

                metrics["epoch"] = epoch + 1

                metrics["lr"] = self.optimizer.param_groups[0]["lr"]

                self.history.append(metrics)

                writer.writerow({k: metrics.get(k, "") for k in writer.fieldnames})

                fcsv.flush()



                score = float(metrics.get(self.validation_metric, 0.0))

                improved = score > self.best_metric + min_delta

                if improved:

                    self.best_metric = score

                    stale = 0

                    self._save_checkpoint(self.detector_dir / "best_detector.pth", epoch, metrics)

                else:

                    stale += 1



                self._save_checkpoint(self.detector_dir / "last_detector.pth", epoch, metrics)

                self.scheduler.step()



                self.logger.info(

                    f"[Detector] Epoch {epoch+1}/{self.epochs} "

                    f"loss={train_loss:.4f} val={metrics['val_loss']:.4f} "

                    f"Fm={metrics['Fm']:.4f} sod_mae={metrics['sod_mae']:.4f} "

                    f"mask_iou={metrics['mask_iou']:.4f} "

                    f"det_p={metrics['det_precision']:.4f} det_r={metrics['det_recall']:.4f} "

                    f"det_f1={metrics['det_f1']:.4f} grains_mae={metrics['grains_mae']:.4f} "

                    f"({time.time()-t0:.1f}s)"

                )



                if es_cfg.get("enabled", True) and stale >= patience:
                    self.logger.info(f"[Detector] Early stopping @ epoch {epoch+1}")
                    break

                if (epoch + 1) % self.vis_every_n_epochs == 0 or epoch == self.epochs - 1:
                    try:
                        save_val_vis_batch(
                            self.model,
                            self.val_loader,
                            device=self.device,
                            config=self.config,
                            output_dir=self.detector_dir / "curves",
                            epoch=epoch + 1,
                            max_samples=int(self.det_cfg.get("vis_samples_per_epoch", 6)),
                        )
                    except Exception as exc:
                        self.logger.warning(f"No se guardaron vis val epoch {epoch+1}: {exc}")

        curve_artifacts = {}
        try:
            curve_artifacts = plot_detector_curves(
                self.history, self.detector_dir / "curves"
            )
        except Exception as exc:
            self.logger.warning(f"No se generaron curvas: {exc}")

        summary = {

            "best_metric": self.best_metric,

            "validation_metric": self.validation_metric,

            "epochs_run": len(self.history),

            "detector_dir": str(self.detector_dir),
            "curve_artifacts": curve_artifacts,
        }

        with open(self.detector_dir / "metrics.json", "w", encoding="utf-8") as f:

            json.dump({"summary": summary, "history": self.history}, f, indent=2)

        return summary


