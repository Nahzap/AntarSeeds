"""Factory for detector training losses (mirrors DML loss factory pattern)."""



from __future__ import annotations



import torch.nn as nn



from src.training.detector_losses import BceDiceLoss, U2NetStructureLoss



DETECTOR_LOSS_TYPES = ("bce_dice", "u2net_structure")





def build_detector_loss(det_cfg: dict) -> nn.Module:

    loss_cfg = det_cfg.get("loss", {}) or {}

    loss_type = str(loss_cfg.get("type", "bce_dice")).lower()



    if loss_type == "bce_dice":

        return BceDiceLoss(

            bce_weight=float(loss_cfg.get("bce_weight", 0.5)),

            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),

        )

    if loss_type == "u2net_structure":

        return U2NetStructureLoss(

            wbce_weight=float(loss_cfg.get("wbce_weight", 1.0)),

            iou_weight=float(loss_cfg.get("iou_weight", 1.0)),

            edge_kernel=int(loss_cfg.get("edge_kernel", 3)),

            edge_boost=float(loss_cfg.get("edge_boost", 5.0)),

        )

    raise ValueError(

        f"detection_training.loss.type={loss_type!r} no soportado. "

        f"Use: {', '.join(DETECTOR_LOSS_TYPES)}"

    )


