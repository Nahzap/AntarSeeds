"""Losses for ViT-dense saliency training (BCE+Dice, U²-Net structure)."""



from __future__ import annotations



import torch

import torch.nn as nn

import torch.nn.functional as F





class BceDiceLoss(nn.Module):

    """BCEWithLogits + Dice on sigmoid probabilities."""



    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):

        super().__init__()

        self.bce_weight = bce_weight

        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()



    @staticmethod

    def _dice_from_probs(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

        pred = pred.reshape(pred.size(0), -1)

        target = target.reshape(target.size(0), -1)

        inter = (pred * target).sum(dim=1)

        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2 * inter + eps) / (union + eps)

        return 1.0 - dice.mean()



    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if target.dim() == 3:

            target = target.unsqueeze(1)

        probs = torch.sigmoid(logits)

        bce = self.bce(logits, target)

        dice = self._dice_from_probs(probs, target)

        return self.bce_weight * bce + self.dice_weight * dice





class U2NetStructureLoss(nn.Module):

    """

    Weighted BCE (edge-emphasized) + IoU loss — Qin et al. (2020) U²-Net formulation.

    Expects logits; edges derived from GT saliency via morphological gradient.

    """



    def __init__(

        self,

        wbce_weight: float = 1.0,

        iou_weight: float = 1.0,

        edge_kernel: int = 3,

        edge_boost: float = 5.0,

    ):

        super().__init__()

        self.wbce_weight = wbce_weight

        self.iou_weight = iou_weight

        self.edge_kernel = max(1, edge_kernel)

        self.edge_boost = edge_boost



    def _edge_weights(self, target: torch.Tensor) -> torch.Tensor:

        k = self.edge_kernel

        pad = k // 2

        dilated = F.max_pool2d(target, kernel_size=k, stride=1, padding=pad)

        eroded = -F.max_pool2d(-target, kernel_size=k, stride=1, padding=pad)

        edge = (dilated - eroded).clamp(0.0, 1.0)

        return 1.0 + self.edge_boost * edge



    @staticmethod

    def _iou_loss(probs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:

        probs = probs.reshape(probs.size(0), -1)

        target = target.reshape(target.size(0), -1)

        inter = (probs * target).sum(dim=1)

        union = probs.sum(dim=1) + target.sum(dim=1) - inter

        iou = (inter + eps) / (union + eps)

        return (1.0 - iou).mean()



    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if target.dim() == 3:

            target = target.unsqueeze(1)

        probs = torch.sigmoid(logits)

        weights = self._edge_weights(target)

        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights, reduction="mean")

        iou = self._iou_loss(probs, target)

        return self.wbce_weight * bce + self.iou_weight * iou


