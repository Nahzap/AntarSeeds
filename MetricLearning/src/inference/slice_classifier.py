"""
Slice-aware classification for dimensional slicing (Sanakoyeu et al., CVPR 2019).

Training partitions embeddings into S disjoint subspaces; inference must use the
same geometry: L2-normalize each slice, cosine similarity to per-slice class
representatives, aggregate by mean over slices.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.losses.slice_utils import resolve_slice_geometry, split_and_normalize_slices

logger = logging.getLogger(__name__)


@dataclass
class SliceRepresentatives:
    """Per-slice class representatives: shape (num_slices, num_classes, slice_dim)."""

    representatives: torch.Tensor
    class_names: List[str]
    num_slices: int
    slice_dim: int
    embedding_dim: int
    mode: str  # "slice_centroids" | "learned_proxies"
    loss_type: str = "sliced_ms"

    def to(self, device: torch.device) -> "SliceRepresentatives":
        return SliceRepresentatives(
            representatives=self.representatives.to(device),
            class_names=list(self.class_names),
            num_slices=self.num_slices,
            slice_dim=self.slice_dim,
            embedding_dim=self.embedding_dim,
            mode=self.mode,
            loss_type=self.loss_type,
        )


class SliceAwareClassifier:
    """Classify embeddings using per-slice representatives (mean cosine aggregation)."""

    def __init__(self, reps: SliceRepresentatives):
        self.reps = reps
        self.device = reps.representatives.device

    @property
    def num_classes(self) -> int:
        return len(self.reps.class_names)

    def predict(
        self,
        embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, D) model outputs (unit-normalized 128D expected)

        Returns:
            preds: (B,) int class indices
            scores: (B,) aggregated similarity to predicted class
            class_scores: (B, C) mean cosine similarity per class
        """
        embeddings = embeddings.to(self.reps.representatives.device)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        b, d = embeddings.shape
        s, c, ds = self.reps.representatives.shape
        assert d == s * ds == self.reps.embedding_dim

        slice_scores = []
        for i, sl_emb in enumerate(split_and_normalize_slices(embeddings, s)):
            sl_rep = F.normalize(self.reps.representatives[i], p=2, dim=1)
            slice_scores.append(sl_emb @ sl_rep.t())

        class_scores = torch.stack(slice_scores, dim=0).mean(dim=0)
        preds = torch.argmax(class_scores, dim=1)
        scores = class_scores.gather(1, preds.unsqueeze(1)).squeeze(1)
        return preds, scores, class_scores

    def distance_to_class(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (preds, min_distance) using 1 - mean_cosine as distance."""
        preds, scores, _ = self.predict(embeddings)
        return preds, 1.0 - scores

    def reconstruct_target_vector(self, class_idx: int) -> torch.Tensor:
        """
        Build 128D Grad-CAM target by concatenating L2-normalized slice representatives.

        Matches AnalogyNet L2-Cat-L2 topology: two unit halves -> concat norm sqrt(2) -> /sqrt(2).
        Here S slices of dim slice_dim are concatenated without global re-normalization.
        """
        slices = []
        for i in range(self.reps.num_slices):
            v = self.reps.representatives[i, class_idx]
            slices.append(F.normalize(v.unsqueeze(0), p=2, dim=1).squeeze(0))
        target = torch.cat(slices, dim=0)
        return target

    def get_target_for_gradcam(self, class_idx: int, device: Optional[torch.device] = None) -> torch.Tensor:
        dev = device or self.reps.representatives.device
        return self.reconstruct_target_vector(class_idx).to(dev)


def save_slice_representatives(
    path: Union[str, Path],
    reps: SliceRepresentatives,
    extra: Optional[dict] = None,
) -> None:
    path = Path(path)
    payload = {
        "representatives": reps.representatives.cpu(),
        "class_names": list(reps.class_names),
        "num_slices": reps.num_slices,
        "slice_dim": reps.slice_dim,
        "embedding_dim": reps.embedding_dim,
        "mode": reps.mode,
        "loss_type": reps.loss_type,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info(
        "Slice representatives saved: %s (%d slices, %d classes, %dD/slice, mode=%s)",
        path,
        reps.num_slices,
        len(reps.class_names),
        reps.slice_dim,
        reps.mode,
    )


def load_slice_representatives(path: Union[str, Path], device: torch.device) -> SliceRepresentatives:
    data = torch.load(path, map_location=device, weights_only=False)
    reps = SliceRepresentatives(
        representatives=data["representatives"].to(device),
        class_names=list(data["class_names"]),
        num_slices=int(data["num_slices"]),
        slice_dim=int(data["slice_dim"]),
        embedding_dim=int(data["embedding_dim"]),
        mode=str(data.get("mode", "slice_centroids")),
        loss_type=str(data.get("loss_type", "sliced_ms")),
    )
    logger.info(
        "Slice representatives loaded: %s (%d slices, %d classes, mode=%s)",
        path,
        reps.num_slices,
        len(reps.class_names),
        reps.mode,
    )
    return reps


def compute_slice_centroids_from_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_names: List[str],
    num_slices: int = 4,
) -> SliceRepresentatives:
    """Mean embedding per class per slice from training embeddings."""
    n = embeddings.shape[0]
    d = embeddings.shape[1]
    _, slice_dim = resolve_slice_geometry(d, num_slices)
    n_classes = len(class_names)
    reps = torch.zeros(num_slices, n_classes, slice_dim)

    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        class_embs = embeddings[mask]
        for s in range(num_slices):
            start = s * slice_dim
            end = start + slice_dim
            reps[s, c] = class_embs[:, start:end].mean(dim=0)

    return SliceRepresentatives(
        representatives=reps,
        class_names=list(class_names),
        num_slices=num_slices,
        slice_dim=slice_dim,
        embedding_dim=d,
        mode="slice_centroids",
        loss_type="sliced_ms",
    )


def export_proxies_tensor_from_loss_state(
    loss_state_dict: dict,
    num_slices: int,
    num_classes: int,
    slice_dim: int,
) -> torch.Tensor:
    """Stack SG-Softmax proxy matrices from SlicedProxyLoss state_dict -> (S, C, D_s)."""
    proxies = []
    for i in range(num_slices):
        key = f"slice_losses.{i}.proxies"
        if key not in loss_state_dict:
            raise KeyError(f"Missing {key} in loss_state_dict")
        proxies.append(loss_state_dict[key])
    stacked = torch.stack(proxies, dim=0)
    if stacked.shape != (num_slices, num_classes, slice_dim):
        raise ValueError(f"Unexpected proxy shape {stacked.shape}")
    return stacked


def extract_proxies_from_optimizer(
    optimizer_state_dict: dict,
    num_slices: int = 4,
    num_classes: int = 16,
    slice_dim: int = 32,
) -> Optional[torch.Tensor]:
    """
    Detect proxy param group in a legacy optimizer_state_dict.

    Note: PyTorch checkpoints store momentum buffers, not parameter values.
    Returns None unless actual proxy tensors are present (rare in legacy runs).
    """
    param_groups = optimizer_state_dict.get("param_groups", [])
    state = optimizer_state_dict.get("state", {})
    for pg in param_groups:
        pids = pg.get("params", [])
        if len(pids) != num_slices:
            continue
        shapes = []
        for pid in pids:
            exp_avg = state.get(pid, {}).get("exp_avg")
            if exp_avg is None or not hasattr(exp_avg, "shape"):
                shapes = []
                break
            shapes.append(tuple(exp_avg.shape))
        if shapes and all(s == (num_classes, slice_dim) for s in shapes):
            logger.info(
                "Detected sliced_proxy optimizer group (%d x %s) — "
                "parameter values not in checkpoint; use slice centroids fallback",
                num_slices,
                shapes[0],
            )
            return None
    return None


def extract_proxies_from_checkpoint(
    checkpoint_path: Union[str, Path],
    num_slices: int = 4,
    num_classes: int = 16,
    slice_dim: int = 32,
) -> Optional[torch.Tensor]:
    """Recover learned proxies from checkpoint loss_state_dict (sliced_proxy only)."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    loss_sd = ckpt.get("loss_state_dict")
    loss_type = ckpt.get("loss_type", "")
    if loss_sd:
        try:
            return export_proxies_tensor_from_loss_state(
                loss_sd, num_slices, num_classes, slice_dim
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Could not load proxies from loss_state_dict: %s", exc)
    opt = ckpt.get("optimizer_state_dict")
    if opt:
        extract_proxies_from_optimizer(opt, num_slices, num_classes, slice_dim)
    return None


def compute_slice_representatives_from_reference(
    reference_path: Union[str, Path],
    num_slices: int = 4,
) -> SliceRepresentatives:
    """Build slice centroids from saved reference_embeddings.pt (no model pass)."""
    data = torch.load(str(reference_path), map_location="cpu", weights_only=False)
    embeddings = data["embeddings"]
    labels = data["labels"]
    class_names = list(data["class_names"])
    return compute_slice_centroids_from_embeddings(
        embeddings, labels, class_names, num_slices=num_slices
    )


def load_slice_classifier_from_run(
    run_dir: Union[str, Path],
    device: torch.device,
    embedding_dim: int = 128,
    num_slices: int = 4,
) -> Optional[SliceAwareClassifier]:
    """
    Load slice classifier from run artifacts.

    Priority: slice_representatives.pt > class_proxies.pt > checkpoint extraction.
    """
    run_dir = Path(run_dir)
    slice_path = run_dir / "slice_representatives.pt"
    proxy_path = run_dir / "class_proxies.pt"

    if slice_path.exists():
        reps = load_slice_representatives(slice_path, device)
        return SliceAwareClassifier(reps)

    if proxy_path.exists():
        data = torch.load(proxy_path, map_location=device, weights_only=False)
        reps = SliceRepresentatives(
            representatives=data["proxies"].to(device),
            class_names=list(data["class_names"]),
            num_slices=int(data["num_slices"]),
            slice_dim=int(data["slice_dim"]),
            embedding_dim=int(data.get("embedding_dim", embedding_dim)),
            mode="learned_proxies",
            loss_type=str(data.get("loss_type", "sliced_proxy")),
        )
        return SliceAwareClassifier(reps)

    ckpt = run_dir / "checkpoints" / "best_model.pth"
    centroids_path = run_dir / "class_centroids.pt"
    if ckpt.exists() and centroids_path.exists():
        cd = torch.load(centroids_path, map_location="cpu", weights_only=False)
        n_classes = len(cd["class_names"])
        slice_dim = embedding_dim // num_slices
        proxies = extract_proxies_from_checkpoint(ckpt, num_slices, n_classes, slice_dim)
        if proxies is not None:
            reps = SliceRepresentatives(
                representatives=proxies.to(device),
                class_names=list(cd["class_names"]),
                num_slices=num_slices,
                slice_dim=slice_dim,
                embedding_dim=embedding_dim,
                mode="learned_proxies",
                loss_type="sliced_proxy",
            )
            return SliceAwareClassifier(reps)

    return None
