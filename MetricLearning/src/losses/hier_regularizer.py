"""
Hierarchical Regularization (HIER) for Deep Metric Learning.
Based on: Kim, Jeong & Kwak (2023) "HIER: Metric Learning Beyond Class Labels
via Hierarchical Regularization" (CVPR 2023)

Requires an explicit taxonomy file — never infer hierarchy from label indices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from src.utils.sota_integrity import load_hier_taxonomy

logger = logging.getLogger(__name__)


class HIERRegularizer(nn.Module):
    """
    Hierarchical regularization with explicit taxonomy mapping.

    ``taxonomy_path`` JSON format::

        {
          "leaf_to_parents": {
            "0": [0, 0],
            "1": [0, 1]
          }
        }
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        num_levels: int = 3,
        lambda_hier: float = 0.1,
        temperature: float = 0.1,
        taxonomy_path: Optional[str] = None,
        leaf_to_parents: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()
        if taxonomy_path is None and leaf_to_parents is None:
            raise ValueError(
                "HIER requiere taxonomy_path o leaf_to_parents explícito "
                "(no se admite label//2)"
            )

        if leaf_to_parents is None:
            leaf_to_parents = load_hier_taxonomy(Path(taxonomy_path))["leaf_to_parents"]

        self.num_levels = num_levels
        self.lambda_hier = lambda_hier
        self.temperature = temperature
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.leaf_to_parents = {int(k): list(v) for k, v in leaf_to_parents.items()}

        max_parents = [0] * num_levels
        for parents in self.leaf_to_parents.values():
            for level_idx, parent in enumerate(parents[:num_levels]):
                max_parents[level_idx] = max(max_parents[level_idx], int(parent) + 1)

        self.leaf_proxies = nn.Parameter(
            torch.randn(num_classes, embedding_dim) * 0.01
        )
        nn.init.kaiming_normal_(self.leaf_proxies.unsqueeze(0))

        self.internal_proxies = nn.ParameterList()
        for level_idx in range(num_levels):
            n_proxies = max(1, max_parents[level_idx])
            self.internal_proxies.append(
                nn.Parameter(torch.randn(n_proxies, embedding_dim) * 0.01)
            )

        logger.info(
            "HIER initialized: %d leaves, %d levels, lambda=%s, taxonomy leaves=%d",
            num_classes,
            num_levels,
            lambda_hier,
            len(self.leaf_to_parents),
        )

    def _get_parent_labels(self, labels: torch.Tensor, level: int) -> torch.Tensor:
        parent = labels.clone()
        for idx in range(labels.shape[0]):
            leaf = int(labels[idx].item())
            mapping = self.leaf_to_parents.get(leaf)
            if mapping is None or level >= len(mapping):
                raise ValueError(
                    f"HIER taxonomy sin padre para clase {leaf} en nivel {level}"
                )
            parent[idx] = int(mapping[level])
        return parent

    def forward(self, embeddings, labels):
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        hier_loss = torch.tensor(0.0, device=embeddings.device)

        leaf_proxies_norm = F.normalize(self.leaf_proxies, p=2, dim=1)
        sim_leaf = embeddings_norm @ leaf_proxies_norm.t() / self.temperature
        hier_loss = hier_loss + F.cross_entropy(sim_leaf, labels)

        for level_idx, proxies in enumerate(self.internal_proxies):
            parent_labels = self._get_parent_labels(labels, level_idx)
            num_parent_classes = proxies.shape[0]
            parent_labels = parent_labels.clamp(0, num_parent_classes - 1)

            proxies_norm = F.normalize(proxies, p=2, dim=1)
            sim = embeddings_norm @ proxies_norm.t() / self.temperature
            hier_loss = hier_loss + F.cross_entropy(sim, parent_labels)

        return self.lambda_hier * hier_loss / (1 + len(self.internal_proxies))
