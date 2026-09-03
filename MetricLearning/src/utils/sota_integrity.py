"""Validate that enabled SOTA modules are actually usable (Phase A integrity)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _sota_block(config: dict, key: str) -> dict:
    return (config.get("sota") or {}).get(key, {}) or {}


def validate_sota_integrity(
    config: dict,
    *,
    check_files: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Return (errors, warnings) for enabled SOTA regularizers.

    Phase A policy:
    - NIR: allowed when min_samples_per_class is satisfiable in batch
    - DADA: blocked until dual-view training pipeline exists
    - HIER: requires taxonomy_path with valid JSON mapping
    """
    errors: List[str] = []
    warnings: List[str] = []

    data = config.get("data", {}) or {}
    batch_size = int(data.get("batch_size", 0) or 0)
    classes_per_batch = int(data.get("classes_per_batch", 0) or 0)
    samples_per_class = int(data.get("samples_per_class", 0) or 0)
    expected_per_class = (
        samples_per_class
        if samples_per_class > 0 and classes_per_batch > 0
        else max(1, batch_size // max(1, classes_per_batch))
    )

    nir = _sota_block(config, "nir")
    if nir.get("enabled", False):
        min_samples = int(nir.get("min_samples_per_class", 2))
        if min_samples < 2:
            errors.append("sota.nir.min_samples_per_class debe ser >= 2")
        if expected_per_class < min_samples:
            warnings.append(
                f"sota.nir habilitado pero el batch aporta ~{expected_per_class} "
                f"muestras/clase (min_samples_per_class={min_samples}); "
                "NIR se omitirá en batches pequeños"
            )

    dada = _sota_block(config, "dada")
    if dada.get("enabled", False):
        if not dada.get("dual_view_implemented", False):
            errors.append(
                "sota.dada.enabled=true pero el pipeline dual-view no está implementado. "
                "Desactiva sota.dada.enabled o espera a una versión con dual_view_implemented."
            )

    hier = _sota_block(config, "hier")
    if hier.get("enabled", False):
        taxonomy_path = hier.get("taxonomy_path")
        if not taxonomy_path:
            errors.append(
                "sota.hier.enabled=true requiere sota.hier.taxonomy_path "
                "(árbol taxonómico explícito; label//2 no es válido científicamente)"
            )
        elif check_files:
            path = Path(taxonomy_path)
            if not path.exists():
                errors.append(f"sota.hier.taxonomy_path no encontrado: {taxonomy_path}")
            else:
                try:
                    load_hier_taxonomy(path)
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    errors.append(f"sota.hier.taxonomy_path inválido: {exc}")

    return errors, warnings


def load_hier_taxonomy(path: Path) -> Dict[str, Any]:
    """
    Load hierarchical parent mapping for HIER.

    Expected JSON:
    {
      "leaf_to_parents": {
        "0": [0, 0],
        "1": [0, 1]
      }
    }
    Keys are leaf class indices (str/int); values are parent label lists per level.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    leaf_map = data.get("leaf_to_parents")
    if not isinstance(leaf_map, dict) or not leaf_map:
        raise ValueError("taxonomy debe contener 'leaf_to_parents' no vacío")

    normalized: Dict[int, List[int]] = {}
    for key, parents in leaf_map.items():
        idx = int(key)
        if not isinstance(parents, list) or not parents:
            raise ValueError(f"leaf_to_parents[{key!r}] debe ser lista no vacía")
        normalized[idx] = [int(p) for p in parents]

    return {"leaf_to_parents": normalized, "num_levels": max(len(v) for v in normalized.values())}
