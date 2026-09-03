"""
Ablation matrix definitions and result aggregation (Phase F).

Matrices F-01 … F-07 patch a base config declaratively; results are collected
from run directories into CSV for publication tables.
"""

from __future__ import annotations

import copy
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# Nested dict patches: dotted keys applied via set_nested.
Variant = Dict[str, Any]

MATRIX_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pooling": {
        "id": "F-01",
        "description": "Pooling strategy ablation (multi_spectral vs GeM vs mean vs FSW)",
        "variants": [
            {
                "id": "pool_multi_spectral",
                "label": "multi_spectral",
                "patch": {"model.pooling.strategy": "multi_spectral"},
            },
            {
                "id": "pool_gem",
                "label": "gem",
                "patch": {"model.pooling.strategy": "gem"},
            },
            {
                "id": "pool_mean",
                "label": "mean",
                "patch": {"model.pooling.strategy": "mean"},
            },
            {
                "id": "pool_fsw",
                "label": "fsw",
                "patch": {
                    "model.pooling.strategy": "fsw",
                    "model.pooling.fsw": {"num_iterations": 3, "temperature": 1.0, "use_external": False},
                },
            },
        ],
    },
    "slices": {
        "id": "F-02",
        "description": "Sliced MS num_slices ablation S in {2, 4, 8}",
        "variants": [
            {"id": "slices_2", "label": "S=2", "patch": {"training.loss.params.num_slices": 2}},
            {"id": "slices_4", "label": "S=4", "patch": {"training.loss.params.num_slices": 4}},
            {"id": "slices_8", "label": "S=8", "patch": {"training.loss.params.num_slices": 8}},
        ],
    },
    "layout": {
        "id": "F-03",
        "description": "Projection layout ablation",
        "variants": [
            {
                "id": "layout_lf_hf_dual",
                "label": "lf_hf_dual",
                "patch": {
                    "model.projection.layout": "lf_hf_dual",
                    "model.pooling.layer_fusion": "per_layer",
                },
            },
            {
                "id": "layout_slice_native",
                "label": "slice_native",
                "patch": {
                    "model.projection.layout": "slice_native",
                    "model.pooling.layer_fusion": "concat",
                },
            },
            {
                "id": "layout_multi_tower",
                "label": "multi_tower",
                "patch": {
                    "model.projection.layout": "multi_tower",
                    "model.pooling.layer_fusion": "per_layer",
                },
            },
        ],
    },
    "masked": {
        "id": "F-04",
        "description": "Masked vs unmasked pooling",
        "variants": [
            {
                "id": "masked_on",
                "label": "masked",
                "patch": {
                    "grain_detection.masked_pooling": True,
                    "grain_detection.use_mask": True,
                    "model.pooling.masked.enabled": True,
                },
            },
            {
                "id": "masked_off",
                "label": "unmasked",
                "patch": {
                    "grain_detection.masked_pooling": False,
                    "grain_detection.use_mask": False,
                    "model.pooling.masked.enabled": False,
                },
            },
        ],
    },
    "augment": {
        "id": "F-05",
        "description": "Augmentation mode ablation",
        "variants": [
            {"id": "aug_full", "label": "full", "patch": {"sota.augmentation_mode": "full"}},
            {
                "id": "aug_deit3_simple",
                "label": "deit3_simple",
                "patch": {"sota.augmentation_mode": "deit3_simple"},
            },
        ],
    },
    "freeze": {
        "id": "F-06",
        "description": "Backbone freeze strategy",
        "variants": [
            {
                "id": "freeze_head_only",
                "label": "head_only",
                "patch": {"model.freeze_backbone": True, "model.num_unfrozen_blocks": 0},
            },
            {
                "id": "freeze_last2",
                "label": "last_2_blocks",
                "patch": {"model.freeze_backbone": True, "model.num_unfrozen_blocks": 2},
            },
            {
                "id": "freeze_full",
                "label": "full_finetune",
                "patch": {"model.freeze_backbone": False},
            },
        ],
    },
    "validation_metric": {
        "id": "F-07",
        "description": "Checkpoint selection metric ablation",
        "variants": [
            {"id": "valmetric_r1", "label": "R@1", "patch": {"training.validation_metric": "recall_at_1"}},
            {"id": "valmetric_mapr", "label": "mAP@R", "patch": {"training.validation_metric": "mAP@R"}},
            {"id": "valmetric_f1", "label": "F1_macro", "patch": {"training.validation_metric": "F1_macro"}},
        ],
    },
}


def list_matrices() -> List[str]:
    return sorted(MATRIX_REGISTRY.keys())


def get_matrix(name: str) -> Dict[str, Any]:
    key = name.lower()
    if key not in MATRIX_REGISTRY:
        raise KeyError(
            f"Matriz de ablación desconocida: {name!r}. "
            f"Opciones: {', '.join(list_matrices())}"
        )
    return MATRIX_REGISTRY[key]


def _set_nested(data: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_patch(base_config: dict, patch: dict) -> dict:
    """Return deep-copied config with dotted-key patches applied."""
    cfg = copy.deepcopy(base_config)
    for key, value in patch.items():
        if isinstance(value, dict) and "." not in key:
            # Merge one-level dict under top key (e.g. model.pooling.fsw)
            top, *rest = key.split(".", 1)
            if rest:
                _set_nested(cfg, key, value)
            else:
                if top not in cfg or not isinstance(cfg[top], dict):
                    cfg[top] = {}
                cfg[top] = {**cfg.get(top, {}), **value}
        else:
            _set_nested(cfg, key, value)
    return cfg


def _get_nested(data: dict, dotted_key: str, default=None):
    cur = data
    for key in dotted_key.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def variant_matches_baseline(base_config: dict, variant: Variant) -> bool:
    """True if the variant patch would not change the current config."""
    for key, new_val in (variant.get("patch") or {}).items():
        if _get_nested(base_config, key) != new_val:
            return False
    return True


ABLATION_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "when": "after_training",
    "skip_baseline_variant": True,
    "on_failure": "continue",
    "epochs": None,
    "matrices": {
        "pooling": False,
        "slices": False,
        "layout": False,
        "masked": False,
        "augment": False,
        "freeze": False,
        "validation_metric": False,
    },
}


def ensure_ablation_defaults(config: dict) -> bool:
    """Merge evaluation.ablation defaults into config (returns True if changed)."""
    eval_block = config.setdefault("evaluation", {})
    raw = eval_block.get("ablation")
    if raw is None:
        eval_block["ablation"] = copy.deepcopy(ABLATION_DEFAULTS)
        return True
    changed = False
    for key, default in ABLATION_DEFAULTS.items():
        if key not in raw:
            raw[key] = copy.deepcopy(default) if isinstance(default, dict) else default
            changed = True
        elif key == "matrices" and isinstance(raw.get("matrices"), dict):
            for mk, mv in ABLATION_DEFAULTS["matrices"].items():
                if mk not in raw["matrices"]:
                    raw["matrices"][mk] = mv
                    changed = True
    return changed


def resolve_ablation_settings(config: dict) -> dict:
    return (config.get("evaluation") or {}).get("ablation") or {}


def resolve_enabled_matrices(ablation_cfg: dict) -> List[str]:
    raw = ablation_cfg.get("matrices")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(m) for m in raw]
    if isinstance(raw, dict):
        return [name for name, enabled in raw.items() if enabled and name in MATRIX_REGISTRY]
    return []


def format_ablation_startup_summary(config: dict) -> str:
    """One-line summary for GUI status bar / training tab."""
    abl = resolve_ablation_settings(config)
    if not abl.get("enabled", False):
        return (
            "Sin estudio de ablación. El informe post-entrenamiento "
            "(métricas MLRC, espectral, full-image) se genera siempre "
            "tras el entrenamiento principal."
        )
    matrices = resolve_enabled_matrices(abl)
    if not matrices:
        return (
            "Estudio de ablación activado — seleccione al menos una matriz "
            "F-01…F-07 en Evaluación (cada variante re-entrena; no es el informe)."
        )
    n = count_planned_variants(config)
    labels = ", ".join(matrices)
    ep = abl.get("epochs")
    ep_note = f", {int(ep)} ép/variante" if ep is not None else ", mismas épocas que el principal"
    return (
        f"Estudio de ablación ON — {n} re-entrenamiento(s) extra ({labels}){ep_note}. "
        "Genera tabla F-01…F-07; el informe del best model corre aparte."
    )


def count_planned_variants(config: dict) -> int:
    """How many ablation training runs will launch (excluding skipped baseline)."""
    abl = resolve_ablation_settings(config)
    if not abl.get("enabled", False):
        return 0
    skip_base = bool(abl.get("skip_baseline_variant", True))
    total = 0
    for matrix_name in resolve_enabled_matrices(abl):
        for variant, _ in iter_variants(matrix_name, config):
            if skip_base and variant_matches_baseline(config, variant):
                continue
            total += 1
    return total


def build_variant_config(
    base_config: dict,
    variant: Variant,
    *,
    matrix_name: str,
    epochs_override: Optional[int] = None,
    disable_nested_ablation: bool = False,
) -> dict:
    cfg = apply_patch(base_config, variant.get("patch", {}))
    exp = cfg.setdefault("experiment", {})
    exp["name"] = variant["id"]
    exp["description"] = f"ablation:{matrix_name}:{variant.get('label', variant['id'])}"
    if epochs_override is not None:
        cfg.setdefault("training", {})["epochs"] = int(epochs_override)
    if disable_nested_ablation:
        cfg.setdefault("evaluation", {}).setdefault("ablation", {})["enabled"] = False
    return cfg


def create_ablation_run_directory(parent_run_dir: Path, variant_id: str) -> Dict[str, str]:
    """Run folder inside parent run: runs/<main>/ablations/runs/<variant_id>/"""
    run_dir = Path(parent_run_dir) / "ablations" / "runs" / variant_id
    paths = {
        "run_dir": run_dir,
        "config_dir": run_dir / "config",
        "checkpoints_dir": run_dir / "checkpoints",
        "logs_dir": run_dir / "logs",
        "images_dir": run_dir / "images",
        "pre_training_dir": run_dir / "images" / "pre_training",
        "post_training_dir": run_dir / "images" / "post_training",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in paths.items()}


def execute_integrated_ablation_study(
    config: dict,
    config_path: str,
    parent_run_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run ablation matrices automatically after main training (GUI / train.py).

    Never raises: failures are logged and optionally continued per on_failure.
    """
    import argparse

    log = logger or logging.getLogger(__name__)
    parent_run_dir = Path(parent_run_dir)
    abl = resolve_ablation_settings(config)

    if not abl.get("enabled", False):
        return {"enabled": False, "status": "skipped", "reason": "disabled"}

    when = str(abl.get("when", "after_training")).lower()
    if when != "after_training":
        return {"enabled": True, "status": "skipped", "reason": f"when={when} not implemented"}

    matrices = resolve_enabled_matrices(abl)
    if not matrices:
        return {"enabled": True, "status": "skipped", "reason": "no matrices selected"}

    epochs_raw = abl.get("epochs")
    epochs_override = int(epochs_raw) if epochs_raw is not None else None
    skip_base = bool(abl.get("skip_baseline_variant", True))
    on_failure = str(abl.get("on_failure", "continue")).lower()

    from scripts.train import run_training

    batch_dir = parent_run_dir / "ablations"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "parent_run_dir": str(parent_run_dir),
        "base_config": config_path,
        "when": when,
        "matrices": matrices,
        "runs": [],
        "errors": [],
    }

    rows: List[Dict[str, Any]] = []
    baseline_row = collect_run_metrics(parent_run_dir)
    baseline_row.update({"matrix": "baseline", "variant_id": "main_run", "variant_label": "main"})
    rows.append(baseline_row)

    completed = 0
    failed = 0
    skipped = 0

    log.info("=" * 70)
    log.info("ABLACION INTEGRADA — inicio (%d matrices)", len(matrices))
    log.info("=" * 70)

    for matrix_name in matrices:
        matrix = get_matrix(matrix_name)
        log.info("Matriz %s (%s)", matrix["id"], matrix_name)

        for variant, variant_cfg in iter_variants(
            matrix_name,
            config,
            epochs_override=epochs_override,
        ):
            variant_cfg = build_variant_config(
                config,
                variant,
                matrix_name=matrix_name,
                epochs_override=epochs_override,
                disable_nested_ablation=True,
            )

            if skip_base and variant_matches_baseline(config, variant):
                log.info("  Skip %s (ya es la config base)", variant["id"])
                skipped += 1
                continue

            paths = create_ablation_run_directory(parent_run_dir, variant["id"])
            cfg_path = Path(paths["config_dir"]) / "config.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.dump(variant_cfg, f, default_flow_style=False, allow_unicode=True)

            entry = {
                "matrix": matrix_name,
                "variant_id": variant["id"],
                "variant_label": variant.get("label"),
                "config_path": str(cfg_path),
                "run_dir": paths["run_dir"],
                "status": "pending",
            }
            manifest["runs"].append(entry)

            args = argparse.Namespace(config=str(cfg_path), resume=None, gpu=None)
            try:
                log.info("  Entrenando variante %s ...", variant["id"])
                run_training(
                    args,
                    estimate_only=False,
                    run_dir=paths["run_dir"],
                    _ablation_child=True,
                )
                entry["status"] = "completed"
                completed += 1
                row = collect_run_metrics(Path(paths["run_dir"]))
                row.update({
                    "matrix": matrix_name,
                    "variant_id": variant["id"],
                    "variant_label": variant.get("label"),
                })
                rows.append(row)
            except Exception as exc:
                failed += 1
                entry["status"] = "failed"
                entry["error"] = str(exc)
                manifest["errors"].append({"variant_id": variant["id"], "error": str(exc)})
                log.error("  Variante %s falló: %s", variant["id"], exc, exc_info=True)
                if on_failure == "abort":
                    break
        if on_failure == "abort" and failed:
            break

    manifest_path = batch_dir / "manifest.json"
    save_manifest(manifest_path, manifest)
    csv_path = batch_dir / "ablation_results.csv"
    write_results_csv(rows, csv_path)

    status = "completed" if failed == 0 else "partial"
    result = {
        "enabled": True,
        "status": status,
        "when": when,
        "matrices": matrices,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "errors": manifest["errors"],
    }
    log.info(
        "ABLACION INTEGRADA — fin: completed=%d failed=%d skipped=%d csv=%s",
        completed,
        failed,
        skipped,
        csv_path,
    )
    return result


def iter_variants(matrix_name: str, base_config: dict, **kwargs) -> Iterable[Tuple[Variant, dict]]:
    matrix = get_matrix(matrix_name)
    for variant in matrix["variants"]:
        yield variant, build_variant_config(base_config, variant, matrix_name=matrix_name, **kwargs)


def find_run_dir_by_experiment(runs_dir: Path, experiment_name: str) -> Optional[Path]:
    """Locate newest run whose saved config matches experiment.name."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return None
    candidates = []
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        for cfg_path in (run_path / "config" / "config.yaml", run_path / "config.yaml"):
            if not cfg_path.exists():
                continue
            try:
                with open(cfg_path, encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                if cfg.get("experiment", {}).get("name") == experiment_name:
                    candidates.append((run_path.stat().st_mtime, run_path))
            except (yaml.YAMLError, OSError):
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def collect_run_metrics(run_dir: Path) -> Dict[str, Any]:
    """Extract MLRC + spectral metrics from a completed run."""
    run_dir = Path(run_dir)
    row: Dict[str, Any] = {"run_dir": str(run_dir)}

    for summary_name in (
        "evaluation_metrics_summary_final.json",
        "evaluation_metrics_summary_test.json",
        "evaluation_metrics_summary_val.json",
    ):
        p = run_dir / summary_name
        if p.exists():
            with open(p, encoding="utf-8") as f:
                summary = json.load(f)
            ml = summary.get("metric_learning") or {}
            row.update({
                "R@1": ml.get("R@1"),
                "R@5": ml.get("R@5"),
                "mAP@R": ml.get("mAP@R"),
                "NMI": ml.get("NMI"),
                "F1_macro": ml.get("F1_macro"),
                "slice_accuracy": ml.get("slice_accuracy"),
                "knn_accuracy": ml.get("knn_accuracy"),
                "distance_ratio": ml.get("distance_ratio_inter_intra"),
            })
            break

    for spectral_rel in (
        "images/post_training/val/spectral_analysis/spectral_summary.json",
        "images/post_training/test/spectral_analysis/spectral_summary.json",
    ):
        sp = run_dir / spectral_rel
        if sp.exists():
            with open(sp, encoding="utf-8") as f:
                spec = json.load(f)
            row["effective_rank"] = spec.get("global_effective_rank")
            row["embedding_dim"] = spec.get("embedding_dim")
            row["collapse_severity"] = spec.get("collapse_severity")
            break

    csv_path = run_dir / "training_metrics.csv"
    if csv_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            if "mAP@R" in df.columns and len(df):
                best_idx = df["mAP@R"].idxmax()
                row["best_epoch_mapr"] = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else None
        except Exception as exc:
            logger.debug("CSV parse skipped for %s: %s", run_dir, exc)

    cfg_path = run_dir / "config" / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        row["experiment_name"] = (cfg.get("experiment") or {}).get("name")
        row["experiment_desc"] = (cfg.get("experiment") or {}).get("description")

    return row


def write_results_csv(rows: List[Dict[str, Any]], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Sin filas para exportar CSV de ablación")

    fieldnames: List[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Ablación CSV: %s (%d filas)", output_path, len(rows))
    return output_path


def save_manifest(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aggregate_manifest_to_csv(manifest_path: Path, output_csv: Optional[Path] = None) -> Path:
    manifest = load_manifest(manifest_path)
    rows = []
    for entry in manifest.get("runs", []):
        run_dir = entry.get("run_dir")
        if not run_dir:
            continue
        metrics = collect_run_metrics(Path(run_dir))
        metrics["matrix"] = manifest.get("matrix")
        metrics["variant_id"] = entry.get("variant_id")
        metrics["variant_label"] = entry.get("variant_label")
        rows.append(metrics)
    out = output_csv or Path(manifest_path).with_suffix(".csv")
    return write_results_csv(rows, out)


def create_batch_dir(base: Path, matrix_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    batch = Path(base) / f"{matrix_name}_{ts}"
    batch.mkdir(parents=True, exist_ok=True)
    (batch / "configs").mkdir(exist_ok=True)
    return batch
