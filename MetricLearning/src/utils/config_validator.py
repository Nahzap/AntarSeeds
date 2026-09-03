"""
Validación de precondiciones del pipeline MetricLearning.

Usado al arranque de la GUI y en scripts CLI para detectar configuraciones
inconsistentes antes de entrenar o inferir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.models.pooling_config import (
    resolve_pooling_config,
    resolve_projection_config,
    validate_pooling_config,
    validate_projection_config,
)
from src.utils.resolution_config import resolve_effective_image_size
from src.utils.detector_resolution_config import resolve_detector_input_size, resolve_detector_patch_size
from src.grain_detection.detector_registry import validate_active_detector, normalize_backend
from src.utils.sota_integrity import validate_sota_integrity
from src.utils.mlrc_protocol import normalize_validation_metric


from src.training.detector_loss_factory import DETECTOR_LOSS_TYPES

DETECTOR_VALIDATION_METRICS = ("det_f1", "Fm", "mask_iou")

LOSS_REQUIRED_PARAMS = {
    "multi_similarity": ["alpha", "beta", "base"],
    "sliced_ms": ["num_slices", "alpha", "beta", "base_margin"],
    "sliced_proxy": ["num_slices", "proxy_scale", "proxy_margin"],
    "ldm_ms": ["alpha", "beta"],
    "hybrid_proxy_ms": ["alpha", "beta", "proxy_scale", "proxy_margin"],
}


def _validate_loss_params(loss_type: str, loss_params: dict) -> List[str]:
    """C-03: required training.loss.params per loss type (no silent defaults)."""
    errors: List[str] = []
    required = LOSS_REQUIRED_PARAMS.get(loss_type, [])
    for key in required:
        if key not in loss_params:
            errors.append(
                f"training.loss.params.{key} es obligatorio para loss.type={loss_type}"
            )
    if loss_type in ("ldm_ms", "hybrid_proxy_ms"):
        if "base_margin" not in loss_params and "base" not in loss_params:
            errors.append(
                f"training.loss.params.base_margin (o base) es obligatorio para "
                f"loss.type={loss_type}"
            )
    return errors


def _get_nested(config: dict, *keys, default=None):
    cur = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_pipeline_summary(config: dict) -> str:
    """Resumen legible del pipeline configurado."""
    loss_type = _get_nested(config, "training", "loss", "type", default="?")
    num_slices = _get_nested(config, "training", "loss", "params", "num_slices", default=4)
    embed_dim = _get_nested(config, "model", "embedding_dim", default=128)
    slice_dim = embed_dim // num_slices if num_slices else "?"
    sd = _get_nested(config, "sota", "self_distillation", "enabled", default=False)
    sd_label = "on" if sd else "off"
    pooling = resolve_pooling_config(config.get("model", {}), config.get("grain_detection", {}))
    layout = resolve_projection_config(config.get("model", {}))["layout"]
    masked = "on" if pooling.get("masked", {}).get("enabled") else "off"
    towers = f" ({layout})" if layout != "multi_tower" else f" (multi_tower×{num_slices})"
    image_size = resolve_effective_image_size(config)
    return (
        f"{loss_type} {num_slices}×{slice_dim}D | "
        f"{image_size}px | "
        f"pooling={pooling.get('strategy')} masked={masked} | "
        f"proj={layout}{towers} | self-distill {sd_label}"
    )


def validate_config(
    config: dict,
    config_path: Optional[str] = None,
    check_data_dirs: bool = True,
) -> Dict[str, Any]:
    """
    Valida coherencia de configuración del pipeline multi-slice.

    Returns:
        dict con keys: valid, errors, warnings, pipeline_summary
    """
    errors: List[str] = []
    warnings: List[str] = []

    embed_dim = int(_get_nested(config, "model", "embedding_dim", default=128))
    loss_type = _get_nested(config, "training", "loss", "type", default="")
    loss_params = _get_nested(config, "training", "loss", "params", default={}) or {}
    num_slices = int(loss_params.get("num_slices", 4))
    classes_per_batch = int(_get_nested(config, "data", "classes_per_batch", default=8))

    if embed_dim % num_slices != 0:
        errors.append(
            f"model.embedding_dim ({embed_dim}) no es divisible por "
            f"num_slices ({num_slices})"
        )

    if loss_type in ("sliced_ms", "sliced_proxy"):
        if classes_per_batch < 2:
            errors.append(
                f"training.loss.type={loss_type} requiere data.classes_per_batch >= 2 "
                f"(actual: {classes_per_batch})"
            )
    errors.extend(_validate_loss_params(loss_type, loss_params))

    model_cfg = config.get("model", {}) or {}
    grain_cfg = config.get("grain_detection", {}) or {}
    pooling_cfg = resolve_pooling_config(model_cfg, grain_cfg)

    embed_dim_allowed = {128, 256, 384}
    if embed_dim not in embed_dim_allowed:
        warnings.append(
            f"model.embedding_dim={embed_dim} fuera de ablación estándar "
            f"{sorted(embed_dim_allowed)} (D-02)"
        )

    proj_head = model_cfg.get("projection_head", {}) or {}
    if proj_head.get("use_batch_norm") and proj_head.get(
        "use_layer_norm", proj_head.get("normalize_embeddings_ln")
    ):
        warnings.append(
            "projection_head: use_batch_norm y use_layer_norm simultáneos; "
            "preferir LN para batch pequeño (D-01)"
        )

    aug_mode = str(_get_nested(config, "sota", "augmentation_mode", default="full")).lower()
    if aug_mode not in ("full", "deit3", "deit3_simple"):
        errors.append(
            f"sota.augmentation_mode debe ser full|deit3|deit3_simple, recibido: {aug_mode!r}"
        )

    pooling_strategy = pooling_cfg.get("strategy", "multi_spectral")
    if pooling_strategy == "fsw":
        fsw_cfg = pooling_cfg.get("fsw") or {}
        if fsw_cfg.get("use_external", False):
            try:
                import fswlib  # type: ignore  # noqa: F401
            except ImportError:
                errors.append(
                    "model.pooling.strategy=fsw con fsw.use_external=true requiere fswlib"
                )

    finetune_phases = _get_nested(config, "training", "finetune_phases", default=[]) or []
    if finetune_phases:
        phase_epochs = sum(int(p.get("epochs", 0)) for p in finetune_phases)
        total_epochs = int(_get_nested(config, "training", "epochs", default=0) or 0)
        if phase_epochs != total_epochs:
            warnings.append(
                f"training.finetune_phases suma {phase_epochs} épocas pero "
                f"training.epochs={total_epochs} (D-05)"
            )

    ac_cfg = (config.get("sota") or {}).get("anti_collapse", {}) or {}
    if ac_cfg.get("enabled", False) and float(ac_cfg.get("lambda", 0.1)) <= 0:
        errors.append("sota.anti_collapse.lambda debe ser > 0 cuando enabled=true")

    loss_distance = _get_nested(config, "training", "loss", "distance")
    if loss_distance is not None and str(loss_distance).lower() not in (
        "cosine", "euclidean", "lp"
    ):
        errors.append(
            f"training.loss.distance debe ser cosine|euclidean|lp, recibido: {loss_distance!r}"
        )

    use_gpu = _get_nested(config, "hardware", "use_gpu", default=True)
    if use_gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                warnings.append("hardware.use_gpu=true pero CUDA no está disponible")
        except ImportError:
            warnings.append("PyTorch no importable; no se pudo verificar GPU")

    if check_data_dirs:
        for split_key in ("train_dir", "val_dir", "test_dir"):
            rel = _get_nested(config, "data", split_key)
            if not rel:
                warnings.append(f"data.{split_key} no definido")
                continue
            path = Path(rel)
            if not path.exists():
                warnings.append(f"Directorio no encontrado: {rel} (data.{split_key})")
            elif not any(path.iterdir()) if path.is_dir() else False:
                warnings.append(f"Directorio vacío: {rel} (data.{split_key})")

    if config_path and not Path(config_path).exists():
        errors.append(f"Archivo de configuración no encontrado: {config_path}")

    # Parámetros huérfanos bajo sliced_ms — error (no deben figurar en el paper)
    if loss_type == "sliced_ms":
        orphan_keys = {"proxy_scale", "proxy_margin", "margin", "scale", "base", "use_ldm"}
        present = orphan_keys.intersection(loss_params.keys())
        if present:
            errors.append(
                f"Parámetros no usados por sliced_ms en training.loss.params: "
                f"{sorted(present)} — elimínelos del config"
            )

    sota = config.get("sota") or {}
    for ghost_key in ("uniformity", "decorrelation", "disentanglement", "amplitude_penalty"):
        block = sota.get(ghost_key) or {}
        if block.get("enabled", False):
            errors.append(
                f"sota.{ghost_key}.enabled=true no está implementado en el trainer; "
                "desactívelo o use NIR/HIER/anti_collapse"
            )

    if config.get("augmentation", {}).get("performance_mode") and (
        config.get("augmentation", {}).get("domain_augmentation", {}).get("enabled")
        or config.get("augmentation", {})
        .get("microscopy_specific", {})
        .get("optical_distortion", {})
        .get("enabled")
    ):
        warnings.append(
            "augmentation.performance_mode=true desactiva domain_augmentation y "
            "optical_distortion aunque aparezcan enabled en YAML"
        )

    projection_layout = resolve_projection_config(model_cfg)["layout"]
    projection_cfg = resolve_projection_config(model_cfg)
    p_errors, p_warnings = validate_pooling_config(pooling_cfg, projection_layout)
    errors.extend(p_errors)
    warnings.extend(p_warnings)
    pr_errors, pr_warnings = validate_projection_config(
        projection_cfg, pooling_cfg, embed_dim, num_slices
    )
    errors.extend(pr_errors)
    warnings.extend(pr_warnings)

    masked_pooling = bool(grain_cfg.get("masked_pooling", False))
    pooling_masked = bool(pooling_cfg.get("masked", {}).get("enabled", False))
    if masked_pooling != pooling_masked:
        warnings.append(
            f"grain_detection.masked_pooling ({masked_pooling}) != "
            f"model.pooling.masked.enabled ({pooling_masked})"
        )

    raw_pooling = model_cfg.get("pooling") or {}
    explicit_threshold = raw_pooling.get("masked", {}).get("threshold")
    saliency = grain_cfg.get("saliency_threshold")
    loc_detector = str(grain_cfg.get("localization_detector", "u2net")).strip().lower()
    active = grain_cfg.get("active_detector") or {}
    active_backend = active.get("backend") or loc_detector
    try:
        normalize_backend(active_backend)
    except ValueError:
        errors.append(
            f"grain_detection.active_detector.backend inválido: {active_backend!r} "
            "(use u2net, vit_dense o vit)"
        )
    errors.extend(validate_active_detector(config))

    crop_source = str(_get_nested(config, "data", "crop_source", default="seg")).lower()
    if crop_source not in ("seg", "detector"):
        errors.append("data.crop_source debe ser 'seg' o 'detector'")

    det_cfg = config.get("detection_training") or {}
    if det_cfg.get("enabled", True):
        input_size = resolve_detector_input_size(config)
        patch_size = resolve_detector_patch_size(config)
        if input_size % patch_size != 0:
            errors.append(
                f"detection_training.input_size ({input_size}) debe ser múltiplo de "
                f"patch_size ({patch_size})"
            )
        if int(det_cfg.get("num_unfrozen_blocks", 2)) < 0:
            errors.append("detection_training.num_unfrozen_blocks debe ser >= 0")
        loss_type = str((det_cfg.get("loss") or {}).get("type", "bce_dice")).lower()
        if loss_type not in DETECTOR_LOSS_TYPES:
            errors.append(
                f"detection_training.loss.type={loss_type!r} inválido; "
                f"use: {', '.join(DETECTOR_LOSS_TYPES)}"
            )
        val_metric = str(det_cfg.get("validation_metric", "det_f1"))
        if val_metric.lower() == "fm":
            val_metric = "Fm"
        if val_metric not in DETECTOR_VALIDATION_METRICS:
            errors.append(
                f"detection_training.validation_metric={val_metric!r} inválido; "
                f"use: {', '.join(DETECTOR_VALIDATION_METRICS)}"
            )
        max_side = grain_cfg.get("max_load_side")
        if max_side is not None and int(max_side) != input_size:
            warnings.append(
                f"grain_detection.max_load_side ({max_side}) != detection_training.input_size "
                f"({input_size}); se sincronizará al guardar"
            )
    if loc_detector not in ("u2net", "pollen", "vit", "model", "analogynet", "vit_attention", "u2-net", "vit_dense"):
        warnings.append(
            f"grain_detection.localization_detector legacy: {loc_detector!r} "
            "(preferir active_detector.backend)"
        )
    resolved_threshold = pooling_cfg.get("masked", {}).get("threshold")
    if (
        explicit_threshold is not None
        and saliency is not None
        and float(explicit_threshold) != float(saliency)
    ):
        warnings.append(
            f"model.pooling.masked.threshold ({explicit_threshold}) != "
            f"grain_detection.saliency_threshold ({saliency}); "
            "unificar para evitar divergencia train/inference"
        )
    elif saliency is not None and explicit_threshold is None:
        if float(resolved_threshold) != float(saliency):
            warnings.append(
                f"masked threshold resuelto ({resolved_threshold}) difiere de "
                f"saliency_threshold ({saliency})"
            )

    image_size = resolve_effective_image_size(config)
    resize_size = _get_nested(config, "data", "resize_size")
    grain_crop = grain_cfg.get("crop_size")
    if resize_size is not None and int(resize_size) != image_size:
        warnings.append(
            f"data.resize_size ({resize_size}) != data.image_size ({image_size}); "
            "se sincronizará al guardar/entrenar"
        )
    if grain_crop is not None and int(grain_crop) != image_size:
        warnings.append(
            f"grain_detection.crop_size ({grain_crop}) != data.image_size ({image_size}); "
            "el entrenamiento usa data.image_size (campo obsoleto en GUI)"
        )

    u2_input = int(grain_cfg.get("input_size", 320))
    if u2_input != image_size:
        warnings.append(
            f"grain_detection.input_size ({u2_input}) != data.image_size ({image_size}): "
            "esperado — U²-Net solo segmenta/recorta; el ViT entrena a image_size (E-05)"
        )

    norm_mode = str(_get_nested(config, "data", "normalize", "mode", default="imagenet")).lower()
    if norm_mode not in ("imagenet", "dataset", "macenko"):
        errors.append(
            f"data.normalize.mode debe ser imagenet|dataset|macenko, recibido: {norm_mode!r}"
        )
    if norm_mode == "dataset":
        stats_path = _get_nested(config, "data", "normalize", "stats_path")
        if check_data_dirs and stats_path and not Path(stats_path).exists():
            warnings.append(
                f"data.normalize.mode=dataset pero stats_path no existe: {stats_path} "
                "(ejecutar scripts/compute_dataset_norm.py)"
            )

    for bg_key in ("train_mask_bg_mode", "val_mask_bg_mode"):
        bg_mode = str(grain_cfg.get(bg_key, "gray")).lower()
        if bg_mode not in ("imagenet_neutral", "random", "gray", "black", "none"):
            errors.append(
                f"grain_detection.{bg_key} inválido: {bg_mode!r} "
                "(imagenet_neutral|random|gray|black|none)"
            )

    sota_errors, sota_warnings = validate_sota_integrity(config, check_files=check_data_dirs)
    errors.extend(sota_errors)
    warnings.extend(sota_warnings)

    eval_cfg = config.get("evaluation", {}) or {}
    validation_metric = _get_nested(config, "training", "validation_metric")
    if validation_metric is not None:
        try:
            normalize_validation_metric(str(validation_metric))
        except ValueError as exc:
            errors.append(str(exc))

    nn_metric = eval_cfg.get("nn_metric")
    if nn_metric is not None and str(nn_metric).lower() not in (
        "cosine", "euclidean", "hyperbolic"
    ):
        errors.append(
            f"evaluation.nn_metric debe ser cosine|euclidean|hyperbolic|null, "
            f"recibido: {nn_metric!r}"
        )

    map_cfg = eval_cfg.get("map_at_r", {}) or {}
    if map_cfg.get("exact") is False:
        warnings.append(
            "evaluation.map_at_r.exact=false: mAP@R aproximado solo para debug, "
            "no válido para publicación (Musgrave et al. MLRC)"
        )

    nmi_cfg = eval_cfg.get("nmi", {}) or {}
    n_init = int(nmi_cfg.get("n_init", 10))
    n_seeds = int(nmi_cfg.get("n_seeds", 1))
    if n_init < 1:
        errors.append("evaluation.nmi.n_init debe ser >= 1")
    if n_seeds < 1:
        errors.append("evaluation.nmi.n_seeds debe ser >= 1")

    n_runs = int(eval_cfg.get("n_runs", 1))
    if n_runs < 1:
        errors.append("evaluation.n_runs debe ser >= 1")
    elif n_runs < 10:
        warnings.append(
            f"evaluation.n_runs={n_runs}: MLRC recomienda ≥10 runs para intervalos de confianza"
        )

    report_final_on = str(eval_cfg.get("report_final_on", "test")).lower()
    test_dir = _get_nested(config, "data", "test_dir")
    if report_final_on == "test" and check_data_dirs:
        if not test_dir or not Path(test_dir).exists():
            warnings.append(
                "evaluation.report_final_on=test pero data.test_dir no existe; "
                "la evaluación final caerá en val (protocolo MLRC incompleto)"
            )

    abl_cfg = eval_cfg.get("ablation") or {}
    if abl_cfg:
        when = str(abl_cfg.get("when", "after_training")).lower()
        if when not in ("after_training",):
            errors.append(
                f"evaluation.ablation.when debe ser after_training, recibido: {when!r}"
            )
        on_failure = str(abl_cfg.get("on_failure", "continue")).lower()
        if on_failure not in ("continue", "abort"):
            errors.append(
                f"evaluation.ablation.on_failure debe ser continue|abort, recibido: {on_failure!r}"
            )
        epochs_raw = abl_cfg.get("epochs")
        if epochs_raw is not None:
            ep = int(epochs_raw)
            # GUI usa 0 = mismas épocas que training.epochs (se guarda como null)
            if ep == 0:
                pass
            elif ep < 1:
                errors.append("evaluation.ablation.epochs debe ser >= 1, 0 (auto) o null")
        matrices = abl_cfg.get("matrices")
        if abl_cfg.get("enabled", False):
            if matrices is None:
                warnings.append(
                    "evaluation.ablation.enabled=true pero matrices vacío; "
                    "no se lanzará ninguna variante"
                )
            elif isinstance(matrices, dict):
                unknown = [k for k in matrices if k not in (
                    "pooling", "slices", "layout", "masked", "augment", "freeze", "validation_metric"
                )]
                if unknown:
                    warnings.append(
                        f"evaluation.ablation.matrices: claves desconocidas {unknown}"
                    )
                if not any(bool(v) for v in matrices.values()):
                    warnings.append(
                        "evaluation.ablation.enabled=true pero ninguna matriz activa"
                    )
            elif isinstance(matrices, list) and not matrices:
                warnings.append(
                    "evaluation.ablation.enabled=true pero matrices=[]"
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "pipeline_summary": build_pipeline_summary(config),
    }


def validate_config_file(
    config_path: str,
    check_data_dirs: bool = True,
) -> Dict[str, Any]:
    """Carga YAML y valida."""
    path = Path(config_path)
    if not path.exists():
        return {
            "valid": False,
            "errors": [f"Archivo no encontrado: {config_path}"],
            "warnings": [],
            "pipeline_summary": "config no cargada",
        }
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    result = validate_config(config, config_path=str(path), check_data_dirs=check_data_dirs)
    result["config_path"] = str(path.resolve())
    return result
