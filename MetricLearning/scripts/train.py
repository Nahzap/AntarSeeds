"""
Training Script for Metric Learning System
Implements SOTA techniques (2024) for honey type classification

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --resume models/checkpoint_epoch_20.pth

Author: MetricLearning Project
Date: 2026-01-05
"""

import argparse
import yaml
import torch
import logging
import time
import statistics
from pathlib import Path
from torch.utils.data import DataLoader

from src.models.analogy_net import AnalogyNet
from src.data.data_engine import GrainDataEngine, resolve_dataloader_workers
from src.training.trainer_enhanced import MetricLearningTrainer
from src.utils.config_utils import get_backbone_name
from src.utils.training_alarms import TrainingAlarmManager


def setup_logging(log_dir: str = 'logs', log_level: str = 'INFO'):
    """Setup logging configuration (ASCII-safe on Windows cp1252 consoles)."""
    from src.utils.unicode_utils import make_logger_ascii_safe

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, log_level))

    file_handler = logging.FileHandler(f'{log_dir}/training.log', encoding='utf-8')
    stream_handler = logging.StreamHandler()
    if hasattr(stream_handler.stream, 'reconfigure'):
        try:
            stream_handler.stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    formatter = logging.Formatter(fmt)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    make_logger_ascii_safe(root)

    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: dict):
    """
    Create train and validation dataloaders with M-Per-Class sampling.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, GrainDataEngine)
    """
    engine = GrainDataEngine.from_config(config)
    data_config = config["data"]
    logger = logging.getLogger(__name__)
    logger.info(
        f"Batch configuration: {data_config['classes_per_batch']} classes x "
        f"{data_config['samples_per_class']} samples = {data_config['batch_size']} batch size"
    )
    return engine.train_loader, engine.val_loader, engine


def create_model(config: dict) -> AnalogyNet:
    """
    Create AnalogyNet model.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AnalogyNet model
    """
    logger = logging.getLogger(__name__)
    
    from src.utils.model_factory import create_analogy_net_from_config
    model = create_analogy_net_from_config(config)
    model_config = config['model']
    
    logger.info(f"Model created: {get_backbone_name(config)}")
    logger.info(f"Embedding dimension: {model_config['embedding_dim']}")
    logger.info(f"Pretrained: {model_config['pretrained']}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def _unpack_batch(batch_data, device):
    """Support (images, labels) or (images, labels, masks) batches."""
    if len(batch_data) == 3:
        images, labels, masks = batch_data
        masks = masks.to(device, non_blocking=True)
    else:
        images, labels = batch_data
        masks = None
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    return images, labels, masks


def estimate_training_time(config: dict, train_loader, val_loader, model, logger, benchmark_batches: int = 8) -> dict:
    """
    Benchmarks real training steps (forward + loss + backward + optimizer)
    and validation batches to estimate total training time accurately.
    Uses the same configured batch size as training for a closer estimate.
    
    Args:
        config: Full configuration dictionary
        train_loader: Training DataLoader (used only for batch count / dataset size)
        val_loader: Validation DataLoader (used only for batch count / dataset size)
        model: The model (already on device)
        logger: Logger instance
        benchmark_batches: Number of batches to benchmark
        
    Returns:
        Dict with detailed time estimates per phase and total
    """
    from torch.amp import autocast, GradScaler
    from pytorch_metric_learning.losses import MultiSimilarityLoss, TripletMarginLoss
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['hardware'].get('use_gpu', True) else 'cpu')
    model.to(device)
    model.train()
    
    use_amp = config['hardware'].get('use_amp', True)
    epochs = config['training']['epochs']
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    batch_size = config['data']['batch_size']
    
    # Setup loss function for realistic benchmark
    # Filter params to only those accepted by each loss type
    loss_type = config['training']['loss']['type']
    all_loss_params = config['training']['loss'].get('params', {})
    if loss_type == 'multi_similarity':
        ms_keys = {'alpha', 'beta', 'base'}
        filtered = {k: v for k, v in all_loss_params.items() if k in ms_keys}
        bench_loss_func = MultiSimilarityLoss(**filtered).to(device)
    elif loss_type == 'triplet':
        tri_keys = {'margin', 'triplet_margin'}
        filtered = {k: v for k, v in all_loss_params.items() if k in tri_keys}
        if 'triplet_margin' in filtered:
            filtered['margin'] = filtered.pop('triplet_margin')
        bench_loss_func = TripletMarginLoss(**filtered).to(device)
    else:
        bench_loss_func = MultiSimilarityLoss().to(device)
    
    # Setup optimizer and scaler for realistic benchmark
    bench_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bench_scaler = GradScaler('cuda') if use_amp else None
    gradient_clip_norm = config['training'].get('gradient_clip_norm', 5.0)
    
    # Create lightweight single-process loaders for benchmarking
    # Keep real batch size to avoid optimistic estimates.
    bench_batch_size = batch_size
    logger.info(f"Using batch_size={bench_batch_size} for benchmark (original: {batch_size})")
    
    collate_fn = getattr(train_loader, "collate_fn", None)
    bench_kw = dict(
        batch_size=bench_batch_size,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    bench_train_loader = DataLoader(
        train_loader.dataset,
        shuffle=True,
        drop_last=True,
        **bench_kw,
    )
    bench_val_loader = DataLoader(
        val_loader.dataset,
        shuffle=False,
        drop_last=False,
        **bench_kw,
    )
    
    # --- CUDA warmup (2 batches, NOT timed) ---
    # First CUDA ops trigger kernel compilation + memory allocation → 5-10x slower.
    # Discard warmup batches so they don't pollute the estimate.
    warmup_batches = 2
    logger.info(f"CUDA warmup ({warmup_batches} batches, not timed)...")
    bench_train_iter = iter(bench_train_loader)
    for i in range(warmup_batches):
        logger.info(f"  Warmup batch {i+1}/{warmup_batches}: loading...")
        try:
            batch_data = next(bench_train_iter)
        except StopIteration:
            bench_train_iter = iter(bench_train_loader)
            batch_data = next(bench_train_iter)
        logger.info(f"  Warmup batch {i+1}/{warmup_batches}: loaded, processing...")
        images, labels, masks = _unpack_batch(batch_data, device)
        bench_optimizer.zero_grad(set_to_none=True)
        
        logger.info(f"    -> Forward pass...")
        if use_amp:
            with autocast('cuda'):
                embeddings = model(images, mask=masks)
                logger.info(f"    -> Loss computation...")
                loss = bench_loss_func(embeddings, labels)
            logger.info(f"    -> Backward pass...")
            bench_scaler.scale(loss).backward()
            bench_scaler.unscale_(bench_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            logger.info(f"    -> Optimizer step...")
            bench_scaler.step(bench_optimizer)
            bench_scaler.update()
        else:
            embeddings = model(images, mask=masks)
            logger.info(f"    -> Loss computation...")
            loss = bench_loss_func(embeddings, labels)
            logger.info(f"    -> Backward pass...")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            logger.info(f"    -> Optimizer step...")
            bench_optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        logger.info(f"  Warmup batch {i+1}/{warmup_batches}: complete")
    logger.info("Warmup complete.")
    
    # --- Benchmark FULL training steps (forward + loss + backward + optimizer) ---
    actual_benchmark_batches = min(benchmark_batches, num_train_batches)
    logger.info(f"Benchmarking {actual_benchmark_batches} training steps (forward + loss + backward + optimizer)...")
    train_times = []
    data_load_times = []
    
    t_data_start = time.perf_counter()
    for i in range(actual_benchmark_batches):
        if i % 2 == 0:
            logger.info(f"  Benchmark batch {i+1}/{actual_benchmark_batches}...")
        try:
            batch_data = next(bench_train_iter)
        except StopIteration:
            bench_train_iter = iter(bench_train_loader)
            batch_data = next(bench_train_iter)
        images, labels, masks = _unpack_batch(batch_data, device)
        data_load_times.append(time.perf_counter() - t_data_start)
        
        t0 = time.perf_counter()
        bench_optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with autocast('cuda'):
                embeddings = model(images, mask=masks)
                loss = bench_loss_func(embeddings, labels)
            bench_scaler.scale(loss).backward()
            bench_scaler.unscale_(bench_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            bench_scaler.step(bench_optimizer)
            bench_scaler.update()
        else:
            embeddings = model(images, mask=masks)
            loss = bench_loss_func(embeddings, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            bench_optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_times.append(time.perf_counter() - t0)
        t_data_start = time.perf_counter()
    
    # --- Benchmark validation batches ---
    logger.info("Benchmarking validation batches...")
    val_times = []
    model.eval()
    bench_val_iter = iter(bench_val_loader)
    with torch.no_grad():
        for i in range(min(benchmark_batches, num_val_batches)):
            try:
                batch_data = next(bench_val_iter)
            except StopIteration:
                break
            images, labels, masks = _unpack_batch(batch_data, device)
            
            t0 = time.perf_counter()
            if use_amp:
                with autocast('cuda'):
                    embeddings = model(images, mask=masks)
            else:
                embeddings = model(images, mask=masks)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            val_times.append(time.perf_counter() - t0)
    
    # --- Calculate estimates (use MEDIAN for robustness against outliers) ---
    avg_train_step = statistics.median(train_times) if train_times else 1.0
    avg_val_batch = statistics.median(val_times) if val_times else 0.5
    avg_data_load = statistics.median(data_load_times) if data_load_times else 0.1
    
    # Data loading overhead:
    #   num_workers > 0: loading is overlapped with GPU work; add ~15% for stalls
    #   num_workers = 0: loading is BLOCKING (sequential), so total = step + data_load
    num_workers = int(getattr(train_loader, "num_workers", 0))
    val_workers = int(getattr(val_loader, "num_workers", 0))
    if num_workers > 0:
        data_overhead_factor = 1.0 + (avg_data_load / avg_train_step) if avg_train_step > 0 else 1.15
    else:
        data_overhead_factor = 1.0 + (avg_data_load / avg_train_step) if avg_train_step > 0 else 1.5
    val_data_factor = 1.0 + (avg_data_load / avg_val_batch) if val_workers > 0 and avg_val_batch > 0 else (
        1.0 + (avg_data_load / avg_val_batch) if avg_val_batch > 0 else 1.0
    )
    
    avg_train_batch_full = avg_train_step * data_overhead_factor
    
    recall_overhead_sec = float(
        config.get("evaluation", {}).get("metrics_overhead_sec", 5.0)
    )
    avg_val_batch_full = avg_val_batch * val_data_factor
    
    # Reference embedding extraction: happens ONCE at the end (final_model.pth),
    # NOT during training epochs.  Centroids are also computed once post-training.
    # So per-epoch overhead = 0; one-time cost added separately.
    ref_embed_one_time_sec = num_train_batches * avg_val_batch  # one full pass over train set
    
    train_epoch_sec = avg_train_batch_full * num_train_batches
    val_epoch_sec = (avg_val_batch_full * num_val_batches) + recall_overhead_sec
    epoch_total_sec = train_epoch_sec + val_epoch_sec
    total_sec = (epoch_total_sec * epochs) + ref_embed_one_time_sec
    
    def fmt_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m {s}s"
        else:
            h, rem = divmod(int(seconds), 3600)
            m, s = divmod(rem, 60)
            return f"{h}h {m}m {s}s"
    
    estimate = {
        'avg_train_batch_sec': avg_train_batch_full,
        'avg_val_batch_sec': avg_val_batch,
        'avg_train_step_sec': avg_train_step,
        'avg_data_load_sec': avg_data_load,
        'data_overhead_factor': data_overhead_factor,
        'train_batches_per_epoch': num_train_batches,
        'val_batches_per_epoch': num_val_batches,
        'train_epoch_sec': train_epoch_sec,
        'val_epoch_sec': val_epoch_sec,
        'ref_embed_overhead_sec': ref_embed_one_time_sec,
        'epoch_total_sec': epoch_total_sec,
        'epochs': epochs,
        'total_sec': total_sec,
        'train_epoch_fmt': fmt_time(train_epoch_sec),
        'val_epoch_fmt': fmt_time(val_epoch_sec),
        'epoch_total_fmt': fmt_time(epoch_total_sec),
        'total_fmt': fmt_time(total_sec),
        'batch_size': config['data']['batch_size'],
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'num_classes': len(train_loader.dataset.classes),
        'class_names': train_loader.dataset.classes,
        'model_backbone': get_backbone_name(config),
        'embedding_dim': config['model']['embedding_dim'],
        'loss_type': loss_type,
        'optimizer': config['training']['optimizer'],
        'learning_rate': config['training']['learning_rate'],
        'scheduler': config['training']['scheduler'],
        'use_amp': use_amp,
        'num_workers': num_workers,
    }
    
    logger.info("=" * 70)
    logger.info("TRAINING TIME ESTIMATE")
    logger.info("=" * 70)
    logger.info(f"Batch size: {estimate['batch_size']}  |  num_workers: {num_workers}")
    logger.info(f"Median train step (fwd+bwd+opt): {avg_train_step:.4f}s  (range: {min(train_times):.4f}-{max(train_times):.4f}s)")
    logger.info(f"Median data load: {avg_data_load:.4f}s  (range: {min(data_load_times):.4f}-{max(data_load_times):.4f}s)")
    logger.info(f"Data overhead factor: {data_overhead_factor:.2f}x  ({'blocking I/O' if num_workers == 0 else 'overlapped'})")
    if num_workers == 0:
        logger.info("num_workers=0: data loading is BLOCKING. GPU idles during disk I/O.")
        logger.info(f"    Consider enabling cache_in_memory in config or setting num_workers > 0.")
    logger.info(f"Train: {num_train_batches} batches x {avg_train_batch_full:.3f}s = {estimate['train_epoch_fmt']}/epoch")
    logger.info(f"Val:   {num_val_batches} batches x {avg_val_batch:.3f}s + recall = {estimate['val_epoch_fmt']}/epoch")
    logger.info(f"Ref embed (one-time, post-training): {fmt_time(ref_embed_one_time_sec)}")
    logger.info(f"Epoch total: {estimate['epoch_total_fmt']}")
    logger.info(f"Total estimated: {estimate['epochs']} epochs x {estimate['epoch_total_fmt']} + ref_embed = {estimate['total_fmt']}")
    logger.info("=" * 70)
    
    # Clean up benchmark objects (same process as GUI train thread — free RAM aggressively)
    del bench_train_loader, bench_val_loader, bench_train_iter, bench_val_iter
    del bench_optimizer, bench_loss_func
    if bench_scaler:
        del bench_scaler
    model.cpu()
    import gc
    from src.data.dataloader_utils import free_training_memory
    gc.collect()
    free_training_memory("benchmark")
    
    return estimate


def run_training(
    args,
    estimate_only: bool = False,
    run_dir: str = None,
    _ablation_child: bool = False,
):
    """
    Core training function that accepts an args Namespace.
    Can be called from CLI (main()) or from GUI.
    
    Args:
        args: Namespace with config, resume, gpu (optional), checkpoint (optional)
        estimate_only: If True, only benchmark and return time estimate + pre-training analysis
        run_dir: If provided, use this as the run directory for checkpoints/models/logs
        _ablation_child: Internal flag — ablation variants must not spawn nested ablations
        
    Returns:
        If estimate_only: dict with time estimates + pre-training analysis paths
        Otherwise: dict with run_dir, status, and optional ablation summary
    """
    
    # Load configuration
    config = load_config(args.config)
    from src.utils.resolution_config import resolve_effective_image_size, sync_resolution_config
    config, res_synced = sync_resolution_config(config)
    
    # Override GPU if specified
    if args.gpu is not None:
        config['hardware']['gpu_ids'] = [args.gpu]
    
    # Setup logging
    logger = setup_logging(
        log_dir=config['paths']['log_dir'],
        log_level=config['logging']['log_level']
    )
    alarm_manager = TrainingAlarmManager(run_dir=run_dir, logger=logger)
    
    logger.info("="*70)
    logger.info("METRIC LEARNING TRAINING - SOTA 2024")
    logger.info("="*70)
    logger.info(f"Configuration file: {args.config}")
    
    # Full config diagnostic dump
    grain_cfg = config.get('grain_detection', {})
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    logger.info("--- CONFIG DUMP ---")
    logger.info(f"  Model: {get_backbone_name(config)}, emb_dim={model_cfg['embedding_dim']}, pretrained={model_cfg['pretrained']}")
    logger.info(f"  Projection: {model_cfg.get('projection_head', {}).get('hidden_dims', [])}, dropout={model_cfg.get('projection_head', {}).get('dropout', 0)}")
    effective_size = resolve_effective_image_size(config)
    logger.info(
        f"  Data: image_size={effective_size}px (canonical), resize={data_cfg.get('resize_size')}, "
        f"batch={data_cfg['batch_size']}, workers={data_cfg.get('num_workers', 4)}"
    )
    if res_synced:
        logger.info("  Resolution sync: data.resize_size y grain_detection.crop_size alineados a image_size")
    stale_crop = grain_cfg.get("crop_size")
    if stale_crop is not None and int(stale_crop) != effective_size:
        logger.warning(
            f"  grain_detection.crop_size={stale_crop} ignorado; pipeline usa data.image_size={effective_size}"
        )
    logger.info(
        f"  Grain: enabled={grain_cfg.get('enabled', True)}, "
        f"use_mask={grain_cfg.get('use_mask')}, masked_pooling={grain_cfg.get('masked_pooling')}"
    )
    logger.info(f"  Training: loss={train_cfg['loss']['type']}, lr={train_cfg['learning_rate']}, epochs={train_cfg['epochs']}, warmup={train_cfg.get('warmup_epochs', 0)}")
    logger.info(f"  Scheduler: {train_cfg.get('scheduler')}, T_max={train_cfg.get('scheduler_params', {}).get('T_max')}")
    logger.info(f"  AMP: {config.get('hardware', {}).get('use_amp', True)}, grad_clip={train_cfg.get('gradient_clip_norm', 5.0)}")
    logger.info(f"  Early stop: patience={train_cfg.get('early_stopping', {}).get('patience')}, min_delta={train_cfg.get('early_stopping', {}).get('min_delta')}")
    logger.info("--- END CONFIG ---")
    
    # Set random seed for reproducibility
    if config['reproducibility']['seed'] is not None:
        torch.manual_seed(config['reproducibility']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['reproducibility']['seed'])
        logger.info(f"Random seed set to: {config['reproducibility']['seed']}")
    
    # Set deterministic mode or enable cudnn benchmark for speed
    if config['reproducibility']['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Deterministic mode enabled")
    elif config['reproducibility'].get('benchmark', True):
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark mode enabled (faster convolutions)")
    
    from src.data.dataloader_utils import (
        dispose_data_engine,
        estimate_worker_ram_mb,
        free_training_memory,
    )

    from src.utils.config_validator import validate_config
    from src.utils.sota_integrity import validate_sota_integrity

    validation = validate_config(config, config_path=args.config, check_data_dirs=not estimate_only)
    if validation["warnings"]:
        for warn in validation["warnings"]:
            logger.warning("Config: %s", warn)
    if not validation["valid"]:
        for err in validation["errors"]:
            logger.error("Config: %s", err)
        raise ValueError(
            "Configuración inválida. Corrige los errores antes de entrenar "
            f"(ver docs/2026-06-12_PLAN_CUMPLIMIENTO_ACADEMICO_SOTA.md Fase A)."
        )
    logger.info("Pipeline: %s", validation["pipeline_summary"])

    # Create dataloaders
    logger.info("\n" + "="*70)
    logger.info("CREATING DATALOADERS")
    logger.info("="*70)
    train_loader, val_loader, data_engine = create_dataloaders(config)
    test_loader = getattr(data_engine, 'test_loader', None)
    ram_est = estimate_worker_ram_mb(config)
    logger.info(
        "Worker RAM budget (estimate): ~%.0f MB for %d train + %d val workers",
        ram_est["total_workers_mb"],
        ram_est["train_workers"],
        ram_est["val_workers"],
    )
    effective_train_workers = int(getattr(train_loader, 'num_workers', -1))
    effective_val_workers = int(getattr(val_loader, 'num_workers', -1))
    requested_train_workers, requested_val_workers = resolve_dataloader_workers(config)
    logger.info(
        f"DataLoader effective workers: train={effective_train_workers}, "
        f"val={effective_val_workers} "
        f"(config: train={requested_train_workers}, val={requested_val_workers})"
    )
    alarm_manager.info(
        "WORKER_REGIME_APPLIED",
        "DataLoader regime initialized.",
        requested_train_workers=requested_train_workers,
        requested_val_workers=requested_val_workers,
        effective_train_workers=effective_train_workers,
        effective_val_workers=effective_val_workers,
    )

    if effective_train_workers != requested_train_workers:
        alarm_manager.error(
            "WORKER_REGIME_MISMATCH",
            "Train DataLoader workers do not match config.",
            requested=requested_train_workers,
            effective=effective_train_workers,
        )
        raise RuntimeError(
            "Worker regime mismatch: train DataLoader workers differ from config."
        )
    if effective_val_workers != requested_val_workers:
        alarm_manager.error(
            "WORKER_REGIME_MISMATCH",
            "Val DataLoader workers do not match config.",
            requested=requested_val_workers,
            effective=effective_val_workers,
        )
        raise RuntimeError(
            "Worker regime mismatch: val DataLoader workers differ from config."
        )

    if effective_train_workers > 0:
        if not bool(getattr(train_loader, "persistent_workers", False)):
            raise RuntimeError(
                "Persistent workers disabled on train loader while num_workers > 0."
            )
    if effective_val_workers > 0:
        if not bool(getattr(val_loader, "persistent_workers", False)):
            raise RuntimeError(
                "Persistent workers disabled on val loader while val_num_workers > 0."
            )
    
    # Create model
    logger.info("\n" + "="*70)
    logger.info("CREATING MODEL")
    logger.info("="*70)
    model = create_model(config)
    
    # --- Time estimation (skip if run_dir already exists = already estimated) ---
    if run_dir and not estimate_only:
        logger.info("\n" + "="*70)
        logger.info("SKIPPING BENCHMARK (already estimated in pre-analysis)")
        logger.info("="*70)
    else:
        logger.info("\n" + "="*70)
        logger.info("BENCHMARKING (estimating training time...)")
        logger.info("="*70)
        time_estimate = estimate_training_time(config, train_loader, val_loader, model, logger)
        dispose_data_engine(data_engine)
        free_training_memory("post-benchmark")
    
    if estimate_only:
        # Run pre-training analysis and create run directory
        from scripts.run_setup import run_pre_training_analysis
        analysis = run_pre_training_analysis(
            config=config,
            config_path=args.config,
            train_dataset=train_loader.dataset,
            val_dataset=val_loader.dataset
        )
        time_estimate['analysis'] = analysis
        time_estimate['run_dir'] = analysis['run_paths']['run_dir']
        
        dispose_data_engine(data_engine)
        del train_loader, val_loader, data_engine, model
        free_training_memory("estimate_only")

        from src.utils.ablation_matrix import count_planned_variants

        n_abl = count_planned_variants(config)
        time_estimate["ablation_variants"] = n_abl
        if n_abl:
            abl_epochs = (config.get("evaluation") or {}).get("ablation", {}).get("epochs")
            ep_note = (
                f" ({abl_epochs} épocas/variante)"
                if abl_epochs is not None
                else " (mismas épocas que el run principal)"
            )
            time_estimate["ablation_note"] = (
                f"+ {n_abl} entrenamiento(s) de ablación tras el run principal{ep_note}"
            )
        return time_estimate
    
    # Merge training and hardware configs for trainer
    sota_config = config.get('sota', {})
    eval_cfg = config.get("evaluation", {})
    trainer_config = {
        **config['training'],
        **config['hardware'],
        'num_classes': config.get('num_classes'),
        'embedding_dim': config['model']['embedding_dim'],
        'run_dir': run_dir,
        'expected_train_workers': effective_train_workers,
        'expected_val_workers': effective_val_workers,
        'use_gpu': config['hardware']['use_gpu'],
        'use_amp': config['hardware']['use_amp'],
        'epochs': config['training']['epochs'],
        'save_every_n_epochs': config['training']['save_every_n_epochs'],
        'progressive_plot_every_epochs': int(eval_cfg.get('progressive_plot_every_epochs', 2)),
        # SOTA regularizers (2022-2024)
        'nir': sota_config.get('nir', {}),
        'hier': sota_config.get('hier', {}),
        'dada': sota_config.get('dada', {}),
        'anti_collapse': sota_config.get('anti_collapse', {}),
        'self_distillation': sota_config.get('self_distillation', {}),
        'evaluation': eval_cfg,
        'sota': sota_config,
        'reproducibility': config.get('reproducibility', {}),
        'inference': config.get('inference', {}),
        'grain_detection': config.get('grain_detection', {}),
        'loader_policy': config.get('data', {}).get('loader_policy', {}),
    }
    
    # Create trainer
    logger.info("\n" + "="*70)
    logger.info("CREATING TRAINER")
    logger.info("="*70)
    trainer = MetricLearningTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        logger=logger,
        checkpoint_dir=run_dir,
        data_engine=data_engine,
    )
    
    # Print training configuration
    logger.info("\n" + "="*70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*70)
    logger.info(f"Loss function: {config['training']['loss']['type']}")
    logger.info(f"Optimizer: {config['training']['optimizer']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Weight decay: {config['training']['weight_decay']}")
    logger.info(f"Scheduler: {config['training']['scheduler']}")
    logger.info(f"Gradient clipping: {config['training']['gradient_clip_norm']}")
    logger.info(f"Mixed precision: {config['hardware']['use_amp']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Warmup epochs: {config['training']['warmup_epochs']}")
    
    # Start training
    logger.info("\n" + "="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    free_training_memory("pre_fit")
    
    try:
        train_losses, val_losses = trainer.fit(
            epochs=config['training']['epochs'],
            resume_from=args.resume
        )
        
        # Save final model to run directory
        final_dir = Path(run_dir) / 'checkpoints' if run_dir else Path('models')
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            val_loss=trainer.best_val_loss,
            filepath=str(final_dir / 'final_model.pth')
        )
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING SUMMARY")
        logger.info("="*70)
        logger.info(f"Total epochs: {len(train_losses)}")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Final train loss: {train_losses[-1]:.4f}")
        logger.info(f"Final val loss: {val_losses[-1]:.4f}")
        logger.info("="*70)
        
        # --- Post-training analysis (curves, t-SNE, confusion matrix, inference, report) ---
        if run_dir:
            from scripts.post_training import (
                create_eval_dataloaders,
                run_post_training_analysis,
                compute_and_save_centroids,
                finalize_self_contained_checkpoint,
                publish_final_evaluation_summary,
            )

            eval_cfg = config.get("evaluation", {}) or {}
            report_final_on = str(eval_cfg.get("report_final_on", "test")).lower()

            del trainer
            dispose_data_engine(data_engine)
            free_training_memory("post_fit_workers_released")

            train_loader, val_loader, test_loader, data_engine = create_eval_dataloaders(config)
            free_training_memory("post_training_start")

            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                best_ckpt = Path(run_dir) / 'checkpoints' / 'best_model.pth'
                if best_ckpt.exists():
                    logger.info("Loading BEST model checkpoint for post-training analysis...")
                    ckpt = torch.load(str(best_ckpt), map_location=device, weights_only=True)
                    model.load_state_dict(ckpt['model_state_dict'])
                    logger.info(
                        f"Best model loaded (epoch {ckpt.get('epoch', '?')+1}, "
                        f"val_loss={ckpt.get('val_loss', '?')})"
                    )
                model.to(device)

                class_names = train_loader.dataset.classes
                loss_params = config.get('training', {}).get('loss', {}).get('params', {})
                num_slices = int(loss_params.get('num_slices', 4))
                ckpt_full = torch.load(str(best_ckpt), map_location='cpu', weights_only=False) if best_ckpt.exists() else {}
                centroids, slice_classifier = compute_and_save_centroids(
                    model,
                    train_loader,
                    device,
                    run_dir,
                    class_names,
                    num_slices=num_slices,
                    loss_state_dict=ckpt_full.get('loss_state_dict'),
                    loss_type=ckpt_full.get('loss_type'),
                )
                free_training_memory("centroids_saved")

                logger.info(
                    "Post-training VAL (intermedio — selección de checkpoint fue en val; "
                    "métricas finales en %s)",
                    report_final_on,
                )
                run_post_training_analysis(
                    model, val_loader, config, run_dir, device,
                    slice_classifier=slice_classifier, centroids=centroids, prefix="val"
                )

                if eval_cfg.get("calibration", {}).get("enabled", True):
                    from src.utils.threshold_calibration import calibrate_inference_thresholds
                    calibrate_inference_thresholds(
                        model, val_loader, device, Path(run_dir), config
                    )

                final_prefix = report_final_on
                if report_final_on == "test":
                    if test_loader is not None:
                        logger.info("Post-training TEST (evaluación final hold-out, 1 pasada)...")
                        run_post_training_analysis(
                            model, test_loader, config, run_dir, device,
                            slice_classifier=slice_classifier, centroids=centroids, prefix="test"
                        )
                    else:
                        logger.warning(
                            "evaluation.report_final_on=test pero no hay test_loader; "
                            "usando val como evaluación final (protocolo MLRC incompleto)"
                        )
                        final_prefix = "val"
                elif test_loader is not None:
                    logger.info("Post-training TEST (exploratorio; reporte final = %s)...", report_final_on)
                    run_post_training_analysis(
                        model, test_loader, config, run_dir, device,
                        slice_classifier=slice_classifier, centroids=centroids, prefix="test"
                    )

                publish_final_evaluation_summary(Path(run_dir), prefix=final_prefix)

                logger.info("Re-saving best_model.pth as self-contained inference checkpoint...")
                finalize_self_contained_checkpoint(Path(run_dir))

            except Exception as post_err:
                alarm_manager.error(
                    "POST_TRAINING_FAILED",
                    "Post-training analysis failed (training completed successfully).",
                    error=str(post_err),
                )
                logger.error(
                    f"\nPost-training analysis failed: {post_err}\n"
                    f"Use the Evaluación tab or run_evaluation_from_checkpoint to regenerate results.",
                    exc_info=True,
                )
            finally:
                dispose_data_engine(data_engine)
                free_training_memory("post_training_end")

        ablation_result = None
        if run_dir and not _ablation_child:
            abl_cfg = (config.get("evaluation") or {}).get("ablation") or {}
            if abl_cfg.get("enabled", False):
                try:
                    from src.utils.ablation_matrix import execute_integrated_ablation_study

                    logger.info("\n" + "=" * 70)
                    logger.info("ABLACION INTEGRADA (post-entrenamiento principal)")
                    logger.info("=" * 70)
                    ablation_result = execute_integrated_ablation_study(
                        config=config,
                        config_path=args.config,
                        parent_run_dir=Path(run_dir),
                        logger=logger,
                    )
                except Exception as abl_err:
                    logger.error(
                        "Ablación integrada falló (entrenamiento principal OK): %s",
                        abl_err,
                        exc_info=True,
                    )
                    ablation_result = {
                        "enabled": True,
                        "status": "failed",
                        "error": str(abl_err),
                    }

        return {
            "run_dir": run_dir,
            "status": "completed",
            "ablation": ablation_result,
        }

    except KeyboardInterrupt:
        alarm_manager.warning(
            "TRAINING_INTERRUPTED",
            "Training interrupted by user.",
        )
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            val_loss=trainer.val_losses[-1] if trainer.val_losses else float('inf'),
            filepath=str(Path(run_dir) / 'checkpoints' / 'interrupted_checkpoint.pth') if run_dir else 'models/interrupted_checkpoint.pth',
            save_reference=False
        )
        logger.info("Checkpoint saved to run directory")
    
    except Exception as e:
        alarm_manager.error(
            "TRAINING_FAILED",
            "Unhandled training exception.",
            error=str(e),
        )
        logger.error(f"\n\nTraining failed with error: {str(e)}", exc_info=True)
        raise


def main():
    """CLI entry point — parses arguments and calls run_training()"""
    parser = argparse.ArgumentParser(description='Train Metric Learning Model')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (overrides config)')
    args = parser.parse_args()
    run_training(args)


if __name__ == '__main__':
    main()
