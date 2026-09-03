"""
Enhanced Metric Learning Trainer with SOTA Techniques (2024)
Supports Multi-Similarity Loss, ArcFace, Triplet Loss, and advanced training strategies

Author: MetricLearning Project
Date: 2026-01-05
Based on: SOTA research investigation (docs/09_SOTA_research_investigation.md)
"""

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm
from pytorch_metric_learning import losses, miners, distances
from pathlib import Path
import logging
import csv
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

# SOTA losses and regularizers (2022-2024)
from src.losses.anti_collapse_loss import AntiCollapseLoss
from src.losses.nir_regularizer import NIRRegularizer
from src.losses.hier_regularizer import HIERRegularizer
from src.losses.sg_softmax import SGSoftmaxLoss
from src.losses.dada_wrapper import DADAWrapper
from src.training.ema_teacher import EMATeacher, SelfDistillationLoss
from src.training.hard_mining import HardNegativeMiner
from src.utils.training_alarms import TrainingAlarmManager
from src.data.dataloader_utils import free_training_memory
from src.utils.mlrc_protocol import (
    is_improvement,
    metric_value_from_dict,
    normalize_validation_metric,
    resolve_map_at_r_config,
    resolve_nmi_config,
    resolve_nn_metric,
)


class MetricLearningTrainer:
    """
    Enhanced Trainer for Metric Learning with SOTA techniques.
    
    Features:
    - Multi-Similarity Loss (SOTA 2024)
    - ArcFace Loss
    - Triplet Loss with flexible mining
    - Mixed Precision Training (AMP)
    - Gradient Clipping
    - Learning Rate Warmup
    - Cosine Annealing Scheduler
    - Early Stopping
    - Comprehensive Logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict,
        logger: Optional[logging.Logger] = None,
        checkpoint_dir: Optional[str] = None,
        data_engine=None,
    ):
        """
        Initialize trainer with model and configuration.
        
        Args:
            model: Neural network model (AnalogyNet)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            logger: Optional logger instance
            checkpoint_dir: Optional run directory for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.data_engine = data_engine
        self.config = config
        self.logger = logger or self._setup_logger()
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_mem = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            self.logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"Model VRAM after load: {torch.cuda.memory_allocated(self.device)/1e6:.0f} MB")
        
        # Loss function setup (SOTA 2024)
        self.loss_func = self._setup_loss_function().to(self.device)
        
        # Miner setup (if needed)
        self.miner = self._setup_miner()
        
        # SOTA Regularizers (active)
        self.nir_reg = self._setup_nir()
        self.hier_reg = self._setup_hier()
        self.anti_collapse = self._setup_anti_collapse()
        self.dada_wrapper = self._setup_dada()
        self.finetune_phases = list(self.config.get("finetune_phases") or [])
        self._finetune_phase_boundaries = self._build_finetune_phase_boundaries()
        # Dimensional Slicing (SlicedMSLoss) reemplaza amplitude/uniformity/
        # decorrelation/disentanglement — ver docs/2026-06-09_INFORME_MODIFICACIONES_PIPELINE_MULTI_SLICE.md
        
        # Optimizer setup
        self.optimizer = self._setup_optimizer()
        
        # Scheduler setup
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Self-distillation (Zeng et al., 2024)
        self.ema_teacher = None
        self.sd_loss = None
        self._setup_self_distillation()
        
        # Hard Negative Mining (OHEM)
        self.hard_miner = None
        self._setup_hard_mining()
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"Gradient Accumulation: {self.gradient_accumulation_steps} steps (effective batch = {self.gradient_accumulation_steps}x)")
        self.best_recall_at_1 = 0.0
        self.best_selection_value = float("-inf")
        self.val_metrics_history = []
        self.epochs_without_improvement = 0
        
        # Validation metric for best model selection (MLRC: mAP@R default)
        self.validation_metric = normalize_validation_metric(
            config.get("validation_metric", "mAP@R")
        )
        if self.validation_metric == "val_loss":
            self.best_selection_value = float("inf")
        self.logger.info(
            f"Checkpoint selection metric: {self.validation_metric} "
            f"(MLRC protocol)"
        )
        
        # Gradient clipping
        self.gradient_clip_norm = config.get('gradient_clip_norm', 5.0)
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping', {}).get('patience', 10)
        self.early_stopping_min_delta = config.get('early_stopping', {}).get('min_delta', 0.001)
        
        # Checkpoint directory (run_dir/checkpoints or default models/)
        if checkpoint_dir:
            self.checkpoint_dir = str(Path(checkpoint_dir) / 'checkpoints')
        else:
            self.checkpoint_dir = 'models'
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        self._logged_first_batch = False
        self.expected_train_workers = int(self.config.get('expected_train_workers', -1))
        self.expected_val_workers = int(self.config.get('expected_val_workers', -1))
        self.batch_stall_threshold_sec = float(self.config.get('batch_stall_threshold_sec', 8.0))
        self.alarm_manager = TrainingAlarmManager(
            run_dir=self.config.get('run_dir'),
            logger=self.logger,
        )
        
        self.logger.info(f"Trainer initialized with loss: {self.config['loss']['type']}")
        self.logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        self.alarm_manager.info(
            "TRAINER_INIT",
            "Trainer initialized with worker regime tracking.",
            expected_train_workers=self.expected_train_workers,
            expected_val_workers=self.expected_val_workers,
            batch_stall_threshold_sec=self.batch_stall_threshold_sec,
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger('MetricLearningTrainer')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
    
    @staticmethod
    def _require_loss_param(params: dict, key: str, loss_type: str):
        if key not in params:
            raise ValueError(
                f"training.loss.params.{key} es obligatorio para loss.type={loss_type} "
                f"(sin defaults silenciosos — ver config_validator)"
            )
        return params[key]

    def _resolve_distance(self):
        """Distance for metric-learning losses (C-09)."""
        loss_cfg = self.config.get("loss", {})
        dist_name = str(loss_cfg.get("distance", "")).lower()
        if not dist_name:
            space = str(self.config.get("sota", {}).get("embedding_space", "euclidean")).lower()
            dist_name = "euclidean" if space == "euclidean" else "cosine"
        if dist_name == "cosine":
            return distances.CosineSimilarity()
        if dist_name in ("euclidean", "lp"):
            return distances.LpDistance(p=2)
        raise ValueError(
            f"training.loss.distance debe ser cosine|euclidean|lp, recibido: {dist_name!r}"
        )

    def _setup_loss_function(self) -> nn.Module:
        """
        Setup loss function based on configuration.
        Supports: Multi-Similarity, Triplet, ArcFace
        """
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'multi_similarity')
        params = loss_config.get('params', {})
        
        distance = self._resolve_distance()
        
        if loss_type == 'multi_similarity':
            # Multi-Similarity Loss (SOTA 2024)
            self.logger.info("Using Multi-Similarity Loss (SOTA 2024)")
            return losses.MultiSimilarityLoss(
                alpha=float(self._require_loss_param(params, "alpha", loss_type)),
                beta=float(self._require_loss_param(params, "beta", loss_type)),
                base=float(self._require_loss_param(params, "base", loss_type)),
                distance=distance
            )
        
        elif loss_type == 'triplet':
            # Triplet Margin Loss
            self.logger.info("Using Triplet Margin Loss")
            return losses.TripletMarginLoss(
                margin=params.get('triplet_margin', 0.2),
                distance=distance
            )
        
        elif loss_type == 'proxy_anchor':
            # Proxy Anchor Loss (Kim et al. CVPR 2020) - SOTA for small datasets
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for Proxy Anchor loss")
            
            self.logger.info(f"Using Proxy Anchor Loss with {num_classes} classes")
            self._proxy_anchor_loss = losses.ProxyAnchorLoss(
                num_classes=num_classes,
                embedding_size=self.config.get('embedding_dim', 128),
                margin=params.get('margin', 0.1),
                alpha=params.get('alpha', 32)
            )
            return self._proxy_anchor_loss
        
        elif loss_type == 'arcface':
            # ArcFace Loss (requires number of classes)
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for ArcFace loss")
            
            margin = params.get('margin', 28.6)
            scale = params.get('scale', 64)
            
            if self.config.get('sota', {}).get('embedding_space') == 'hyperbolic':
                from src.losses.hyperbolic_arcface import HyperbolicArcFaceLoss
                curvature = self.config.get('sota', {}).get('hyperbolic_curvature', 1.0)
                self.logger.info(
                    f"Using Hyperbolic ArcFace Loss with {num_classes} classes "
                    f"(margin={margin}, scale={scale}, curvature={curvature})"
                )
                return HyperbolicArcFaceLoss(
                    num_classes=num_classes,
                    embedding_size=self.config.get('embedding_dim', 128),
                    margin=params.get('hyperbolic_margin', 0.2),
                    scale=scale,
                    curvature=curvature,
                )

            self.logger.info(
                f"Using ArcFace Loss with {num_classes} classes "
                f"(margin={margin}Â°, scale={scale})"
            )
            return losses.ArcFaceLoss(
                num_classes=num_classes,
                embedding_size=self.config.get('embedding_dim', 128),
                margin=margin,
                scale=scale,
            )
        
        elif loss_type == 'sg_softmax':
            # SG-Softmax Loss (Yang et al., AAAI 2023)
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for SG-Softmax loss")
            
            self.logger.info(f"Using SG-Softmax Loss with {num_classes} classes")
            self._sg_softmax_loss = SGSoftmaxLoss(
                embedding_dim=self.config.get('embedding_dim', 128),
                num_classes=num_classes,
                scale=float(params.get('scale', 30.0)),
                margin=float(params.get('margin', 0.4)),
                ortho_penalty_weight=float(params.get('ortho_penalty_weight', 1.0)),
            )
            return self._sg_softmax_loss
        
        elif loss_type == 'ldm_ms':
            # Learnable Dynamic Margin MS Loss
            from src.losses.ldm_loss import LearnableDynamicMarginMSLoss
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for LDM-MS loss")
            self.logger.info(f"Using LDM-MS Loss with {num_classes} classes")
            base_margin = params.get("base_margin", params.get("base"))
            if base_margin is None:
                base_margin = self._require_loss_param(params, "base_margin", loss_type)
            self._ldm_ms_loss = LearnableDynamicMarginMSLoss(
                num_classes=num_classes,
                alpha=float(self._require_loss_param(params, "alpha", loss_type)),
                beta=float(self._require_loss_param(params, "beta", loss_type)),
                base_margin=float(base_margin),
            )
            return self._ldm_ms_loss

        elif loss_type == 'hybrid_proxy_ms':
            # Hybrid Proxy MS Loss (SG-Softmax + MS Loss/LDM)
            from src.losses.hybrid_loss import HybridProxyMSLoss
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for Hybrid Proxy MS loss")
            self.logger.info(f"Using Hybrid Proxy MS Loss with {num_classes} classes")
            self._hybrid_loss = HybridProxyMSLoss(
                embedding_dim=self.config.get('embedding_dim', 128),
                num_classes=num_classes,
                proxy_scale=params.get('proxy_scale', 30.0),
                proxy_margin=params.get('proxy_margin', 0.4),
                ms_alpha=float(self._require_loss_param(params, "alpha", loss_type)),
                ms_beta=float(self._require_loss_param(params, "beta", loss_type)),
                ms_base_margin=float(
                    params.get("base_margin", params.get("base"))
                    if params.get("base_margin", params.get("base")) is not None
                    else self._require_loss_param(params, "base_margin", loss_type)
                ),
                w_global=params.get('w_global', 1.0),
                w_local=params.get('w_local', 1.0),
                use_ldm=params.get('use_ldm', True)
            )
            return self._hybrid_loss
        
        elif loss_type == 'sliced_ms':
            # Sliced MS Loss: Spectral Embedding Expansion (Ko & Gu, CVPR 2020)
            from src.losses.sliced_ms_loss import SlicedMSLoss
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for Sliced MS loss")
            self.logger.info(f"Using Sliced MS Loss with {num_classes} classes")
            self._sliced_ms_loss = SlicedMSLoss(
                num_classes=num_classes,
                num_slices=int(self._require_loss_param(params, "num_slices", loss_type)),
                embedding_dim=self.config.get('embedding_dim', 128),
                alpha=float(self._require_loss_param(params, "alpha", loss_type)),
                beta=float(self._require_loss_param(params, "beta", loss_type)),
                base_margin=float(self._require_loss_param(params, "base_margin", loss_type)),
            )
            return self._sliced_ms_loss
        
        elif loss_type == 'sliced_proxy':
            # Sliced Proxy Loss: Dimensional Slicing (CVPR 2019) + SG-Softmax (AAAI 2023)
            from src.losses.sliced_proxy_loss import SlicedProxyLoss
            num_classes = self.config.get('num_classes')
            if num_classes is None:
                raise ValueError("num_classes required for Sliced Proxy loss")
            self.logger.info(f"Using Sliced Proxy Loss with {num_classes} classes")
            self._sliced_proxy_loss = SlicedProxyLoss(
                num_classes=num_classes,
                num_slices=int(self._require_loss_param(params, "num_slices", loss_type)),
                embedding_dim=self.config.get('embedding_dim', 128),
                scale=float(self._require_loss_param(params, "proxy_scale", loss_type)),
                margin=float(self._require_loss_param(params, "proxy_margin", loss_type)),
            )
            return self._sliced_proxy_loss
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _get_loss_type(self) -> str:
        """Loss type from flattened trainer config (train.py merges training.*)."""
        return self.config.get('loss', {}).get('type', 'multi_similarity')

    def _loss_forward_embeddings(
        self,
        embeddings: torch.Tensor,
        prenorm_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Embeddings fed to the primary loss.

        Sliced losses operate on per-slice unit vectors (prenorm concat for multi_tower).
        """
        loss_type = self._get_loss_type()
        if loss_type in ("sliced_ms", "sliced_proxy") and prenorm_embeddings is not None:
            return prenorm_embeddings
        return embeddings

    def _setup_miner(self) -> Optional[nn.Module]:
        """
        Setup mining strategy if needed.
        Multi-Similarity Loss doesn't require mining.
        """
        loss_type = self._get_loss_type()

        # Proxy / angular losses do not use triplet miners (saves VRAM per batch)
        if loss_type in ('multi_similarity', 'arcface', 'proxy_anchor', 'sg_softmax', 'sliced_proxy'):
            if loss_type != 'multi_similarity':
                self.logger.info(
                    f"Triplet miner skipped for loss '{loss_type}' (not used with this loss)"
                )
            return None

        # Setup miner for Triplet Loss
        mining_config = self.config.get('mining', {})
        if not mining_config.get('enabled', True):
            return None
        
        mining_type = mining_config.get('type', 'semihard')
        margin = mining_config.get('margin', 0.2)
        
        self.logger.info(f"Using {mining_type} mining strategy")
        
        if mining_type == 'semihard':
            return miners.TripletMarginMiner(
                margin=margin,
                type_of_triplets='semihard'
            )
        elif mining_type == 'hard':
            return miners.TripletMarginMiner(
                margin=margin,
                type_of_triplets='hard'
            )
        elif mining_type == 'distance_weighted':
            return miners.DistanceWeightedMiner(
                cutoff=float(mining_config.get("cutoff", 0.5)),
                nonzero_loss_cutoff=float(mining_config.get("nonzero_loss_cutoff", 1.4)),
            )
        else:
            return None
    
    def _setup_nir(self) -> Optional[NIRRegularizer]:
        """Setup Non-Isotropy Regularization (Roth et al., CVPR 2022)."""
        nir_config = self.config.get('nir', {})
        if not nir_config.get('enabled', False):
            return None
        
        reg = NIRRegularizer(
            lambda_nir=nir_config.get('lambda', 0.5),
            eps=nir_config.get('eps', 1e-4),
            min_samples_per_class=int(nir_config.get('min_samples_per_class', 2)),
        )
        self.logger.info(f"NIR regularization enabled (lambda={nir_config.get('lambda', 0.5)})")
        return reg
    
    def _build_finetune_phase_boundaries(self) -> List[int]:
        """Cumulative epoch indices where finetune phases switch (D-05)."""
        boundaries = []
        total = 0
        for phase in self.finetune_phases:
            total += int(phase.get("epochs", 0))
            if total > 0:
                boundaries.append(total)
        return boundaries

    def _apply_finetune_phase(self, phase_index: int):
        """Apply freeze/LR settings for finetune phase *phase_index*."""
        if not self.finetune_phases or phase_index >= len(self.finetune_phases):
            return
        phase = self.finetune_phases[phase_index]
        freeze = bool(phase.get("freeze_backbone", True))
        num_unfrozen = int(phase.get("num_unfrozen_blocks", 2))
        lr_factor = float(phase.get("lr_factor", 1.0))
        base_lr = float(self.config.get("learning_rate", 1e-4))
        backbone_factor = float(self.config.get("backbone_lr_factor", 1.0))

        if hasattr(self.model, "set_backbone_frozen"):
            self.model.set_backbone_frozen(freeze, num_unfrozen_blocks=num_unfrozen)

        for group in self.optimizer.param_groups:
            init_lr = group.get("initial_lr", group["lr"])
            name = group.get("name", "")
            if name == "backbone":
                group["lr"] = init_lr * lr_factor
            else:
                group["lr"] = base_lr * lr_factor if init_lr <= base_lr * 1.01 else init_lr * lr_factor

        self.logger.info(
            f"Finetune phase {phase_index + 1}/{len(self.finetune_phases)}: "
            f"freeze_backbone={freeze}, lr_factor={lr_factor}"
        )

    def _setup_anti_collapse(self) -> Optional[AntiCollapseLoss]:
        """Anti-collapse regularization (IEEE Trans. Multimedia 2024, D-04)."""
        ac_cfg = self.config.get("anti_collapse", {}) or {}
        if not ac_cfg.get("enabled", False):
            return None
        reg = AntiCollapseLoss(
            lambda_ac=float(ac_cfg.get("lambda", 0.1)),
            t=float(ac_cfg.get("t", 2.0)),
        )
        self.logger.info(
            f"Anti-Collapse regularization enabled (lambda={ac_cfg.get('lambda', 0.1)})"
        )
        return reg

    def _setup_hier(self) -> Optional[HIERRegularizer]:
        """Setup Hierarchical Regularization (Kim et al., CVPR 2023)."""
        hier_config = self.config.get('hier', {})
        if not hier_config.get('enabled', False):
            return None
        
        num_classes = self.config.get('num_classes')
        if num_classes is None:
            self.logger.warning("HIER requires num_classes, skipping")
            return None
        
        reg = HIERRegularizer(
            embedding_dim=self.config.get('embedding_dim', 128),
            num_classes=num_classes,
            num_levels=hier_config.get('num_levels', 3),
            lambda_hier=hier_config.get('lambda', 0.1),
            temperature=hier_config.get('temperature', 0.1),
            taxonomy_path=hier_config.get('taxonomy_path'),
        ).to(self.device)
        self.logger.info(f"HIER regularization enabled (lambda={hier_config.get('lambda', 0.1)}, levels={hier_config.get('num_levels', 3)})")
        return reg
    
    def _setup_dada(self) -> Optional[DADAWrapper]:
        """Setup DADA wrapper (Ren et al., AAAI 2024)."""
        dada_config = self.config.get('dada', {})
        if not dada_config.get('enabled', False):
            return None
        
        wrapper = DADAWrapper(
            base_loss=self.loss_func,
            lambda_dada=dada_config.get('lambda', 0.1),
            kernel_bandwidth=dada_config.get('kernel_bandwidth', 1.0)
        )
        self.logger.info(f"DADA wrapper enabled (lambda={dada_config.get('lambda', 0.1)})")
        return wrapper
    
    def _setup_self_distillation(self):
        """Setup Self-Distillation with EMA Teacher (Zeng et al., 2024)."""
        sd_config = self.config.get('self_distillation', {})
        if not sd_config.get('enabled', False):
            return
        
        self.ema_teacher = EMATeacher(
            self.model,
            alpha=sd_config.get('ema_alpha', 0.999)
        )
        self.ema_teacher.to(self.device)
        
        self.sd_loss = SelfDistillationLoss(
            temperature=sd_config.get('temperature', 0.1),
            lambda_sd=sd_config.get('lambda', 0.5)
        )
        self.logger.info(
            f"Self-distillation enabled (alpha={sd_config.get('ema_alpha', 0.999)}, "
            f"temp={sd_config.get('temperature', 0.1)}, lambda={sd_config.get('lambda', 0.5)})"
        )
    
    def _setup_hard_mining(self):
        """Setup Hard Negative Mining (OHEM) if enabled in config."""
        hard_mining_config = self.config.get('hard_mining', {})
        loss_type = self._get_loss_type()

        if not hard_mining_config.get('enabled', False):
            self.logger.info("Hard Negative Mining: Disabled")
            return

        if loss_type == 'multi_similarity':
            self.logger.info("Hard Negative Mining: Auto-disabled (Multi-Similarity loss handles it natively intra-graph).")
            self.hard_miner = None
            return

        if loss_type not in ('arcface', 'triplet'):
            self.logger.info(
                f"Hard Negative Mining: Auto-disabled (not supported with loss '{loss_type}')."
            )
            self.hard_miner = None
            return

        hard_ratio = hard_mining_config.get('ratio', 0.3)
        min_samples = hard_mining_config.get('min_samples', 32)
        warmup_epochs = hard_mining_config.get('warmup_epochs', 0)
        adaptive = hard_mining_config.get('adaptive', False)
        
        self.hard_miner = HardNegativeMiner(
            hard_ratio=hard_ratio,
            min_samples=min_samples,
            warmup_epochs=warmup_epochs,
            adaptive_ratio=adaptive
        )
        
        self.logger.info(
            f"Hard Negative Mining: Enabled for '{loss_type}' "
            f"(ratio={hard_ratio}, min_samples={min_samples}, warmup={warmup_epochs})"
        )

    def _compute_training_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        images: torch.Tensor,
        masks: torch.Tensor = None,
        prenorm_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Training loss with optional OHEM (hard negative mining).

        ArcFace: per-sample CE on angular-margin logits via loss_func.compute_loss(),
        then mean only over the hardest examples in the batch.
        Validation always uses the full-batch loss (see validate()).
        """
        if self.hard_miner is None:
            return self._compute_total_loss(embeddings, labels, images, masks=masks, prenorm_embeddings=prenorm_embeddings)

        loss_per_sample = self._compute_loss_per_sample(embeddings, labels)
        hard_mask = self.hard_miner.mine_hard_examples(loss_per_sample)
        if not hard_mask.any():
            base_loss = loss_per_sample.sum() / len(loss_per_sample)
        else:
            base_loss = loss_per_sample[hard_mask].sum() / len(loss_per_sample)
            
        return self._compute_total_loss(embeddings, labels, images, masks=masks, base_loss=base_loss, prenorm_embeddings=prenorm_embeddings)
    
    def _compute_loss_per_sample(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computa pÃ©rdida por sample individual (sin reducciÃ³n) para Hard Negative Mining.
        
        Args:
            embeddings: Tensor (B, D) de embeddings normalizados
            labels: Tensor (B,) de etiquetas
        
        Returns:
            loss_per_sample: Tensor (B,) con pÃ©rdida de cada sample
        """
        loss_type = self._get_loss_type()

        if loss_type == 'arcface':
            # Use library implementation: W is nn.Parameter (num_classes, emb), not Linear.weight
            if self.loss_func.__class__.__name__ == 'ArcFaceLoss':
                loss_dict = self.loss_func.compute_loss(
                    embeddings, labels, None, embeddings, labels
                )
            else:
                loss_dict = self.loss_func.compute_loss(
                    embeddings,
                    labels,
                    indices_tuple=None,
                    ref_emb=embeddings,
                    ref_labels=labels,
                )
            loss_per_sample = loss_dict['loss']['losses']

        elif loss_type in ('proxy_anchor', 'sg_softmax'):
            raise NotImplementedError(
                f"Hard negative mining is not supported with loss '{loss_type}'. "
                "Set training.hard_mining.enabled to false or switch to arcface."
            )

        else:
            # Fallback vectorizado acelerado por GPU (O(1) operaciones Python)
            dist_mat = torch.cdist(embeddings, embeddings, p=2)
            labels_expanded = labels.unsqueeze(1)
            pos_mask = (labels_expanded == labels_expanded.t()) & (torch.eye(len(labels), device=embeddings.device) == 0)
            neg_mask = labels_expanded != labels_expanded.t()
            
            # Max_pos_dist y Min_neg_dist matricial
            pos_dist = torch.where(pos_mask, dist_mat, torch.zeros_like(dist_mat)).max(dim=1)[0]
            neg_dist = torch.where(neg_mask, dist_mat, torch.full_like(dist_mat, float('inf'))).min(dim=1)[0]
            
            # Hardcoded 0.2 margin fallback, we use actual margin if loss_func exposes it
            margin = getattr(self.loss_func, 'margin', 0.2)
            loss_per_sample = torch.clamp(pos_dist - neg_dist + margin, min=0.0)
        
        return loss_per_sample
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer (AdamW recommended by SOTA).
        For proxy-based losses, creates separate param groups for model and proxies.
        Supports differential LR: backbone gets lr * backbone_lr_factor."""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        # Differential learning rate: backbone (pretrained) vs projection head (from scratch)
        backbone_lr_factor = self.config.get('backbone_lr_factor', 1.0)
        if backbone_lr_factor < 1.0 and hasattr(self.model, 'backbone'):
            backbone_params = list(self.model.backbone.parameters())
            backbone_param_ids = {id(p) for p in backbone_params}
            head_params = [p for p in self.model.parameters() if id(p) not in backbone_param_ids]
            param_groups = [
                {'params': backbone_params, 'lr': lr * backbone_lr_factor, 'name': 'backbone'},
                {'params': head_params, 'lr': lr, 'name': 'projection'},
            ]
            self.logger.info(
                f"Differential LR: backbone={lr * backbone_lr_factor:.2e} "
                f"({len(backbone_params)} tensors), "
                f"projection={lr:.2e} ({len(head_params)} tensors)"
            )
        else:
            param_groups = [{'params': self.model.parameters(), 'lr': lr}]
        
        proxy_lr_factor = float(self.config.get("proxy_lr_factor", 100.0))
        margin_lr_factor = float(self.config.get("margin_lr_factor", 10.0))

        if hasattr(self, '_proxy_anchor_loss'):
            proxy_lr = lr * proxy_lr_factor
            param_groups.append({'params': self._proxy_anchor_loss.proxies, 'lr': proxy_lr})
            self.logger.info(f"Proxy Anchor: model lr={lr:.2e}, proxy lr={proxy_lr:.2e}")
        
        if hasattr(self, '_sg_softmax_loss'):
            proxy_lr = lr * proxy_lr_factor
            param_groups.append({'params': self._sg_softmax_loss.proxies, 'lr': proxy_lr})
            self.logger.info(f"SG-Softmax: model lr={lr:.2e}, proxy lr={proxy_lr:.2e}")
            
        if hasattr(self, '_ldm_ms_loss'):
            margin_lr = lr * margin_lr_factor
            param_groups.append({'params': [self._ldm_ms_loss.pos_margin, self._ldm_ms_loss.neg_margin], 'lr': margin_lr, 'weight_decay': 0.0})
            self.logger.info(f"LDM-MS: added learnable margins to optimizer (lr={margin_lr:.2e})")

        if hasattr(self, '_hybrid_loss'):
            proxy_lr = lr * proxy_lr_factor
            param_groups.append({'params': self._hybrid_loss.proxy_loss.proxies, 'lr': proxy_lr})
            self.logger.info(f"Hybrid: added proxies to optimizer (lr={proxy_lr:.2e})")
            if self._hybrid_loss.use_ldm:
                margin_lr = lr * margin_lr_factor
                param_groups.append({'params': [self._hybrid_loss.ms_loss.pos_margin, self._hybrid_loss.ms_loss.neg_margin], 'lr': margin_lr, 'weight_decay': 0.0})
                self.logger.info(f"Hybrid: added learnable margins to optimizer (lr={margin_lr:.2e})")
        
        if hasattr(self, '_sliced_ms_loss'):
            margin_lr = lr * margin_lr_factor
            margin_params = self._sliced_ms_loss.get_learnable_margins()
            param_groups.append({'params': margin_params, 'lr': margin_lr, 'weight_decay': 0.0})
            self.logger.info(f"SlicedMS: added {len(margin_params)} learnable margins to optimizer (lr={margin_lr:.2e})")
            
        if hasattr(self, '_sliced_proxy_loss'):
            proxy_lr = lr * proxy_lr_factor
            proxy_params = self._sliced_proxy_loss.get_proxies()
            param_groups.append({'params': proxy_params, 'lr': proxy_lr, 'weight_decay': 0.0})
            self.logger.info(f"SlicedProxy: added {len(proxy_params)} learnable proxy matrices to optimizer (lr={proxy_lr:.2e})")
        
        if self.hier_reg is not None:
            hier_lr = lr * margin_lr_factor
            param_groups.append({'params': self.hier_reg.parameters(), 'lr': hier_lr})
            self.logger.info(f"HIER proxies added to optimizer (lr={hier_lr:.2e})")

        # ArcFace / large-margin losses: trainable class weight matrix W
        if self._get_loss_type() == 'arcface' and hasattr(self.loss_func, 'parameters'):
            arcface_params = list(self.loss_func.parameters())
            if arcface_params:
                param_groups.append(
                    {'params': arcface_params, 'lr': lr, 'name': 'arcface_weights'}
                )
                self.logger.info(
                    f"ArcFace weights added to optimizer ({len(arcface_params)} tensors, lr={lr:.2e})"
                )

        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            opt_params = self.config.get("optimizer_params", {}) or {}
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=float(opt_params.get("momentum", 0.9)),
                nesterov=bool(opt_params.get("nesterov", False)),
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Store initial LR for warmup to reference
        for pg in optimizer.param_groups:
            pg['initial_lr'] = pg['lr']
        
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')
        params = self.config.get('scheduler_params', {})
        
        if scheduler_type == 'cosine':
            epochs = int(self.config.get('epochs', 50))
            warmup = int(self.config.get('warmup_epochs', 0) or 0)
            eta_min = float(params.get('eta_min', 1e-6))
            t_max_raw = params.get('T_max')
            if t_max_raw is not None:
                t_max = max(1, int(t_max_raw))
            else:
                t_max = max(1, epochs - warmup)

            if warmup > 0:
                lr = float(self.config.get('learning_rate', 1e-4))
                warmup_start_lr = self.config.get('warmup_start_lr')
                if warmup_start_lr is None:
                    warmup_start_lr = lr / max(1.0, 10.0 * warmup)
                warmup_start_factor = max(1e-8, float(warmup_start_lr) / max(lr, 1e-12))
                if 'warmup_start_factor' in params:
                    warmup_start_factor = float(params['warmup_start_factor'])
                warmup_sched = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=warmup_start_factor,
                    end_factor=1.0,
                    total_iters=warmup,
                )
                cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=t_max,
                    eta_min=eta_min,
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer,
                    schedulers=[warmup_sched, cosine_sched],
                    milestones=[warmup],
                )

            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        elif scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=params.get('mode', 'min'),
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 5),
                min_lr=params.get('min_lr', 1e-7)
            )
        elif scheduler_type == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.get('learning_rate', 1e-4),
                epochs=self.config.get('epochs', 50),
                steps_per_epoch=len(self.train_loader),
                pct_start=float(params.get('pct_start', 0.3)),
            )
        else:
            return None
    
    def _reset_cuda_peak_memory(self):
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def _log_cuda_memory(self, stage: str):
        if self.device.type != 'cuda':
            return
        alloc_mb = torch.cuda.memory_allocated(self.device) / 1e6
        reserved_mb = torch.cuda.memory_reserved(self.device) / 1e6
        peak_alloc_mb = torch.cuda.max_memory_allocated(self.device) / 1e6
        peak_reserved_mb = torch.cuda.max_memory_reserved(self.device) / 1e6
        self.logger.info(
            f"  VRAM [{stage}]: active={alloc_mb:.0f} MB | reserved={reserved_mb:.0f} MB | "
            f"peak_active={peak_alloc_mb:.0f} MB | peak_reserved={peak_reserved_mb:.0f} MB"
        )
        vram_warn_fraction = float(self.config.get("vram_warn_fraction", 0.9))
        total_mb = torch.cuda.get_device_properties(self.device).total_memory / 1e6
        warn_threshold_mb = total_mb * vram_warn_fraction
        if peak_reserved_mb > warn_threshold_mb:
            self.alarm_manager.warning(
                "VRAM_PEAK_HIGH",
                "GPU reserved memory above configured fraction of device VRAM.",
                stage=stage,
                peak_reserved_mb=round(peak_reserved_mb, 1),
                peak_allocated_mb=round(peak_alloc_mb, 1),
                warn_threshold_mb=round(warn_threshold_mb, 1),
                vram_warn_fraction=vram_warn_fraction,
            )

    def _iter_optimized_parameters(self):
        """All parameters updated by the optimizer (model + loss heads)."""
        seen = set()
        for group in self.optimizer.param_groups:
            for param in group['params']:
                pid = id(param)
                if pid not in seen:
                    seen.add(pid)
                    yield param

    def _warmup_lr(self, epoch: int):
        """
        Apply learning rate warmup for first few epochs.
        SOTA best practice to stabilize training.
        Respects per-group base LR (differential LR for backbone vs projection).
        """
        warmup_epochs = self.config.get('warmup_epochs', 5)
        if epoch < warmup_epochs:
            progress = (epoch + 1) / warmup_epochs  # reach full LR on last warmup epoch
            lr_strs = []
            global_lr = float(self.config.get('learning_rate', 1e-4))
            warmup_start_lr_cfg = self.config.get('warmup_start_lr')
            if warmup_start_lr_cfg is None:
                warmup_start_lr_cfg = global_lr / max(1.0, 10.0 * warmup_epochs)
            for param_group in self.optimizer.param_groups:
                base_lr = param_group.get('initial_lr', param_group['lr'])
                start_lr = float(warmup_start_lr_cfg) * (base_lr / max(global_lr, 1e-12))
                warmup_lr = start_lr + (base_lr - start_lr) * progress
                param_group['lr'] = warmup_lr
                name = param_group.get('name', 'default')
                lr_strs.append(f"{name}={warmup_lr:.2e}")
            self.logger.info(f"Warmup LR: {', '.join(lr_strs)}")
        elif epoch == warmup_epochs:
            lr_strs = []
            for param_group in self.optimizer.param_groups:
                base_lr = param_group.get('initial_lr', param_group['lr'])
                param_group['lr'] = base_lr
                name = param_group.get('name', 'default')
                lr_strs.append(f"{name}={base_lr:.2e}")
            self.logger.info(f"Warmup complete â€” restored LR: {', '.join(lr_strs)}")
    
    def _compute_base_metric_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        prenorm_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Loss métrica sin regularizadores SOTA (comparable train vs val P×K)."""
        loss_embeddings = self._loss_forward_embeddings(embeddings, prenorm_embeddings)
        if self.miner is not None:
            hard_pairs = self.miner(loss_embeddings, labels)
            return self.loss_func(loss_embeddings, labels, hard_pairs)
        return self.loss_func(loss_embeddings, labels)

    def _estimate_pk_val_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        *,
        prenorm_embeddings: Optional[torch.Tensor] = None,
        num_batches: int = 16,
    ) -> float:
        """
        Estima val_loss en mini-batches P×K muestreados del set val.

        El DataLoader de val no usa MPerClassSampler; el promedio por batch
        secuencial no es comparable con train_loss (sliced_ms / multi-similarity).
        """
        import random

        data_cfg = self.config.get("data", {}) or {}
        p = int(data_cfg.get("classes_per_batch", 8))
        k = int(data_cfg.get("samples_per_class", 2))
        labels_cpu = labels.detach().cpu()
        by_class: Dict[int, List[int]] = {}
        for idx, lab in enumerate(labels_cpu.tolist()):
            by_class.setdefault(int(lab), []).append(idx)

        eligible = [c for c, ids in by_class.items() if len(ids) >= k]
        if len(eligible) < 2:
            return float("nan")

        p_eff = min(p, len(eligible))
        seed = int(self.config.get("reproducibility", {}).get("seed", 42))
        rng = random.Random(seed + int(self.current_epoch))
        losses: List[float] = []

        with torch.inference_mode():
            for _ in range(num_batches):
                chosen = rng.sample(eligible, p_eff)
                idxs: List[int] = []
                for cls_id in chosen:
                    idxs.extend(rng.sample(by_class[cls_id], k))
                batch_emb = embeddings[idxs].to(self.device)
                batch_lab = labels_cpu[idxs].to(self.device)
                batch_prenorm = (
                    prenorm_embeddings[idxs].to(self.device)
                    if prenorm_embeddings is not None
                    else batch_emb
                )
                loss = self._compute_base_metric_loss(batch_emb, batch_lab, batch_prenorm)
                losses.append(float(loss.item()))

        return float(sum(losses) / max(1, len(losses)))

    def _compute_total_loss(self, embeddings, labels, images, masks=None, base_loss=None, prenorm_embeddings=None):
        """
        Compute total loss = base loss + SOTA regularizers.
        """
        # Base loss
        if base_loss is not None:
            loss = base_loss
        else:
            loss_embeddings = self._loss_forward_embeddings(embeddings, prenorm_embeddings)
            if self.miner is not None:
                hard_pairs = self.miner(loss_embeddings, labels)
                loss = self.loss_func(loss_embeddings, labels, hard_pairs)
            else:
                loss = self.loss_func(loss_embeddings, labels)
        
        if self.dada_wrapper is not None:
            raise RuntimeError(
                "DADA está habilitado pero no debe alcanzar el loop de entrenamiento "
                "sin dual_view_implemented (validar con validate_sota_integrity)."
            )

        # NIR: Non-Isotropy Regularization
        if self.nir_reg is not None:
            nir_loss = self.nir_reg(embeddings, labels)
            loss += nir_loss
        
        # HIER: Hierarchical Regularization
        if self.hier_reg is not None:
            hier_loss = self.hier_reg(embeddings, labels)
            loss += hier_loss

        if self.anti_collapse is not None:
            loss += self.anti_collapse(embeddings, labels)

        # Self-distillation with EMA Teacher
        if self.ema_teacher is not None and self.sd_loss is not None:
            teacher_embeddings = self.ema_teacher.get_teacher_embeddings(images, masks=masks)
            sd_loss_val = self.sd_loss(embeddings, teacher_embeddings)
            loss += sd_loss_val
        
        return loss

    @staticmethod
    def _is_dataloader_worker_failure(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "dataloader worker" in msg
            and "exited unexpectedly" in msg
        )

    def _raise_if_dataloader_worker_failure(self, exc: Exception, stage: str):
        if self._is_dataloader_worker_failure(exc):
            workers = getattr(self.train_loader, "num_workers", "?")
            self.alarm_manager.error(
                "WORKER_CRASH",
                "DataLoader worker process crashed.",
                stage=stage,
                configured_workers=workers,
                error=str(exc),
            )
            raise RuntimeError(
                f"DataLoader worker crash during {stage}. "
                f"Configured num_workers={workers}. "
                "Training was paused without fallback. "
                "Fix multiprocessing/dataset issues or adjust num_workers manually."
            ) from exc
        raise exc

    def _assert_worker_regime(self, stage: str):
        train_workers = int(getattr(self.train_loader, "num_workers", -1))
        val_workers = int(getattr(self.val_loader, "num_workers", -1))

        if self.expected_train_workers >= 0 and train_workers != self.expected_train_workers:
            self.alarm_manager.error(
                "WORKER_REGIME_DRIFT",
                "Train DataLoader worker count drift detected.",
                stage=stage,
                expected=self.expected_train_workers,
                actual=train_workers,
            )
            raise RuntimeError(
                f"Train DataLoader workers drifted from {self.expected_train_workers} to {train_workers}."
            )
        if self.expected_val_workers >= 0 and val_workers != self.expected_val_workers:
            self.alarm_manager.error(
                "WORKER_REGIME_DRIFT",
                "Validation DataLoader worker count drift detected.",
                stage=stage,
                expected=self.expected_val_workers,
                actual=val_workers,
            )
            raise RuntimeError(
                f"Val DataLoader workers drifted from {self.expected_val_workers} to {val_workers}."
            )

        self.alarm_manager.info(
            "WORKER_REGIME_OK",
            "Worker regime verified.",
            stage=stage,
            train_workers=train_workers,
            val_workers=val_workers,
        )

    def _check_alive_workers(self, loader, expected_workers: int, stage: str):
        if expected_workers <= 0:
            return
        iterator = getattr(loader, "_iterator", None)
        workers = getattr(iterator, "_workers", None) if iterator is not None else None
        if workers is None:
            return
        alive = sum(1 for w in workers if w.is_alive())
        if alive < expected_workers:
            self.alarm_manager.error(
                "WORKER_COUNT_DROPPED",
                "One or more DataLoader workers are no longer alive.",
                stage=stage,
                expected_workers=expected_workers,
                alive_workers=alive,
            )
            raise RuntimeError(
                f"DataLoader workers dropped: alive={alive}, expected={expected_workers}."
            )
    
    def train_epoch(self) -> float:
        """Train one epoch with mixed precision and SOTA regularizers."""
        self._assert_worker_regime(stage="train_epoch_start")
        self._reset_cuda_peak_memory()
        self.model.train()
        total_loss = 0
        num_batches = 0
        grad_norm = 0.0

        # GPU utilization tracking
        data_time_total = 0.0
        gpu_time_total = 0.0
        t_data_start = time.time()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')

        try:
            for batch_idx, batch_data in enumerate(pbar):
                # Support both (images, labels) and (images, labels, masks)
                if len(batch_data) == 3:
                    images, labels, masks = batch_data
                    masks = masks.to(self.device, non_blocking=True)
                else:
                    images, labels = batch_data
                    masks = None
                data_time_total += time.time() - t_data_start
                self._check_alive_workers(
                    self.train_loader,
                    expected_workers=self.expected_train_workers,
                    stage=f"train_batch_{batch_idx}",
                )
                if (time.time() - t_data_start) > self.batch_stall_threshold_sec:
                    self.alarm_manager.warning(
                        "BATCH_DATA_STALL",
                        "Batch data wait exceeded stall threshold.",
                        stage="train",
                        batch_idx=batch_idx,
                        wait_sec=round(time.time() - t_data_start, 3),
                        threshold_sec=self.batch_stall_threshold_sec,
                    )

                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # First-batch diagnostic (once per training run)
                if not self._logged_first_batch:
                    self._logged_first_batch = True
                    n_classes_batch = len(torch.unique(labels))
                    self.logger.info(f"  [DIAG] First batch shapes: images={tuple(images.shape)}, labels={tuple(labels.shape)}")
                    self.logger.info(f"  [DIAG] Batch: {images.shape[0]} samples, {n_classes_batch} unique classes")
                    self.logger.info(f"  [DIAG] Image range: [{images.min():.3f}, {images.max():.3f}] (post-normalize)")
                    self.logger.info(f"  [DIAG] Masks: {'provided '+str(tuple(masks.shape)) if masks is not None else 'None (standard pooling)'}")
                    self.logger.info(f"  [DIAG] AMP: {self.use_amp}, device: {self.device}")

                t_gpu_start = time.time()
                
                # Gradient Accumulation: only zero_grad at the start of each accumulation cycle
                accum_step = batch_idx % self.gradient_accumulation_steps
                if accum_step == 0:
                    self.optimizer.zero_grad(set_to_none=True)

                optimizer_stepped = False
                is_accumulation_boundary = (accum_step == self.gradient_accumulation_steps - 1) or (batch_idx == len(self.train_loader) - 1)
                
                if self.use_amp:
                    with autocast('cuda'):
                        embeddings, prenorm_embeddings = self.model(images, mask=masks, return_prenorm=True)
                        loss = self._compute_training_loss(embeddings, labels, images, masks=masks, prenorm_embeddings=prenorm_embeddings)
                    
                    # Scale loss by accumulation steps for correct gradient magnitude
                    scaled_loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    
                    if is_accumulation_boundary:
                        self.scaler.unscale_(self.optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._iter_optimized_parameters(),
                            max_norm=self.gradient_clip_norm
                        )
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_stepped = (scale_before <= scale_after)
                    else:
                        grad_norm = torch.tensor(0.0)
                else:
                    embeddings, prenorm_embeddings = self.model(images, mask=masks, return_prenorm=True)
                    loss = self._compute_training_loss(embeddings, labels, images, masks=masks, prenorm_embeddings=prenorm_embeddings)
                    
                    scaled_loss = loss / self.gradient_accumulation_steps
                    scaled_loss.backward()
                    
                    if is_accumulation_boundary:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self._iter_optimized_parameters(),
                            max_norm=self.gradient_clip_norm
                        )
                        self.optimizer.step()
                        optimizer_stepped = True
                    else:
                        grad_norm = torch.tensor(0.0)

                gpu_time_total += time.time() - t_gpu_start

                if optimizer_stepped:
                    # Update EMA teacher after each actual optimizer step
                    if self.ema_teacher is not None:
                        self.ema_teacher.update(self.model)

                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{total_loss/num_batches:.4f}',
                    'gnorm': f'{grad_norm:.2f}'
                })
                t_data_start = time.time()
                
                if optimizer_stepped and self.config.get('scheduler') == 'onecycle' and self.scheduler is not None:
                    self.scheduler.step()
        except torch.cuda.OutOfMemoryError as e:
            self.alarm_manager.error("CUDA_OOM", "Out of memory during training.", stage="train", error=str(e))
            torch.cuda.empty_cache()
            raise e
        except RuntimeError as e:
            self._raise_if_dataloader_worker_failure(e, stage="training")

        avg_loss = total_loss / max(1, num_batches)
        if self.hard_miner is not None:
            mining_stats = self.hard_miner.get_stats()
            self.logger.info(
                f"  OHEM: selected {mining_stats.get('avg_mining_ratio', 0):.1%} of samples/batch "
                f"(hard_loss={mining_stats.get('avg_loss_hard', 0):.4f}, "
                f"easy_loss={mining_stats.get('avg_loss_easy', 0):.4f})"
            )
        if not np.isfinite(float(grad_norm)):
            self.alarm_manager.warning(
                "GRAD_NORM_NON_FINITE",
                "Gradient norm became non-finite.",
                grad_norm=str(grad_norm),
            )

        # Log GPU utilization stats (single sync â€” avoids per-batch GPU idle in Task Manager)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        total_time = data_time_total + gpu_time_total
        if total_time > 0:
            gpu_pct = gpu_time_total / total_time * 100
            self.logger.info(
                f"  GPU: {gpu_pct:.0f}% util ({gpu_time_total:.1f}s compute, {data_time_total:.1f}s data)"
            )
            batches_per_sec = num_batches / max(gpu_time_total, 1e-6)
            self.logger.info(
                f"  Throughput: {num_batches} batches, ~{batches_per_sec:.2f} train-steps/s (GPU-bound if >1.5)"
            )
            self._log_cuda_memory(stage="train_epoch_end")
            loader_policy = self.config.get("loader_policy", {}) or {}
            alarm_min_workers = int(loader_policy.get("alarm_min_workers", 8))
            alarm_gpu_util_pct = float(loader_policy.get("alarm_gpu_util_pct", 45.0))
            if self.expected_train_workers >= alarm_min_workers and gpu_pct < alarm_gpu_util_pct:
                self.alarm_manager.warning(
                    "LOW_GPU_UTIL_WITH_HIGH_WORKERS",
                    "GPU utilization is low despite high worker regime.",
                    gpu_util_pct=round(gpu_pct, 2),
                    expected_train_workers=self.expected_train_workers,
                    alarm_min_workers=alarm_min_workers,
                    alarm_gpu_util_pct=alarm_gpu_util_pct,
                    data_time_sec=round(data_time_total, 2),
                    gpu_time_sec=round(gpu_time_total, 2),
                )
        self.logger.info(f"  Train loss: {avg_loss:.6f} (last grad_norm: {grad_norm:.3f})")

        return avg_loss

    def validate(self) -> Tuple[float, Dict]:
        """
        Validate the model: forward pass on val set, then full SOTA metrics on embeddings.

        Computes ArcFace val loss and mandatory retrieval/clustering metrics each epoch.

        Returns:
            Tuple of (avg_loss, metrics_dict) with R@1, R@5, mAP@R, NMI, F1-macro
        """
        import torch.nn.functional as F
        from src.utils.metrics_utils import compute_all_metrics

        self._assert_worker_regime(stage="validation_start")
        self._check_alive_workers(
            self.val_loader,
            expected_workers=self.expected_val_workers,
            stage="validation_start",
        )
        self.model.eval()
        emb_chunks = []
        prenorm_chunks = []
        label_chunks = []
        val_data_time = 0.0
        val_gpu_time = 0.0
        t_batch_wait = time.time()

        t_embed_start = time.time()
        with torch.inference_mode():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
            try:
                for batch_idx, batch_data in enumerate(pbar):
                    batch_wait = time.time() - t_batch_wait
                    val_data_time += batch_wait
                    if batch_idx == 0:
                        self.logger.info(
                            f"  Val first-batch wait: {batch_wait:.1f}s "
                            f"(workers={self.expected_val_workers})"
                        )
                    if batch_idx > 0 and batch_wait > self.batch_stall_threshold_sec:
                        self.alarm_manager.warning(
                            "BATCH_DATA_STALL",
                            "Val batch data wait exceeded stall threshold.",
                            stage="validation",
                            batch_idx=batch_idx,
                            wait_sec=round(batch_wait, 3),
                            threshold_sec=self.batch_stall_threshold_sec,
                        )

                    if len(batch_data) == 3:
                        images, labels, masks = batch_data
                        masks = masks.to(self.device, non_blocking=True)
                    else:
                        images, labels = batch_data
                        masks = None
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    t_gpu_start = time.time()
                    if self.use_amp and self.device.type == "cuda":
                        with autocast("cuda"):
                            embeddings, _prenorm = self.model(
                                images, mask=masks, return_prenorm=True
                            )
                    else:
                        embeddings, _prenorm = self.model(
                            images, mask=masks, return_prenorm=True
                        )

                    emb_chunks.append(embeddings.detach())
                    prenorm_chunks.append(_prenorm.detach())
                    label_chunks.append(labels.detach())

                    val_gpu_time += time.time() - t_gpu_start
                    t_batch_wait = time.time()
            except torch.cuda.OutOfMemoryError as e:
                self.alarm_manager.error("CUDA_OOM", "Out of memory during validation.", stage="validation", error=str(e))
                torch.cuda.empty_cache()
                raise e
            except RuntimeError as e:
                self._raise_if_dataloader_worker_failure(e, stage="validation")

        embed_sec = time.time() - t_embed_start

        if not emb_chunks:
            self.logger.warning("Validation set is empty (0 batches). Skipping validation metrics.")
            return 0.0, {'val_loss': 0.0, 'recall_at_1': 0.0, 'recall_at_5': 0.0}

        all_embeddings = torch.cat(emb_chunks, dim=0)
        all_prenorm = torch.cat(prenorm_chunks, dim=0) if prenorm_chunks else None
        all_labels = torch.cat(label_chunks, dim=0)
        del emb_chunks, prenorm_chunks, label_chunks

        avg_loss = self._estimate_pk_val_loss(
            all_embeddings, all_labels, prenorm_embeddings=all_prenorm
        )
        if not np.isfinite(avg_loss):
            self.logger.warning(
                "val_loss P×K no disponible (pocas muestras/clase en val); "
                "use mAP@R / R@1 como métrica principal."
            )
            avg_loss = 0.0
        elif self.current_epoch == 0:
            self.logger.info(
                "val_loss se estima con mini-batches P×K en val "
                "(misma estructura que train). El checkpoint se selecciona por %s.",
                self.validation_metric,
            )

        embeddings_np = all_embeddings.cpu().numpy()
        labels_np = all_labels.cpu().numpy()
        del all_embeddings, all_labels

        class_names = getattr(self.val_loader.dataset, 'classes', None)
        t_metrics_start = time.time()
        eval_cfg = self.config.get("evaluation", {}) or {}
        nn_metric = resolve_nn_metric(self.config)
        map_exact, map_multiplier = resolve_map_at_r_config(self.config)
        nmi_n_init, nmi_n_seeds, nmi_base_seed = resolve_nmi_config(self.config)

        sota_metrics = compute_all_metrics(
            embeddings_np,
            labels_np,
            nn_metric=nn_metric,
            k_list=[1, 5, 10],
            class_names=class_names,
            include_per_class=False,
            nmi_n_init=nmi_n_init,
            nmi_n_seeds=nmi_n_seeds,
            nmi_base_seed=nmi_base_seed,
            map_at_r_exact=map_exact,
            map_at_r_candidate_multiplier=map_multiplier,
        )
        metrics_sec = time.time() - t_metrics_start
        compute_sec = max(0.0, embed_sec - val_data_time)
        self.logger.info(
            f"  Val timing: total={embed_sec:.1f}s, data_wait={val_data_time:.1f}s, "
            f"gpu={val_gpu_time:.1f}s, metrics={metrics_sec:.1f}s (n={embeddings_np.shape[0]})"
        )

        # Embedding diagnostics
        norms = np.linalg.norm(embeddings_np, axis=1)
        self.logger.info(
            f"  Val embeddings: n={embeddings_np.shape[0]}, dim={embeddings_np.shape[1]}, "
            f"norm: {norms.mean():.4f}Â±{norms.std():.4f}"
        )

        metrics = {
            'val_loss': avg_loss,
            'R@1': sota_metrics.get('R@1', 0.0),
            'R@5': sota_metrics.get('R@5', 0.0),
            'R@10': sota_metrics.get('R@10', 0.0),
            'mAP@R': sota_metrics.get('mAP@R', 0.0),
            'NMI': sota_metrics.get('NMI', 0.0),
            'NMI_std': sota_metrics.get('NMI_std', 0.0),
            'F1_macro': sota_metrics.get('F1_macro', 0.0),
            'Accuracy': sota_metrics.get('Accuracy', 0.0),
            'Balanced_Accuracy': sota_metrics.get('Balanced_Accuracy', 0.0),
            'Precision_macro': sota_metrics.get('Precision_macro', 0.0),
            'Precision_weighted': sota_metrics.get('Precision_weighted', 0.0),
            'F1_weighted': sota_metrics.get('F1_weighted', 0.0),
        }
        del embeddings_np, labels_np, sota_metrics

        self._log_cuda_memory(stage="validation_end")
        return avg_loss, metrics
    
    def _extract_reference_embeddings(self):
        """Extrae embeddings de referencia del conjunto de entrenamiento"""
        self.logger.info("Extracting reference embeddings from training set...")
        self.model.eval()
        
        embeddings_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch_data in tqdm(self.train_loader, desc="Building reference"):
                if len(batch_data) == 3:
                    images, labels, masks = batch_data
                    masks = masks.to(self.device, non_blocking=True)
                else:
                    images, labels = batch_data[0], batch_data[1]
                    masks = None
                images = images.to(self.device, non_blocking=True)
                embeddings = self.model(images, mask=masks)
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(labels.numpy())
        
        reference_embeddings = np.vstack(embeddings_list)
        reference_labels = np.concatenate(labels_list)
        
        # Obtener nombres de clases del dataset
        class_names = self.train_loader.dataset.classes
        
        self.logger.info(f"Reference embeddings: {reference_embeddings.shape}")
        return reference_embeddings, reference_labels, class_names
    
    def save_checkpoint(self, epoch: int, val_loss: float, filepath: str, save_reference: bool = True):
        """
        Save model checkpoint with optional reference embeddings.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            filepath: Path to save checkpoint
            save_reference: If True, includes reference embeddings for inference
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'selection_metric': self.validation_metric,
            'selection_value': float(self.best_selection_value),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss_state_dict': self.loss_func.state_dict(),
            'loss_type': self._get_loss_type(),
        }
        
        # Guardar embeddings de referencia para inferencia
        if save_reference:
            ref_embeddings, ref_labels, class_names = self._extract_reference_embeddings()
            checkpoint['reference_embeddings'] = ref_embeddings
            checkpoint['reference_labels'] = ref_labels
            checkpoint['class_names'] = class_names
            self.logger.info("Reference embeddings included in checkpoint")
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if checkpoint.get('loss_state_dict'):
            try:
                self.loss_func.load_state_dict(checkpoint['loss_state_dict'])
                self.logger.info("Loss state restored from checkpoint")
            except Exception as exc:
                self.logger.warning(f"Could not restore loss state: {exc}")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        # Maintain history if already populated (e.g., during early stopping rollback)
        if not self.train_losses:
            self.train_losses = checkpoint.get('train_losses', [])
        if not self.val_losses:
            self.val_losses = checkpoint.get('val_losses', [])
        
        self.logger.info(f"Checkpoint loaded: {filepath} (epoch {self.current_epoch})")
    
    def fit(self, epochs: int, resume_from: Optional[str] = None) -> Tuple[list, list]:
        """
        Train the model for N epochs.
        
        Args:
            epochs: Number of epochs to train
            resume_from: Optional checkpoint path to resume from
            
        Returns:
            Tuple of (train_losses, val_losses)
        """
        from src.data.dataloader_utils import free_training_memory

        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        # Initialize metrics CSV in run directory for progressive tracking
        run_dir = str(Path(self.checkpoint_dir).parent)
        self.metrics_csv_path = Path(run_dir) / 'training_metrics.csv'
        with open(self.metrics_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss',
                'selection_metric', 'selection_value',
                'recall_at_1', 'recall_at_5', 'recall_at_10',
                'mAP@R', 'NMI', 'F1_macro',
                'accuracy', 'balanced_accuracy',
                'precision_macro', 'precision_weighted', 'F1_weighted',
                'learning_rate', 'is_best', 'best_selection_value', 'epoch_time_sec'
            ])
        self.logger.info(f"Metrics CSV: {self.metrics_csv_path}")
        
        if self.finetune_phases and start_epoch == 0:
            self._apply_finetune_phase(0)

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            if self.finetune_phases:
                for phase_idx, boundary in enumerate(self._finetune_phase_boundaries):
                    if epoch == boundary and phase_idx + 1 < len(self.finetune_phases):
                        self._apply_finetune_phase(phase_idx + 1)
            epoch_start_time = time.time()
            self.alarm_manager.info(
                "EPOCH_HEARTBEAT",
                "Epoch started.",
                epoch=epoch + 1,
                total_epochs=epochs,
                expected_train_workers=self.expected_train_workers,
                expected_val_workers=self.expected_val_workers,
            )
            
            self.logger.info(f"\n{'='*70}")
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            self.logger.info(f"{'='*70}")
            
            if self.hard_miner is not None:
                self.hard_miner.set_epoch(epoch)

            # Free val workers before train so only one worker pool runs at a time.
            if self.data_engine is not None:
                self.data_engine.release_val_workers()

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate every epoch — full SOTA metrics always computed after embeddings.
            if self.data_engine is not None:
                self.data_engine.release_train_workers()
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics_history.append(val_metrics)
            
            if self.data_engine is not None:
                self.data_engine.release_val_workers()

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.config.get('scheduler') != 'onecycle':
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            recall_at_1 = val_metrics.get('R@1', 0.0)
            recall_at_5 = val_metrics.get('R@5', 0.0)
            recall_at_10 = val_metrics.get('R@10', 0.0)
            map_at_r = val_metrics.get('mAP@R', 0.0)
            nmi = val_metrics.get('NMI', 0.0)
            f1_macro = val_metrics.get('F1_macro', 0.0)
            accuracy = val_metrics.get('Accuracy', 0.0)
            balanced_accuracy = val_metrics.get('Balanced_Accuracy', 0.0)
            precision_macro = val_metrics.get('Precision_macro', 0.0)
            precision_weighted = val_metrics.get('Precision_weighted', 0.0)
            f1_weighted = val_metrics.get('F1_weighted', 0.0)
            selection_value = metric_value_from_dict(self.validation_metric, val_metrics)
            
            # Log SOTA metrics
            self.logger.info(f"\n--- SOTA Metrics (CVPR 2016-2024) ---")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            self.logger.info(f"  Val Loss:   {val_loss:.4f}  (P×K, auxiliar)")
            self.logger.info(f"  Sel. metric: {self.validation_metric}={selection_value:.4f}")
            self.logger.info(f"  R@1:        {recall_at_1:.4f}")
            self.logger.info(f"  R@5:        {recall_at_5:.4f}")
            self.logger.info(f"  R@10:       {recall_at_10:.4f}")
            self.logger.info(f"  mAP@R:      {map_at_r:.4f}")
            self.logger.info(f"  NMI:        {nmi:.4f}")
            self.logger.info(f"  F1-Macro:   {f1_macro:.4f}")
            self.logger.info(f"  Accuracy:   {accuracy:.4f}")
            self.logger.info(f"  BalAcc:     {balanced_accuracy:.4f}")
            self.logger.info(f"  Prec-Macro: {precision_macro:.4f}")
            self.logger.info(f"  Prec-Wght:  {precision_weighted:.4f}")
            self.logger.info(f"  F1-Wght:    {f1_weighted:.4f}")
            self.logger.info(f"  LR:         {current_lr:.2e}")

            # Determine if this is the best model (MLRC validation_metric)
            is_best = is_improvement(
                self.validation_metric,
                selection_value,
                self.best_selection_value,
                self.early_stopping_min_delta,
            )

            if is_best:
                self.best_selection_value = selection_value
                self.best_val_loss = val_loss
                if recall_at_1 > self.best_recall_at_1:
                    self.best_recall_at_1 = recall_at_1
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, val_loss, str(Path(self.checkpoint_dir) / 'best_model.pth'), save_reference=False)
                self.logger.info(
                    f"  >> New best model saved! "
                    f"({self.validation_metric}={selection_value:.4f}, R@1={recall_at_1:.4f})"
                )
            else:
                self.epochs_without_improvement += 1
                self.logger.info(f"  -- No improvement for {self.epochs_without_improvement} epochs")
            
            # Save epoch metrics to CSV (progressive â€” survives interruptions)
            epoch_time = time.time() - epoch_start_time
            with open(self.metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, f'{train_loss:.6f}', f'{val_loss:.6f}',
                    self.validation_metric, f'{selection_value:.6f}',
                    f'{recall_at_1:.6f}', f'{recall_at_5:.6f}', f'{recall_at_10:.6f}',
                    f'{map_at_r:.6f}', f'{nmi:.6f}', f'{f1_macro:.6f}',
                    f'{accuracy:.6f}', f'{balanced_accuracy:.6f}',
                    f'{precision_macro:.6f}', f'{precision_weighted:.6f}', f'{f1_weighted:.6f}',
                    f'{current_lr:.2e}', is_best, f'{self.best_selection_value:.6f}',
                    f'{epoch_time:.1f}'
                ])
            self.alarm_manager.info(
                "EPOCH_COMPLETED",
                "Epoch metrics persisted.",
                epoch=epoch + 1,
                epoch_time_sec=round(epoch_time, 2),
                is_best=bool(is_best),
                recall_at_1=round(float(recall_at_1), 6),
                recall_at_5=round(float(recall_at_5), 6),
                map_at_r=round(float(map_at_r), 6),
                nmi=round(float(nmi), 6),
                f1_macro=round(float(f1_macro), 6),
                val_loss=round(float(val_loss), 6),
            )
            free_training_memory(f"epoch_{epoch + 1}_end")

            # Progressive charts (CSV always updated; plots less often to save RAM/time).
            plot_every = int(self.config.get("progressive_plot_every_epochs", 2))
            is_last_epoch = (epoch + 1) >= epochs
            if plot_every > 0 and (is_last_epoch or (epoch + 1) % plot_every == 0):
                try:
                    from scripts.post_training import plot_training_curves
                    post_dir = Path(run_dir) / 'images' / 'post_training'
                    post_dir.mkdir(parents=True, exist_ok=True)
                    plot_training_curves(str(self.metrics_csv_path), str(post_dir))
                    self.logger.info(f"  Training curves updated: {post_dir}")
                    free_training_memory("post_plot_curves")
                except Exception as curve_err:
                    self.logger.warning(f"  Could not update training curves: {curve_err}")

            # Periodic checkpoint (no reference embeddings â€” only on final save)
            if (epoch + 1) % self.config.get('save_every_n_epochs', 10) == 0:
                self.save_checkpoint(epoch, val_loss, str(Path(self.checkpoint_dir) / f'checkpoint_epoch_{epoch+1}.pth'), save_reference=False)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                self.logger.info(f"\n[STOP] Early stopping: no improvement for {self.early_stopping_patience} epochs")
                
                # Restore best weights before breaking
                if hasattr(self, 'best_val_loss') and Path(self.checkpoint_dir, 'best_model.pth').exists():
                    self.logger.info("Restoring best model weights in memory...")
                    self.load_checkpoint(str(Path(self.checkpoint_dir) / 'best_model.pth'))
                
                break
        
        if self.data_engine is not None:
            self.data_engine.release_all_workers(close_datasets=True)

        self.logger.info(f"\n{'='*70}")
        self.logger.info("Training completed!")
        self.logger.info(f"Best {self.validation_metric}: {self.best_selection_value:.4f}")
        self.logger.info(f"Best val_loss (auxiliary P×K): {self.best_val_loss:.4f}")
        self.logger.info(f"{'='*70}\n")
        
        return self.train_losses, self.val_losses
