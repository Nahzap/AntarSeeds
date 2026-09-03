"""Phase C — config ↔ code integrity tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.transforms_enhanced import _coarse_dropout_transform, _resolve_normalize_stats
from src.models.analogy_net import AnalogyNet
from src.models.pooling_config import resolve_pooling_config
from src.training.trainer_enhanced import MetricLearningTrainer
from src.utils.config_validator import validate_config


def _base_config():
    return {
        "model": {
            "embedding_dim": 128,
            "pooling": {"masked": {"enabled": True}},
            "freeze_backbone": False,
            "num_unfrozen_blocks": 2,
        },
        "training": {
            "epochs": 10,
            "warmup_epochs": 3,
            "learning_rate": 0.0001,
            "validation_metric": "mAP@R",
            "loss": {
                "type": "sliced_ms",
                "distance": "euclidean",
                "params": {
                    "num_slices": 4,
                    "alpha": 1.0,
                    "beta": 50.0,
                    "base_margin": 0.5,
                },
            },
            "scheduler": "cosine",
            "scheduler_params": {"T_max": 10, "eta_min": 1e-6},
            "proxy_lr_factor": 100.0,
            "margin_lr_factor": 10.0,
            "mining": {"enabled": False, "cutoff": 0.5, "nonzero_loss_cutoff": 1.4},
        },
        "data": {
            "classes_per_batch": 8,
            "test_dir": "data/processed/test",
            "loader_policy": {"alarm_min_workers": 8, "alarm_gpu_util_pct": 45.0},
        },
        "hardware": {"use_gpu": False, "vram_warn_fraction": 0.9},
        "evaluation": {"report_final_on": "test", "n_runs": 10},
        "augmentation": {
            "grayscale": {"enabled": False, "override_norm": False},
            "domain_augmentation": {
                "enabled": True,
                "coarse_dropout_holes": 4,
                "coarse_dropout_hole_min": 20,
                "coarse_dropout_hole_max": 80,
            },
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "grain_detection": {
            "masked_pooling": True,
            "saliency_threshold": 0.3,
        },
        "sota": {
            "embedding_space": "euclidean",
            "nir": {"enabled": False},
            "dada": {"enabled": False, "dual_view_implemented": False},
            "hier": {"enabled": False},
        },
    }


class TestConfigValidator:
    def test_valid_sliced_ms_config(self):
        result = validate_config(_base_config(), check_data_dirs=False)
        assert result["valid"] is True

    def test_missing_sliced_ms_params_fails(self):
        cfg = _base_config()
        del cfg["training"]["loss"]["params"]["alpha"]
        result = validate_config(cfg, check_data_dirs=False)
        assert result["valid"] is False
        assert any("alpha" in e for e in result["errors"])


class TestPoolingThresholdSync:
    def test_saliency_drives_masked_threshold(self):
        cfg = _base_config()
        del cfg["model"]["pooling"]
        pooling = resolve_pooling_config(cfg["model"], cfg["grain_detection"])
        assert pooling["masked"]["threshold"] == pytest.approx(0.3)


class TestCoarseDropoutConfig:
    def test_holes_from_config(self):
        t = _coarse_dropout_transform({"coarse_dropout_holes": 4})
        assert t.num_holes_range == (4, 4)


class TestGrayscaleNorm:
    def test_override_norm_disabled_uses_dataset_stats(self):
        cfg = _base_config()["augmentation"]
        mean, std = _resolve_normalize_stats(cfg, is_grayscale=True)
        assert mean == cfg["normalize"]["mean"]

    def test_override_norm_enabled_uses_symmetric(self):
        cfg = _base_config()["augmentation"]
        cfg["grayscale"]["override_norm"] = True
        mean, std = _resolve_normalize_stats(cfg, is_grayscale=True)
        assert mean == [0.5, 0.5, 0.5]


class TestTrainerConfigWiring:
    def _minimal_trainer(self, config):
        model = nn.Linear(4, 4)
        train_loader = MagicMock()
        train_loader.__len__ = lambda _: 2
        val_loader = MagicMock()
        return MetricLearningTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_dir=None,
        )

    def test_t_max_from_scheduler_params(self):
        cfg = {
            "loss": {"type": "triplet", "params": {}},
            "epochs": 10,
            "warmup_epochs": 3,
            "learning_rate": 1e-4,
            "scheduler": "cosine",
            "scheduler_params": {"T_max": 10, "eta_min": 1e-6},
            "optimizer": "adamw",
            "num_classes": 4,
            "embedding_dim": 128,
            "validation_metric": "mAP@R",
        }
        trainer = self._minimal_trainer(cfg)
        sched = trainer._setup_scheduler()
        cosine = sched._schedulers[1] if hasattr(sched, "_schedulers") else sched
        assert cosine.T_max == 10

    def test_proxy_lr_factor_configurable(self):
        cfg = {
            "loss": {
                "type": "sg_softmax",
                "params": {"scale": 30.0, "margin": 0.4, "ortho_penalty_weight": 2.0},
            },
            "epochs": 2,
            "learning_rate": 1e-3,
            "proxy_lr_factor": 50.0,
            "margin_lr_factor": 5.0,
            "optimizer": "adamw",
            "num_classes": 4,
            "embedding_dim": 32,
            "validation_metric": "mAP@R",
        }
        trainer = self._minimal_trainer(cfg)
        proxy_group = trainer.optimizer.param_groups[-1]
        assert proxy_group["lr"] == pytest.approx(0.05)


class TestNumUnfrozenBlocks:
    def test_resolve_unfreeze_keys_last_n_blocks(self):
        backbone = nn.Module()
        backbone.blocks = nn.ModuleList([nn.Linear(1, 1) for _ in range(12)])
        model = AnalogyNet(backbone="resnet18", embedding_dim=128, pretrained=False)
        model.backbone = backbone
        keys = model._resolve_unfreeze_keys(num_unfrozen_blocks=2)
        assert keys == ["blocks.10", "blocks.11"]
