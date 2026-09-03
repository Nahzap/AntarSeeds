"""
Enhanced Data Transforms with Microscopy-Specific Augmentations
Based on SOTA research (2024) for microscopy image analysis

Author: MetricLearning Project
Date: 2026-01-05
"""

import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _resolve_normalize_stats(config: dict, is_grayscale: bool) -> tuple:
    """Return (mean, std) from data.normalize.mode (E-01) and grayscale override (C-11)."""
    from src.utils.dataset_norm import resolve_normalization_config

    resolved = resolve_normalization_config(config)
    norm_mean = list(resolved["mean"])
    norm_std = list(resolved["std"])
    gs = config.get("grayscale", {}) or {}
    if is_grayscale and gs.get("override_norm", False):
        logger.warning(
            "augmentation.grayscale.override_norm=true: usando [0.5,0.5,0.5] "
            "en lugar de data.normalize (ImageNet/dataset stats)"
        )
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    return norm_mean, norm_std


def _stain_normalize_step(config: dict):
    """Optional stain normalization before Albumentations Normalize (E-02)."""
    from src.data.stain_normalization import build_stain_normalizer
    from src.utils.dataset_norm import resolve_normalization_config

    resolved = resolve_normalization_config(config)
    method = resolved.get("stain_method")
    if resolved.get("mode") == "macenko" and not method:
        method = "macenko"
    stain_cfg = (config.get("data") or {}).get("normalize") or config.get("normalize") or {}
    normalizer = build_stain_normalizer(method, stain_cfg)
    if normalizer is None:
        return None

    def _apply(image, **kwargs):
        return normalizer(image)

    return A.Lambda(image=_apply, p=1.0)


def _coarse_dropout_transform(domain_aug: dict) -> A.CoarseDropout:
    """Build CoarseDropout from domain_augmentation config (C-02)."""
    holes = int(domain_aug.get("coarse_dropout_holes", 4))
    hole_min = int(domain_aug.get("coarse_dropout_hole_min", 20))
    hole_max = int(domain_aug.get("coarse_dropout_hole_max", 80))
    fill = int(domain_aug.get("coarse_dropout_fill", 128))
    p = float(domain_aug.get("coarse_dropout_p", 0.7))
    return A.CoarseDropout(
        num_holes_range=(max(1, holes), max(1, holes)),
        hole_height_range=(hole_min, hole_max),
        hole_width_range=(hole_min, hole_max),
        fill=fill,
        p=p,
    )


def get_input_size_for_backbone(backbone_name):
    """
    Obtiene la resolución de entrada óptima para cada backbone.
    
    Args:
        backbone_name: Nombre del backbone
        
    Returns:
        int: Tamaño de imagen (asume cuadrada, e.g., 224 → 224×224)
        
    Notes:
        - DINOv2 ViT-S/14: 252×252 (18×14 parches, sin fractales)
        - DeiT-S: 224×224 (estándar ImageNet, parche 16×16)
        - ResNet-50: 224×224 (estándar ImageNet)
        - ConvNeXt-V2 Tiny: 224×224 (timm default)
        - EfficientNet-B4: 380×380 (compound scaling)
        - Swin-T: 224×224 (window size 7×7)
    """
    # Tabla estática de resoluciones óptimas
    BACKBONE_RESOLUTIONS = {
        'resnet18': 224,
        'resnet50': 224,
        'convnext_v2_tiny': 224,
        'dinov2_vits14': 252,
        'dinov2_vitb14': 252,
        'dinov2_vits14_reg': 252,
        'dinov2_vitb14_reg': 252,
        'deit_small_patch16_224': 224,
        'efficientnet_b4': 380,
        'swin_tiny_patch4_window7_224': 224,
    }
    
    if backbone_name in BACKBONE_RESOLUTIONS:
        return BACKBONE_RESOLUTIONS[backbone_name]
    
    # Fallback: intentar consultar timm si está disponible
    try:
        import timm
        from timm.data import resolve_model_data_config
        
        # Intentar obtener config del modelo en timm
        data_config = resolve_model_data_config(pretrained_cfg=None, model=None, 
                                                  model_name=backbone_name)
        input_size = data_config.get('input_size', (3, 224, 224))
        resolution = input_size[1]  # Asume (C, H, W)
        logger.info(f"Resolved input size for '{backbone_name}' from timm: {resolution}×{resolution}")
        return resolution
    except Exception:
        # Fallback final: 224×224 (estándar ImageNet)
        logger.warning(
            f"Could not resolve input size for '{backbone_name}'. "
            f"Using default 224×224."
        )
        return 224


class MicroscopyAugmentation:
    """
    Microscopy-specific augmentation pipeline using Albumentations.
    Designed for pollen/honey microscopy images with domain-specific transforms.
    """
    
    def __init__(self, config=None):
        """
        Initialize microscopy augmentation pipeline.
        
        Args:
            config (dict): Configuration dictionary with augmentation parameters
        """
        if config is None:
            config = self._get_default_config()

        if config.get("performance_mode"):
            micro = config.setdefault("microscopy_specific", {})
            micro.setdefault("optical_distortion", {})["enabled"] = False
            config.setdefault("domain_augmentation", {})["enabled"] = False
        
        self.config = config
        self.transform = self._build_transform()
    
    def _get_default_config(self):
        """Default configuration for microscopy augmentation"""
        return {
            'image_size': 224,
            'resize_size': 256,
            'color_jitter': {
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.2,
                'hue': 0.1
            },
            'microscopy_specific': {
                'random_gamma': {'enabled': True, 'gamma_limit': [80, 120], 'p': 0.5},
                'gaussian_noise': {'enabled': True, 'var_limit': [10.0, 50.0], 'p': 0.5},
                'blur': {'enabled': True, 'blur_limit': 3, 'p': 0.3},
                'optical_distortion': {'enabled': True, 'distort_limit': 0.1, 'p': 0.3},
                'clahe': {'enabled': True, 'clip_limit': 2.0, 'tile_grid_size': [8, 8], 'p': 0.1}
            },
            'grayscale': {'enabled': True, 'p': 0.5},
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }

    def _scale_transforms(self, image_size: int, resize_size: int, random_crop: bool):
        """Resize / crop block shared by train pipeline and preview."""
        if resize_size > image_size:
            return [
                A.Resize(resize_size, resize_size),
                A.RandomCrop(image_size, image_size),
            ]
        if random_crop:
            return [
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.3, 1.0),
                    ratio=(0.8, 1.2),
                    p=1.0,
                )
            ]
        return [A.Resize(image_size, image_size)]

    def _build_transform(self):
        """Build Albumentations transform pipeline — optimized for DINOv2 fine-tuning."""
        transform_list = []
        image_size = self.config['image_size']
        resize_size = self.config.get('resize_size', image_size)
        random_crop = bool(self.config.get('random_crop', True))

        transform_list.extend(
            self._scale_transforms(image_size, resize_size, random_crop)
        )

        # Geometric transforms — read probabilities from config.yaml / GUI
        h_flip_p = float(self.config.get('random_horizontal_flip', 0.5))
        v_flip_p = float(self.config.get('random_vertical_flip', 0.5))
        rot_limit = int(self.config.get('random_rotation', 180))
        import cv2
        transform_list.extend([
            A.HorizontalFlip(p=h_flip_p),
            A.VerticalFlip(p=v_flip_p),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.1,
                rotate_limit=rot_limit,
                border_mode=cv2.BORDER_CONSTANT,
                value=[128, 128, 128],
                p=0.8
            )
        ])

        # Color augmentations — ONE unified block (no redundancy)
        grayscale_cfg = self.config.get('grayscale', {})
        is_grayscale = grayscale_cfg.get('enabled', False)
        
        if not is_grayscale:
            cj = self.config.get('color_jitter', {})
            transform_list.append(
                A.ColorJitter(
                    brightness=cj.get('brightness', 0.3),
                    contrast=cj.get('contrast', 0.3),
                    saturation=cj.get('saturation', 0.2),
                    hue=cj.get('hue', 0.1),
                    p=0.8
                )
            )


        # Microscopy-specific augmentations (SOTA 2024)
        micro = self.config.get('microscopy_specific', {})

        if micro.get('random_gamma', {}).get('enabled', False):
            transform_list.append(
                A.RandomGamma(
                    gamma_limit=tuple(micro['random_gamma']['gamma_limit']),
                    p=micro['random_gamma']['p']
                )
            )
        
        if micro.get('gaussian_noise', {}).get('enabled', False):
            from src.utils.sensor_noise import resolve_gauss_noise_var_limit

            v_lo, v_hi = resolve_gauss_noise_var_limit(self.config)
            transform_list.append(
                A.GaussNoise(
                    std_range=(v_lo / 255.0, v_hi / 255.0),
                    p=micro['gaussian_noise']['p']
                )
            )
        
        if micro.get('blur', {}).get('enabled', False):
            transform_list.append(
                A.Blur(
                    blur_limit=micro['blur']['blur_limit'],
                    p=micro['blur']['p']
                )
            )
        
        if micro.get('optical_distortion', {}).get('enabled', False):
            transform_list.append(
                A.OpticalDistortion(
                    distort_limit=micro['optical_distortion']['distort_limit'],
                    p=micro['optical_distortion']['p']
                )
            )
        
        if micro.get('clahe', {}).get('enabled', False):
            transform_list.append(
                A.CLAHE(
                    clip_limit=micro['clahe']['clip_limit'],
                    tile_grid_size=tuple(micro['clahe']['tile_grid_size']),
                    p=micro['clahe']['p']
                )
            )
        
        # Domain-robustness augmentations (simulate different microscopes/preparations)
        # NOTE: removed redundant RandomBrightnessContrast (already covered by ColorJitter above)
        domain_aug = self.config.get('domain_augmentation', {})
        if domain_aug.get('enabled', True):
            rgb_shift = domain_aug.get('rgb_shift_limit', 20)
            tone_scale = domain_aug.get('tone_curve_scale', 0.15)
            ds_min = domain_aug.get('downscale_min', 0.6)
            # Avoid destructive resolution/color drops on grayscale/low-res
            if not is_grayscale:
                transform_list.extend([
                    A.RGBShift(
                        r_shift_limit=rgb_shift, g_shift_limit=rgb_shift,
                        b_shift_limit=rgb_shift, p=0.3
                    ),
                    A.RandomToneCurve(scale=tone_scale, p=0.05),
                ])
                
            # Only apply Downscale/Dropout if explicitly configured (default skip for B/W to preserve morphology)
            if not is_grayscale or domain_aug.get('force_geometric_in_bw', False):
                transform_list.extend([
                    A.Downscale(
                        scale_range=(ds_min, 0.8), p=0.5
                    ),
                    _coarse_dropout_transform(domain_aug),
                ])
            
        # RGB → grayscale: aplicado DESPUÉS de alteraciones de color para no re-colorear
        gs = self.config.get('grayscale', {})
        if gs.get('enabled', False):
            transform_list.append(A.ToGray(p=float(gs.get('p', 0.5))))
        
        stain_step = _stain_normalize_step(self.config)
        if stain_step is not None:
            transform_list.append(stain_step)

        norm_mean, norm_std = _resolve_normalize_stats(self.config, is_grayscale)

        transform_list.extend([
            A.Normalize(
                mean=norm_mean,
                std=norm_std
            ),
            ToTensorV2()
        ])
        
        return A.Compose(transform_list, additional_targets={"mask": "mask"})

    def apply_with_mask(self, image, mask):
        """Apply pipeline to image + mask (same geometry)."""
        from src.data.grain_transform import to_mask_float_tensor, to_mask_uint8_numpy

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        out = self.transform(image=image, mask=to_mask_uint8_numpy(mask))
        return out["image"], to_mask_float_tensor(out["mask"])

    def __call__(self, image, mask=None):
        """
        Apply augmentation to image (and optional mask for masked pooling).
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if mask is not None:
            return self.apply_with_mask(image, mask)

        augmented = self.transform(image=image)
        return augmented["image"]


def _preview_compose(transform_list, image):
    """Apply Albumentations list to a numpy RGB image (no normalize)."""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    pipeline = A.Compose(transform_list)
    return pipeline(image=image)['image']


def get_augmentation_preview_columns(config: dict):
    """
    Build (title, transform_list) for augmentation preview grid.
    One column per enabled augmentation (p=1.0) plus full training pipeline.
    """
    if config is None:
        config = {}
    image_size = int(config.get('image_size', 224))
    resize_size = int(config.get('resize_size', image_size))
    random_crop = bool(config.get('random_crop', True))
    columns = []

    base = MicroscopyAugmentation(config)
    columns.append(('Original', base._scale_transforms(image_size, resize_size, False)))

    h_flip_p = float(config.get('random_horizontal_flip', 0.5))
    if h_flip_p > 0:
        columns.append((
            f'H-Flip (p={h_flip_p:.2f})',
            base._scale_transforms(image_size, resize_size, False)
            + [A.HorizontalFlip(p=1.0)],
        ))

    v_flip_p = float(config.get('random_vertical_flip', 0.5))
    if v_flip_p > 0:
        columns.append((
            f'V-Flip (p={v_flip_p:.2f})',
            base._scale_transforms(image_size, resize_size, False)
            + [A.VerticalFlip(p=1.0)],
        ))

    rot_limit = int(config.get('random_rotation', 180))
    if rot_limit > 0:
        import cv2
        columns.append((
            f'Rotate ±{rot_limit}°',
            base._scale_transforms(image_size, resize_size, False)
            + [A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=rot_limit, p=1.0, border_mode=cv2.BORDER_CONSTANT, value=[128, 128, 128])],
        ))

    cj = config.get('color_jitter', {})
    if any(cj.get(k, 0) > 0 for k in ('brightness', 'contrast', 'saturation', 'hue')):
        columns.append((
            'ColorJitter',
            base._scale_transforms(image_size, resize_size, False)
            + [
                A.ColorJitter(
                    brightness=cj.get('brightness', 0.3),
                    contrast=cj.get('contrast', 0.3),
                    saturation=cj.get('saturation', 0.2),
                    hue=cj.get('hue', 0.1),
                    p=1.0,
                )
            ],
        ))

    gs = config.get('grayscale', {})
    if gs.get('enabled', False):
        columns.append((
            f'Gris B&N (p={float(gs.get("p", 0.5)):.2f})',
            base._scale_transforms(image_size, resize_size, False)
            + [A.ToGray(p=1.0)],
        ))

    micro = config.get('microscopy_specific', {})
    micro_map = [
        ('random_gamma', 'Gamma', lambda m: A.RandomGamma(
            gamma_limit=tuple(m['gamma_limit']), p=1.0)),
        ('gaussian_noise', 'GaussNoise', lambda m: A.GaussNoise(
            std_range=tuple(v / 255.0 for v in m['var_limit']), p=1.0)),
        ('blur', 'Blur', lambda m: A.Blur(blur_limit=m['blur_limit'], p=1.0)),
        ('optical_distortion', 'Opt.Distort', lambda m: A.OpticalDistortion(
            distort_limit=m['distort_limit'], p=1.0)),
        ('clahe', 'CLAHE', lambda m: A.CLAHE(
            clip_limit=m['clip_limit'],
            tile_grid_size=tuple(m['tile_grid_size']),
            p=1.0)),
    ]
    for key, label, builder in micro_map:
        block = micro.get(key, {})
        if block.get('enabled', False):
            columns.append((
                label,
                base._scale_transforms(image_size, resize_size, False) + [builder(block)],
            ))

    domain = config.get('domain_augmentation', {})
    if domain.get('enabled', True):
        rgb_shift = domain.get('rgb_shift_limit', 20)
        tone_scale = domain.get('tone_curve_scale', 0.15)
        ds_min = domain.get('downscale_min', 0.6)
        domain_steps = [
            ('RGBShift', [A.RGBShift(
                r_shift_limit=rgb_shift, g_shift_limit=rgb_shift,
                b_shift_limit=rgb_shift, p=1.0)]),
            ('ToneCurve', [A.RandomToneCurve(scale=tone_scale, p=1.0)]),
            ('Downscale', [A.Downscale(scale_range=(ds_min, 0.8), p=1.0)]),
            ('CoarseDropout', [_coarse_dropout_transform({**domain, "coarse_dropout_p": 1.0})]),
        ]
        for label, steps in domain_steps:
            columns.append((
                label,
                base._scale_transforms(image_size, resize_size, False) + steps,
            ))

    columns.append(('Pipeline completo', None))
    return columns


def get_train_transforms(config=None, mode='full', backbone_name=None):
    """
    Get training transforms with microscopy-specific augmentations.
    
    Args:
        config (dict, optional): Configuration dictionary
        mode (str): 'full' for microscopy augmentations, 'deit3' for DeiT III 3-Augment
        backbone_name (str, optional): Backbone name for dynamic input size resolution
        
    Returns:
        MicroscopyAugmentation or DeiT3Augmentation or transforms.Compose
    """
    if backbone_name is not None:
        if config is None:
            config = {}
        
        # Respetar el image_size si viene explícito en el config (para no limitar el rendimiento)
        if 'image_size' not in config or config.get('image_size') is None:
            resolved_size = get_input_size_for_backbone(backbone_name)
            config['image_size'] = resolved_size
            logger.info(f"Train transforms using dynamic size for '{backbone_name}': {resolved_size}x{resolved_size}")
        else:
            resolved_size = config['image_size']
            logger.info(f"Train transforms using user config size: {resolved_size}x{resolved_size}")

        # Configurar resize_size si no viene dado
        if 'resize_size' not in config:
            # Para resize_size, usar un poco más grande para crop (excepto DINOv2 que usa exacto)
            if 'dinov2' in backbone_name:
                config['resize_size'] = resolved_size
            else:
                config['resize_size'] = int(resolved_size * 1.14)  # ~256 para 224
    
    if mode in ('deit3', 'deit3_simple'):
        return DeiT3Augmentation(config)
    
    # Usar directamente MicroscopyAugmentation sin ramas ocultas ni fallbacks
    return MicroscopyAugmentation(config)


class ValidationTransform:
    """Wrapper for validation transforms using Albumentations"""
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'image_size': 224,
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }
        
        # Aplicar grayscale en val/inference determinista
        grayscale_cfg = config.get("grayscale", {})
        is_grayscale = grayscale_cfg.get("enabled", False)
        grayscale_p = 1.0 if is_grayscale else 0.0
        
        transforms_list = [
            A.Resize(config["image_size"], config["image_size"]),
        ]
        
        if grayscale_p > 0:
            transforms_list.append(A.ToGray(p=grayscale_p))
            
        stain_step = _stain_normalize_step(config)
        if stain_step is not None:
            transforms_list.append(stain_step)

        norm_mean, norm_std = _resolve_normalize_stats(config, is_grayscale)

        transforms_list.extend([
            A.Normalize(
                mean=norm_mean,
                std=norm_std,
            ),
            ToTensorV2(),
        ])
        
        self.transform = A.Compose(
            transforms_list,
            additional_targets={"mask": "mask"},
        )

    def apply_with_mask(self, image, mask):
        from src.data.grain_transform import to_mask_float_tensor, to_mask_uint8_numpy

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        out = self.transform(image=image, mask=to_mask_uint8_numpy(mask))
        return out["image"], to_mask_float_tensor(out["mask"])

    def __call__(self, image, mask=None):
        """Apply validation transform to image (optional mask)."""
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        if mask is not None:
            return self.apply_with_mask(image, mask)

        augmented = self.transform(image=image)
        return augmented["image"]


def get_val_transforms(config=None, backbone_name=None):
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        config (dict, optional): Configuration dictionary
        backbone_name (str, optional): Backbone name for dynamic input size resolution
        
    Returns:
        ValidationTransform or transforms.Compose: Transform pipeline
    """
    if config is None:
        config = {
            'image_size': 224,
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
    
    # Si se proporciona backbone_name, ajustar image_size dinámicamente si no está definido
    if backbone_name is not None:
        if 'image_size' not in config or config.get('image_size') is None:
            resolved_size = get_input_size_for_backbone(backbone_name)
            config['image_size'] = resolved_size
            logger.info(f"Val transforms using dynamic size for '{backbone_name}': {resolved_size}x{resolved_size}")
        else:
            resolved_size = config['image_size']
            logger.info(f"Val transforms using user config size: {resolved_size}x{resolved_size}")
    
    return ValidationTransform(config)



class DeiT3Augmentation:
    """
    Simplified 3-Augment recipe from DeiT III (Touvron et al., ECCV 2022).
    
    Uses only 3 augmentations (Grayscale, Solarize, GaussianBlur) with
    RandomResizedCrop and HorizontalFlip. Shown to outperform complex
    augmentation pipelines for fine-grained recognition with ViT backbones.
    
    Args:
        config (dict): Configuration with 'image_size' and 'normalize' keys.
    """
    
    def __init__(self, config=None):
        if config is None:
            config = {
                'image_size': 224,
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }
        
        image_size = config.get('image_size', 224)
        is_grayscale = config.get("grayscale", {}).get("enabled", False)
        norm_mean, norm_std = _resolve_normalize_stats(config, is_grayscale)
        
        self.transform = A.Compose(
            [
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.3, 1.0),
                    ratio=(0.75, 1.333),
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.OneOf(
                    [
                        A.ToGray(p=1.0),
                        A.Solarize(threshold_range=(0.5, 0.5), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    ],
                    p=0.3,
                ),
                A.Normalize(mean=norm_mean, std=norm_std),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )

    def apply_with_mask(self, image, mask):
        from src.data.grain_transform import to_mask_float_tensor, to_mask_uint8_numpy

        if not isinstance(image, np.ndarray):
            image = np.array(image)
        out = self.transform(image=image, mask=to_mask_uint8_numpy(mask))
        return out["image"], to_mask_float_tensor(out["mask"])

    def __call__(self, image, mask=None):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if mask is not None:
            return self.apply_with_mask(image, mask)
        augmented = self.transform(image=image)
        return augmented["image"]


# Backward-compatible re-export (implementación en stain_normalization.py, E-02)
from src.data.stain_normalization import ReinhardStainNormalization as _ReinhardImpl


class ReinhardStainNormalization(_ReinhardImpl):
    """Alias de stain_normalization.ReinhardStainNormalization (Reinhard et al., 2001)."""

    @staticmethod
    def compute_dataset_stats(image_paths, max_images=100):
        """
        Compute mean and std of a dataset in LAB space for reference stats.
        
        Args:
            image_paths: List of image paths
            max_images: Max images to sample
            
        Returns:
            (mean_lab, std_lab) each as list of 3 floats
        """
        import cv2
        from PIL import Image
        
        all_means = []
        all_stds = []
        
        sample = image_paths[:max_images]
        for path in sample:
            img = np.array(Image.open(path).convert('RGB'))
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            all_means.append([lab[:, :, i].mean() for i in range(3)])
            all_stds.append([lab[:, :, i].std() for i in range(3)])
        
        mean_lab = np.mean(all_means, axis=0).tolist()
        std_lab = np.mean(all_stds, axis=0).tolist()
        
        return mean_lab, std_lab
