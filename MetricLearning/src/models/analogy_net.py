import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.models.pooling import PoolingModule
from src.models.pooling_config import (
    compute_pooling_output_dim,
    resolve_pooling_config,
    resolve_projection_config,
    resolve_slice_layer_map,
)

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    """Channel Attention Module (parte de CBAM). Ref: [R5] Zolfaghari & Sajedi 2024."""
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.sigmoid(self.fc(x))
        return x * out


class SE1D(nn.Module):
    """
    Channel-Based Attention Module for 1D feature vectors.
    Simplified Squeeze-and-Excitation (SE) for embeddings (no spatial dimension).
    Ref: [R5] Zolfaghari & Sajedi 2024.
    """
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
    
    def forward(self, x):
        return self.channel_attention(x)


class DINOv2Backbone(nn.Module):
    """
    Wrapper for DINOv2 that uses multi-spectral pooling of patch tokens
    (mean, max, and std) instead of the default CLS token. This ensures 
    gradients flow through spatial tokens for meaningful Grad-CAM heatmaps
    and preserves spectral richness.
    
    Supports masked multi-spectral pooling: when a spatial mask is provided,
    only patch tokens inside the grain (mask > 0) contribute to the
    embedding. Background patches are excluded.
    """
    
    def __init__(self, dino_model, pooling_config: dict):
        super().__init__()
        self.model = dino_model
        self.blocks = dino_model.blocks
        self.patch_embed = dino_model.patch_embed
        self.norm = dino_model.norm
        self.patch_size = dino_model.patch_embed.patch_size[0]
        self.pooling_config = pooling_config
        self.pooling_layers = list(pooling_config.get("layers") or [11])
        self.norm_intermediate = bool(pooling_config.get("norm_intermediate", True))
        self._logged_first_forward = False

        token_dim = getattr(dino_model, "embed_dim", 384)
        self.pooling = PoolingModule(pooling_config, token_dim, self.patch_size)
        self.num_features = self.pooling.output_dim
    
    def forward(self, x, mask=None):
        outputs = self.model.get_intermediate_layers(
            x, n=self.pooling_layers, reshape=False, norm=False
        )
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]

        layer_tokens = {}
        for layer_idx, tokens in zip(self.pooling_layers, outputs):
            if self.norm_intermediate and hasattr(self.model, "norm"):
                tokens = self.model.norm(tokens)
            layer_tokens[layer_idx] = tokens

        if not self._logged_first_forward:
            self._logged_first_forward = True
            first = layer_tokens[self.pooling_layers[0]]
            _, N, D = first.shape
            H_p = W_p = int(N ** 0.5)
            logger.info(
                f"[DINOv2] First forward: input={tuple(x.shape)}, "
                f"patches={N} ({H_p}x{W_p}), dim={D}, "
                f"layers={self.pooling_layers}, strategy={self.pooling_config.get('strategy')}, "
                f"masked={mask is not None}"
            )

        return self.pooling(
            layer_tokens,
            spatial_mask=mask,
            image_hw=(x.shape[2], x.shape[3]),
        )

    def forward_for_attention(self, x):
        """
        Standard forward pass for attention map extraction.
        """
        raw = self.model
        if hasattr(raw, 'forward_features'):
            out = raw.forward_features(x)
            if isinstance(out, dict):
                cls_token = out.get('x_norm_clstoken', out.get('cls_token', None))
                patch_tokens = out.get('x_norm_patchtokens', out.get('x', None))
                if cls_token is not None and patch_tokens is not None:
                    if cls_token.dim() == 2:
                        cls_token = cls_token.unsqueeze(1)
                    return torch.cat([cls_token, patch_tokens], dim=1)
                else:
                    raise TypeError(f"Unexpected dict keys from forward_features: {list(out.keys())}")
            return out
        return raw(x)
    
    @property
    def num_register_tokens(self):
        """Number of register tokens in the DINOv2 model (0 for standard, 4 for _reg variants)."""
        return getattr(self.model, 'num_register_tokens', 0)


class AnalogyNet(nn.Module):
    """
    Encoder para Metric Learning con soporte multi-backbone.
    
    Transforma imágenes en vectores de características normalizados (embeddings)
    que residen en una hiperesfera unitaria.
    
    Args:
        backbone (str): Tipo de backbone. Opciones:
            'resnet18', 'resnet50', 'convnext_v2_tiny', 'dinov2_vits14', 'dinov2_vitb14'
        embedding_dim (int): Dimensión del vector de salida
        pretrained (bool): Usar pesos pre-entrenados
        projection_head_config (dict, optional): Configuración del projection head.
            Keys: 'hidden_dims' (list[int]), 'use_batch_norm' (bool), 'dropout' (float).
            Si es None, usa default [256] sin BatchNorm (backward compatible).
        use_attention (bool): Si True, inserta SE entre backbone y projection head.
        freeze_backbone (bool): Si True, congela el backbone (útil para DINOv2).
    
    Example:
        >>> model = AnalogyNet(backbone='resnet18', embedding_dim=128)
        >>> x = torch.randn(4, 3, 224, 224)
        >>> embeddings = model(x)
        >>> print(embeddings.shape)  # torch.Size([4, 128])
        >>> print(torch.norm(embeddings[0]))  # tensor(1.0000)
    """
    
    BACKBONE_FEATURES = {
        'resnet18': 512,
        'resnet50': 2048,
        'convnext_v2_tiny': 768,
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vits14_reg': 384,
        'dinov2_vitb14_reg': 768,
        'deit_small_patch16_224': 384,
        'efficientnet_b4': 1792,
        'swin_tiny_patch4_window7_224': 768,
    }
    
    def __init__(self, backbone='resnet18', embedding_dim=128, pretrained=True,
                 projection_head_config=None, use_attention=False, freeze_backbone=False,
                 embedding_space='euclidean', hyperbolic_curvature=1.0,
                 pooling_config=None, projection_layout='lf_hf_dual', grain_config=None,
                 num_slices=4, projection_config=None,
                 num_unfrozen_blocks=2, unfreeze_block_indices=None):
        super(AnalogyNet, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        self.freeze_backbone = freeze_backbone
        self.embedding_space = embedding_space
        self.pooling_config = resolve_pooling_config(
            {"pooling": pooling_config or {}}, grain_config
        )
        self.projection_layout = projection_layout
        self.num_slices = int(num_slices)
        self.slice_dim = embedding_dim // self.num_slices if self.num_slices else embedding_dim
        proj_cfg = projection_config or {}
        self.slice_layer_map = resolve_slice_layer_map(
            proj_cfg, self.pooling_config, self.num_slices
        ) if projection_layout == "multi_tower" else None
        
        self.backbone, num_features = self._build_backbone(backbone, pretrained)
        per_layer_dim = num_features if self.pooling_config.get("layer_fusion") != "concat" else num_features
        
        if freeze_backbone:
            unfreeze_keys = self._resolve_unfreeze_keys(
                num_unfrozen_blocks=int(num_unfrozen_blocks),
                unfreeze_block_indices=unfreeze_block_indices,
            )
            for name, param in self.backbone.named_parameters():
                if not any(key in name for key in unfreeze_keys):
                    if name not in ['norm.weight', 'norm.bias']:
                        param.requires_grad = False
            logger.info(
                f"Backbone '{self.backbone_name}' partially frozen "
                f"(unfrozen: {unfreeze_keys or 'norm only'}, projection head trainable)"
            )
        
        self.attention = SE1D(per_layer_dim) if use_attention else None
        self.projection = None
        self.projection_lf = None
        self.projection_hf = None
        self.projection_towers = None

        if projection_layout == 'slice_native':
            self.projection = self._build_projection_head(
                num_features, embedding_dim, projection_head_config, embedding_space
            )
        elif projection_layout == 'multi_tower':
            self.projection_towers = nn.ModuleList([
                self._build_projection_head(
                    per_layer_dim, self.slice_dim, projection_head_config, embedding_space
                )
                for _ in range(self.num_slices)
            ])
            logger.info(
                f"multi_tower: {self.num_slices}×{self.slice_dim}D, "
                f"slice_layer_map={self.slice_layer_map}"
            )
        else:
            emb_dim_half = embedding_dim // 2
            self.projection_lf = self._build_projection_head(
                per_layer_dim, emb_dim_half, projection_head_config, embedding_space
            )
            self.projection_hf = self._build_projection_head(
                per_layer_dim, emb_dim_half, projection_head_config, embedding_space
            )
        
        self._init_projection_weights()
        
        # Hyperbolic projection (Ermolov et al., CVPR 2022)
        self.hyp_projection = None
        if embedding_space == 'hyperbolic':
            from src.models.hyperbolic import PoincareBallProjection
            self.hyp_projection = PoincareBallProjection(curvature=hyperbolic_curvature)
            logger.info(f"Hyperbolic embedding space enabled (curvature={hyperbolic_curvature})")
    
    def set_backbone_frozen(self, freeze: bool, num_unfrozen_blocks: int = 2):
        """Runtime freeze/unfreeze for multi-phase fine-tuning (D-05)."""
        self.freeze_backbone = bool(freeze)
        if not freeze:
            for param in self.backbone.parameters():
                param.requires_grad = True
            self.backbone.train(self.training)
            logger.info("Backbone fully unfrozen for fine-tune phase")
            return

        unfreeze_keys = self._resolve_unfreeze_keys(num_unfrozen_blocks=num_unfrozen_blocks)
        for name, param in self.backbone.named_parameters():
            if not any(key in name for key in unfreeze_keys):
                if name not in ['norm.weight', 'norm.bias']:
                    param.requires_grad = False
            else:
                param.requires_grad = True
        self.backbone.eval()
        logger.info(f"Backbone partially frozen (unfrozen keys: {unfreeze_keys})")

    def _resolve_unfreeze_keys(
        self,
        num_unfrozen_blocks: int = 2,
        unfreeze_block_indices=None,
    ) -> list:
        """Return name substrings for ViT blocks to keep trainable when freeze_backbone=True."""
        if unfreeze_block_indices is not None:
            return [f"blocks.{int(i)}" for i in unfreeze_block_indices]

        if hasattr(self.backbone, "blocks"):
            n_blocks = len(self.backbone.blocks)
            n_unfreeze = max(0, min(int(num_unfrozen_blocks), n_blocks))
            start = max(0, n_blocks - n_unfreeze)
            return [f"blocks.{i}" for i in range(start, n_blocks)]

        return []

    def _build_backbone(self, backbone, pretrained, pooling_config=None):
        """
        Construye el backbone según el tipo especificado.
        
        Returns:
            (nn.Module, int): backbone module y número de features de salida
        """
        if backbone not in self.BACKBONE_FEATURES:
            supported = ', '.join(self.BACKBONE_FEATURES.keys())
            raise ValueError(f"Backbone '{backbone}' no soportado. Opciones: {supported}")
        
        num_features = self.BACKBONE_FEATURES[backbone]
        
        if backbone == 'resnet18':
            model = models.resnet18(weights='DEFAULT' if pretrained else None)
            model.fc = nn.Identity()
        elif backbone == 'resnet50':
            model = models.resnet50(weights='DEFAULT' if pretrained else None)
            model.fc = nn.Identity()
        elif backbone == 'convnext_v2_tiny':
            try:
                import timm
                model = timm.create_model(
                    'convnextv2_tiny.fcmae_ft_in22k_in1k',
                    pretrained=pretrained
                )
                model.reset_classifier(0)
            except ImportError:
                logger.warning("timm not available, falling back to torchvision ConvNeXt V1")
                model = models.convnext_tiny(weights='DEFAULT' if pretrained else None)
                model.classifier = nn.Sequential(
                    model.classifier[0],
                    nn.Identity()
                )
            except Exception:
                raise ValueError("ConvNeXt V2 Tiny requires timm >= 0.9.0: pip install timm>=0.9.0")
        elif backbone == 'deit_small_patch16_224':
            # DeiT-S: Vision Transformer supervisado (Touvron et al. ICML 2021)
            try:
                import timm
                model = timm.create_model(
                    'deit_small_patch16_224.fb_in1k',
                    pretrained=pretrained,
                    num_classes=0  # Remove classification head
                )
            except ImportError:
                raise ValueError("DeiT requires timm >= 0.9.0: pip install timm>=0.9.0")
        elif backbone == 'efficientnet_b4':
            # EfficientNet-B4: CNN eficiente (Tan & Le, ICML 2019)
            try:
                import timm
                model = timm.create_model(
                    'efficientnet_b4.ra2_in1k',
                    pretrained=pretrained,
                    num_classes=0
                )
            except ImportError:
                raise ValueError("EfficientNet requires timm >= 0.9.0: pip install timm>=0.9.0")
        elif backbone == 'swin_tiny_patch4_window7_224':
            # Swin Transformer Tiny: ViT jerárquico (Liu et al. ICCV 2021)
            try:
                import timm
                model = timm.create_model(
                    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
                    pretrained=pretrained,
                    num_classes=0
                )
            except ImportError:
                raise ValueError("Swin Transformer requires timm >= 0.9.0: pip install timm>=0.9.0")
        elif backbone.startswith('dinov2_'):
            # Map our backbone names to torch.hub model names
            hub_name_map = {
                'dinov2_vits14': 'dinov2_vits14',
                'dinov2_vitb14': 'dinov2_vitb14',
                'dinov2_vits14_reg': 'dinov2_vits14_reg4',
                'dinov2_vitb14_reg': 'dinov2_vitb14_reg4',
            }
            hub_name = hub_name_map.get(backbone, backbone)
            try:
                raw_model = torch.hub.load('facebookresearch/dinov2', hub_name, pretrained=pretrained)
                pcfg = pooling_config or self.pooling_config
                model = DINOv2Backbone(raw_model, pcfg)
            except Exception:
                raise ValueError(
                    f"DINOv2 '{backbone}' (hub: '{hub_name}') requires internet and torch.hub. "
                    "Install with: pip install timm"
                )
        else:
            raise ValueError(f"Backbone '{backbone}' no soportado.")
        
        # Obtener num_features dinámicamente si está disponible
        num_features = self._get_num_features(backbone, model)
        logger.info(f"Backbone '{backbone}' loaded: {num_features} features")
        return model, num_features
    
    def _get_num_features(self, backbone_name, backbone_module):
        """
        Obtiene el número de features del backbone dinámicamente.
        
        Args:
            backbone_name: Nombre del backbone
            backbone_module: Módulo del backbone ya instanciado
            
        Returns:
            int: Número de features de salida
        """
        # Para modelos timm, usar atributo num_features si existe
        if hasattr(backbone_module, 'num_features'):
            return backbone_module.num_features
        
        # Para DINOv2 y otros sin num_features, usar dict estático
        if backbone_name in self.BACKBONE_FEATURES:
            return self.BACKBONE_FEATURES[backbone_name]
        
        # Fallback: intentar inferir del último módulo
        logger.warning(
            f"Could not detect num_features for '{backbone_name}'. "
            f"Using static value from BACKBONE_FEATURES."
        )
        return self.BACKBONE_FEATURES.get(backbone_name, 512)
    
    def _build_projection_head(self, num_features, embedding_dim, config, embedding_space='euclidean'):
        """
        Construye el cabezal de proyección lineal.
        
        Args:
            num_features: Dimensión de salida del backbone
            embedding_dim: Dimensión final del embedding
            config: Dict con 'hidden_dims', 'use_batch_norm', 'dropout'.
                    Si None, usa default [256] sin BN (backward compatible).
        
        Returns:
            nn.Sequential: Projection head
        """
        use_bias = (embedding_space != 'hyperbolic')
        
        if config is None:
            return nn.Sequential(
                nn.Linear(num_features, 256, bias=use_bias),
                nn.ReLU(inplace=True),
                nn.Linear(256, embedding_dim, bias=use_bias)
            )
        
        hidden_dims = config.get('hidden_dims', [256])
        use_bn = bool(config.get('use_batch_norm', False))
        use_ln_hidden = bool(
            config.get('use_layer_norm', config.get('normalize_embeddings_ln', True))
        )
        dropout = config.get('dropout', 0.0)

        layers = []
        in_dim = num_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim, bias=use_bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            elif use_ln_hidden:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, embedding_dim, bias=use_bias))
        
        if config is not None:
            use_ln_out = config.get(
                'use_layer_norm',
                config.get('normalize_embeddings_ln', True),
            )
            if use_ln_out:
                layers.append(nn.LayerNorm(embedding_dim))
            if config.get('use_hardtanh', False):
                min_val = config.get('hardtanh_min', -2.0)
                max_val = config.get('hardtanh_max', 2.0)
                layers.append(nn.Hardtanh(min_val=min_val, max_val=max_val))
        
        return nn.Sequential(*layers)
    
    def _init_projection_weights(self):
        """Inicializa los pesos del Projection Head con Kaiming Normal."""
        emb_dim_half = self.embedding_dim // 2
        projections = []
        if self.projection is not None:
            projections.append((self.projection, self.embedding_dim))
        if self.projection_lf is not None:
            projections.append((self.projection_lf, emb_dim_half))
        if self.projection_hf is not None:
            projections.append((self.projection_hf, emb_dim_half))
        if self.projection_towers is not None:
            for tower in self.projection_towers:
                projections.append((tower, self.slice_dim))

        for proj, out_dim in projections:
            for m in proj.modules():
                if isinstance(m, nn.Linear):
                    if m.out_features == out_dim:
                        nn.init.orthogonal_(m.weight)
                    else:
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        # Inicia pesos de atencion si existen
        if self.attention is not None:
            for m in self.attention.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def train(self, mode=True):
        """Override train to maintain frozen backbone in eval mode."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
    
    def forward(self, x, mask=None, return_prenorm=False):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Batch de imágenes (B, 3, 224, 224)
            mask (torch.Tensor, optional): Spatial mask (B, H, W) for masked pooling.
                Only used with DINOv2 backbone. None = standard pooling.
            return_prenorm (bool): Si True, devuelve (embeddings, prenorm_embeddings)
        
        Returns:
            torch.Tensor o tuple: Embeddings normalizados (B, embedding_dim) o tupla si return_prenorm=True
        """
        if mask is not None:
            # Zero activation (autor): fondo rellenado con imagenet_neutral en el dataset
            # (ver grain_detection.train_mask_bg_mode) + masked pooling en salida.
            # No se multiplica x*mask aquí: el ViT ve contexto espacial coherente y el
            # pooling excluye parches de fondo (evita focos de atención espurios).
            if hasattr(self.backbone, 'patch_size'):
                features = self.backbone(x, mask=mask)
            else:
                logger.warning("Mask provided but backbone does not support spatial masking. Ignoring mask.")
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        
        embeddings, prenorm_embeddings = self._project_features(features)

        if return_prenorm:
            return embeddings, prenorm_embeddings
        return embeddings

    def _layer_feature_dict(self, features):
        """Map ViT layer index → pooled feature vector (B, D)."""
        layers = list(self.pooling_config.get("layers") or [11])
        if isinstance(features, tuple):
            sorted_layers = sorted(layers)
            return {layer: features[i] for i, layer in enumerate(sorted_layers)}
        if features.dim() > 2:
            features = features.mean(dim=list(range(2, features.dim())))
        return {layer: features for layer in layers}

    def _resolve_backbone_features(self, features):
        """Normalize backbone outputs to projection inputs."""
        if self.projection_layout == 'multi_tower':
            return self._layer_feature_dict(features)

        if self.projection_layout == 'slice_native':
            if isinstance(features, tuple):
                if self.pooling_config.get("layer_fusion") == "per_layer":
                    return torch.cat(features, dim=1)
                return features[0] if len(features) == 1 else torch.cat(features, dim=1)
            if features.dim() > 2:
                return features.mean(dim=list(range(2, features.dim())))
            return features

        if isinstance(features, tuple) and len(features) == 2:
            # Pooling returns layers in ascending index; LF=head uses deeper layer first
            layers = self.pooling_config.get("layers") or []
            if len(layers) == 2 and layers[0] < layers[1]:
                return features[1], features[0]
            return features[0], features[1]
        if features.dim() > 2:
            features = features.mean(dim=list(range(2, features.dim())))
        return features, features

    def _project_features(self, features):
        if self.projection_layout == 'multi_tower':
            layer_feats = self._resolve_backbone_features(features)
            slice_vectors = []
            for i, tower in enumerate(self.projection_towers):
                layer_idx = self.slice_layer_map[i]
                feat = layer_feats[layer_idx]
                if self.attention is not None:
                    feat = self.attention(feat)
                prenorm_sl = tower(feat)
                slice_vectors.append(F.normalize(prenorm_sl, p=2, dim=1))
            prenorm_embeddings = torch.cat(slice_vectors, dim=1)
            scale = math.sqrt(self.num_slices)
            if self.hyp_projection is not None and self.embedding_space == 'hyperbolic':
                embeddings = self.hyp_projection(prenorm_embeddings)
            else:
                embeddings = prenorm_embeddings / scale
            return embeddings, prenorm_embeddings

        if self.projection_layout == 'slice_native':
            feat = self._resolve_backbone_features(features)
            if self.attention is not None:
                feat = self.attention(feat)
            prenorm_embeddings = self.projection(feat)
            if self.hyp_projection is not None and self.embedding_space == 'hyperbolic':
                embeddings = self.hyp_projection(prenorm_embeddings)
            else:
                embeddings = F.normalize(prenorm_embeddings, p=2, dim=1)
            return embeddings, prenorm_embeddings

        feat_lf, feat_hf = self._resolve_backbone_features(features)
        if self.attention is not None:
            feat_lf = self.attention(feat_lf)
            feat_hf = self.attention(feat_hf)

        prenorm_lf = self.projection_lf(feat_lf)
        prenorm_hf = self.projection_hf(feat_hf)
        norm_lf = F.normalize(prenorm_lf, p=2, dim=1)
        norm_hf = F.normalize(prenorm_hf, p=2, dim=1)
        prenorm_embeddings = torch.cat([norm_lf, norm_hf], dim=1)

        if self.hyp_projection is not None:
            if self.embedding_space == 'hyperbolic':
                embeddings = self.hyp_projection(prenorm_embeddings)
            else:
                embeddings = prenorm_embeddings / math.sqrt(2.0)
        else:
            embeddings = prenorm_embeddings / math.sqrt(2.0)
        return embeddings, prenorm_embeddings
    
    def get_embedding_dim(self):
        """Retorna la dimensión del embedding."""
        return self.embedding_dim
