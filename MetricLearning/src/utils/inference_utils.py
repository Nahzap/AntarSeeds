"""
Utilidades para inferencia y extracción de embeddings.
Centraliza operaciones comunes de inferencia.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    return_paths: bool = False,
    show_progress: bool = True,
    grain_config: Optional[dict] = None,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, List[str]]]:
    """
    Extrae embeddings usando el forward canónico (con máscara si el loader la provee).
    """
    from src.models.embedding_extractor import extract_embeddings as _canonical_extract

    result = _canonical_extract(
        model,
        dataloader,
        device,
        grain_config=grain_config,
        return_paths=return_paths,
        show_progress=show_progress,
    )
    if return_paths:
        embeddings, labels, paths = result
        logger.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        return embeddings, labels, paths

    embeddings, labels = result
    logger.info(f"Extracted {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embeddings, labels


def build_embedding_database(
    model: nn.Module,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Construye base de datos de embeddings desde un dataset.
    
    Args:
        model: Modelo para extraer embeddings
        dataset: Dataset con imágenes
        device: Dispositivo (cuda/cpu)
        batch_size: Tamaño de batch
        num_workers: Workers para DataLoader
        
    Returns:
        (embeddings, labels, paths, class_names)
    """
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    embeddings, labels, paths = extract_embeddings(
        model, dataloader, device, return_paths=True, show_progress=True
    )
    
    class_names = dataset.classes if hasattr(dataset, 'classes') else None
    
    return embeddings, labels, paths, class_names


def get_single_embedding(
    model: nn.Module,
    image_path: str,
    transform,
    device: torch.device
) -> np.ndarray:
    """
    Obtiene embedding de una sola imagen.
    
    Args:
        model: Modelo para extraer embedding
        image_path: Ruta a la imagen
        transform: Transformación a aplicar
        device: Dispositivo (cuda/cpu)
        
    Returns:
        Embedding (1, embedding_dim)
    """
    from PIL import Image
    
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding.cpu().numpy()


class EmbeddingSimilarityTarget:
    """
    Target para Grad-CAM en Metric Learning.
    Maximiza la similitud coseno entre el embedding de salida y un centroide
    de clase dado, generando gradientes significativos para el heatmap.
    """
    def __init__(self, centroid_vector: torch.Tensor):
        self.centroid = centroid_vector
    
    def __call__(self, model_output):
        if self.centroid.device != model_output.device:
            self.centroid = self.centroid.to(model_output.device)
        similarity = torch.nn.functional.cosine_similarity(
            model_output, self.centroid.unsqueeze(0), dim=1
        )
        return similarity


def _is_vit_backbone(model):
    """Check if the model uses a ViT-based backbone (DINOv2, DeiT, ViT, etc.)."""
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        # Direct ViT from timm (DeiT, ViT, etc.)
        if hasattr(backbone, 'blocks') and hasattr(backbone, 'patch_embed'):
            return True
        # DINOv2Backbone wrapper (has .model attribute pointing to raw ViT)
        if hasattr(backbone, 'model') and hasattr(backbone.model, 'blocks'):
            return True
    return False


def _get_dino_model(model):
    """
    Get the inner ViT model from AnalogyNet.
    Supports:
    - DINOv2: AnalogyNet > DINOv2Backbone > model
    - Generic ViT (DeiT, ViT): AnalogyNet > backbone (direct timm model)
    """
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        # DINOv2 wrapped structure
        if hasattr(backbone, 'model'):
            return backbone.model
        # Generic ViT from timm (DeiT, ViT, etc.)
        elif hasattr(backbone, 'blocks') and hasattr(backbone, 'patch_embed'):
            return backbone
    return None


def _run_vit_forward_for_attention(model, dino_model, image_tensor, enable_grad=False):
    """
    Run a standard ViT forward pass that preserves ALL tokens (CLS + patches).
    
    CRITICAL: This bypasses DINOv2Backbone.forward() which uses
    get_intermediate_layers() and discards CLS token. Instead, it calls
    forward_features() directly on the raw ViT model, ensuring:
    - CLS token is preserved at position 0
    - Standard attention patterns between CLS and patches
    - Clean gradient flow (when enable_grad=True)
    
    Args:
        model: AnalogyNet (used only if dino_model doesn't have forward_features)
        dino_model: Raw ViT model (from _get_dino_model)
        image_tensor: (B, 3, H, W) tensor on correct device
        enable_grad: If True, enables gradient computation
        
    Returns:
        Output tensor from forward_features (typically (B, N_tokens, D))
    """
    ctx = torch.enable_grad() if enable_grad else torch.no_grad()
    with ctx:
        # Prefer forward_for_attention on the backbone (DINOv2Backbone)
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'forward_for_attention'):
            return model.backbone.forward_for_attention(image_tensor)
        # timm models (DeiT, ViT, etc.) have forward_features
        elif hasattr(dino_model, 'forward_features'):
            return dino_model.forward_features(image_tensor)
        # Fallback: standard forward (not ideal but works)
        else:
            return dino_model(image_tensor)


def extract_dino_attention(model, image_tensor, device):
    """
    Extract CLS-to-patch attention maps from ViT's last transformer block.
    Supports DINOv2, DeiT, and other ViT architectures.
    
    IMPORTANT ARCHITECTURAL NOTE:
    - DINOv2Backbone wrapper uses get_intermediate_layers() for training
      (excludes CLS token, uses mean pooling for optimal metric learning embeddings)
    - For attention extraction, we bypass the wrapper and execute forward_features()
      directly on the raw ViT model to preserve CLS token
    - This ensures meaningful CLS-to-patch attention maps
    
    Ref: Caron et al. (DINO, ICCV 2021) — self-attention heads naturally
    segment foreground objects. This is the correct approach for ViTs
    (GradCAM was designed for CNNs and often fails on ViT architectures).

    Args:
        model: AnalogyNet with ViT-based backbone (DINOv2, DeiT, ViT, etc.)
        image_tensor: (1, 3, H, W) normalized tensor
        device: torch.device

    Returns:
        attn_map: numpy array (H_img, W_img) in [0, 1], attention heatmap
        head_maps: numpy array (num_heads, h_patches, w_patches) per-head attention
    """
    import cv2

    model.eval()
    image_tensor = image_tensor.to(device)
    dino_model = _get_dino_model(model)
    if dino_model is None:
        raise ValueError("Model does not have ViT backbone (expected DINOv2, DeiT, or similar)")

    # Detect if model has distillation token (DeiT-specific)
    has_dist_token = hasattr(dino_model, 'dist_token') and dino_model.dist_token is not None
    
    # Detect DINOv2 register tokens (dinov2_*_reg variants have 4 register tokens)
    num_register_tokens = 0
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'num_register_tokens'):
        num_register_tokens = model.backbone.num_register_tokens
    elif hasattr(dino_model, 'num_register_tokens'):
        num_register_tokens = dino_model.num_register_tokens
    
    last_block = dino_model.blocks[-1]
    attn_module = last_block.attn
    num_heads = attn_module.num_heads

    # Hook to capture QKV from last block's attention
    captured = {}
    def qkv_hook(module, input_args, output):
        captured['qkv'] = output

    hook = attn_module.qkv.register_forward_hook(qkv_hook)

    try:
        # CRITICAL: bypass DINOv2Backbone wrapper — use forward_features() directly
        _ = _run_vit_forward_for_attention(model, dino_model, image_tensor, enable_grad=False)
    finally:
        hook.remove()

    qkv = captured['qkv']  # (B, N_tokens, 3 * dim)
    B, N, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)
    
    # Determine patch start index based on prefix tokens:
    # DeiT: [CLS, DIST, patches...] → start=2
    # DINOv2: [CLS, patches...] → start=1
    # DINOv2_reg: [CLS, reg0, ..., regN, patches...] → start=1+num_register_tokens
    if has_dist_token:
        patch_start_idx = 2
    else:
        patch_start_idx = 1 + num_register_tokens
    n_expected_patches = N - patch_start_idx
    
    logger.info(f"[Attention] N_tokens={N}, has_dist_token={has_dist_token}, "
                f"num_register_tokens={num_register_tokens}, "
                f"patch_start_idx={patch_start_idx}, n_patches={n_expected_patches}, "
                f"backbone={type(dino_model).__name__}")

    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)  # each: (B, heads, N, head_dim)

    # Attention weights: softmax(Q @ K^T / sqrt(d_k))
    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)  # (B, heads, N, N)

    # CLS token (index 0) attention to patch tokens
    # DeiT has [CLS, DIST, patches...], others have [CLS, patches...]
    cls_attn = attn[0, :, 0, patch_start_idx:]  # (heads, N_patches)
    n_patches = cls_attn.shape[-1]
    
    h_p = w_p = int(n_patches ** 0.5)
    if h_p * w_p != n_patches:
        logger.warning(f"Non-square patch grid: {n_patches} patches, using {h_p}x{w_p}")

    # Per-head attention maps
    head_maps = cls_attn.reshape(num_heads, h_p, w_p).cpu().numpy()

    # Average over heads for the combined map
    avg_attn = cls_attn.mean(dim=0)  # (N_patches,)
    attn_spatial = avg_attn.reshape(h_p, w_p).cpu().numpy()

    # Normalize to [0, 1]
    attn_spatial = (attn_spatial - attn_spatial.min()) / (attn_spatial.max() - attn_spatial.min() + 1e-8)

    # Resize to image resolution
    H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
    attn_map = cv2.resize(attn_spatial, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
    attn_map = np.clip(attn_map, 0, 1)

    return attn_map, head_maps


def extract_gradient_weighted_attention(model, image_tensor, device, centroid):
    """
    Gradient-weighted attention for metric learning: combines ViT
    self-attention with gradients of cosine similarity to class centroid.
    This produces task-specific attention maps that show which patches
    contribute most to the predicted embedding's similarity with the centroid.
    
    IMPORTANT: Bypasses DINOv2Backbone wrapper by using forward_features()
    directly on the raw ViT, then manually pipes through the projection head
    to compute cosine similarity for gradient weighting.

    Ref: Chefer et al. (Transformer Interpretability, CVPR 2021)

    Args:
        model: AnalogyNet with ViT-based backbone (DINOv2, DeiT, ViT, etc.)
        image_tensor: (1, 3, H, W) normalized tensor
        device: torch.device
        centroid: (embed_dim,) class centroid tensor

    Returns:
        attn_map: numpy array (H_img, W_img) in [0, 1]
    """
    import cv2

    model.eval()
    image_tensor = image_tensor.to(device).requires_grad_(False)
    centroid = centroid.to(device)
    dino_model = _get_dino_model(model)
    if dino_model is None:
        raise ValueError("Model does not have ViT backbone (expected DINOv2, DeiT, or similar)")

    # Detect if model has distillation token (DeiT-specific)
    has_dist_token = hasattr(dino_model, 'dist_token') and dino_model.dist_token is not None
    
    # Detect DINOv2 register tokens
    num_register_tokens = 0
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'num_register_tokens'):
        num_register_tokens = model.backbone.num_register_tokens
    elif hasattr(dino_model, 'num_register_tokens'):
        num_register_tokens = dino_model.num_register_tokens
    
    last_block = dino_model.blocks[-1]
    attn_module = last_block.attn
    num_heads = attn_module.num_heads

    captured = {}
    def qkv_hook(module, input_args, output):
        captured['qkv'] = output
        # Guard retain_grad: only works if tensor has requires_grad=True
        if output.requires_grad:
            output.retain_grad()
            captured['qkv_ref'] = output
        else:
            captured['qkv_ref'] = None

    hook = attn_module.qkv.register_forward_hook(qkv_hook)

    try:
        # CRITICAL: bypass wrapper — forward_features() with gradients enabled
        features_out = _run_vit_forward_for_attention(
            model, dino_model, image_tensor, enable_grad=True
        )
        
        # Determine patch start index for feature extraction
        if has_dist_token:
            patch_start = 2
        else:
            patch_start = 1 + num_register_tokens
        
        # Extract features for embedding computation:
        # forward returns (B, N_tokens, D) — use mean pooling of patch tokens
        # to match the DINOv2Backbone training behavior
        if features_out.dim() == 3:
            patch_features = features_out[:, patch_start:]  # (B, N_patches, D)
            pooled_features = patch_features.mean(dim=1)  # (B, D)
        else:
            pooled_features = features_out
        
        # Pipe through projection head + normalize (mirrors AnalogyNet.forward)
        if hasattr(model, 'attention') and model.attention is not None:
            pooled_features = model.attention(pooled_features)
        
        if hasattr(model, "projection") and model.projection is not None:
            embedding = model.projection(pooled_features)
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        elif hasattr(model, "projection_lf") and model.projection_lf is not None:
            embedding = model.projection_lf(pooled_features)
            embedding = nn.functional.normalize(embedding, p=2, dim=1)
        else:
            embedding = pooled_features
        
        similarity = torch.nn.functional.cosine_similarity(
            embedding, centroid.unsqueeze(0), dim=1
        )
        similarity.backward()
    finally:
        hook.remove()

    qkv = captured['qkv'].detach()
    qkv_grad = captured['qkv_ref'].grad if captured.get('qkv_ref') is not None else None

    B, N, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)
    
    # Patch start index (same as above)
    if has_dist_token:
        patch_start_idx = 2
    else:
        patch_start_idx = 1 + num_register_tokens
    
    logger.info(f"[GradAttention] N_tokens={N}, has_dist_token={has_dist_token}, "
                f"num_register_tokens={num_register_tokens}, "
                f"grad_available={qkv_grad is not None}, "
                f"backbone={type(dino_model).__name__}")

    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)

    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(dim=-1)  # (B, heads, N, N)

    # CLS attention to patches
    cls_attn = attn[0, :, 0, patch_start_idx:]  # (heads, N_patches)

    # Weight by gradient magnitude (which heads matter for this centroid)
    if qkv_grad is not None:
        grad_importance = qkv_grad.abs().mean(dim=(0, 1))  # (3*dim,)
        # Reshape to per-head importance
        head_importance = grad_importance.reshape(3, num_heads, head_dim).mean(dim=(0, 2))  # (heads,)
        head_importance = head_importance / (head_importance.sum() + 1e-8)
        weighted_attn = (cls_attn * head_importance.unsqueeze(-1)).sum(dim=0)  # (N_patches,)
    else:
        logger.warning("[GradAttention] No gradients captured — falling back to mean attention")
        weighted_attn = cls_attn.mean(dim=0)

    n_patches = weighted_attn.shape[-1]
    h_p = w_p = int(n_patches ** 0.5)
    attn_spatial = weighted_attn.reshape(h_p, w_p).cpu().numpy()

    # Normalize to [0, 1]
    attn_spatial = (attn_spatial - attn_spatial.min()) / (attn_spatial.max() - attn_spatial.min() + 1e-8)

    H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
    attn_map = cv2.resize(attn_spatial, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
    attn_map = np.clip(attn_map, 0, 1)

    return attn_map


def _vit_reshape_transform(tensor):
    """
    Reshape transform for ViT-based Grad-CAM.
    ViT blocks output (B, num_tokens, embed_dim) where num_tokens = 1 (CLS) + H*W patches.
    This removes the CLS token and reshapes to (B, embed_dim, H, W) spatial format.
    """
    if tensor.dim() == 3:
        result = tensor[:, 1:, :]  # skip CLS token
        num_patches = result.shape[1]
        h = w = int(num_patches ** 0.5)
        result = result.reshape(result.shape[0], h, w, result.shape[2])
        result = result.permute(0, 3, 1, 2)
        return result
    return tensor


def _detect_target_layer(model):
    """
    Auto-detecta la última capa convolucional del backbone para Grad-CAM.
    Prueba múltiples candidatos y valida que tengan salida 4D (B, C, H, W).
    Soporta: ResNet, ConvNeXt V1/V2, DINOv2, EfficientNet, Swin, etc.
    """
    if not hasattr(model, 'backbone'):
        raise ValueError("Model doesn't have 'backbone' attribute")
    
    backbone = model.backbone
    candidates = []
    
    # Strategy 1: ResNet-style (layer4)
    if hasattr(backbone, 'layer4'):
        candidates.append(('layer4[-1]', backbone.layer4[-1]))
    
    # Strategy 2: ConvNeXt from timm (stages)
    if hasattr(backbone, 'stages'):
        # ConvNeXt V2 from timm has stages -> last stage is best
        try:
            last_stage = backbone.stages[-1]
            # Last stage puede ser Sequential, tomar último bloque
            if hasattr(last_stage, '__getitem__'):
                candidates.append(('stages[-1][-1]', last_stage[-1]))
            else:
                candidates.append(('stages[-1]', last_stage))
        except:
            pass
    
    # Strategy 3: torchvision ConvNeXt V1 (features)
    if hasattr(backbone, 'features'):
        # features es Sequential, tomar el penúltimo (último suele ser norm/pool)
        try:
            features = backbone.features
            if len(features) > 1:
                # Probar penúltimo y último
                candidates.append(('features[-2]', features[-2]))
                candidates.append(('features[-1]', features[-1]))
        except:
            pass
    
    # Strategy 4: EfficientNet/timm genérico (blocks)
    if hasattr(backbone, 'blocks'):
        try:
            candidates.append(('blocks[-1]', backbone.blocks[-1]))
        except:
            pass
    
    # Strategy 5: Último hijo genérico
    try:
        children = list(backbone.children())
        if children:
            candidates.append(('children[-1]', children[-1]))
            if len(children) > 1:
                candidates.append(('children[-2]', children[-2]))
    except:
        pass
    
    if not candidates:
        raise ValueError("No target layer candidates found in backbone")
    
    # Validar cada candidato: debe ser un módulo con parámetros
    valid_candidates = []
    for name, layer in candidates:
        # Verificar que sea un nn.Module con parámetros
        if isinstance(layer, nn.Module):
            params = list(layer.parameters())
            if len(params) > 0:
                valid_candidates.append((name, layer))
    
    if not valid_candidates:
        raise ValueError(f"No valid target layers found. Tried: {[c[0] for c in candidates]}")
    
    # Retornar el primer candidato válido
    # (típicamente el más específico/profundo)
    selected_name, selected_layer = valid_candidates[0]
    logger.info(f"GradCAM target_layer auto-detected: {selected_name}")
    return selected_layer


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    target_layer=None,
    centroid: torch.Tensor = None,
    mask: Optional[torch.Tensor] = None,
    grain_config: Optional[dict] = None,
) -> np.ndarray:
    """
    Genera mapa de calor de atención para explicabilidad visual.

    Delegates to the unified ``visual_explainer`` module which uses:
      - ViT/DINOv2/DeiT: Gradient-weighted self-attention (Chefer et al., CVPR 2021)
        with correct token topology parsing and bilinear upsampling.
      - ConvNeXt: Grad-CAM with ``reshape_transform`` for 3-D stage outputs.
      - ResNet/EfficientNet: Standard Grad-CAM (Selvaraju et al., ICCV 2017).

    Args:
        model: AnalogyNet
        image_tensor: (1, 3, H, W) normalizado
        device: torch.device
        target_layer: Capa objetivo (solo para CNN fallback, ignored by new module)
        centroid: Centroide de clase (embed_dim,) para gradient weighting

    Returns:
        Heatmap numpy array (H, W) en [0, 1]
    """
    try:
        from src.utils.visual_explainer import create_explainer
        explainer = create_explainer(model, device)
        return explainer.generate_heatmap(
            image_tensor,
            centroid,
            mask=mask,
            grain_config=grain_config,
        )
    except Exception as e:
        logger.warning(f"[generate_gradcam] New explainer failed ({e}), falling back to legacy")
        return _generate_gradcam_legacy(model, image_tensor, device, target_layer, centroid)


def _generate_gradcam_legacy(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    target_layer=None,
    centroid: torch.Tensor = None
) -> np.ndarray:
    """Legacy generate_gradcam kept as fallback."""
    model.eval()
    image_tensor = image_tensor.to(device)

    # ViT/DINOv2: use native attention maps (SOTA for transformers)
    if _is_vit_backbone(model):
        try:
            if centroid is not None:
                return extract_gradient_weighted_attention(model, image_tensor, device, centroid)
            else:
                attn_map, _ = extract_dino_attention(model, image_tensor, device)
                return attn_map
        except Exception as e:
            logger.warning(f"DINOv2 attention extraction failed: {e}, falling back to GradCAM")

    # CNN fallback: classic Grad-CAM
    from pytorch_grad_cam import GradCAM

    if target_layer is None:
        target_layer = _detect_target_layer(model)

    # Intentar con el target_layer detectado
    reshape_fn = _vit_reshape_transform if _is_vit_backbone(model) else None
    
    try:
        cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_fn)
        targets = None
        if centroid is not None:
            targets = [EmbeddingSimilarityTarget(centroid)]
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        return grayscale_cam
    except ValueError as e:
        if "Invalid grads shape" in str(e):
            logger.warning(f"GradCAM failed with detected layer: {e}")
            logger.info("Trying alternative layers...")
            
            alternative_layers = []
            backbone = model.backbone
            
            if hasattr(backbone, 'stages'):
                try:
                    for i in range(len(backbone.stages) - 1, -1, -1):
                        stage = backbone.stages[i]
                        if hasattr(stage, '__getitem__') and len(stage) > 0:
                            alternative_layers.append(stage[-1])
                            if len(stage) > 1:
                                alternative_layers.append(stage[-2])
                except:
                    pass
            
            for alt_layer in alternative_layers:
                try:
                    logger.info(f"  Trying alternative layer: {type(alt_layer).__name__}")
                    cam = GradCAM(model=model, target_layers=[alt_layer], reshape_transform=reshape_fn)
                    targets = None
                    if centroid is not None:
                        targets = [EmbeddingSimilarityTarget(centroid)]
                    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
                    grayscale_cam = grayscale_cam[0, :]
                    logger.info(f"  SUCCESS with {type(alt_layer).__name__}")
                    return grayscale_cam
                except Exception as inner_e:
                    logger.debug(f"  Failed with {type(alt_layer).__name__}: {inner_e}")
                    continue
            
            raise ValueError(f"GradCAM failed with all attempted layers. Original error: {e}")
        else:
            raise


def overlay_gradcam_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Superpone heatmap sobre imagen original.
    
    Args:
        image: Imagen RGB (H, W, 3) en [0, 255] o [0, 1]
        heatmap: Heatmap (H, W) en [0, 1]
        alpha: Transparencia del heatmap
        
    Returns:
        Imagen con heatmap superpuesto (H, W, 3) en [0, 255] uint8
    """
    import cv2

    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    blended = (1 - alpha) * image + alpha * heatmap_rgb
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return blended


def project_heatmap_to_full_image(
    crop_heatmap: np.ndarray,
    grain_bbox: Tuple[int, int, int, int],
    crop_padding: float,
    full_image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Proyecta un heatmap de un recorte (crop) de vuelta a las dimensiones
    de la imagen completa original, usando las coordenadas del bounding box.
    
    Args:
        crop_heatmap: Heatmap generado sobre el tensor del crop (H_crop, W_crop)
        grain_bbox: Tupla (x, y, w, h) del grano en la imagen original
        crop_padding: Padding usado al recortar (ej. 0.2)
        full_image_shape: Forma de la imagen completa (H, W, ...)
        
    Returns:
        Heatmap de tamaño (full_H, full_W) proyectado en la ubicación del grano.
    """
    import cv2
    
    h_img, w_img = full_image_shape[:2]
    bx, by, bw, bh = grain_bbox

    pad_x = int(bw * crop_padding)
    pad_y = int(bh * crop_padding)
    side = max(bw + 2 * pad_x, bh + 2 * pad_y)

    cx = bx + bw // 2
    cy = by + bh // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    # Redimensionar el heatmap (que típicamente es 224x224 o similar) al tamaño real del recorte (side x side)
    heatmap_resized = cv2.resize(crop_heatmap, (side, side), interpolation=cv2.INTER_LINEAR)

    # Crear lienzo en blanco del tamaño de la imagen original
    full_heatmap = np.zeros((h_img, w_img), dtype=np.float32)

    # Calcular intersección válida con la imagen original (evitar bordes fuera de límite)
    valid_x1 = max(0, x1)
    valid_y1 = max(0, y1)
    valid_x2 = min(w_img, x2)
    valid_y2 = min(h_img, y2)

    crop_x1 = valid_x1 - x1
    crop_y1 = valid_y1 - y1
    crop_x2 = side - (x2 - valid_x2)
    crop_y2 = side - (y2 - valid_y2)

    if valid_x2 > valid_x1 and valid_y2 > valid_y1:
        full_heatmap[valid_y1:valid_y2, valid_x1:valid_x2] = heatmap_resized[crop_y1:crop_y2, crop_x1:crop_x2]

    return full_heatmap


def overlay_full_image_gradcam(
    full_image: np.ndarray,
    heatmaps_and_bboxes: List[Tuple[np.ndarray, Tuple[int, int, int, int]]],
    crop_padding: float = 0.2,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Superpone múltiples heatmaps de granos sobre la imagen original completa.
    
    Args:
        full_image: Imagen BGR original completa (H, W, 3)
        heatmaps_and_bboxes: Lista de tuplas (crop_heatmap, (x, y, w, h))
        crop_padding: Padding usado en la extracción de recortes
        alpha: Transparencia de la superposición
    """
    full_canvas = np.zeros(full_image.shape[:2], dtype=np.float32)
    
    for crop_heatmap, bbox in heatmaps_and_bboxes:
        projected_hm = project_heatmap_to_full_image(
            crop_heatmap, bbox, crop_padding, full_image.shape
        )
        # Combinar heatmaps (pueden superponerse)
        full_canvas = np.maximum(full_canvas, projected_hm)
        
    return overlay_gradcam_on_image(full_image, full_canvas, alpha=alpha)


def get_embedding_with_tta(
    model: nn.Module,
    image_path: str,
    tta_transforms: list,
    device: torch.device
) -> np.ndarray:
    """
    Obtiene embedding con Test-Time Augmentation (TTA).
    Aplica N transforms, obtiene N embeddings, promedia y re-normaliza L2.
    Ref: Zolfaghari & Sajedi 2024 [R5], Goncalves et al. 2022 [R7].
    
    Args:
        model: Modelo para extraer embeddings
        image_path: Ruta a la imagen
        tta_transforms: Lista de transforms a aplicar
        device: Dispositivo
        
    Returns:
        Embedding promediado y L2-normalizado (1, embedding_dim)
    """
    from PIL import Image
    
    model.eval()
    image = Image.open(image_path).convert('RGB')
    
    all_embeddings = []
    with torch.no_grad():
        for transform in tta_transforms:
            image_tensor = transform(image).unsqueeze(0).to(device)
            embedding = model(image_tensor).cpu().numpy()
            all_embeddings.append(embedding)
    
    avg_embedding = np.mean(all_embeddings, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding, axis=1, keepdims=True)
    
    logger.info(f"TTA embedding: {len(tta_transforms)} augmentations averaged")
    return avg_embedding


def compute_pairwise_distances(
    embeddings1: np.ndarray,
    embeddings2: Optional[np.ndarray] = None,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Calcula distancias por pares entre embeddings.
    
    Args:
        embeddings1: Primer conjunto de embeddings (N, D)
        embeddings2: Segundo conjunto (M, D). Si None, usa embeddings1
        metric: 'euclidean' o 'cosine'
        
    Returns:
        Matriz de distancias (N, M)
    """
    from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
    
    if embeddings2 is None:
        embeddings2 = embeddings1
    
    if metric == 'euclidean':
        distances = euclidean_distances(embeddings1, embeddings2)
    elif metric == 'cosine':
        distances = cosine_distances(embeddings1, embeddings2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances
