"""
Visualization Utilities for Metric Learning System
Provides denormalization and image conversion functions for visual inspection.
"""

import torch
import numpy as np
from PIL import Image
from typing import Union, Tuple


# ImageNet normalization constants (used during training)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(tensor: torch.Tensor, 
                      mean: torch.Tensor = IMAGENET_MEAN,
                      std: torch.Tensor = IMAGENET_STD) -> torch.Tensor:
    """
    Reverse ImageNet normalization to recover original image colors.
    
    Args:
        tensor: Normalized image tensor (C, H, W) or (B, C, H, W)
        mean: Mean values used during normalization
        std: Std values used during normalization
    
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    if tensor.dim() == 3:
        # Single image (C, H, W)
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
        denorm = tensor * std + mean
    elif tensor.dim() == 4:
        # Batch of images (B, C, H, W)
        mean = mean.to(tensor.device).unsqueeze(0)
        std = std.to(tensor.device).unsqueeze(0)
        denorm = tensor * std + mean
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")
    
    # Clamp to valid range [0, 1]
    denorm = torch.clamp(denorm, 0, 1)
    
    return denorm


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a PyTorch tensor to PIL Image.
    
    Args:
        tensor: Image tensor (C, H, W) with values in [0, 1]
    
    Returns:
        PIL Image in RGB format
    """
    if tensor.dim() == 4:
        # If batch, take first image
        tensor = tensor[0]
    
    # Ensure tensor is on CPU and convert to numpy
    tensor = tensor.cpu().detach()
    
    # Convert from (C, H, W) to (H, W, C)
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy and scale to [0, 255]
    numpy_image = (tensor.numpy() * 255).astype(np.uint8)
    
    # Create PIL Image
    pil_image = Image.fromarray(numpy_image, mode='RGB')
    
    return pil_image


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to numpy array for OpenCV operations.
    
    Args:
        tensor: Image tensor (C, H, W) with values in [0, 1]
    
    Returns:
        Numpy array (H, W, C) in BGR format for OpenCV, values in [0, 255]
    """
    if tensor.dim() == 4:
        # If batch, take first image
        tensor = tensor[0]
    
    # Ensure tensor is on CPU
    tensor = tensor.cpu().detach()
    
    # Convert from (C, H, W) to (H, W, C)
    if tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy and scale to [0, 255]
    numpy_image = (tensor.numpy() * 255).astype(np.uint8)
    
    # Convert RGB to BGR for OpenCV
    numpy_image = np.ascontiguousarray(numpy_image[:, :, ::-1])
    
    return numpy_image


def create_color_palette(num_classes: int) -> list:
    """
    Create a color palette for visualization.
    
    Args:
        num_classes: Number of classes
    
    Returns:
        List of RGB tuples
    """
    import colorsys
    
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    
    return colors


def get_distance_color(distance: float, threshold: float = 0.8) -> Tuple[int, int, int]:
    """
    Get color based on distance (traffic light system).
    
    Args:
        distance: Distance value
        threshold: Threshold for safe/unsafe classification
    
    Returns:
        BGR color tuple for OpenCV
    """
    if distance < threshold * 0.5:
        # Very confident - Dark Green
        return (0, 200, 0)
    elif distance < threshold:
        # Confident - Light Green
        return (0, 255, 0)
    elif distance < threshold * 1.5:
        # Warning - Yellow
        return (0, 255, 255)
    else:
        # Anomaly/Uncertain - Red
        return (0, 0, 255)


def add_text_with_background(image: np.ndarray, 
                             text: str, 
                             position: Tuple[int, int],
                             font_scale: float = 0.6,
                             thickness: int = 2,
                             text_color: Tuple[int, int, int] = (255, 255, 255),
                             bg_color: Tuple[int, int, int] = (0, 0, 0),
                             padding: int = 5) -> np.ndarray:
    """
    Add text with background rectangle to image (for better readability).
    
    Args:
        image: Input image (numpy array)
        text: Text to add
        position: (x, y) position for text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color (BGR)
        bg_color: Background color (BGR)
        padding: Padding around text
    
    Returns:
        Image with text added
    """
    import cv2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    # Calculate background rectangle coordinates
    x, y = position
    bg_x1 = x - padding
    bg_y1 = y - text_height - padding
    bg_x2 = x + text_width + padding
    bg_y2 = y + baseline + padding
    
    # Draw background rectangle
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # Draw text
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
    
    return image
