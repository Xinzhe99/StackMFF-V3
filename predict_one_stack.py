# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

"""
Single image stack fusion prediction script for StackMFF_V3 network.
"""

import argparse
import torch
import cv2
import numpy as np
import os
from datetime import datetime
import matplotlib
from network import StackMFF_V3
import torch.nn.functional as F

def gray_to_colormap(img, cmap='rainbow'):
    """
    Convert grayscale image to colormap visualization
    Args:
        img: Input grayscale image (normalized to [0,1])
        cmap: Colormap name (default: 'rainbow')
    Returns:
        colormap: RGB colormap visualization
    """
    img = np.clip(img, 0, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    return colormap


def parse_args():
    parser = argparse.ArgumentParser(description="Image Stack Fusion Inference Script")
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input image stack')
    parser.add_argument('--output_dir', type=str, default='./results_single_stack',
                       help='Directory for saving results')
    parser.add_argument('--model_path', type=str, default='./weights/stackmffv3.pth',
                       help='Path to the trained model weights')
    return parser.parse_args()


def load_image_stack(input_dir):
    """
    Load image stack from directory
    Args:
        input_dir: Directory containing image files
    Returns:
        gray_tensors: Stack of grayscale images as tensor [N, H, W]
        color_images: List of original RGB images
    """
    # Get all image files and sort them numerically
    image_files = sorted([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg','bmp'))],
                        key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    gray_tensors = []
    color_images = []

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        # Read BGR image and convert to RGB
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            print(f"Warning: Could not load {img_path}")
            continue
            
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        color_images.append(rgb_img)

        # Convert to grayscale (Y channel)
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        # Normalize and convert to tensor
        gray_tensor = torch.from_numpy(gray_img.astype(np.float32) / 255.0)
        gray_tensors.append(gray_tensor)

    if not gray_tensors:
        raise ValueError(f"No valid images could be loaded from {input_dir}")
        
    # Stack to create [N, H, W] tensor
    return torch.stack(gray_tensors), color_images


def create_fused_color_image(fused_image, focus_indices, color_stack):
    """
    Create color fused image based on focus indices (vectorized implementation)
    Args:
        fused_image: Fused grayscale image
        focus_indices: Index map indicating which image to sample from
        color_stack: List of original color images
    Returns:
        fused_color_bgr: Color fused image in BGR format
    """
    height, width = fused_image.shape
    num_images = len(color_stack)

    # Ensure indices are within valid range
    focus_indices = np.clip(focus_indices, 0, num_images - 1).astype(int)

    # Convert color stack to single numpy array (N, H, W, 3)
    color_array = np.stack(color_stack, axis=0)

    # Use advanced indexing to get corresponding color values
    fused_color = color_array[focus_indices, np.arange(height)[:, None], np.arange(width)]

    # Convert RGB to BGR for saving
    fused_color_bgr = cv2.cvtColor(fused_color.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return fused_color_bgr


def resize_to_multiple_of_32(image):
    """
    Resize image to multiple of 32 for network processing
    Args:
        image: Input image tensor
    Returns:
        resized_image: Resized image tensor
        original_size: Original image dimensions
    """
    h, w = image.shape[-2:]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized_image, (h, w)


def process_stack(model, image_stack, color_stack, device):
    """
    Process image stack and generate results
    Args:
        model: Neural network model
        image_stack: Stack of input images [N, H, W]
        color_stack: List of original color images
        device: Computing device (CPU/GPU)
    Returns:
        fused_image: Grayscale fused image
        focus_indices: Focus index map
        color_fused: Color fused image
        focus_colormap: Colormap visualization of focus indices
    """
    model.eval()
    with torch.no_grad():
        # Save original size
        original_size = image_stack.shape[-2:]

        # Resize to multiple of 32 and add batch dimension
        resized_stack, _ = resize_to_multiple_of_32(image_stack.unsqueeze(0))
        resized_stack = resized_stack.to(device)

        # Model inference - StackMFF_V3 returns (fused_image, focus_indices)
        fused_image, focus_indices = model(resized_stack)

        # Convert to numpy and resize back to original size
        fused_image = cv2.resize(fused_image.cpu().numpy().squeeze(),
                               (original_size[1], original_size[0]))
        focus_indices = cv2.resize(focus_indices.cpu().numpy().squeeze().astype(np.float32),
                                   (original_size[1], original_size[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(int)

        # Generate color fused image
        color_fused = create_fused_color_image(fused_image, focus_indices, color_stack)

        # Generate colormap visualization of focus indices
        # Normalize focus indices to [0, 1] for colormap
        focus_normalized = focus_indices.astype(np.float32) / max(len(color_stack) - 1, 1)
        focus_colormap = gray_to_colormap(focus_normalized)

        return fused_image, focus_indices, color_fused, focus_colormap


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = StackMFF_V3()
    # Load model state dict and handle potential 'module.' prefix from DataParallel
    state_dict = torch.load(args.model_path, map_location=device)
    # Remove 'module.' prefix if it exists (from DataParallel training)
    if any(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict
    model.load_state_dict(new_state_dict)
    model.to(device)

    # Load image stack
    image_stack, color_stack = load_image_stack(args.input_dir)
    print(f"Loaded {len(color_stack)} images from {args.input_dir}")
    
    # Process image stack
    fused_image, focus_indices, color_fused, focus_colormap = process_stack(
        model, image_stack, color_stack, device)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results (ensure values are in 0-255 range)
    cv2.imwrite(os.path.join(args.output_dir, f'fused_gray_{timestamp}.png'),
                (fused_image * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.output_dir, f'focus_indices_{timestamp}.png'),
                focus_indices.astype(np.uint8) * (255 // max(len(color_stack) - 1, 1)))
    cv2.imwrite(os.path.join(args.output_dir, f'color_fused_{timestamp}.png'),
                color_fused)
    cv2.imwrite(os.path.join(args.output_dir, f'focus_colormap_{timestamp}.png'),
                cv2.cvtColor(focus_colormap, cv2.COLOR_RGB2BGR))

    print(f"Results saved to {args.output_dir} with timestamp {timestamp}")
    print(f"Focus index range: [{focus_indices.min()}, {focus_indices.max()}]")


if __name__ == "__main__":
    main()