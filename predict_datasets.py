# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

"""
Batch prediction script for StackMFF_V3 network on multiple datasets.
"""

import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
from network import StackMFF_V3
import random
import re
import time
import cv2
import numpy as np
import torch.nn.functional as F
import os
import matplotlib
import torch
import torch.nn as nn
from datetime import datetime

def parse_args():
    """
    Parse command line arguments for batch evaluation of multiple datasets
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
            - model_path: Path to the trained model weights
            - test_root: Root directory containing test datasets
            - test_datasets: List of dataset names to evaluate
            - batch_size: Batch size for evaluation
            - num_workers: Number of data loading workers
            - output_dir: Directory for saving results
    """
    parser = argparse.ArgumentParser(description="Batch evaluation script for multiple datasets")
    parser.add_argument('--model_path', type=str, default='./weights/stackmffv3.pth',
                        help='Path to the trained model weights')
    parser.add_argument('--test_root', type=str, required=True,
                        help='Path to test data root directory')
    parser.add_argument('--test_datasets', nargs='+',
                        default=['Mobile Depth','FlyingThings3D','Middlebury','Road-MF'],
                        help='List of test datasets to evaluate')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./results_stack',
                        help='Directory for saving evaluation results')
    return parser.parse_args()

def resize_to_multiple_of_32(image):
    """
    Resize input image tensor to dimensions that are multiples of 32
    
    Args:
        image (torch.Tensor): Input image tensor
        
    Returns:
        tuple: (resized_image, (original_height, original_width))
            - resized_image: Image resized to multiple of 32
            - tuple of original dimensions
    """
    h, w = image.shape[-2:]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return resized_image, (h, w)

def gray_to_colormap(img, cmap='rainbow'):
    """
    Convert grayscale image to colormap visualization
    
    Args:
        img (numpy.ndarray): Input grayscale image (normalized to [0,1])
        cmap (str): Matplotlib colormap name
        
    Returns:
        numpy.ndarray: RGB colormap visualization
    """
    img = np.clip(img, 0, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    return colormap

class TestDataset(Dataset):
    """
    Dataset class for loading and processing image stacks
    
    Args:
        root_dir (str): Root directory containing image stacks
        transform (callable, optional): Optional transform to be applied on images
        subset_fraction (float, optional): Fraction of total stacks to use
    """
    def __init__(self, root_dir, transform=None, subset_fraction=1.0):
        self.root_dir = root_dir
        self.transform = transform
        self.image_stacks = []
        self.stack_names = []

        # Get all stack directories and optionally sample a subset
        all_stacks = sorted(os.listdir(root_dir))
        subset_size = int(len(all_stacks) * subset_fraction)
        selected_stacks = random.sample(all_stacks, subset_size)

        # Load image paths for each stack
        for stack_name in selected_stacks:
            stack_path = os.path.join(root_dir, stack_name)
            if os.path.isdir(stack_path):
                image_stack = []
                for img_name in sorted(os.listdir(stack_path), key=self.sort_key):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(stack_path, img_name)
                        image_stack.append(img_path)

                if image_stack:
                    self.image_stacks.append(image_stack)
                    self.stack_names.append(stack_name)
                else:
                    print(f'Failed to read image stack: {stack_name}')

    def __len__(self):
        return len(self.image_stacks)

    def __getitem__(self, idx):
        """
        Get a single image stack
        
        Args:
            idx (int): Index of the stack
            
        Returns:
            tuple: (stack_tensor, stack_name, num_images)
                - stack_tensor: Tensor containing the image stack
                - stack_name: Name of the stack
                - num_images: Number of images in the stack
        """
        image_stack = self.image_stacks[idx]
        stack_name = self.stack_names[idx]

        images = []
        for img_path in image_stack:
            # Read image and convert to grayscale
            bgr_img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
            # Normalize to [0,1]
            gray_img = gray_img.astype(np.float32) / 255.0
            if self.transform:
                gray_img = self.transform(gray_img)
            images.append(gray_img.squeeze(0))

        stack_tensor = torch.stack(images)
        return stack_tensor, stack_name, len(images)

    @staticmethod
    def sort_key(filename):
        """
        Extract numerical value from filename for sorting
        """
        numbers = re.findall(r'\d+\.?\d*', filename)
        return float(numbers[0]) if numbers else 0


def infer_dataset(model, dataset_loader, device, save_path):
    """
    Perform inference on a dataset and save results
    
    Args:
        model: Neural network model
        dataset_loader: DataLoader containing the test dataset
        device: Computing device (CPU/GPU)
        save_path: Directory to save results
    
    Returns:
        float: Average inference time per stack
    """
    model.eval()

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_path, f'results_{timestamp}')
    
    # Create output subdirectories
    subdirs = ['fused_gray', 'focus_indices', 'color_fused', 'focus_colormap']
    for subdir in subdirs:
        os.makedirs(os.path.join(save_path, subdir), exist_ok=True)

    # Initialize timing metrics
    total_inference_time = 0
    total_stacks = 0

    with torch.no_grad():
        for idx, (image_stack, stack_name, _) in tqdm(enumerate(dataset_loader)):
            # Save original dimensions
            original_size = image_stack.shape[-2:]

            # Resize input to multiple of 32 for network processing
            resized_image_stack, _ = resize_to_multiple_of_32(image_stack)
            resized_image_stack = resized_image_stack.to(device)

            # Load original color images for color fusion
            color_stack = load_color_stack(dataset_loader.dataset.image_stacks[idx])

            # Warmup on first batch (not timed) to stabilize kernels
            if idx == 0:
                for _ in range(3):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    _ = model(resized_image_stack)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            # Measure inference time with CUDA synchronization to ensure strict timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            # Model inference - StackMFF_V3 returns (fused_image, focus_indices)
            fused_image, focus_indices = model(resized_image_stack)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            inference_time = end_time - start_time
            total_inference_time += inference_time
            total_stacks += 1

            # Process model outputs
            fused_image, focus_indices = process_model_output(
                fused_image, focus_indices, original_size)

            # Create color fused image and focus colormap
            color_fused_bgr = create_fused_color_image(fused_image, focus_indices, color_stack)
            # Normalize focus indices to [0, 1] for colormap
            focus_normalized = focus_indices.astype(np.float32) / max(len(color_stack) - 1, 1)
            focus_colormap = gray_to_colormap(focus_normalized)
            focus_colormap_bgr = cv2.cvtColor(focus_colormap, cv2.COLOR_RGB2BGR)
            
            # Generate output filename with stack name
            filename = f'{stack_name[0]}.png'
            
            # Save all results
            try:
                cv2.imwrite(os.path.join(save_path, subdirs[0], filename), 
                           (fused_image * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(save_path, subdirs[1], filename), 
                           focus_indices.astype(np.uint8) * (255 // max(len(color_stack) - 1, 1)))
                cv2.imwrite(os.path.join(save_path, subdirs[2], filename), 
                           color_fused_bgr)
                cv2.imwrite(os.path.join(save_path, subdirs[3], filename), 
                           focus_colormap_bgr)
            except Exception as e:
                print(f"Error saving images: {str(e)}")
                continue

    # Calculate average inference time
    avg_inference_time = total_inference_time / total_stacks if total_stacks > 0 else 0
    return avg_inference_time


def main():
    """
    Main function to run batch evaluation
    
    Directory structure should be:
    test_root/
        dataset1/
            dof_stack/
                scene1/
                    img1.png
                    img2.png
                    ...
                scene2/
                    ...
        dataset2/
            dof_stack/
                ...
    """
    # Parse command line arguments
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    # Only use DataParallel if CUDA is available and multiple GPUs exist (match star script behavior)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # Initialize test data loaders for each dataset
    test_loaders = {}
    for dataset_name in args.test_datasets:
        dataset_root = os.path.join(args.test_root, dataset_name)
        if not os.path.exists(dataset_root):
            print(f"Warning: Dataset directory {dataset_root} not found. Skipping...")
            continue

        # Set up data transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Create dataset and dataloader
        dataset = TestDataset(
            root_dir=os.path.join(dataset_root, 'dof_stack'),
            transform=transform
        )

        test_loaders[dataset_name] = DataLoader(
            dataset, 
            shuffle=False, 
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # Process each dataset
    dataset_avg_times = {}
    for dataset, loader in test_loaders.items():
        dataset_save_path = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_save_path, exist_ok=True)
        print(f"Processing dataset: {dataset}")
        avg_time = infer_dataset(model, loader, device, dataset_save_path)
        dataset_avg_times[dataset] = avg_time

    # Print results summary
    print(f"\nResults saved to: {args.output_dir}")
    print("\nAverage inference times per image stack:")
    for dataset, avg_time in dataset_avg_times.items():
        print(f"{dataset}: {avg_time:.4f} seconds")


def create_fused_color_image(fused_image, focus_indices, color_stack):
    """
    Create color fused image based on focus indices (vectorized implementation)
    
    Args:
        fused_image: Fused grayscale image
        focus_indices: Index map indicating which image to sample from
        color_stack: List of original color images
    
    Returns:
        numpy.ndarray: Color fused image in BGR format
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


def load_color_stack(image_paths):
    """
    Load color image stack from file paths
    
    Args:
        image_paths: List of paths to image files
    
    Returns:
        list: List of RGB images
    """
    color_images = []
    for img_path in image_paths:
        # Read BGR image and convert to RGB
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        color_images.append(rgb_img)
    return color_images


def process_model_output(fused_image, focus_indices, original_size):
    """
    Process model outputs and resize to original dimensions
    
    Args:
        fused_image: Fused image from model
        focus_indices: Focus index map from model
        original_size: Original image dimensions
    
    Returns:
        tuple: (fused_image, focus_indices) resized to original dimensions
    """
    # Convert to numpy and resize back to original size
    fused_image = cv2.resize(fused_image.cpu().numpy().squeeze(),
                             (original_size[1], original_size[0]))
    focus_indices = cv2.resize(focus_indices.cpu().numpy().squeeze().astype(np.float32),
                               (original_size[1], original_size[0]),
                               interpolation=cv2.INTER_NEAREST).astype(int)

    return fused_image, focus_indices


if __name__ == "__main__":
    main()