# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# @Description : This script implements a data loader for focus image stacks and their corresponding focus index maps.
# It provides functionality for loading image stacks and focus index maps, with support for data augmentation,
# subset sampling, and grouped batching based on stack sizes.

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset

class FocusStackDataset(Dataset):
    """
    Dataset class for handling stacks of focus images and their corresponding focus index maps.
    Supports data augmentation and subset sampling.
    """
    def __init__(self, root_dir, focus_index_dir, transform=None, augment=True, subset_fraction=1):
        """
        Initialize the dataset.
        Args:
            root_dir: Directory containing focus image stacks  
            focus_index_dir: Directory containing focus index maps (.npy files)
            transform: Optional transforms to be applied
            augment: Whether to apply data augmentation
            subset_fraction: Fraction of the dataset to use (0-1)
        """
        self.root_dir = root_dir
        self.focus_index_dir = focus_index_dir
        self.transform = transform
        self.augment = augment
        self.image_stacks = []
        self.focus_index_maps = []
        self.stack_sizes = []

        all_stacks = sorted(os.listdir(root_dir))
        subset_size = int(len(all_stacks) * subset_fraction)
        selected_stacks = random.sample(all_stacks, subset_size)

        for stack_name in selected_stacks:
            stack_path = os.path.join(root_dir, stack_name)
            if os.path.isdir(stack_path):
                image_stack = []
                # Exclude layer_order.npy file, only load image files
                for img_name in sorted(os.listdir(stack_path), key=self.sort_key):
                    if img_name.lower().endswith(('.png', '.jpg', '.bmp')) and img_name != 'layer_order.npy':
                        img_path = os.path.join(stack_path, img_name)
                        image_stack.append(img_path)

                if image_stack:
                    focus_index_map_path = os.path.join(focus_index_dir, stack_name + '.npy')
                    if os.path.exists(focus_index_map_path):
                        self.image_stacks.append(image_stack)
                        self.focus_index_maps.append(focus_index_map_path)
                        self.stack_sizes.append(len(image_stack))
                    else:
                        print(f"Warning: Focus index map not found for {stack_name}")
                else:
                    print(f'Failed to read image stack: {stack_name}')

    def __len__(self):
        return len(self.image_stacks)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Returns:
            stack_tensor: Tensor of stacked images (N, H, W)
            focus_index_gt: Corresponding focus index map [H, W] as torch.long
            len(images): Number of images in the stack
        """
        image_stack = self.image_stacks[idx]
        focus_index_map_path = self.focus_index_maps[idx]

        images = []
        for img_path in image_stack:
            image = Image.open(img_path).convert('YCbCr')
            image = image.split()[0]  # Keep only Y channel
            images.append(image)

        # Load focus index map (.npy format)
        focus_index_gt = np.load(focus_index_map_path)  # [H, W] format, np.int64
        
        # Validate focus index validity: indices must be in [0, len(images)-1] range
        max_index = len(images) - 1
        if focus_index_gt.max() > max_index or focus_index_gt.min() < 0:
            print(f"Warning: Focus index out of valid range [0, {max_index}]: actual range [{focus_index_gt.min()}, {focus_index_gt.max()}]")
            focus_index_gt = np.clip(focus_index_gt, 0, max_index)
            
        focus_index_gt = torch.from_numpy(focus_index_gt).long()  # Convert to torch.long

        if self.augment:
            images, focus_index_gt = self.consistent_transform(images, focus_index_gt)

        # Apply other transformations
        if self.transform:
            images = [self.transform(img) for img in images]
            # Apply the same size adjustment to the focus index map to ensure consistency
            # Get target size
            target_size = None
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    target_size = t.size
                    break
            
            if target_size is not None:
                # Adjust focus index map size, using nearest neighbor interpolation to keep index values unchanged
                focus_index_gt = TF.resize(
                    focus_index_gt.unsqueeze(0).float(), 
                    target_size, 
                    interpolation=transforms.InterpolationMode.NEAREST
                ).squeeze(0).long()
            
        # Convert to tensor and remove channel dimension
        images = [img.squeeze(0) for img in images]
        stack_tensor = torch.stack(images)  # Shape will be (N, H, W)

        return stack_tensor, focus_index_gt, len(images)

    def consistent_transform(self, images, focus_index_gt):
        """
        Apply consistent transformations to both images and focus index map.
        Includes random horizontal and vertical flips.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            images = [TF.hflip(img) for img in images]
            focus_index_gt = TF.hflip(focus_index_gt.unsqueeze(0)).squeeze(0)  # Temporarily add dimension for flipping

        # Random vertical flip
        if random.random() > 0.5:
            images = [TF.vflip(img) for img in images]
            focus_index_gt = TF.vflip(focus_index_gt.unsqueeze(0)).squeeze(0)  # Temporarily add dimension for flipping

        return images, focus_index_gt

    @staticmethod
    def sort_key(filename):
        """
        Helper function to sort filenames based on their numerical values.
        Returns 0 if no digits are found to handle non-numeric filenames.
        """
        digits = ''.join(filter(str.isdigit, filename))
        return int(digits) if digits else 0

class GroupedBatchSampler(Sampler):
    """
    Custom batch sampler that groups samples by stack size for efficient batching.
    Ensures that each batch contains stacks of the same size.
    """
    def __init__(self, stack_sizes, batch_size):
        """
        Initialize the sampler.
        Args:
            stack_sizes: List of stack sizes for each sample
            batch_size: Number of samples per batch
        """
        self.stack_size_groups = defaultdict(list)
        for idx, size in enumerate(stack_sizes):
            self.stack_size_groups[size].append(idx)
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        """
        Create batches of indices grouped by stack size.
        Returns shuffled batches for random sampling.
        """
        batches = []
        for size, indices in self.stack_size_groups.items():
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        random.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)

def get_updated_dataloader(dataset_params, batch_size, num_workers=4, augment=True, target_size=384):
    """
    Create a DataLoader with multiple datasets combined.
    Args:
        dataset_params: List of parameter dictionaries for each dataset
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        augment: Whether to apply data augmentation
        target_size: Size to resize images to
    Returns:
        DataLoader object with combined datasets
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])

    datasets = []
    for params in dataset_params:
        dataset = FocusStackDataset(
            root_dir=params['root_dir'],
            focus_index_dir=params['focus_index_gt'],  # Use correct parameter name
            transform=transform,
            augment=augment,
            subset_fraction=params['subset_fraction']
        )
        datasets.append(dataset)

    combined_dataset = CombinedDataset(datasets)

    sampler = GroupedBatchSampler(combined_dataset.stack_sizes, batch_size)

    dataloader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=num_workers)
    return dataloader

class CombinedDataset(ConcatDataset):
    """
    Extension of ConcatDataset that maintains stack size information
    when combining multiple datasets.
    """
    def __init__(self, datasets):
        """
        Initialize the combined dataset.
        Args:
            datasets: List of FocusStackDataset objects to combine
        """
        super(CombinedDataset, self).__init__(datasets)
        self.stack_sizes = []
        for dataset in datasets:
            self.stack_sizes.extend(dataset.stack_sizes)

    def __getitem__(self, idx):
        return super(CombinedDataset, self).__getitem__(idx)