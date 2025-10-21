# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# @Description : This script implements a data loader for image stack datasets with continuous depth maps.
# It provides functionality for loading image stacks and their corresponding continuous depth maps,
# with support for data augmentation and grouped batching based on stack sizes.

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from PIL import Image
from collections import defaultdict
import torchvision.transforms.functional as TF

class UpdatedImageStackDataset(Dataset):
    """Dataset class for loading image stacks and continuous depth maps."""
    
    def __init__(self, root_dir, continuous_depth_dir, transform=None, augment=True, subset_fraction=1):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory containing image stacks
            continuous_depth_dir (str): Directory containing continuous depth maps
            transform (callable, optional): Transform to be applied on images
            augment (bool): Whether to apply data augmentation
            subset_fraction (float): Fraction of data to use (0.0 to 1.0)
        """
        self.root_dir = root_dir
        self.continuous_depth_dir = continuous_depth_dir
        self.transform = transform
        self.augment = augment
        self.image_stacks = []
        self.continuous_depth_maps = []
        self.stack_sizes = []

        all_stacks = sorted(os.listdir(root_dir))
        subset_size = int(len(all_stacks) * subset_fraction)
        selected_stacks = random.sample(all_stacks, subset_size)

        for stack_name in selected_stacks:
            stack_path = os.path.join(root_dir, stack_name)
            if os.path.isdir(stack_path):
                image_stack = []
                for img_name in sorted(os.listdir(stack_path), key=self.sort_key):
                    if img_name.lower().endswith(('.png', '.jpg', '.bmp')):
                        img_path = os.path.join(stack_path, img_name)
                        image_stack.append(img_path)

                if image_stack:
                    continuous_depth_map_path = os.path.join(continuous_depth_dir, stack_name + '.png')
                    if os.path.exists(continuous_depth_map_path):
                        self.image_stacks.append(image_stack)
                        self.continuous_depth_maps.append(continuous_depth_map_path)
                        self.stack_sizes.append(len(image_stack))
                    else:
                        print(f"Warning: Depth map not found for {stack_name}")
                else:
                    print(f'Failed to read image stack: {stack_name}')

    def __len__(self):
        """Return the number of image stacks in the dataset."""
        return len(self.image_stacks)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (stack_tensor, continuous_depth_map, stack_length)
        """
        image_stack = self.image_stacks[idx]
        continuous_depth_map_path = self.continuous_depth_maps[idx]

        images = []
        for img_path in image_stack:
            image = Image.open(img_path).convert('YCbCr')
            image = image.split()[0]  # Keep only Y channel
            images.append(image)

        continuous_depth_map = Image.open(continuous_depth_map_path).convert('L')

        if self.augment:
            images, continuous_depth_map = self.consistent_transform(images, continuous_depth_map)

        # Apply other transformations
        if self.transform:
            images = [self.transform(img) for img in images]
            continuous_depth_map = self.transform(continuous_depth_map)

        # Convert to tensor and remove channel dimension
        images = [img.squeeze(0) for img in images]
        stack_tensor = torch.stack(images)  # Shape will be (N, H, W)

        return stack_tensor, continuous_depth_map, len(images)

    def consistent_transform(self, images, continuous_depth_map):
        """
        Apply consistent transformations to images and depth map.
        
        Args:
            images (list): List of PIL images
            continuous_depth_map (PIL.Image): Continuous depth map
            
        Returns:
            tuple: Transformed images and depth map
        """
        # Random horizontal flip
        if random.random() > 0.5:
            images = [TF.hflip(img) for img in images]
            continuous_depth_map = TF.hflip(continuous_depth_map)

        # Random vertical flip
        if random.random() > 0.5:
            images = [TF.vflip(img) for img in images]
            continuous_depth_map = TF.vflip(continuous_depth_map)

        return images, continuous_depth_map

    @staticmethod
    def sort_key(filename):
        """Sort files by numeric value in filename."""
        return int(''.join(filter(str.isdigit, filename)))

class GroupedBatchSampler(Sampler):
    """Sampler that groups samples by stack size for efficient batching."""
    
    def __init__(self, stack_sizes, batch_size):
        """
        Initialize the sampler.
        
        Args:
            stack_sizes (list): List of stack sizes
            batch_size (int): Size of each batch
        """
        self.stack_size_groups = defaultdict(list)
        for idx, size in enumerate(stack_sizes):
            self.stack_size_groups[size].append(idx)
        self.batch_size = batch_size
        self.batches = self._create_batches()

    def _create_batches(self):
        """Create batches grouped by stack size."""
        batches = []
        for size, indices in self.stack_size_groups.items():
            for i in range(0, len(indices), self.batch_size):
                batches.append(indices[i:i + self.batch_size])
        random.shuffle(batches)
        return batches

    def __iter__(self):
        """Return an iterator over the batches."""
        return iter(self.batches)

    def __len__(self):
        """Return the number of batches."""
        return len(self.batches)

def get_updated_dataloader(dataset_params, batch_size, num_workers=4, augment=True, target_size=384):
    """
    Create a DataLoader with the specified parameters.
    
    Args:
        dataset_params (list): List of dictionaries containing dataset parameters
        batch_size (int): Size of each batch
        num_workers (int): Number of worker processes
        augment (bool): Whether to apply data augmentation
        target_size (int): Target size for image resizing
        
    Returns:
        DataLoader: Configured DataLoader instance
    """
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ])

    datasets = []
    for params in dataset_params:
        dataset = UpdatedImageStackDataset(
            root_dir=params['root_dir'],
            continuous_depth_dir=params['continuous_depth_dir'],
            transform=transform,
            augment=augment,
            subset_fraction=params['subset_fraction']
        )
        datasets.append(dataset)

    combined_dataset = CombinedDataset(datasets)

    sampler = GroupedBatchSampler(combined_dataset.stack_sizes, batch_size)

    dataloader = DataLoader(combined_dataset, batch_sampler=sampler, num_workers=num_workers)
    return dataloader

from torch.utils.data import ConcatDataset

class CombinedDataset(ConcatDataset):
    """Combined dataset that concatenates multiple datasets and maintains stack sizes."""
    
    def __init__(self, datasets):
        """
        Initialize the combined dataset.
        
        Args:
            datasets (list): List of Dataset instances to combine
        """
        super(CombinedDataset, self).__init__(datasets)
        self.stack_sizes = []
        for dataset in datasets:
            self.stack_sizes.extend(dataset.stack_sizes)

    def __getitem__(self, idx):
        """Get an item from the combined dataset."""
        return super(CombinedDataset, self).__getitem__(idx)