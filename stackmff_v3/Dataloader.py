# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

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
                # 排除layer_order.npy文件，只加载图像文件
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
            image = image.split()[0]  # 只保留 Y 通道
            images.append(image)

        # 加载焦点索引图（.npy格式）
        focus_index_gt = np.load(focus_index_map_path)  # [H, W] 格式，np.int64
        
        # 验证焦点索引的有效性：索引必须在 [0, len(images)-1] 范围内
        max_index = len(images) - 1
        if focus_index_gt.max() > max_index or focus_index_gt.min() < 0:
            print(f"Warning: 焦点索引超出有效范围 [0, {max_index}]: 实际范围 [{focus_index_gt.min()}, {focus_index_gt.max()}]")
            focus_index_gt = np.clip(focus_index_gt, 0, max_index)
            
        focus_index_gt = torch.from_numpy(focus_index_gt).long()  # 转换为torch.long

        if self.augment:
            images, focus_index_gt = self.consistent_transform(images, focus_index_gt)

        # 应用其他变换
        if self.transform:
            images = [self.transform(img) for img in images]
            # 对焦点索引图进行相同的尺寸调整以确保一致性
            # 获取目标尺寸
            target_size = None
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    target_size = t.size
                    break
            
            if target_size is not None:
                # 调整焦点索引图尺寸，使用最近邻插值保持索引值不变
                focus_index_gt = TF.resize(
                    focus_index_gt.unsqueeze(0).float(), 
                    target_size, 
                    interpolation=transforms.InterpolationMode.NEAREST
                ).squeeze(0).long()
            
        # 转换为张量并移除通道维度
        images = [img.squeeze(0) for img in images]
        stack_tensor = torch.stack(images)  # 形状将是 (N, H, W)

        return stack_tensor, focus_index_gt, len(images)

    def consistent_transform(self, images, focus_index_gt):
        """
        Apply consistent transformations to both images and focus index map.
        Includes random horizontal and vertical flips.
        """
        # 随机水平翻转
        if random.random() > 0.5:
            images = [TF.hflip(img) for img in images]
            focus_index_gt = TF.hflip(focus_index_gt.unsqueeze(0)).squeeze(0)  # 临时添加维度进行翻转

        # 随机垂直翻转
        if random.random() > 0.5:
            images = [TF.vflip(img) for img in images]
            focus_index_gt = TF.vflip(focus_index_gt.unsqueeze(0)).squeeze(0)  # 临时添加维度进行翻转

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
            focus_index_dir=params['focus_index_gt'],  # 使用正确的参数名
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