#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xinzhe Xie
# @University  : ZheJiang University

"""
This script generates multi-focus image stacks from single images and their depth maps.
It creates depth-based layers and generates randomized focus index maps. The process includes:
1. Depth map quantization into layers
2. Layer generation with varying blur levels
3. Layer order randomization
4. Focus index map generation (0 to N-1)
5. Data organization with npy format for focus indices
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


def generate_focus_layers_and_index(img, img_depth, num_regions):
    """
    Generate focus layers from depth map and create randomized focus index map.
    
    Args:
        img (numpy.ndarray): Input image
        img_depth (numpy.ndarray): Corresponding depth map
        num_regions (int): Number of depth regions/layers
    
    Returns:
        tuple: (list of focus layers, focus_index_gt map, layer_order_mapping)
    """
    # 创建不同模糊程度的图像层
    blur_kernels = [2 * i + 1 for i in range(num_regions)]
    blurred_images = []
    for kernel_size in blur_kernels:
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        blurred_images.append(blurred)
    
    # 深度图量化分层
    ref_points = np.linspace(0, 255, num_regions + 1)
    quantized_depth = np.digitize(img_depth, ref_points) - 1
    # 确保值在有效范围内
    quantized_depth = np.clip(quantized_depth, 0, num_regions - 1)
    
    # 为每个深度层创建对应的聚焦图像
    focus_layers = []
    for focus_layer_idx in range(num_regions):
        layer_result = np.zeros_like(img)
        
        for depth_level in range(num_regions):
            mask = (quantized_depth == depth_level)
            # 计算该深度层相对于当前聚焦层的模糊程度
            blur_distance = abs(depth_level - focus_layer_idx)
            blur_idx = min(blur_distance, len(blurred_images) - 1)
            layer_result[mask] = blurred_images[blur_idx][mask]
        
        # 填充可能的黑色区域
        black_mask = np.all(layer_result == 0, axis=2)
        layer_result[black_mask] = img[black_mask]
        
        focus_layers.append(layer_result)
    
    # 创建原始顺序到打乱顺序的映射
    original_order = list(range(num_regions))
    shuffled_order = original_order.copy()
    random.shuffle(shuffled_order)
    
    # 创建从原始深度层索引到新顺序索引的映射
    depth_to_new_index = {original_idx: new_idx for new_idx, original_idx in enumerate(shuffled_order)}
    
    # 根据打乱的顺序重新排列图层
    shuffled_layers = [focus_layers[i] for i in shuffled_order]
    
    # 创建焦点索引图：将原始深度索引映射到新的图层顺序索引
    focus_index_gt = np.zeros_like(quantized_depth, dtype=np.int64)
    for original_depth_idx in range(num_regions):
        mask = (quantized_depth == original_depth_idx)
        new_layer_idx = depth_to_new_index[original_depth_idx]
        focus_index_gt[mask] = new_layer_idx
    
    return shuffled_layers, focus_index_gt, shuffled_order


def process_single_image(args):
    """
    Process a single image for multi-threading.
    
    Args:
        args (tuple): (pic_path, depth_path, output_path, num_regions_list)
    
    Returns:
        tuple: (success, image_name, message)
    """
    pic_path, depth_path, output_path, num_regions_list = args
    
    try:
        filename = os.path.basename(pic_path)
        name, _ = os.path.splitext(filename)

        img = cv2.imread(pic_path)
        img_depth_path = os.path.join(depth_path, name + '.png')
        
        if not os.path.exists(img_depth_path):
            return False, name, f"Depth map not found: {img_depth_path}"
            
        img_depth = cv2.imread(img_depth_path, 0)
        
        if img is None or img_depth is None:
            return False, name, f"Failed to load image or depth map"

        num_regions = random.choice(num_regions_list)
        focus_layers, focus_index_gt, layer_order = generate_focus_layers_and_index(img, img_depth, num_regions)

        # Create separate folders for data organization
        original_folder = os.path.join(output_path, 'AiF')
        depth_folder = os.path.join(output_path, 'depth')
        focus_stack_folder = os.path.join(output_path, 'focus_stack')
        focus_index_folder = os.path.join(output_path, 'focus_index_gt')

        for folder in [original_folder, depth_folder, focus_stack_folder, focus_index_folder]:
            os.makedirs(folder, exist_ok=True)

        # Save the original image
        cv2.imwrite(os.path.join(original_folder, f'{name}.jpg'), img)

        # Save the depth image
        cv2.imwrite(os.path.join(depth_folder, f'{name}.png'), img_depth)

        # Save the focus index ground truth as npy file
        np.save(os.path.join(focus_index_folder, f'{name}.npy'), focus_index_gt)

        # Save the focus stack images
        focus_image_folder = os.path.join(focus_stack_folder, name)
        os.makedirs(focus_image_folder, exist_ok=True)
        for i, layer in enumerate(focus_layers):
            cv2.imwrite(os.path.join(focus_image_folder, f'{i}.jpg'), layer)
        
        # Save layer order mapping for reference
        order_info = {
            'original_order': list(range(num_regions)),
            'shuffled_order': layer_order,
            'num_layers': num_regions
        }
        np.save(os.path.join(focus_image_folder, 'layer_order.npy'), order_info)
        
        return True, name, f"Successfully processed {num_regions} layers"
        
    except Exception as e:
        return False, name, f"Error processing: {str(e)}"


def process_images(original_path, depth_path, output_path, max_workers=None):
    """
    Process a set of images and their depth maps to create multi-focus image stacks using multi-threading.
    
    Args:
        original_path (str): Path to original images
        depth_path (str): Path to depth maps
        output_path (str): Path to save generated data
        max_workers (int): Maximum number of worker threads. If None, uses CPU count.
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)  # 限制最大线程数避免过载
    
    num_regions_list = list(range(2, 25))  # 2到24的所有数量
    os.makedirs(output_path, exist_ok=True)

    # 获取所有图像文件
    original_images = glob.glob(os.path.join(original_path, '*.jpg')) + glob.glob(os.path.join(original_path, '*.png'))
    
    if not original_images:
        print(f"No images found in {original_path}")
        return
    
    print(f"Found {len(original_images)} images to process")
    print(f"Using {max_workers} worker threads")
    
    # 准备参数列表
    task_args = [(pic_path, depth_path, output_path, num_regions_list) for pic_path in original_images]
    
    # 统计结果
    success_count = 0
    failed_count = 0
    failed_images = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_path = {executor.submit(process_single_image, args): args[0] for args in task_args}
        
        # 使用tqdm显示进度
        with tqdm(total=len(original_images), desc="Processing images") as pbar:
            for future in as_completed(future_to_path):
                pic_path = future_to_path[future]
                try:
                    success, name, message = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_count += 1
                        failed_images.append((name, message))
                        print(f"\nFailed: {message}")
                except Exception as e:
                    failed_count += 1
                    failed_images.append((os.path.basename(pic_path), str(e)))
                    print(f"\nException: {str(e)}")
                finally:
                    pbar.update(1)
    
    # 打印处理结果统计
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed: {success_count} images")
    print(f"Failed: {failed_count} images")
    
    if failed_images:
        print("\nFailed images:")
        for name, error in failed_images:
            print(f"  - {name}: {error}")


if __name__ == "__main__":
    original_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/NYU-V2/NYU-OURS-TR/AiF"
    depth_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/NYU-V2/NYU-OURS-TR/depth"
    output_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/stackmffv3_training_datasets/NYU-V2/TR"

    # 你可以通过调整max_workers参数来控制并行线程数
    # None表示自动使用CPU核心数（最多8个）
    # 也可以手动指定，如 max_workers=4
    process_images(original_path, depth_path, output_path, max_workers=8)

    original_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/dff_and_mff/NYU-OURS-TE/AiF"
    depth_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/dff_and_mff/NYU-OURS-TE/depth"
    output_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/stackmffv3_training_datasets/NYU-V2/TE"

    # 你可以通过调整max_workers参数来控制并行线程数
    # None表示自动使用CPU核心数（最多8个）
    # 也可以手动指定，如 max_workers=4
    process_images(original_path, depth_path, output_path, max_workers=8)

    original_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/ADE/ADE-TR"
    depth_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/ADE/ADE-TR-Depth"
    output_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/stackmffv3_training_datasets/ADE/TR"

    # 你可以通过调整max_workers参数来控制并行线程数
    # None表示自动使用CPU核心数（最多8个）
    # 也可以手动指定，如 max_workers=4
    process_images(original_path, depth_path, output_path, max_workers=8)

    original_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/ADE/ADE-TE"
    depth_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/ADE/ADE-TE-Depth"
    output_path = r"/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/stackmffv3_training_datasets/ADE/TE"

    # 你可以通过调整max_workers参数来控制并行线程数
    # None表示自动使用CPU核心数（最多8个）
    # 也可以手动指定，如 max_workers=4
    process_images(original_path, depth_path, output_path, max_workers=8)