# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University

"""
Training script for StackMFF V3 Star model for depth estimation from focal stacks.
"""

import argparse
import time
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from tqdm import tqdm
from Dataloader_star import get_updated_dataloader
from network_star import StackMFF_V3_Star
from utils import (to_image, count_parameters, config_model_dir, 
                   print_banner, print_model_info, print_device_info, 
                   print_dataset_info, print_training_config, 
                   print_epoch_results, print_training_complete)

def parse_args():
    """Parse command line arguments for the training script."""
    # Dataset configuration
    parser = argparse.ArgumentParser(description="StackMFF V3 Star Training Script")
    parser.add_argument('--save_name', default='train_runs_star')
    parser.add_argument('--datasets_root', 
                        default='training_datasets',
                        type=str, help='Root path to all datasets')

    parser.add_argument('--train_datasets', nargs='+', 
                        default=['DUTS', 'DIODE', 'Cityscapes', 'NYU-V2', 'ADE'],
                        help='List of datasets to use for training')
    parser.add_argument('--val_datasets', nargs='+',
                        default=['DUTS', 'DIODE', 'Cityscapes', 'NYU-V2', 'ADE'],
                        help='List of datasets to use for validation')
    parser.add_argument('--subset_fraction_train', type=float, default=1.0,
                        help='Fraction of training data to use')
    parser.add_argument('--subset_fraction_val', type=float, default=0.1,
                        help='Fraction of validation data to use')

    # Training and model configuration
    parser.add_argument('--training_image_size', type=int, default=256,
                        help='Target image size for training')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=10, 
                        help='Interval of epochs between evaluations')
    parser.add_argument('--lr', type=float, default=1e-3, 
                        help='Initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.9, 
                        help='Learning rate decay factor')
    parser.add_argument('--num_workers', type=int, default=8, 
                        help='Number of data loading workers')
    
    # Device configuration
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help='Specific GPU IDs to use (e.g., 0 1 for GPU 0 and 1). If not specified, use all available GPUs in parallel')
    
    return parser.parse_args()

def create_dataset_loaders(args):
    """Create training and validation data loaders."""
    # Create training dataset parameters
    train_dataset_params = []
    for dataset_name in args.train_datasets:
        dataset_path = os.path.join(args.datasets_root, dataset_name, 'TR')
        if os.path.exists(dataset_path):
            train_dataset_params.append({
                'root_dir': os.path.join(dataset_path, 'dof_stack'),
                'continuous_depth_dir': os.path.join(dataset_path, 'depth'),
                'subset_fraction': args.subset_fraction_train
            })
        else:
            print(f"⚠️  Warning: Training dataset path not found: {dataset_path}")
    
    # Create training data loader
    train_loader = None
    if train_dataset_params:
        train_loader = get_updated_dataloader(
            train_dataset_params,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment=True,
            target_size=args.training_image_size
        )
    
    # Create validation dataset loaders (separate for each dataset)
    val_loaders = []
    for dataset_name in args.val_datasets:
        dataset_path = os.path.join(args.datasets_root, dataset_name, 'TE')
        if os.path.exists(dataset_path):
            val_loader = get_updated_dataloader(
                [{
                    'root_dir': os.path.join(dataset_path, 'dof_stack'),
                    'continuous_depth_dir': os.path.join(dataset_path, 'depth'),
                    'subset_fraction': args.subset_fraction_val
                }],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                augment=False,
                target_size=args.training_image_size
            )
            val_loaders.append(val_loader)
        else:
            print(f"⚠️  Warning: Validation dataset path not found: {dataset_path}")
    
    return train_loader, val_loaders

def train(model, train_loader, criterion_depth, optimizer, device, epoch, total_epochs):
    """Train one epoch."""
    model.train()
    train_loss = 0.0
    loss_depth_total = 0.0
    

    progress_bar = tqdm(
        train_loader, 
        desc=f"🔥 Epoch {epoch+1}/{total_epochs}",
        ncols=120,
        bar_format='{l_bar}{bar:20}{r_bar}'
    )

    for batch_idx, (image_stack, depth_map_gt, stack_size) in enumerate(progress_bar):
        image_stack, depth_map_gt = image_stack.to(device), depth_map_gt.to(device)

        optimizer.zero_grad()
        _, depth_map, _ = model(image_stack)

        loss_depth = criterion_depth(depth_map, depth_map_gt)

        loss_depth.backward()
        optimizer.step()

        train_loss += loss_depth.item()
        loss_depth_total += loss_depth.item()


        progress_bar.set_postfix({
            "Loss": f"{loss_depth.item():.4f}",
            "Avg": f"{train_loss/(batch_idx+1):.4f}",
            "Device": device.type.upper()
        })

    return (train_loss / len(train_loader),
            loss_depth_total / len(train_loader))

def validate_dataset(model, val_loader, criterion_depth, device, epoch, save_path, dataset_name):
    """Validate model on one dataset."""
    model.eval()
    val_loss = 0.0
    loss_depth_total = 0.0

    depth_mse_metric = MeanSquaredError().to(device)
    depth_mae_metric = MeanAbsoluteError().to(device)

    total_depth_mse = 0.0
    total_depth_mae = 0.0
    

    os.makedirs(save_path, exist_ok=True)

    progress_bar = tqdm(
        val_loader, 
        desc=f"📊 Validating {dataset_name}",
        ncols=120,
        bar_format='{l_bar}{bar:30}{r_bar}',
        colour='blue'
    )

    with torch.no_grad():
        for i, (image_stack, depth_map_gt, stack_size) in enumerate(progress_bar):
            image_stack, depth_map_gt = image_stack.to(device), depth_map_gt.to(device)

            _, depth_map, _ = model(image_stack)

            loss_depth = criterion_depth(depth_map, depth_map_gt)

            val_loss += loss_depth.item()
            loss_depth_total += loss_depth.item()

            depth_mse = depth_mse_metric(depth_map, depth_map_gt)
            depth_mae = depth_mae_metric(depth_map, depth_map_gt)

            total_depth_mse += depth_mse.item()
            total_depth_mae += depth_mae.item()

            progress_bar.set_postfix({
                "📉 Loss": f"{loss_depth.item():.6f}",
                "🎯 MSE": f"{depth_mse.item():.6f}",
                "📊 MAE": f"{depth_mae.item():.6f}"
            })

            if i == len(val_loader) - 1:
                visualization_path = os.path.join(save_path, f'epoch_{epoch}')
                to_image(depth_map_gt, epoch, 'depth_map_gt', visualization_path)
                to_image(depth_map, epoch, 'depth_map', visualization_path)

    num_batches = len(val_loader)
    avg_depth_mse = total_depth_mse / num_batches
    avg_depth_mae = total_depth_mae / num_batches

    return (val_loss / num_batches,
            loss_depth_total / num_batches,
            avg_depth_mse, avg_depth_mae)

def main():
    """Main training function."""
    args = parse_args()
    
    # Print banner information
    print_banner()
    
    # Initialization
    model_save_path = config_model_dir(resume=False, subdir_name=args.save_name)
    writer = SummaryWriter(log_dir=model_save_path)
    train_loader, val_loaders = create_dataset_loaders(args)

    # Create model
    model = StackMFF_V3_Star()
    num_params = count_parameters(model)
    print_model_info(model, num_params)
    
    # Device configuration
    use_parallel = False
    gpu_count = 0
    
    if args.gpu_ids is not None:
        # Ensure gpu_ids is always a list
        if isinstance(args.gpu_ids, int):
            args.gpu_ids = [args.gpu_ids]
        
        # Use specific GPUs
        if torch.cuda.is_available():
            # Validate GPU IDs
            valid_gpu_ids = []
            for gpu_id in args.gpu_ids:
                if gpu_id < torch.cuda.device_count():
                    valid_gpu_ids.append(gpu_id)
                else:
                    print(f"⚠️  Warning: GPU {gpu_id} not available, skipping...")
            
            if valid_gpu_ids:
                if len(valid_gpu_ids) == 1:
                    # Single specific GPU
                    device = torch.device(f"cuda:{valid_gpu_ids[0]}")
                    model.to(device)
                    use_parallel = False
                    gpu_count = 1
                    print(f"🔧 Using single GPU: {valid_gpu_ids[0]}")
                else:
                    # Multiple specific GPUs
                    device = torch.device(f"cuda:{valid_gpu_ids[0]}")
                    model.to(device)
                    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, valid_gpu_ids))
                    model = nn.DataParallel(model, device_ids=list(range(len(valid_gpu_ids))))
                    use_parallel = True
                    gpu_count = len(valid_gpu_ids)
                    print(f"🔧 Using multiple GPUs: {valid_gpu_ids}")
            else:
                print("⚠️  No valid GPUs specified, falling back to CPU")
                device = torch.device("cpu")
                model.to(device)
        else:
            print("⚠️  CUDA not available, using CPU")
            device = torch.device("cpu")
            model.to(device)
    else:
        # Use all available GPUs with DataParallel (default behavior)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            use_parallel = True
            gpu_count = torch.cuda.device_count()
            print(f"🔧 Using all available GPUs: {list(range(gpu_count))}")
        elif torch.cuda.is_available():
            gpu_count = 1
            print(f"🔧 Using single GPU: 0")
        else:
            print(f"🔧 Using CPU")
    
    print_device_info(device, use_parallel, gpu_count)
    print_dataset_info(train_loader, val_loaders, args)

    # Loss functions, optimizer, and scheduler
    criterion_depth = torch.nn.MSELoss()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)
    
    print_training_config(args, optimizer, scheduler)

    # Training variables
    best_val_loss = float('inf')
    best_epoch = -1
    start_time = time.time()
    val_results_data = []

    # Training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        # Training
        train_loss = None
        train_depth_loss = None
        if train_loader:
            train_loss, train_depth_loss = train(model, train_loader, criterion_depth, optimizer, device, epoch, args.num_epochs)

            writer.add_scalar('Loss/train/total', train_loss, epoch)
            writer.add_scalar('Loss/train/depth_loss', train_depth_loss, epoch)

        # Validation phase
        val_results = []
        epoch_val_data = {'epoch': epoch + 1}
        for i, val_loader in enumerate(val_loaders, 1):
            dataset_name = args.val_datasets[i-1] if i-1 < len(args.val_datasets) else f"dataset_{i}"
            results = validate_dataset(model, val_loader, criterion_depth, device, epoch, 
                                     os.path.join(model_save_path, f'val_{dataset_name}'), 
                                     dataset_name)
            val_results.append(results)

            (val_loss, val_depth_loss, avg_depth_mse, avg_depth_mae) = results

            writer.add_scalar(f'Loss/val_dataset_{i}/total', val_loss, epoch)
            writer.add_scalar(f'Loss/val_dataset_{i}/depth_loss', val_depth_loss, epoch)

            writer.add_scalar(f'Metrics/val_dataset_{i}/Depth_MSE', avg_depth_mse, epoch)
            writer.add_scalar(f'Metrics/val_dataset_{i}/Depth_MAE', avg_depth_mae, epoch)

            epoch_val_data.update({
                f'val_dataset_{i}_loss': val_loss,
                f'val_dataset_{i}_depth_loss': val_depth_loss,
                f'val_dataset_{i}_depth_mse': avg_depth_mse,
                f'val_dataset_{i}_depth_mae': avg_depth_mae
            })

            print(f"Validation Dataset {i} Results:")
            print(f"  Loss: {val_loss:.6f}")
            print(f"  Depth MSE: {avg_depth_mse:.6f}")
            print(f"  Depth MAE: {avg_depth_mae:.6f}")

        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0])

        if train_loader:
            epoch_val_data.update({
                'train_loss': train_loss,
                'learning_rate': scheduler.get_last_lr()[0]
            })

        val_results_data.append(epoch_val_data)

        # Save validation results to CSV after each epoch
        val_results_df = pd.DataFrame(val_results_data)
        val_results_df.to_csv(os.path.join(model_save_path, 'validation_results.csv'), index=False)

        # Save model checkpoint
        os.makedirs(os.path.join(model_save_path, 'model_save'), exist_ok=True)
        torch.save(model.state_dict(), f"{os.path.join(model_save_path, 'model_save')}/epoch_{epoch}.pth")

        # Check for best model and save
        improved = False
        if val_loaders:
            avg_val_loss = sum(results[0] for results in val_results) / len(val_results)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                improved = True
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save(state_dict, f"{model_save_path}/best_model.pth")
        
        dataset_names = args.val_datasets[:len(val_loaders)]
        train_loss_val = train_loss if 'train_loss' in locals() else None
        
        # Format validation results for display
        formatted_val_results = []
        for results in val_results:
            val_loss, val_depth_loss, avg_depth_mse, avg_depth_mae = results
            formatted_val_results.append((val_loss, val_depth_loss, avg_depth_mse))
        
        print_epoch_results(epoch, args.num_epochs, train_loss_val, formatted_val_results, 
                           dataset_names, scheduler.get_last_lr()[0], best_val_loss, improved,
                           metric_name="Depth MSE", metric_format="{:.6f}")

        scheduler.step()

    # Training complete
    print_training_complete(start_time, model_save_path)
    
    # Print best epoch information
    if best_epoch >= 0:
        best_model_path = os.path.join(model_save_path, "best_model.pth")
        print(f"\n🏆 Best Model Information:")
        print(f"   📊 Best Epoch: {best_epoch + 1}/{args.num_epochs}")
        print(f"   📉 Best Validation Loss: {best_val_loss:.6f}")
        print(f"   💾 Best Model Path: {best_model_path}")
    else:
        print(f"\n⚠️  No validation performed, no best model selected.")
    
    writer.close()

if __name__ == "__main__":
    main()