#!/bin/bash
# Night run 1 (2026-10-14): two architecture sizes in parallel, one per GPU, cached 256x256 data.
# Model selection: best mean SSIM on the 4 benchmarks (best_bench.pth), per user decision.
W=/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/v3merge_work
cd $W/new_model/code
source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate xxz
COMMON="--cache_root $W/cache256 --batch_size 8 --num_workers 12 --val_workers 4 --subset_fraction_val 0.05 --log_every 500"
CUDA_VISIBLE_DEVICES=0 setsid nohup python train.py --out_dir $W/runs/A_c32_r121 --C1 32 --C2 64 --heads 2 4 --rounds 1 2 1 --num_epochs 16 $COMMON > $W/runs/A_c32_r121.out 2>&1 < /dev/null &
CUDA_VISIBLE_DEVICES=1 setsid nohup python train.py --out_dir $W/runs/H_c48_r121_p2 --C1 48 --C2 96 --heads 3 6 --rounds 1 2 1 --inter_pool 2 --num_epochs 14 $COMMON > $W/runs/H_c48_r121_p2.out 2>&1 < /dev/null &
echo launched
