#!/bin/bash
# Training command of the final model (paper): equal to the launcher used for the reported run, with the server paths
# replaced by placeholders. Two GPUs, batch 8 per GPU, 6 epochs (about 14 min per epoch on two RTX A6000).
# All flags below are also the defaults of train.py; they are spelled out for clarity.
#   DATA   : training sources, <DATA>/<NYU-V2|DUTS|DIODE|Cityscapes|ADE>/<TR|TE>/{AiF,depth,focus_stack,focus_index_gt}
#   CACHE  : output of tools/make_cache.py --size 256 (validation stacks)
#   PAIRS  : output of tools/make_cache_aif.py (320x320 all-in-focus/depth pairs for on-the-fly synthesis)
#   BENCH  : benchmarks, <BENCH>/<Mobile Depth|Middlebury|FlyingThings3D|Road-MF>/{image stack,AiF} (monitoring only)
set -e
DATA=${DATA:-/path/to/stackmffv3_training_datasets}
CACHE=${CACHE:-/path/to/cache256}
PAIRS=${PAIRS:-/path/to/cache_aif320}
BENCH=${BENCH:-/path/to/benchmarks}
RUN=${RUN:-runs/final}
cd "$(dirname "$0")/.."
mkdir -p "$(dirname "$RUN")"
torchrun --nproc_per_node=2 --master_port=29519 train.py --out_dir $RUN --datasets_root $DATA \
    --arch loop --C1 48 --C2 96 --heads 3 6 --inter_pool 2 --kmax 1 --k_eval 1 --single_scale \
    --cache_root $CACHE --synth_root $PAIRS --subset_fraction_train 1.0 --subset_fraction_val 0.05 \
    --num_epochs 6 --batch_size 8 --num_workers 10 --val_workers 4 --log_every 1000 --scale_aug 1.0 \
    --synth_hdr 0.5 --ema 0.999 --w_spread 1 --bench_root $BENCH 2>&1 | tee $RUN.out
