# -*- coding: utf-8 -*-
"""
Benchmark inference with hard selection on grey images (the per-epoch monitoring protocol of train.py).

Protocol:
  * the stack is reflect-padded (bottom/right) to a multiple of 64 for the forward pass and the
    focus map is cropped back. No resizing (input_mode='pad'; 'resize' is the protocol of the
    earlier StackMFF V4 code and is kept for reference only).
  * hard fusion: fused = gather(stack, argmax image) on the uint8 grey stack;
  * grey: cv2 BGR->GRAY for both stacks and ground truth;
  * PSNR (data_range=255) and SSIM (skimage defaults, data_range=255) per image.
The numbers of the paper are computed from the colour results with evaluation/run_eval.py instead.

Ground truth: <test_root>/<dataset>/AiF/<stack name>.(png|jpg|...).
Stacks: <test_root>/<dataset>/image stack/<stack name>/*.(png|jpg|jpeg).

run_benchmark() is imported by train.py for per-epoch monitoring.

  python predict_benchmark.py --test_root <benchmarks> --out_dir bench_out [--ckpt weights.pth | --run_dir R --ckpt epoch_2.pth]
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DEFAULT_DATASETS = ['Mobile Depth', 'FlyingThings3D', 'Middlebury', 'Road-MF']
STACK_EXT = ('.png', '.jpg', '.jpeg')
GT_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def sort_key(filename):
    numbers = re.findall(r'\d+\.?\d*', filename)
    return (float(numbers[0]) if numbers else 0.0, filename)


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'cannot read {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def list_stacks(stack_root):
    out = []
    if not os.path.isdir(stack_root):
        return out
    for name in sorted(os.listdir(stack_root)):
        d = os.path.join(stack_root, name)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.lower().endswith(STACK_EXT)]
        files = sorted(files, key=sort_key)
        if len(files) >= 2:
            out.append((name, [os.path.join(d, f) for f in files]))
    return out


def index_gt(gt_dir):
    m = {}
    if not os.path.isdir(gt_dir):
        return m
    for f in sorted(os.listdir(gt_dir)):
        stem, ext = os.path.splitext(f)
        if ext.lower() in GT_EXT:
            m.setdefault(stem, os.path.join(gt_dir, f))
    return m


def pad_to_64(x):
    H, W = x.shape[-2:]
    ph, pw = (-H) % 64, (-W) % 64
    if ph or pw:
        mode = 'reflect' if (ph < H and pw < W) else 'replicate'
        x = F.pad(x, (0, pw, 0, ph), mode=mode)
    return x


def resize_dims_v4(H, W):
    """Same rule as StackMFFV4/predict_datasets.py resize_to_multiple_of_32."""
    nh = ((H - 1) // 32 + 1) * 32
    nw = ((W - 1) // 32 + 1) * 32
    if nh % 64 != 0:
        nh = ((nh - 1) // 64 + 1) * 64
    if nw % 64 != 0:
        nw = ((nw - 1) // 64 + 1) * 64
    return nh, nw


@torch.no_grad()
def run_benchmark(model, test_root, datasets, device, save_root=None, log=print, limit=None,
                  input_mode='pad'):
    """Returns (summary: {dataset: {...}}, rows: [per-image dict])."""
    model.eval()
    summary, rows = {}, []
    sync = device.type == 'cuda'
    for ds in datasets:
        ds_root = os.path.join(test_root, ds)
        stacks = list_stacks(os.path.join(ds_root, 'image stack'))
        gts = index_gt(os.path.join(ds_root, 'AiF'))
        if not stacks:
            log(f'[bench] {ds}: no stacks found under {ds_root}')
            continue
        if save_root:
            os.makedirs(os.path.join(save_root, ds), exist_ok=True)
        psnrs, ssims, times = [], [], []
        n_missing_gt, n_size_mismatch, n_inf_psnr = 0, 0, 0
        for i, (name, paths) in enumerate(stacks):
            if limit is not None and i >= limit:
                break
            if name not in gts:
                n_missing_gt += 1
                continue
            stack = np.stack([read_gray(p) for p in paths])          # N, H, W uint8
            N, H, W = stack.shape
            x = torch.from_numpy(stack).float().div_(255.0).unsqueeze(0).to(device)
            if input_mode == 'pad':
                xin = pad_to_64(x)
            elif input_mode == 'resize':
                Hr, Wr = resize_dims_v4(H, W)
                xin = F.interpolate(x, size=(Hr, Wr), mode='bilinear', align_corners=False)
            else:
                raise ValueError(input_mode)
            if sync:
                torch.cuda.synchronize(device)
            t0 = time.time()
            _, focus = model(xin)
            if sync:
                torch.cuda.synchronize(device)
            dt = time.time() - t0
            if input_mode == 'pad':
                focus = focus[0, :H, :W].cpu().numpy().astype(np.int64)
            else:
                focus = cv2.resize(focus[0].cpu().numpy().astype(np.float32), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int64)
            fused = np.take_along_axis(stack, focus[None], axis=0)[0]  # H, W uint8
            gt = read_gray(gts[name])
            if gt.shape != fused.shape:
                n_size_mismatch += 1
                gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_LINEAR)
            psnr = float(peak_signal_noise_ratio(gt, fused, data_range=255))
            if not np.isfinite(psnr):
                n_inf_psnr += 1
                psnr = 100.0
            ssim = float(structural_similarity(gt, fused, data_range=255))
            psnrs.append(psnr)
            ssims.append(ssim)
            times.append(dt)
            rows.append({'dataset': ds, 'stack': name, 'N': N, 'H': H, 'W': W,
                         'psnr': round(psnr, 4), 'ssim': round(ssim, 5), 'time_s': round(dt, 4)})
            if save_root:
                cv2.imwrite(os.path.join(save_root, ds, name + '.png'), fused)
        n = len(psnrs)
        summary[ds] = {
            'n': n,
            'psnr': float(np.mean(psnrs)) if n else float('nan'),
            'ssim': float(np.mean(ssims)) if n else float('nan'),
            'sec_per_stack': float(np.mean(times)) if n else float('nan'),
            'missing_gt': n_missing_gt, 'gt_size_mismatch': n_size_mismatch, 'inf_psnr': n_inf_psnr,
        }
        log(f"[bench] {ds:15s} n={n:4d} PSNR={summary[ds]['psnr']:.3f} SSIM={summary[ds]['ssim']:.4f} "
            f"sec/stack={summary[ds]['sec_per_stack']:.3f} missing_gt={n_missing_gt} "
            f"size_mismatch={n_size_mismatch}")
    if summary:
        w = [(v['n'], v) for v in summary.values() if v['n']]
        tot = sum(n for n, _ in w)
        if tot:
            log(f"[bench] ALL n={tot} mean-of-datasets PSNR={np.mean([v['psnr'] for _, v in w]):.3f} "
                f"SSIM={np.mean([v['ssim'] for _, v in w]):.4f}")
    return summary, rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='', help='checkpoint (default: weights/stackmff_v3_final.pth); with --run_dir a '
                                              'file name in <run_dir>/model_save')
    p.add_argument('--run_dir', default='', help='rebuild the architecture from this training run')
    p.add_argument('--test_root', default=os.environ.get('STACKMFF_BENCH_ROOT', 'data/benchmarks'))
    p.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--save_images', action='store_true')
    p.add_argument('--limit', type=int, default=None, help='max stacks per dataset (debug)')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from stackmff_v3.model_io import DEFAULT_WEIGHTS, load
    ckpt = args.ckpt or DEFAULT_WEIGHTS
    model = load(args.run_dir, ckpt, device)
    save_root = os.path.join(args.out_dir, 'fused_gray') if args.save_images else None
    print(f'ckpt={ckpt} run_dir={args.run_dir or "-"}', flush=True)
    summary, rows = run_benchmark(model, args.test_root, args.datasets, device,
                                  save_root=save_root, limit=args.limit, input_mode='pad')
    with open(os.path.join(args.out_dir, 'per_image.csv'), 'w', newline='') as f:
        if rows:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump({'ckpt': ckpt, 'run_dir': args.run_dir, 'input_mode': 'pad', 'summary': summary}, f, indent=2)


if __name__ == '__main__':
    main()
