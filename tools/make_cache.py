# -*- coding: utf-8 -*-
"""Pre-resize the synthetic training/validation stacks into a uint8 cache (one .npz per stack).

Same preprocessing as the V4 Dataloader (Dataloader.py): Y channel of PIL 'YCbCr', torchvision
Resize((S, S)) (bilinear, antialias) for the layers, nearest for the focus-index map. Flips are applied
at training time. Layout: <out>/<dataset>/<split>/<stack>.npz with arrays x (N,S,S) uint8, y (S,S) uint8.

  python make_cache.py --size 256 --out <root>/v3merge_work/cache256 --workers 48
"""
import argparse, os, re
from multiprocessing import Pool
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

ROOT = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/stackmffv3_training_datasets'
DATASETS = ['NYU-V2', 'DUTS', 'DIODE', 'Cityscapes', 'ADE']
EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def jobs(out, size):
    for ds in DATASETS:
        for split in ('TR', 'TE'):
            base = os.path.join(ROOT, ds, split)
            sd = os.path.join(base, 'focus_stack')
            if not os.path.isdir(sd):
                continue
            for name in sorted(os.listdir(sd)):
                gt = os.path.join(base, 'focus_index_gt', name + '.npy')
                d = os.path.join(sd, name)
                if os.path.isdir(d) and os.path.exists(gt):
                    yield d, gt, os.path.join(out, ds, split, name + '.npz'), size


def work(a):
    d, gt, dst, size = a
    if os.path.exists(dst):
        return 0
    torch.set_num_threads(1)
    files = sorted([f for f in os.listdir(d) if f.lower().endswith(EXT)], key=nat)
    rs = transforms.Resize((size, size))
    x = np.stack([np.asarray(rs(Image.open(os.path.join(d, f)).convert('YCbCr').split()[0])) for f in files])
    y = np.load(gt)
    y = np.clip(y, 0, len(files) - 1)
    y = TF.resize(torch.from_numpy(y).unsqueeze(0).float(), [size, size],
                  interpolation=transforms.InterpolationMode.NEAREST).squeeze(0).numpy().astype(np.uint8)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.tmp.npz'
    np.savez(tmp, x=x.astype(np.uint8), y=y)
    os.replace(tmp, dst)
    return 1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=48)
    a = ap.parse_args()
    js = list(jobs(a.out, a.size))
    print(len(js), 'stacks', flush=True)
    n = 0
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(work, js, chunksize=16)):
            n += r
            if (i + 1) % 2000 == 0:
                print(i + 1, 'processed', flush=True)
    print('done, new', n, flush=True)
