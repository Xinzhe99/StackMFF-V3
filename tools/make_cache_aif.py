# -*- coding: utf-8 -*-
"""Cache (grey AiF, depth) pairs of the training sources, resized as a whole to 320x320 (like the original
whole-image resize policy), for on-the-fly stack synthesis during training.
  python make_cache_aif.py --out cache_aif320 --workers 24
"""
import argparse, os, glob
from multiprocessing import Pool
import cv2, numpy as np

ROOT = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/stackmffv3_training_datasets'
S = 320


def job(a):
    aif, dep, out = a
    if os.path.exists(out):
        return 1
    im = cv2.imread(aif, cv2.IMREAD_GRAYSCALE); d = cv2.imread(dep, cv2.IMREAD_GRAYSCALE)
    if im is None or d is None:
        return 0
    im = cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA)
    d = cv2.resize(d, (S, S), interpolation=cv2.INTER_NEAREST)
    np.savez(out, aif=im, depth=d)
    return 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    jobs = []
    for ds in ['NYU-V2', 'DUTS', 'DIODE', 'Cityscapes', 'ADE']:
        for sp in ['TR', 'TE']:
            od = os.path.join(a.out, ds, sp); os.makedirs(od, exist_ok=True)
            for f in glob.glob(os.path.join(ROOT, ds, sp, 'AiF', '*')):
                n = os.path.splitext(os.path.basename(f))[0]
                dp = os.path.join(ROOT, ds, sp, 'depth', n + '.png')
                if os.path.exists(dp):
                    jobs.append((f, dp, os.path.join(od, n + '.npz')))
    print('jobs', len(jobs), flush=True)
    with Pool(a.workers) as p:
        ok = sum(p.imap_unordered(job, jobs, chunksize=64))
    print('done', ok, flush=True)


if __name__ == '__main__':
    main()
