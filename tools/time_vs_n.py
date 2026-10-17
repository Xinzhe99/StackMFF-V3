# -*- coding: utf-8 -*-
"""Runtime and peak GPU memory of our model versus the stack size N (random 640x360 grey stacks, fp32, batch 1, idle
GPU, median of 5 runs after one warm-up), optionally with StackMFF V2 (official code and weights) for comparison.
  python tools/time_vs_n.py --out time_vs_n.csv --scales 1,0.75
  python tools/time_vs_n.py --out time_vs_n.csv --v2_dir <StackMFF-V2 checkout> --v2_ckpt <StackMFF-V2 weights>
"""
import argparse, csv, os, sys, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.model_io import DEFAULT_WEIGHTS, load
from predict_benchmark import pad_to_64


def timeit(fn, reps=5):
    fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    return float(np.median(ts)), torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_WEIGHTS); ap.add_argument('--run_dir', default='')
    ap.add_argument('--out', required=True)
    ap.add_argument('--Ns', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64])
    ap.add_argument('--scales', default='1,0.75', help='inference scales (1 = one pass)')
    ap.add_argument('--v2_dir', default='', help='StackMFF V2 repository (Stack_MFF_V2_Net_simple.py); empty: skip V2')
    ap.add_argument('--v2_ckpt', default='')
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False; torch.set_grad_enabled(False)
    H, W = 360, 640
    m = load(a.run_dir, a.ckpt, 'cuda')
    m.ms_scales = tuple(float(v) for v in a.scales.split(','))
    v2 = None
    if a.v2_dir:
        sys.path.insert(0, a.v2_dir)
        from Stack_MFF_V2_Net_simple import StackMFF_V2
        v2 = StackMFF_V2(); sd = torch.load(a.v2_ckpt, map_location='cpu')
        v2.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}); v2 = v2.cuda().eval()
    rows = []
    for N in a.Ns:
        x = torch.rand(1, N, H, W, device='cuda')
        t_o, m_o = timeit(lambda: (m.forward_probs(pad_to_64(x))[:, :, :H, :W] * x).sum(1))
        row = {'N': N, 'ours_s': t_o, 'ours_MiB': m_o}
        if v2 is not None:
            xv = F.interpolate(x, size=(384, 640), mode='bilinear', align_corners=False)
            try:
                row['v2_s'], row['v2_MiB'] = timeit(lambda: v2(xv))
            except RuntimeError:
                row['v2_s'], row['v2_MiB'] = float('nan'), float('nan'); torch.cuda.empty_cache()
        rows.append(row)
        print(row, flush=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]


if __name__ == '__main__':
    main()
