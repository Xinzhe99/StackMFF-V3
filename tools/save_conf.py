# -*- coding: utf-8 -*-
"""Save, for selected benchmark scenes, the per-pixel confidence max_n P_n, the argmax index and the weighted
grey fusion (for the confidence / failure-case figure).
  python tools/save_conf.py --scenes "Mobile Depth/keyboard" "Middlebury/Jadeplant" --out conf_maps.npz --bench_root <benchmarks>
"""
import argparse, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.model_io import DEFAULT_WEIGHTS, load
from predict_benchmark import list_stacks, read_gray, pad_to_64

ROOT = os.environ.get('STACKMFF_BENCH_ROOT', 'data/benchmarks')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_WEIGHTS); ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--run_dir', default='')
    ap.add_argument('--bench_root', default=ROOT)
    ap.add_argument('--sigma', type=float, default=float(os.environ.get('SIGMA', 0.5)), help='smoothing of P (paper: 0.5)')
    a = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    m = load(a.run_dir, a.ckpt, dev)
    out = {}
    for sc in a.scenes:
        ds, name = sc.split('/')
        paths = dict(list_stacks(os.path.join(a.bench_root, ds, 'image stack')))[name]
        st = np.stack([read_gray(p) for p in paths]).astype(np.float32)
        N, H, W = st.shape
        with torch.no_grad():
            P = m.forward_probs(pad_to_64(torch.from_numpy(st / 255).unsqueeze(0).to(dev)), sigma=a.sigma)[0, :, :H, :W].float().cpu().numpy()
        key = sc.replace(' ', '_').replace('/', '__')
        out[key + '__conf'] = P.max(0).astype(np.float16)
        out[key + '__idx'] = P.argmax(0).astype(np.uint8)
        out[key + '__soft'] = np.clip((P * st).sum(0), 0, 255).astype(np.uint8)
    np.savez_compressed(a.out, **out)
    print('saved', list(out))


if __name__ == '__main__':
    main()
