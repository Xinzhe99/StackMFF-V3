# -*- coding: utf-8 -*-
"""Save, for selected benchmark scenes, the per-pixel confidence max_n P_n, the argmax index and the weighted
fusion luma (Pillow BT.601 from the rounded RGB output, as in the paper's evaluation).
  python tools/save_conf.py --scenes "Middlebury/Motorcycle" "Mobile Depth/keyboard" --out conf_maps.npz --bench_root <benchmarks>
"""
import argparse, os, sys
import numpy as np, torch
import cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.model_io import DEFAULT_WEIGHTS, load
from stackmff_v3.network import smooth_probs
from predict_benchmark import list_stacks, pad_to_64

ROOT = os.environ.get('STACKMFF_BENCH_ROOT', 'data/benchmarks')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_WEIGHTS); ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', nargs='+', required=True)
    ap.add_argument('--run_dir', default='')
    ap.add_argument('--bench_root', default=ROOT)
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--sigma', type=float, default=float(os.environ.get('SIGMA', 0.5)), help='smoothing of P (paper: 0.5)')
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.set_num_threads(a.threads)
    m = load(a.run_dir, a.ckpt, dev)
    out = {}
    for sc in a.scenes:
        ds, name = sc.split('/')
        paths = dict(list_stacks(os.path.join(a.bench_root, ds, 'image stack')))[name]
        bgr = np.stack([cv2.imread(p) for p in paths])
        st = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in bgr]).astype(np.float32)
        N, H, W = st.shape
        with torch.no_grad():
            P = m.forward_probs(pad_to_64(torch.from_numpy(st / 255).unsqueeze(0).to(dev)))[:, :, :H, :W]
            P = smooth_probs(P, a.sigma)[0].float().cpu().numpy()
        key = sc.replace(' ', '_').replace('/', '__')
        out[key + '__conf'] = P.max(0).astype(np.float32)
        out[key + '__idx'] = P.argmax(0).astype(np.uint8)
        fused = np.round((P[..., None] * bgr.astype(np.float32)).sum(0)).clip(0, 255).astype(np.uint8)
        out[key + '__soft'] = np.array(Image.fromarray(fused[..., ::-1]).convert('L'))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, **out)
    print('saved', list(out))


if __name__ == '__main__':
    main()
