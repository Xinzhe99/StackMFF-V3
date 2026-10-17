# -*- coding: utf-8 -*-
"""Fuse every stack below one or more dataset folders (benchmarks, robustness variants, real-data subsets).
Layout in:  <root>/<Dataset>/image stack/<scene>/<images>  (natural-sort order of file names; the order does not matter)
Layout out: <out>/<Dataset>/<scene>.png   colour, probability-weighted (or hard) fusion.

  python tools/run_ours_roots.py --root <benchmarks> --out results/Ours_soft                          # released weights
  python tools/run_ours_roots.py --root <benchmarks> --out results/Ours_hard --fusion hard
  python tools/run_ours_roots.py --run_dir runs/final --ckpt epoch_2.pth --root <benchmarks> --out ...  # a training run
"""
import argparse, os, re, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import smooth_probs
from stackmff_v3.model_io import DEFAULT_WEIGHTS
from predict_benchmark import pad_to_64

EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt', default=DEFAULT_WEIGHTS, help='checkpoint file; with --run_dir a file name in <run_dir>/model_save')
    ap.add_argument('--run_dir', default='', help='build the model from this training run (any arch)')
    ap.add_argument('--fusion', choices=['soft', 'hard'], default='soft')
    ap.add_argument('--sigma', type=float, default=None,
                    help='Gaussian smoothing of the probability maps (default: 0.5 for soft, 0 for hard)')
    ap.add_argument('--datasets', nargs='*', help='restrict to these dataset folders')
    ap.add_argument('--K', type=int, default=None, help='loop iterations (loop arch)')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    sigma = a.sigma if a.sigma is not None else (0.5 if a.fusion == 'soft' else 0.0)
    dev = torch.device(a.device)
    sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
    from stackmff_v3.model_io import load
    m = load(a.run_dir, a.ckpt, dev, a.K)
    times = []
    for ds in sorted(os.listdir(a.root)):
        sroot = os.path.join(a.root, ds, 'image stack')
        if not os.path.isdir(sroot) or (a.datasets and ds not in a.datasets):
            continue
        od = os.path.join(a.out, ds); os.makedirs(od, exist_ok=True)
        for scene in sorted(os.listdir(sroot), key=nat):
            dst = os.path.join(od, scene + '.png')
            if os.path.exists(dst):
                continue
            d = os.path.join(sroot, scene)
            files = sorted([f for f in os.listdir(d) if f.lower().endswith(EXT)], key=nat)
            if len(files) < 2:
                continue
            bgr = [cv2.imread(os.path.join(d, f), cv2.IMREAD_COLOR) for f in files]
            H, W = bgr[0].shape[:2]
            bgr = [b if b.shape[:2] == (H, W) else cv2.resize(b, (W, H), interpolation=cv2.INTER_CUBIC) for b in bgr]
            gray = np.stack([cv2.cvtColor(b, cv2.COLOR_BGR2GRAY) for b in bgr]).astype(np.float32) / 255.0
            x = torch.from_numpy(gray).unsqueeze(0).to(dev)
            sync(); t0 = time.time()
            with torch.no_grad():
                P = m.forward_probs(pad_to_64(x))[:, :, :H, :W].float()
                if sigma > 0:
                    P = smooth_probs(P, sigma)
                P = P[0]
            sync(); times.append(time.time() - t0)
            P = P.cpu().numpy()
            st = np.stack(bgr).astype(np.float32)
            if a.fusion == 'hard':
                f = np.take_along_axis(st, P.argmax(0)[None, :, :, None], 0)[0]
            else:
                f = (P[..., None] * st).sum(0)
            cv2.imwrite(dst, np.clip(np.round(f), 0, 255).astype(np.uint8))
    if times:
        print(f'{a.root}: {len(times)} stacks, mean forward {np.mean(times):.4f}s', flush=True)


if __name__ == '__main__':
    main()
