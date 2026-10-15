# -*- coding: utf-8 -*-
"""Run the selected checkpoint on arbitrary stack roots (robustness variants, real-data subsets).
Layout in:  <root>/<Dataset>/image stack/<scene>/<layers>  (natural-sort order of file names)
Layout out: <out>/<Dataset>/<scene>.png   colour, probability-weighted (or hard) fusion.

  python run_ours_roots.py --ckpt .../epoch_K.pth --C1 48 --C2 96 --heads 3 6 --inter_pool 2 --fusion soft \
      --root <variants>/<variant> --out <results>/<variant>/Ours
"""
import argparse, os, re, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import StackMFFV3, smooth_probs
from predict_benchmark import pad_to_64

EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def nat(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--C1', type=int, default=48); ap.add_argument('--C2', type=int, default=96)
    ap.add_argument('--heads', type=int, nargs=2, default=[3, 6]); ap.add_argument('--inter_pool', type=int, default=2)
    ap.add_argument('--fusion', choices=['soft', 'hard'], default='soft')
    ap.add_argument('--sigma', type=float, default=1.0, help='Gaussian smoothing of the probability maps (0: off)')
    ap.add_argument('--datasets', nargs='*', help='restrict to these dataset folders')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    m = StackMFFV3(C1=a.C1, C2=a.C2, heads=tuple(a.heads), inter_pool=a.inter_pool).to(dev).eval()
    sd = torch.load(a.ckpt, map_location='cpu'); m.load_state_dict(sd.get('model', sd))
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
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                P = m.forward_probs(pad_to_64(x))[:, :, :H, :W].float()
                if a.sigma > 0:
                    P = smooth_probs(P, a.sigma)
                P = P[0]
            torch.cuda.synchronize(); times.append(time.time() - t0)
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
