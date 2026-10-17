# -*- coding: utf-8 -*-
"""Effect of spatially smoothing the layer probabilities before weighted fusion (Gaussian, sigma in pixels,
applied to each P_n then renormalised). Benchmarks: old convention as eval_epochs.py. Also writes the colour
fusion of a few real stacks for visual inspection.
  python tools/eval_smooth.py --sigmas 0 0.5 1 2 --out smooth.csv --bench_root <benchmarks>        # released weights
  python tools/eval_smooth.py --run_dir runs/final --ckpt epoch_2.pth --sigmas 0 0.5 1 --out smooth.csv --bench_root <benchmarks>
"""
import argparse, csv, os, sys
import cv2, numpy as np, torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.model_io import DEFAULT_WEIGHTS, load
from predict_benchmark import list_stacks, index_gt, read_gray, DEFAULT_DATASETS, pad_to_64

ROOT = os.environ.get('STACKMFF_BENCH_ROOT', 'data/benchmarks')


def gauss(P, s):
    if s <= 0:
        return P
    r = int(3 * s + 0.5); x = torch.arange(-r, r + 1, dtype=torch.float32, device=P.device)
    k = torch.exp(-x ** 2 / (2 * s * s)); k = k / k.sum()
    N = P.shape[1]
    Q = F.conv2d(F.pad(P, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1).repeat(N, 1, 1, 1), groups=N)
    Q = F.conv2d(F.pad(Q, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1).repeat(N, 1, 1, 1), groups=N)
    return Q / Q.sum(1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_WEIGHTS); ap.add_argument('--out', required=True)
    ap.add_argument('--bench_root', default=ROOT, help='<bench_root>/<Dataset>/{image stack,AiF}')
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0, 1, 2, 4])
    ap.add_argument('--real_root'); ap.add_argument('--real_ids', nargs='*', default=[])
    ap.add_argument('--run_dir', default='')
    a = ap.parse_args()
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    m = load(a.run_dir, a.ckpt, dev)
    rows = []
    acc = {s: {ds: [[], []] for ds in DEFAULT_DATASETS} for s in a.sigmas}
    for ds in DEFAULT_DATASETS:
        gts = index_gt(os.path.join(a.bench_root, ds, 'AiF'))
        for n, paths in list_stacks(os.path.join(a.bench_root, ds, 'image stack')):
            if n not in gts:
                continue
            st = np.stack([read_gray(p) for p in paths]).astype(np.float32); gt = read_gray(gts[n])
            N, H, W = st.shape
            with torch.no_grad():
                P = m.forward_probs(pad_to_64(torch.from_numpy(st / 255).unsqueeze(0).to(dev)))[:, :, :H, :W].float()
                for s in a.sigmas:
                    f = (gauss(P, s)[0].cpu().numpy() * st).sum(0)
                    f = np.clip(np.round(f), 0, 255).astype(np.uint8)
                    acc[s][ds][0].append(structural_similarity(gt, f, data_range=255))
                    acc[s][ds][1].append(peak_signal_noise_ratio(gt, f, data_range=255))
    for s in a.sigmas:
        r = {'sigma': s}
        for ds in DEFAULT_DATASETS:
            r[ds + '_SSIM'] = float(np.mean(acc[s][ds][0])); r[ds + '_PSNR'] = float(np.mean(acc[s][ds][1]))
        r['mean_SSIM'] = float(np.mean([r[d + '_SSIM'] for d in DEFAULT_DATASETS]))
        r['mean_PSNR'] = float(np.mean([r[d + '_PSNR'] for d in DEFAULT_DATASETS]))
        rows.append(r); print(r, flush=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]
    od = os.path.splitext(a.out)[0] + '_real'; os.makedirs(od, exist_ok=True)
    for rid in a.real_ids:
        ds, sid = rid.split('/')
        d = os.path.join(a.real_root, ds, 'image stack', sid)
        fs = sorted(os.listdir(d))
        st = np.stack([read_gray(os.path.join(d, f)) for f in fs]).astype(np.float32)
        rgb = np.stack([cv2.cvtColor(cv2.imread(os.path.join(d, f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for f in fs]).astype(np.float32)
        N, H, W = st.shape
        with torch.no_grad():
            P = m.forward_probs(pad_to_64(torch.from_numpy(st / 255).unsqueeze(0).to(dev)))[:, :, :H, :W].float()
            for s in a.sigmas:
                Q = gauss(P, s)[0].cpu().numpy()
                f = np.clip(np.round((Q[..., None] * rgb).sum(0)), 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(od, f'{ds}_{sid}_s{s:g}.png'), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
