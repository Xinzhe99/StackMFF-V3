# -*- coding: utf-8 -*-
"""Evaluate every epoch checkpoint of a run with hard and probability-weighted fusion (old convention:
cv2 grey, skimage SSIM defaults, PSNR), select the best epoch by mean SSIM over the four benchmarks
(user decision: selection on the test benchmarks, as in the earlier StackMFF papers), and write the colour
fused results of the selected epoch for the unified evaluation:
  <out>/<tag>_<fusion>/<Dataset>/<scene>.png
Colour: the probabilities computed on the grey stack weight (or select) the RGB values of the layers.

  python eval_epochs.py --run_dir runs/H_c48_r121_p2 --C1 48 --C2 96 --heads 3 6 --inter_pool 2 --tag H --out results_new
"""
import argparse, csv, glob, os, re, sys
import cv2, numpy as np, torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import StackMFFV3, smooth_probs
from predict_benchmark import list_stacks, index_gt, read_gray, DEFAULT_DATASETS, pad_to_64

ROOT = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/dff_and_mff'


SIGMA = 1.0   # Gaussian smoothing of P for weighted fusion (final default); hard selection uses the raw P


def probs(m, st_gray, dev):
    N, H, W = st_gray.shape
    x = torch.from_numpy(st_gray.astype(np.float32) / 255.0).unsqueeze(0).to(dev)
    with torch.no_grad():
        P = m.forward_probs(pad_to_64(x))[:, :, :H, :W].float()
        return np.stack([P[0].cpu().numpy(), smooth_probs(P, SIGMA)[0].cpu().numpy()])


def fuse(PP, st, mode):
    P = PP[0] if mode == 'hard' else PP[1]
    if mode == 'hard':
        idx = P.argmax(0)
        return np.take_along_axis(st, idx[(None,) + (slice(None),) * 2 + ((None,) if st.ndim == 4 else ())], 0)[0]
    w = P if st.ndim == 3 else P[..., None]
    return np.clip(np.round((w * st.astype(np.float32)).sum(0)), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--C1', type=int, default=32); ap.add_argument('--C2', type=int, default=64)
    ap.add_argument('--heads', type=int, nargs=2, default=[2, 4]); ap.add_argument('--inter_pool', type=int, default=1)
    ap.add_argument('--out', default='/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/v3merge_work/results_new')
    ap.add_argument('--skip_select', action='store_true')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    from stackmff_v3.model_io import build, run_args
    try:
        m = build(run_args(a.run_dir)).to(dev).eval()
    except FileNotFoundError:
        m = StackMFFV3(C1=a.C1, C2=a.C2, heads=tuple(a.heads), inter_pool=a.inter_pool).to(dev).eval()
    data = {}
    for ds in DEFAULT_DATASETS:
        gts = index_gt(os.path.join(ROOT, ds, 'AiF'))
        data[ds] = [(n, p, gts[n]) for n, p in list_stacks(os.path.join(ROOT, ds, 'image stack')) if n in gts]
    ckpts = sorted(glob.glob(os.path.join(a.run_dir, 'model_save', 'epoch_*.pth')), key=lambda f: int(re.findall(r'(\d+)', os.path.basename(f))[0]))
    rows = []
    for ck in ckpts:
        ep = int(re.findall(r'(\d+)', os.path.basename(ck))[0])
        sd = torch.load(ck, map_location='cpu'); m.load_state_dict(sd.get('model', sd))
        res = {}
        for ds, items in data.items():
            acc = {'hard': [[], []], 'soft': [[], []]}
            for name, paths, g in items:
                st = np.stack([read_gray(p) for p in paths]); gt = read_gray(g)
                P = probs(m, st, dev)
                for mode in acc:
                    f = fuse(P, st, mode)
                    acc[mode][0].append(structural_similarity(gt, f, data_range=255))
                    acc[mode][1].append(peak_signal_noise_ratio(gt, f, data_range=255))
            for mode, v in acc.items():
                res[(ds, mode)] = (float(np.mean(v[0])), float(np.mean(v[1])))
        for mode in ('hard', 'soft'):
            row = {'epoch': ep, 'fusion': mode}
            for ds in DEFAULT_DATASETS:
                row[f'{ds}_SSIM'], row[f'{ds}_PSNR'] = res[(ds, mode)]
            row['mean_SSIM'] = float(np.mean([res[(ds, mode)][0] for ds in DEFAULT_DATASETS]))
            row['mean_PSNR'] = float(np.mean([res[(ds, mode)][1] for ds in DEFAULT_DATASETS]))
            rows.append(row)
            print(f"epoch {ep} {mode}: " + ' '.join(f"{ds[:6]} {res[(ds, mode)][0]:.4f}/{res[(ds, mode)][1]:.2f}" for ds in DEFAULT_DATASETS)
                  + f" | mean {row['mean_SSIM']:.4f}/{row['mean_PSNR']:.2f}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f'{a.tag}_epochs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    if a.skip_select:
        return
    for mode in ('soft', 'hard'):
        best = max((r for r in rows if r['fusion'] == mode), key=lambda r: r['mean_SSIM'])
        ck = os.path.join(a.run_dir, 'model_save', f"epoch_{best['epoch']}.pth")
        sd = torch.load(ck, map_location='cpu'); m.load_state_dict(sd.get('model', sd))
        print(f'selected {mode}: epoch {best["epoch"]} mean SSIM {best["mean_SSIM"]:.4f}', flush=True)
        for ds, items in data.items():
            od = os.path.join(a.out, f'{a.tag}_{mode}', ds); os.makedirs(od, exist_ok=True)
            for name, paths, _ in items:
                st = np.stack([read_gray(p) for p in paths])
                rgb = np.stack([cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for p in paths])
                f = fuse(probs(m, st, dev), rgb, mode)
                cv2.imwrite(os.path.join(od, name + '.png'), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        with open(os.path.join(a.out, f'{a.tag}_{mode}', 'SELECTED_EPOCH.txt'), 'w') as fh:
            fh.write(f"{best['epoch']}\n")


if __name__ == '__main__':
    main()
