# -*- coding: utf-8 -*-
"""Evaluate every epoch of each ablation run (hard and probability-weighted fusion, old convention as in
eval_epochs.py) and pick, per run and fusion mode, the epoch with the best mean SSIM over the four benchmarks
(same selection rule as the main model). The architecture is rebuilt from the args line of <run>.out.

  python eval_ablation.py --abl_dir ablation --out ablation/ablation_summary.csv
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np, torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import StackMFFV3, StackMFFLoop, smooth_probs
from predict_benchmark import list_stacks, index_gt, read_gray, DEFAULT_DATASETS, pad_to_64

ROOT = '/media/user/68fdd01e-c642-4deb-9661-23b76592afb1/xxz/datasets/dff_and_mff'


from stackmff_v3.model_io import build  # noqa: E402  (any architecture, from the run's args)

def fuse(P, st, mode, regression):
    if mode == 'soft':
        return np.clip(np.round((P * st.astype(np.float32)).sum(0)), 0, 255).astype(np.uint8)
    if regression:
        idx = np.clip(np.round((P * np.arange(P.shape[0])[:, None, None]).sum(0)), 0, P.shape[0] - 1).astype(int)
    else:
        idx = P.argmax(0)
    return np.take_along_axis(st, idx[None], 0)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--abl_dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--runs', nargs='*')
    ap.add_argument('--sigma', type=float, default=0.0, help='Gaussian smoothing of the probabilities for weighted fusion')
    ap.add_argument('--perm_seed', type=int, default=-1, help='>=0: shuffle the layers of every stack with this seed')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    data = {}
    for ds in DEFAULT_DATASETS:
        gts = index_gt(os.path.join(ROOT, ds, 'AiF'))
        data[ds] = [(np.stack([read_gray(p) for p in paths]), read_gray(gts[n]))
                    for n, paths in list_stacks(os.path.join(ROOT, ds, 'image stack')) if n in gts]
        if a.perm_seed >= 0:
            rng = np.random.default_rng(a.perm_seed)
            data[ds] = [(st[rng.permutation(len(st))], gt) for st, gt in data[ds]]
    runs = a.runs or sorted(os.path.basename(d) for d in glob.glob(os.path.join(a.abl_dir, 'abl_*')) if os.path.isdir(d))
    rows = []
    for run in runs:
        line = next(l for l in open(os.path.join(a.abl_dir, run + '.out')) if l.startswith('args: '))
        args = json.loads(line[6:])
        m = build(args).to(dev).eval()
        n_params = sum(p.numel() for p in m.parameters())
        reg = args.get('loss', 'ce') == 'reg'
        cks = sorted(glob.glob(os.path.join(a.abl_dir, run, 'model_save', 'epoch_*.pth')),
                     key=lambda f: int(re.findall(r'(\d+)', os.path.basename(f))[0]))
        for ck in cks:
            ep = int(re.findall(r'(\d+)', os.path.basename(ck))[0])
            sd = torch.load(ck, map_location='cpu'); m.load_state_dict(sd.get('model', sd))
            acc = {(ds, md): [[], []] for ds in DEFAULT_DATASETS for md in ('hard', 'soft')}
            for ds, items in data.items():
                for st, gt in items:
                    N, H, W = st.shape
                    x = torch.from_numpy(st.astype(np.float32) / 255.0).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        Pt = m.forward_probs(pad_to_64(x))[:, :, :H, :W].float()
                        Ps = smooth_probs(Pt, a.sigma)[0].cpu().numpy() if a.sigma > 0 else None
                        P = Pt[0].cpu().numpy()
                    for md in ('hard', 'soft'):
                        f = fuse(Ps if (md == 'soft' and Ps is not None) else P, st, md, reg)
                        acc[(ds, md)][0].append(structural_similarity(gt, f, data_range=255))
                        acc[(ds, md)][1].append(peak_signal_noise_ratio(gt, f, data_range=255))
            for md in ('hard', 'soft'):
                r = {'run': run, 'epoch': ep, 'fusion': md, 'params_M': round(n_params / 1e6, 3)}
                for ds in DEFAULT_DATASETS:
                    r[f'{ds}_SSIM'] = float(np.mean(acc[(ds, md)][0])); r[f'{ds}_PSNR'] = float(np.mean(acc[(ds, md)][1]))
                r['mean_SSIM'] = float(np.mean([r[f'{ds}_SSIM'] for ds in DEFAULT_DATASETS]))
                r['mean_PSNR'] = float(np.mean([r[f'{ds}_PSNR'] for ds in DEFAULT_DATASETS]))
                rows.append(r)
                print(run, ep, md, f"{r['mean_SSIM']:.4f}/{r['mean_PSNR']:.2f}", flush=True)
        with open(a.out.replace('.csv', '_all.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]
    best = []
    for run in runs:
        for md in ('hard', 'soft'):
            c = [r for r in rows if r['run'] == run and r['fusion'] == md]
            if c:
                best.append(max(c, key=lambda r: r['mean_SSIM']))
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(best[0])); w.writeheader(); [w.writerow(x) for x in best]
    print('saved', a.out)


if __name__ == '__main__':
    main()
