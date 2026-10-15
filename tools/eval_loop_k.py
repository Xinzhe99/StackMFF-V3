# -*- coding: utf-8 -*-
"""Loop models: benchmark scores (old convention, weighted fusion sigma=1 and hard) of one checkpoint for fixed K and
for adaptive stopping (tolerances), with the mean number of loop iterations and the time per stack.
  python eval_loop_k.py --run_dir simplify/loop48 --ckpt best_bench.pth --out simplify/loop48_K.csv
"""
import argparse, csv, json, os, sys, time
import numpy as np, torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stackmff_v3.network import smooth_probs
from predict_benchmark import list_stacks, index_gt, read_gray, DEFAULT_DATASETS, pad_to_64
from eval_ablation import build, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True); ap.add_argument('--ckpt', default='best_bench.pth'); ap.add_argument('--out', required=True)
    ap.add_argument('--Ks', type=int, nargs='+', default=[1, 2, 3, 4, 6])
    ap.add_argument('--tols', type=float, nargs='+', default=[0.05, 0.02, 0.01])
    a = ap.parse_args()
    line = next(l for l in open(a.run_dir.rstrip('/') + '.out') if l.startswith('args: '))
    args = json.loads(line[6:])
    m = build(args).cuda().eval()
    sd = torch.load(os.path.join(a.run_dir, 'model_save', a.ckpt), map_location='cpu'); m.load_state_dict(sd.get('model', sd))
    data = {}
    for ds in DEFAULT_DATASETS:
        gts = index_gt(os.path.join(ROOT, ds, 'AiF'))
        data[ds] = [(np.stack([read_gray(p) for p in paths]), read_gray(gts[n])) for n, paths in list_stacks(os.path.join(ROOT, ds, 'image stack')) if n in gts]
    settings = [('K', k) for k in a.Ks] + [('tol', t) for t in a.tols]
    rows = []
    for kind, val in settings:
        if kind == 'K':
            m.adaptive = False; m.cur_K = val
        else:
            m.adaptive = True; m.tol = val; m.kmax_infer = 8
        r = {'setting': f'{kind}={val}'}
        ks, ts = [], []
        for ds, items in data.items():
            acc = {'soft': [[], []], 'hard': [[], []]}
            for st, gt in items:
                N, H, W = st.shape
                x = torch.from_numpy(st.astype(np.float32) / 255).unsqueeze(0).cuda()
                torch.cuda.synchronize(); t0 = time.time()
                with torch.no_grad():
                    P0 = m.forward_probs(pad_to_64(x))[:, :, :H, :W].float()
                    P1 = smooth_probs(P0, 1.0)
                torch.cuda.synchronize(); ts.append(time.time() - t0); ks.append(m.last_K)
                P0 = P0[0].cpu().numpy(); P1 = P1[0].cpu().numpy()
                fs = np.clip(np.round((P1 * st).sum(0)), 0, 255).astype(np.uint8)
                fh = np.take_along_axis(st, P0.argmax(0)[None], 0)[0]
                for md, f in (('soft', fs), ('hard', fh)):
                    acc[md][0].append(structural_similarity(gt, f, data_range=255)); acc[md][1].append(peak_signal_noise_ratio(gt, f, data_range=255))
            for md in acc:
                r[f'{ds}_{md}_SSIM'] = float(np.mean(acc[md][0])); r[f'{ds}_{md}_PSNR'] = float(np.mean(acc[md][1]))
        for md in ('soft', 'hard'):
            r[f'mean_{md}_SSIM'] = float(np.mean([r[f'{d}_{md}_SSIM'] for d in DEFAULT_DATASETS]))
            r[f'mean_{md}_PSNR'] = float(np.mean([r[f'{d}_{md}_PSNR'] for d in DEFAULT_DATASETS]))
        r['mean_K'] = float(np.mean(ks)); r['sec_per_stack'] = float(np.mean(ts))
        rows.append(r)
        print(r['setting'], f"soft {r['mean_soft_SSIM']:.4f}/{r['mean_soft_PSNR']:.2f} hard {r['mean_hard_SSIM']:.4f} K {r['mean_K']:.2f} t {r['sec_per_stack']:.3f}", flush=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(x) for x in rows]


if __name__ == '__main__':
    main()
