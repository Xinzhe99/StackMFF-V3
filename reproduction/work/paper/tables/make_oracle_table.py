"""Supplementary table: per-pixel oracle selection (the input image closest to the reference at every pixel) versus our
weighted fusion and the best other method, per benchmark (mean over scenes) and for the Middlebury scenes discussed in
the paper. Data: work/eval/oracle/oracle_classical.csv, work/eval/per_image.csv (same SSIM as the main tables).
  python make_oracle_table.py [--ours Ours_final_soft] -> tab_supp_oracle.tex"""
import argparse, csv, os
from collections import defaultdict
import numpy as np

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = ['Mobile Depth', 'Middlebury', 'FlyingThings3D', 'Road-MF']
SCENES = ['Jadeplant', 'Vintage', 'Pipes']
SKIP = ('Ours', 'StackMFF V4')   # our own variants and the unpublished intermediate model are not competitors


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ours', default='Ours_final_soft')
    a = ap.parse_args()
    orc = {(r['dataset'], r['scene'].lower()): (float(r['oracle_SSIM']), float(r['oracle_PSNR']))
           for r in csv.DictReader(open(f'{W}/eval/oracle/oracle_classical.csv'))}
    res = defaultdict(dict)   # (ds, scene) -> method -> (ssim, psnr)
    for r in csv.DictReader(open(f'{W}/eval/per_image.csv')):
        if r['primary'] != 'True' or not r['SSIM']:
            continue
        res[(r['dataset'], r['scene'].lower())][r['method']] = (float(r['SSIM']), float(r['PSNR']))

    def row(keys):
        o = np.array([orc[k] for k in keys]); u = np.array([res[k][a.ours] for k in keys])
        others = {m for k in keys for m in res[k] if not m.startswith(SKIP) and m != a.ours}
        full = [m for m in others if all(m in res[k] for k in keys)]
        best = max(full, key=lambda m: np.mean([res[k][m][0] for k in keys]))
        b = np.array([res[k][best][0] for k in keys])
        return o.mean(0), u.mean(0), b.mean(), best
    L = [r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{4pt}',
         r'\caption{Reference-guided nearest-observation selection. The oracle copies the input value closest to the reference at each pixel. It maximizes PSNR among pixel-wise selection rules, but its SSIM is descriptive; it is not an upper bound for convex mixtures. It is computed on the gray images with the SSIM and PSNR of the main tables. Benchmarks: mean over all scenes; below, the Middlebury scenes discussed in Section~5 of the main paper. Best other: the highest SSIM among all other evaluated methods.}',
         r'\label{tab:supp_oracle}', r'\begin{tabular}{llccccc}', r'\toprule',
         r' & & \multicolumn{2}{c}{Oracle} & \multicolumn{2}{c}{Ours (weighted)} & Best other \\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}',
         r'Benchmark & Scene & SSIM & PSNR & SSIM & PSNR & SSIM \\', r'\midrule']
    for ds in DS:
        keys = [k for k in orc if k[0] == ds and a.ours in res.get(k, {})]
        o, u, b, _ = row(keys)
        L.append(f'{ds} & all ({len(keys)}) & {o[0]:.3f} & ' + (r'$\infty$' if not np.isfinite(o[1]) else f'{o[1]:.1f}') + f' & {u[0]:.3f} & {u[1]:.1f} & {b:.3f} \\\\')
    L.append(r'\midrule')
    for sc in SCENES:
        k = ('Middlebury', sc.lower()); o, u, b, best = row([k])
        L.append(f'Middlebury & {sc} & {o[0]:.3f} & {o[1]:.1f} & {u[0]:.3f} & {u[1]:.1f} & {b:.3f} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tab_supp_oracle.tex')
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[10:]))


if __name__ == '__main__':
    main()
