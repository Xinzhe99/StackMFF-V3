"""Statistical comparison from per_image.csv (primary variant only).

For each dataset and metric (default SSIM and PSNR):
  * Friedman test over all methods with complete per-image results, average ranks (1 = best);
  * Nemenyi critical difference at alpha = 0.05 (Demsar 2006);
  * Wilcoxon signed-rank test (two-sided) of REF against every other method, Holm correction across methods.
Writes stats.md and stats.csv next to per_image.csv (the output folder of run_eval.py).

  python stats.py --dir <out_dir of run_eval.py> --ref "Ours" [--metrics SSIM PSNR VIF] [--exclude A B]
"""
import argparse, csv, math, os
from collections import defaultdict
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Mobile Depth', 'Middlebury', 'FlyingThings3D', 'Road-MF']
# q_alpha for Nemenyi (alpha=0.05), k = 2..30 (Demsar 2006, studentized range / sqrt(2))
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
       13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544, 21: 3.569, 22: 3.593,
       23: 3.616, 24: 3.637, 25: 3.658, 26: 3.678, 27: 3.696, 28: 3.714, 29: 3.732, 30: 3.749}


def q_nemenyi(k):
    """Table value for k <= 30; beyond, the studentized range quantile / sqrt(2) (agrees with the table to 1e-3)."""
    if k in Q05:
        return Q05[k]
    from scipy.stats import studentized_range
    return studentized_range.ppf(0.95, k, 1e6) / math.sqrt(2)


def holm(p):
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(order):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.environ.get('STACKMFF_EVAL_OUT', 'eval_out'), help='folder with per_image.csv')
    ap.add_argument('--ref', required=True, help='reference method (folder name)')
    ap.add_argument('--metrics', nargs='+', default=['SSIM', 'PSNR'])
    ap.add_argument('--exclude', nargs='*', default=[], help='methods to leave out of the comparison')
    a = ap.parse_args()
    data = defaultdict(dict)  # (ds, metric) -> method -> {scene: value}
    for r in csv.DictReader(open(os.path.join(a.dir, 'per_image.csv'))):
        if r['primary'] != 'True' or r['method'] in a.exclude:
            continue
        for k in a.metrics:
            if r[k] not in ('', 'inf'):
                data[(r['dataset'], k)].setdefault(r['method'], {})[r['scene']] = float(r[k])
    L = [f'# Statistical comparison (reference: {a.ref})', '']
    rows = []
    for ds in DATASETS:
        for k in a.metrics:
            d = data[(ds, k)]
            scenes = sorted(set.intersection(*[set(v) for v in d.values()]))
            ms = [m for m in d if all(s in d[m] for s in scenes)]
            X = np.array([[d[m][s] for m in ms] for s in scenes])  # n x k
            n, kk = X.shape
            ranks = np.array([st.rankdata(-row) for row in X])
            avg = ranks.mean(0)
            fr = st.friedmanchisquare(*X.T) if kk >= 3 else None
            cd = q_nemenyi(kk) * math.sqrt(kk * (kk + 1) / (6.0 * n))
            L.append(f'## {ds} / {k}: n={n} images, k={kk} methods, Friedman p={fr.pvalue:.2e}, Nemenyi CD={cd:.2f}')
            L.append('')
            L.append('| method | mean | avg rank | Wilcoxon p (vs ref, Holm) | ref better on |')
            L.append('|---|---|---|---|---|')
            pv, others = [], []
            if a.ref in ms:
                ri = ms.index(a.ref)
                for j, m in enumerate(ms):
                    if m == a.ref:
                        continue
                    diff = X[:, ri] - X[:, j]
                    p = st.wilcoxon(X[:, ri], X[:, j]).pvalue if np.any(diff != 0) else 1.0
                    pv.append(p); others.append((m, float(np.mean(diff > 0))))
                adj = dict(zip([m for m, _ in others], holm(np.array(pv)))) if pv else {}
                better = dict(others)
            else:
                adj, better = {}, {}
            for j in np.argsort(avg):
                m = ms[j]
                ps = '' if m == a.ref else (f'{adj[m]:.2e}' if m in adj else '')
                bt = '' if m == a.ref else (f'{100 * better[m]:.0f}%' if m in better else '')
                L.append(f'| {m} | {X[:, j].mean():.4f} | {avg[j]:.2f} | {ps} | {bt} |')
                rows.append({'dataset': ds, 'metric': k, 'method': m, 'mean': X[:, j].mean(), 'avg_rank': avg[j],
                             'wilcoxon_holm_p_vs_ref': adj.get(m, ''), 'ref_better_frac': better.get(m, ''),
                             'friedman_p': fr.pvalue, 'nemenyi_cd': cd, 'n': n, 'k': kk})
            L.append('')
    open(os.path.join(a.dir, 'stats.md'), 'w').write('\n'.join(L) + '\n')
    with open(os.path.join(a.dir, 'stats.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
    print('written stats.md')


if __name__ == '__main__':
    main()
