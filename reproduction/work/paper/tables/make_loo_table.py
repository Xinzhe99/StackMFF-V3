"""Leave-one-out ablations of the final configuration (17 Oct: one resolution, window attention, SACA r=2, linear head,
contrast-weighted CE, highlight synthesis), quick protocol of the roadmap (35% of the pairs, 3 epochs, no EMA),
single-scale inference and sigma = 1 weighted fusion, best epoch per run by the mean SSIM. Rows with three seeds give the
mean (range in the caption). Reads the single-scale eval_ablation summaries in work/new_model/roadmap/ (and seeds/).
Previous version (two resolutions + head): make_loo_table_finalws.py.bak.  -> tab_ablation.tex"""
import csv, glob, os
import numpy as np

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = f'{W}/new_model/roadmap'
FILES = ['summary_single.csv', 'summary_single_zs.csv', 'summary_rm3_single.csv', 'zs_rm2_summary_single_b_all.csv',
         'zs_rm2_summary_single_a_all.csv', 'seeds/*single.csv']
DS = [('Mobile Depth', 'MD'), ('FlyingThings3D', 'FT3D'), ('Middlebury', 'Mid.'), ('Road-MF', 'Road-MF')]
# label, runs (seeds), params (M), GFLOPs (two 256x256 images, one inference scale; new_model/cost/cost_final17.py)
ROWS = [('Final configuration', ['n6_ws', 'n6_s1', 'n6_s2'], 0.156, 1.94),
        (None,),
        ('Plain cross-entropy', ['n5_hdr', 'plaince_s1', 'plaince_s2'], 0.156, 1.94),
        ('Original synthesis', ['olddata'], 0.156, 1.94),
        ('Without highlight synthesis', ['nohdr'], 0.156, 1.94),
        ('Index regression', ['regression'], 0.156, 1.94),
        ('Conv.\\ block for intra-image attention', ['conv_r2'], 0.125, 1.09),
        ('No aggregation ($r=1$)', ['c7_attn', 'c7_attn_s1', 'c7_attn_s2'], 0.156, 2.86),
        (None,),
        ('+ Full-resolution head', ['head'], 0.165, 2.81),
        ('+ Second resolution ($1/8$)', ['twores'], 0.496, 2.95),
        ('+ Image-index embedding', ['layerpos'], 0.159, 1.94)]


def load():
    res = {}
    for pat in FILES:
        for f in sorted(glob.glob(f'{D}/{pat}')):
            for r in csv.DictReader(open(f)):
                if r.get('fusion') != 'soft':
                    continue
                v = {'mean': float(r['mean_SSIM']), 'psnr': float(r['mean_PSNR']), **{d: float(r[f'{d}_SSIM']) for d, _ in DS}}
                if r['run'] not in res or v['mean'] > res[r['run']]['mean']:
                    res[r['run']] = v
    return res


def main():
    res = load()
    rg = []
    for row in ROWS:
        if row[0] is not None and len([r for r in row[1] if r in res]) > 1:
            v = [res[r]['mean'] for r in row[1] if r in res]
            rg.append(f'{min(v):.4f}--{max(v):.4f} for ' + ('the reference' if row[0].startswith('Final') else row[0][0].lower() + row[0][1:].replace('No ', 'no ')))
    RANGES = 'ranges ' + ', '.join(rg) if rg else ''
    L = [r'\begin{table}[t]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{2pt}',
         r'\caption{Leave-one-out ablations of the final configuration: the middle block removes one component, the last block adds one that the final model does not use. Every variant is trained with the short protocol of Table~\ref{tab:roadmap} (35\% of the training pairs, three epochs, no weight averaging) and evaluated with one inference scale and $\sigma=1$; rows marked $^{3}$ are means over three seeds (mean SSIM ' + RANGES + r'). SSIM: mean SSIM$_7$ ($7\times7$ window, as in Table~\ref{tab:roadmap}). GFLOPs for a stack of two $256\times256$ images. $\Delta$: difference of the mean SSIM to the reference; the seed-to-seed standard deviation is about $3\times10^{-4}$.}',
         r'\label{tab:ablation}', r'\begin{tabular}{lcccccccc}', r'\toprule',
         r'Variant & GFLOPs & MD & FT3D & Mid. & Road-MF & SSIM & PSNR & $\Delta$ \\', r'\midrule']
    ref = None
    for row in ROWS:
        if row[0] is None:
            L.append(r'\midrule'); continue
        lab, runs, par, gfl = row
        got = [res[r] for r in runs if r in res]
        if not got:
            L.append(lab + r' & \multicolumn{8}{c}{(running)} \\'); continue
        m = {k: float(np.mean([g[k] for g in got])) for k in got[0]}
        sup = r'$^{3}$' if len(got) > 1 else ''
        if ref is None:
            ref = m['mean']
        if len(got) > 1:
            rng = f'{min(g["mean"] for g in got):.4f}--{max(g["mean"] for g in got):.4f}'
        else:
            rng = ''
        d = '' if lab.startswith('Final') else ('0.0000' if abs(m['mean'] - ref) < 5e-5 else f'${m["mean"] - ref:+.4f}$')
        L.append(f'{lab}{sup} & {gfl:.2f} & ' + ' & '.join(f'{m[x]:.4f}' for x, _ in DS) + f' & {m["mean"]:.4f} & {m["psnr"]:.2f} & {d} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tab_ablation.tex')
    open(out, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[9:]))


if __name__ == '__main__':
    main()
