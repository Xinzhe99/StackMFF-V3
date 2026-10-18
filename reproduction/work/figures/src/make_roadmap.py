"""ConvNeXt-style roadmap: from a StackMFF V2-style network to ours, one cumulative change per step (quick protocol:
35% of the training pairs, 3 epochs, no EMA, seed 0; best epoch by the mean SSIM over the four benchmarks; single-scale
inference up to the last step). Reads the eval_ablation summaries collected in work/new_model/roadmap/ and writes
  ../out/fig_roadmap.pdf (+ _preview.png) and paper/tables/tab_roadmap.tex
  python make_roadmap.py"""
import csv, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
FONT = 'C:/Windows/Fonts/times.ttf' if os.name == 'nt' else '/mnt/c/Windows/Fonts/times.ttf'
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    if os.path.exists(FONT.replace('times.ttf', 'timesbd.ttf')):
        font_manager.fontManager.addfont(FONT.replace('times.ttf', 'timesbd.ttf'))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams['mathtext.fontset'] = 'stix'
DRED, DBLUE, GREY, INK, MUTED, GRID, PALEBLUE = '#B23A48', '#244A68', '#B9BEC3', '#2b2b2b', '#6b6b6b', '#e6e6e6', '#DBE9F2'
W = str(Path(__file__).resolve().parents[2])
D = f'{W}/new_model/roadmap'
DS = ['Mobile Depth', 'FlyingThings3D', 'Middlebury', 'Road-MF']
NOISE = 0.0005   # changes below this are within the seed spread of the quick protocol (SD about 0.0003)
# step id, group, label (table), short label (figure), summary file, run, fusion, params (M), GFLOPs (two 256x256 images)
STEPS = [
    ('R0', 'Start', 'V2-style network$^{a}$', 'V2-style start', 'single', ['R0'], 'hard', 0.125, 2.00),
    ('R1', 'Data', '+ on-the-fly synthesis', '+ on-the-fly synthesis', 'single', ['R1'], 'hard', 0.125, 2.00),
    ('R2', 'Formulation', '+ classification (hard sel.)', '+ classification', 'single', ['R2', 'R2_s1', 'R2_s2'], 'hard', 0.125, 2.00),
    ('R2b', 'Formulation', '+ weighted fusion (Eq.~\\eqref{eq:soft})', '+ weighted fusion', 'single', ['R2', 'R2_s1', 'R2_s2'], 'soft', 0.125, 2.00),
    ('c5', 'Data', '+ highlight synthesis', '+ highlights', 'single', ['c5_hdr', 'c5_hdr_s1', 'c5_hdr_s2'], 'soft', 0.125, 2.00),
    ('c6', 'Objective', '+ contrast-weighted CE (Eq.~\\eqref{eq:ce})', '+ weighted CE', 'single', ['c6_ws', 'c6_ws_s1', 'c6_ws_s2'], 'soft', 0.125, 2.00),
    ('c7', 'Architecture', '+ window attention (intra-image)', '+ window attention', 'single', ['c7_attn', 'c7_attn_s1', 'c7_attn_s2'], 'soft', 0.156, 2.86),
    ('n6', 'Architecture', '+ SACA, $r=2$', '+ SACA $r{=}2$', 'single', ['n6_ws', 'n6_s1', 'n6_s2'], 'soft', 0.156, 1.94),
    ('n6t', 'Inference', '+ two-scale inference', '+ two scales', 'two', ['n6_ws', 'n6_s1', 'n6_s2'], 'soft', 0.156, 3.03),
]
import glob
FILES = {'single': ['summary_single.csv', 'summary_single_zs.csv', 'summary_rm3_single.csv', 'zs_rm2_summary_single_b_all.csv', 'zs_rm2_summary_single_a_all.csv', 'seeds/*single.csv'],
         'two': ['summary_two.csv', 'zs_rm2_summary_two_b_all.csv', 'seeds/*two.csv']}
SHUF = {'single': ['summary_single_shuf.csv', 'summary_rm3_single_shuf.csv', 'zs_rm2_summary_single_b_shuf_all.csv', 'zs_rm2_summary_single_a_shuf_all.csv', 'seeds/*single_shuf.csv'],
        'two': ['zs_rm2_summary_two_b_shuf_all.csv', 'seeds/*two_shuf.csv']}


def rows_of(pats):
    out = []
    for pat in pats:
        for f in sorted(glob.glob(f'{D}/{pat}')):
            out += list(csv.DictReader(open(f)))
    return out


def best(rows, run, fusion):
    r = [x for x in rows if x['run'] == run and x['fusion'] == fusion]
    if not r:
        return None
    b = max(r, key=lambda x: float(x['mean_SSIM']))
    return {'epoch': b['epoch'], 'mean': float(b['mean_SSIM']), 'psnr': float(b['mean_PSNR']), **{d: float(b[f'{d}_SSIM']) for d in DS}}


def main():
    rows = {k: rows_of(p) for k, p in FILES.items()}
    shuf = {k: rows_of(p) for k, p in SHUF.items()}
    res = []
    for s in STEPS:
        vs = []
        for run in s[5]:
            v = best(rows[s[4]], run, s[6])
            if v is None:
                continue
            sh = [x for x in shuf[s[4]] if x['run'] == run and x['fusion'] == s[6] and x['epoch'] == v['epoch']]
            v['shuf'] = float(sh[0]['mean_SSIM']) - v['mean'] if sh else None
            vs.append(v)
        if not vs:
            print('missing', s[0], s[5]); continue
        m = {k: float(np.mean([v[k] for v in vs])) for k in ('mean', 'psnr', *DS)}
        sh = [v['shuf'] for v in vs if v['shuf'] is not None]
        m['shuf'] = max(sh, key=abs) if sh else None
        m['nseeds'] = len(vs); m['range'] = (min(v['mean'] for v in vs), max(v['mean'] for v in vs))
        print(s[0], len(vs), 'seeds', f"{m['mean']:.4f}", [f"{v['mean']:.4f}" for v in vs])
        res.append((s, m))
    final = None
    if os.path.exists(f'{D}/final.csv'):   # full schedule (all pairs, 6 epochs, EMA), two scales, the paper's model
        f = next(csv.DictReader(open(f'{D}/final.csv')))
        final = {'mean': float(f['mean_SSIM']), 'psnr': float(f['mean_PSNR']), **{d: float(f[f'{d}_SSIM']) for d in DS}}

    # table
    L = [r'\begin{table}[htbp]', r'\centering', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
         r'\caption{From a StackMFF V2-style network to ours, one cumulative change per row. Every row is trained with the same short protocol (35\% of the training pairs, three epochs, no weight averaging; rows marked $^{3}$ are means over three seeds) and evaluated at the epoch with the highest mean SSIM; up to the last change, inference uses a single scale. Hard selection (Eq.~\eqref{eq:hard}) is used until the weighted fusion is introduced, then weighted fusion with $\sigma=1$, the default when these variants were trained. MD: SSIM on Mobile Depth, the only benchmark with optical blur; per-benchmark scores of the short-protocol reference configuration are in main-paper Table~5. SSIM: SSIM$_7$, mean over the four benchmarks ($7\times7$ window, as in training monitoring, so the values differ slightly from main-paper Table~1). GFLOPs for a stack of two $256\times256$ images. $\Delta$: change of the mean SSIM with respect to the previous row; the seed-to-seed standard deviation of this protocol is about $3\times10^{-4}$, so changes of at most $5\times10^{-4}$ (gray) are within noise. Shuffled: change of the mean SSIM when the images of every stack are randomly permuted ($0$: below $5\times10^{-5}$). The last row is the final model, trained on all pairs for six epochs. $^{a}$Index regression, convolutional intra-image block, $1/4$ resolution only, original synthesis in focal order.}',
         r'\label{tab:roadmap}', r'\begin{tabular}{lcccccc}', r'\toprule',
         r'Change & GFLOPs & MD & SSIM & PSNR & $\Delta$ & Shuffled \\', r'\midrule']
    prev, grp = None, None
    for s, v in res:
        d = '' if prev is None else v['mean'] - prev
        ds = '' if prev is None else (f'{d:+.4f}' if abs(d) >= NOISE else rf'\textcolor{{gray}}{{{d:+.4f}}}')
        if s[1] != grp:
            L.append(rf'\multicolumn{{7}}{{l}}{{\textit{{{s[1]}}}}} \\')
        grp = s[1]
        L.append(f'{s[2]}' + (f'$^{{{v["nseeds"]}}}$' if v['nseeds'] > 1 else '') + f' & {s[8]:.2f} & {v["Mobile Depth"]:.4f} & {v["mean"]:.4f} & {v["psnr"]:.2f} & {ds} & ' + ('--' if v['shuf'] is None else ('$0$' if abs(v['shuf']) < 5e-5 else f'{v["shuf"]:+.4f}')) + ' \\\\')
        prev = v['mean']
    if final:
        L.append(r'\midrule')
        L.append(f'Full training (6 epochs, all pairs) & 3.03 & {final["Mobile Depth"]:.4f} & \\textbf{{{final["mean"]:.4f}}} & \\textbf{{{final["psnr"]:.2f}}} & & $0$ \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    out_t = f'{W}/figures/out/tab_roadmap_regen.tex'   # T2: the paper copy is owned by the orchestrator; regenerate to out/ and diff
    Path(out_t).parent.mkdir(parents=True, exist_ok=True)
    open(out_t, 'w').write('\n'.join(L) + '\n'); print('\n'.join(L[8:]))
    if os.environ.get('STACKMFF_TABLE_ONLY') == '1':
        return

    # figure: ConvNeXt-style roadmap; left: mean SSIM per cumulative step, right: GFLOPs
    PALEBLUE, PALERED, SLATE = '#DBE9F2', '#ECD5DD', '#8E959C'
    items = [(s, v) for s, v in res]
    labels = [s[3] for s, _ in items] + (['full training'] if final else [])
    groups = [s[1] for s, _ in items] + (['Final'] if final else [])
    n = len(labels)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 0.27 * n + 0.75), sharey=True,
                                 gridspec_kw={'width_ratios': [3.2, 1.0], 'wspace': 0.06})
    y = np.arange(n)[::-1].astype(float)
    vals = [v['mean'] for _, v in items] + ([final['mean']] if final else [])
    gfl = [s[8] for s, _ in items] + ([3.03] if final else [])
    lo = min(vals) - 0.0012; hi = max(vals)
    # waterfall: light base bar up to the previous step, the increment of the step coloured
    MID, SOFT, BASE = '#5E86AD', '#A9C3DA', '#E8EEF4'
    prev = None
    for i in range(n):
        last = final and i == n - 1
        if prev is None:
            ax.barh(y[i], vals[i] - lo, left=lo, height=0.6, color='#C9CDD2', lw=0)
            tc, txt = MUTED, ''
        else:
            d = vals[i] - prev
            gain = d >= NOISE
            ax.barh(y[i], prev - lo, left=lo, height=0.6, color=PALERED if last else BASE, lw=0)
            inc = DRED if last else (MID if gain else SOFT)
            if d > 2e-5:
                ax.barh(y[i], d, left=prev, height=0.6, color=inc, lw=0)
            else:
                ax.plot([prev, prev], [y[i] - 0.3, y[i] + 0.3], color=SOFT, lw=1.6, solid_capstyle='butt')
            tc = DRED if last else (MID if gain else MUTED)
            txt = '' if last else f'{d:+.4f}'
        ax.text(vals[i] + 0.00008, y[i], f'{vals[i]:.4f}', va='center', fontsize=6.6,
                color=DRED if last else INK, fontweight='bold' if last else 'normal')
        if txt:
            ax.text(hi + 0.0021, y[i], txt, va='center', ha='right', fontsize=6.6, color=tc,
                    fontweight='bold' if tc == MID else 'normal')
        if not last:
            prev = vals[i]
        bx.barh(y[i], gfl[i], height=0.6, color=PALERED if last else '#DADDE1', lw=0)
        bx.text(gfl[i] + 0.12, y[i], f'{gfl[i]:.2f}', va='center', fontsize=6.6, color=DRED if last else INK)
    # group separators and names
    for i in range(1, n):
        if groups[i] != groups[i - 1]:
            for a_ in (ax, bx):
                a_.axhline(y[i] + 0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.2, color=INK)
    for i, g in enumerate(groups):
        if i == 0 or groups[i - 1] != g:
            ax.text(-0.012, y[i] + 0.36, g.upper(), transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                    fontsize=6.6, color=SLATE, fontweight='bold')   # T2: was 5.4 (printed 4.99 pt at \textwidth)
    ax.set_xlim(lo, hi + 0.0023)
    ax.set_xlabel('mean SSIM over the four benchmarks', fontsize=7.5, color=INK)
    ax.text(hi + 0.0021, y[0] + 0.75, r'$\Delta$', ha='right', fontsize=7.2, color=INK)
    bx.set_xlim(0, max(gfl) * 1.35); bx.set_xlabel('GFLOPs', fontsize=7.5, color=INK)
    for a_ in (ax, bx):
        a_.tick_params(axis='x', labelsize=6.5, colors=MUTED, length=2); a_.tick_params(axis='y', length=0)
        for sp in ('top', 'right', 'left'):
            a_.spines[sp].set_visible(False)
        a_.spines['bottom'].set_color(SLATE)
    ax.set_ylim(-0.6, n - 0.2)
    out = f'{W}/figures/out/fig_roadmap.pdf'
    fig.savefig(out, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(out.replace('.pdf', '_preview.png'), bbox_inches='tight', pad_inches=0.02, dpi=200)


if __name__ == '__main__':
    main()
