"""Supplementary table: colour fidelity (PSNR on RGB and mean CIEDE2000 to the reference) of every method whose results
are in colour. Data: work/eval/color/color_summary.csv (color_eval.py).
  python make_color_table.py [--ours Ours_final_soft] -> tab_supp_color.tex"""
import argparse, csv, os, runpy
from collections import defaultdict

W = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = ['Mobile Depth', 'Middlebury', 'FlyingThings3D', 'Road-MF']
_main = runpy.run_path(os.path.join(os.path.dirname(__file__), 'make_tables.py'))
GROUPS = [
    ('Spatial domain', [(m, m) for m in ['GFF', 'IFM', 'QB', 'BFMF', 'GFDF', 'MGFF']]),
    ('Transform domain', [('CVT', 'CVT'), ('DWT', 'DWT'), ('DCT', 'DCT'), ('DTCWT', 'DTCWT'), ('NSCT', 'NSCT')]),
    ('Pairwise, adapted to stacks', [('IFCNN-MAX', 'IFCNN (MAX)'), ('U2Fusion', 'U2Fusion'), ('SDNet', 'SDNet'), ('MFF-GAN', 'MFF-GAN'),
                                     ('SwinFusion', 'SwinFusion'), ('MUFusion', 'MUFusion'), ('SwinMFF', 'SwinMFF'),
                                     ('DDBFusion', 'DDBFusion'), ('CCSR-Net', 'CCSR-Net'), ('MCCSR-Net', 'MCCSR-Net'),
                                     ('DeFusion', 'DeFusion'), ('MFIF-GAN', 'MFIF-GAN'), ('GIFNet', 'GIFNet'), ('CSNet', 'CSNet'), ('ZMFF', 'ZMFF'), ('ReDiffuse', 'ReDiffuse')]),
    ('Commercial software', [('Zerene Stacker - DMap', 'Zerene (DMap)'), ('Zerene Stacker - PMax', 'Zerene (PMax)')]),
    ('Native stack', [('StackMFF_release', 'StackMFF'), ('StackMFF V2', 'StackMFF V2'), ('VAEEDOF', 'VAEEDOF'),
                      ('Ours_final_hard', 'Ours (hard selection)'), (None, 'Ours (weighted)')]),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ours', default='Ours_final_soft')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tab_supp_color.tex'))
    a = ap.parse_args()
    d = defaultdict(dict)
    for r in csv.DictReader(open(f'{W}/eval/color/color_summary.csv')):
        d[r['method']][r['dataset']] = (float(r['PSNR_RGB']), float(r['DE2000']))
    rows = [(key or a.ours, lab) for _, g in GROUPS for key, lab in g]
    best = {}
    for ds in DS:   # best and second best per column (PSNR high, CIEDE2000 low)
        ps = sorted({d[k][ds][0] for k, _ in rows if ds in d[k]}, reverse=True)
        de = sorted({d[k][ds][1] for k, _ in rows if ds in d[k]})
        best[ds] = (ps[:2], de[:2])

    def fmt(v, top, nd):
        s = f'{v:.{nd}f}'
        return r'\best{' + s + '}' if v == top[0] else (r'\second{' + s + '}' if len(top) > 1 and v == top[1] else s)
    L = [r'\begin{table}[!htbp]', r'\centering', r'\scriptsize', r'\renewcommand{\arraystretch}{0.90}', r'\setlength{\tabcolsep}{3pt}',
         r'\caption{Color fidelity: RGB PSNR (dB, higher is better) and mean CIEDE2000 difference ($\Delta E_{00}$, lower is better). Helicon outputs, aligned and evaluated on luma, are omitted. Pink/blue indicate the best/second-best unrounded values.}',
         r'\label{tab:supp_color}', r'\begin{tabular}{lcccccccc}', r'\toprule',
         r'Method & \multicolumn{2}{c}{MD} & \multicolumn{2}{c}{Mid.} & \multicolumn{2}{c}{FT3D} & \multicolumn{2}{c}{Road-MF} \\',
         r'\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}',
         r' & PSNR & $\Delta E_{00}$ & PSNR & $\Delta E_{00}$ & PSNR & $\Delta E_{00}$ & PSNR & $\Delta E_{00}$ \\', r'\midrule']
    for gi, (gname, g) in enumerate(GROUPS):
        if gi:
            L.append(r'\midrule')
        L.append(r'\multicolumn{9}{l}{\textit{' + gname + r'}} \\')
        for key, lab in g:
            key = key or a.ours
            if key in _main['CITE']:
                lab += ' (' + _main['VENUE'][key].split(r'~\cite')[0] + r')~\cite{' + _main['CITE'][key] + '}'
            cells = []
            for ds in DS:
                if ds in d[key]:
                    p, e = d[key][ds]
                    cells += [fmt(p, best[ds][0], 2), fmt(e, best[ds][1], 2)]
                else:
                    cells += ['--', '--']
            L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    open(a.out, 'w').write('\n'.join(L) + '\n')
    print('written', a.out)


if __name__ == '__main__':
    main()
