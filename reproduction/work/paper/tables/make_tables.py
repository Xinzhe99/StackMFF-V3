"""LaTeX tables for the paper from work/eval/summary.csv (unified evaluation).

Rows are grouped (transform domain / adapted pairwise / native stack / commercial). In every column the best
value is bold and the second-best underlined, over all rows. Old StackMFF V3/V4 trunks are not shown
(paper requirement). 'Ours' is read from the summary under OURS_NAME once the new model has been evaluated.

  python make_tables.py            -> tab_main_fr.tex (SSIM, PSNR, VIF, NMI per benchmark)
                                      tab_main_nr.tex (Q^{X/F}, MEF-SSIM per benchmark)
"""
import csv, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'eval', 'summary.csv')
OURS_NAME = 'Ours_final_soft'
DATASETS = [('Mobile Depth', 'MD'), ('Middlebury', 'Mid.'), ('FlyingThings3D', 'FT3D'), ('Road-MF', 'Road-MF')]
GROUPS = [
    ('Spatial domain', ['GFF', 'IFM', 'QB', 'BFMF', 'GFDF', 'MGFF']),
    ('Transform domain', ['CVT', 'DWT', 'DCT', 'DTCWT', 'NSCT']),
    ('Pairwise, adapted to stacks', ['IFCNN-MAX', 'U2Fusion', 'SDNet', 'MFF-GAN', 'SwinFusion', 'MUFusion', 'SwinMFF',
                                     'DDBFusion', 'CCSR-Net', 'MCCSR-Net', 'DeFusion', 'MFIF-GAN', 'GIFNet', 'CSNet',
                                     'ReDiffuse', 'ZMFF']),
    ('Commercial software', ['Helicon Focus 8 - A', 'Helicon Focus 8 - B', 'Helicon Focus 8 - C',
                             'Zerene Stacker - DMap', 'Zerene Stacker - PMax']),
    ('Native stack', ['StackMFF_release', 'StackMFF V2', 'VAEEDOF', 'Ours_final_hard', OURS_NAME]),
    # StackMFF: results recomputed with the officially released weights (GitHub Xinzhe99/StackMFF @47515e1);
    # the older stored results came from an unrecorded checkpoint and are not used.
]
DISPLAY = {'GFF': 'GFF', 'IFM': 'IFM', 'DSIFT': 'DSIFT', 'QB': 'QB', 'BFMF': 'BFMF', 'MWGF': 'MWGF',
           'GFDF': 'GFDF', 'MGFF': 'MGFF',
           'Ours_final_hard': 'Ours (hard selection)', 'StackMFF_release': 'StackMFF', 'IFCNN-MAX': 'IFCNN (MAX)', 'Helicon Focus 8 - A': 'Helicon Focus (A)', 'Helicon Focus 8 - B': 'Helicon Focus (B)',
           'Helicon Focus 8 - C': 'Helicon Focus (C)', 'Zerene Stacker - DMap': 'Zerene (DMap)',
           'Zerene Stacker - PMax': 'Zerene (PMax)', OURS_NAME: 'Ours (weighted)'}
CITE = {'GFF': 'li2013image', 'IFM': 'li2013matting', 'DSIFT': 'LIU2015139', 'QB': 'bai2015quadtree',
        'BFMF': 'zhang2017boundary', 'MWGF': 'zhou2014multi', 'GFDF': 'qiu2019guided',
        'MGFF': 'bavirisetti2019multiscale',
        'CVT': 'guo2012multifocus', 'DWT': 'li1995multisensor', 'DCT': 'HAGHIGHAT2011789', 'DTCWT': 'lewis2007pixel',
        'NSCT': 'yang2007image', 'IFCNN-MAX': 'zhang2020ifcnn', 'U2Fusion': 'xu2020u2fusion', 'SDNet': 'zhang2021sdnet',
        'MFF-GAN': 'zhang2021mff', 'SwinFusion': 'ma2022swinfusion', 'MUFusion': 'cheng2023mufusion',
        'SwinMFF': 'xie2024swinmff', 'DDBFusion': 'zhang2025ddbfusion', 'CCSR-Net': 'ZHENG2025102974',
        'MCCSR-Net': 'ZHENG2025102974', 'DeFusion': 'liang2022defusion', 'MFIF-GAN': 'wang2021mfifgan',
        'GIFNet': 'cheng2025gifnet', 'CSNet': 'csnet2026', 'ReDiffuse': 'rediffuse2026', 'ZMFF': 'hu2023zmff',
        'StackMFF': 'xie2025stackmff', 'StackMFF_release': 'xie2025stackmff', 'StackMFF V2': 'xie2025stackmffv2', 'VAEEDOF': 'vaeedof2025'}
# Venue of the reference, printed as a second column of the main table (main.tex Table 1). The cell already
# carries the citation, because the venue and the citation are the same reference.
VENUE = {
    'GFF': 'TIP~\\cite{li2013image}', 'IFM': 'Inf.\\ Fusion~\\cite{li2013matting}',
    'QB': 'Inf.\\ Fusion~\\cite{bai2015quadtree}', 'BFMF': 'Inf.\\ Fusion~\\cite{zhang2017boundary}',
    'GFDF': 'SPIC~\\cite{qiu2019guided}', 'MGFF': 'CSSP~\\cite{bavirisetti2019multiscale}',
    'CVT': 'Opt.\\ Express~\\cite{guo2012multifocus}', 'DWT': 'GMIP~\\cite{li1995multisensor}',
    'DCT': 'CEE~\\cite{HAGHIGHAT2011789}', 'DTCWT': 'Inf.\\ Fusion~\\cite{lewis2007pixel}',
    'NSCT': 'ICIG~\\cite{yang2007image}', 'IFCNN-MAX': 'Inf.\\ Fusion~\\cite{zhang2020ifcnn}',
    'U2Fusion': 'TPAMI~\\cite{xu2020u2fusion}', 'SDNet': 'IJCV~\\cite{zhang2021sdnet}',
    'MFF-GAN': 'Inf.\\ Fusion~\\cite{zhang2021mff}', 'SwinFusion': 'JAS~\\cite{ma2022swinfusion}',
    'MUFusion': 'Inf.\\ Fusion~\\cite{cheng2023mufusion}', 'SwinMFF': 'Vis.\\ Comput.~\\cite{xie2024swinmff}',
    'DDBFusion': 'Inf.\\ Fusion~\\cite{zhang2025ddbfusion}', 'CCSR-Net': 'Inf.\\ Fusion~\\cite{ZHENG2025102974}',
    'MCCSR-Net': 'Inf.\\ Fusion~\\cite{ZHENG2025102974}', 'DeFusion': 'ECCV~\\cite{liang2022defusion}',
    'MFIF-GAN': 'SPIC~\\cite{wang2021mfifgan}', 'GIFNet': 'CVPR~\\cite{cheng2025gifnet}',
    'CSNet': 'arXiv~\\cite{csnet2026}', 'ReDiffuse': 'arXiv~\\cite{rediffuse2026}',
    'ZMFF': 'Inf.\\ Fusion~\\cite{hu2023zmff}',
    'Helicon Focus 8 - A': '---', 'Helicon Focus 8 - B': '---',
    'Helicon Focus 8 - C': '---', 'Zerene Stacker - DMap': '---', 'Zerene Stacker - PMax': '---',
    'StackMFF': 'Appl.\\ Intell.~\\cite{xie2025stackmff}', 'StackMFF_release': 'Appl.\\ Intell.~\\cite{xie2025stackmff}',
    'StackMFF V2': 'EAAI~\\cite{xie2025stackmffv2}', 'VAEEDOF': 'arXiv~\\cite{vaeedof2025}',
    'Ours_final_hard': '---', OURS_NAME: '---',
}
EXCLUDE = set()  # DWT for MD/Mid/FT3D regenerated (stored files were NSCT); nothing excluded now
FMT = {'SSIM_sk7': '{:.4f}', 'MI': '{:.3f}', 'SSIM': '{:.4f}', 'PSNR': '{:.2f}', 'VIF': '{:.4f}', 'NMI': '{:.4f}', 'Qabf_N': '{:.4f}', 'MEF_SSIM': '{:.4f}'}
HEAD = {'SSIM_sk7': 'SSIM$_{7}$', 'MI': 'MI', 'SSIM': 'SSIM', 'PSNR': 'PSNR', 'VIF': 'VIF', 'NMI': 'NMI', 'Qabf_N': '$Q^{\\mathbf{X}/F}$', 'MEF_SSIM': 'MEF-SSIM'}


def load():
    v = {}
    for r in csv.DictReader(open(SUMMARY)):
        if (r['method'], r['dataset']) in EXCLUDE:
            continue
        v[(r['method'], r['dataset'], r['metric'])] = float(r['mean']) if r['mean'] else float('nan')
    return v


def table(v, metrics, caption, label, dsets=None, font='\\scriptsize', venue=False,
          env='table*', stretch=None, rules=False, citations=True):
    global DATASETS
    full = DATASETS
    if dsets is not None:
        DATASETS = [d for d in DATASETS if d[0] in dsets]
    try:
        return _table(v, metrics, caption, label, font, full, venue, env, stretch, rules, citations)
    finally:
        DATASETS = full


def _table(v, metrics, caption, label, font='\\scriptsize', all_ds=None, venue=False,
           env='table*', stretch=None, rules=False, citations=True):
    present = [(g, [m for m in ms if any((m, d, metrics[0]) in v for d, _ in (all_ds or DATASETS))]) for g, ms in GROUPS]
    rows = [m for _, ms in present for m in ms]
    rank = {}
    for d, _ in DATASETS:
        for k in metrics:
            vals = sorted({v[(m, d, k)] for m in rows if (m, d, k) in v and v[(m, d, k)] == v[(m, d, k)]}, reverse=True)
            for m in rows:
                x = v.get((m, d, k))
                rank[(m, d, k)] = vals.index(x) + 1 if x is not None and x in vals else None
    ncol = len(DATASETS) * len(metrics)
    ven = 1 if venue else 0
    L = [f'\\begin{{{env}}}[t]', '\\centering', font,
         '\\renewcommand{\\arraystretch}{%s}' % (stretch or ('0.78' if venue else '0.94')),
         '\\setlength{\\tabcolsep}{3.2pt}']
    if rules or venue:
        L += ['\\setlength{\\aboverulesep}{0.2ex}', '\\setlength{\\belowrulesep}{0.2ex}']
    if venue:
        L += ['\\setlength{\\abovecaptionskip}{2pt}', '\\setlength{\\belowcaptionskip}{3pt}']
    elif rules:
        L += ['\\setlength{\\abovecaptionskip}{1pt}', '\\setlength{\\belowcaptionskip}{2pt}']
    L += [f'\\caption{{{caption}}}', f'\\label{{{label}}}',
          '\\resizebox{\\textwidth}{!}{%',
          '\\begin{tabular}{l' + 'l' * ven + 'c' * ncol + '}', '\\toprule',
          'Method & ' + ('Venue & ' if venue else '') +
          ' & '.join(f'\\multicolumn{{{len(metrics)}}}{{c}}{{{s}}}' for _, s in DATASETS) + ' \\\\']
    if venue:
        L.append('\\cmidrule(lr){1-1} ' +
                 ' '.join(f'\\cmidrule(lr){{{3 + i * len(metrics)}-{2 + (i + 1) * len(metrics)}}}' for i in range(len(DATASETS))))
    else:
        L.append(' '.join(f'\\cmidrule(lr){{{2 + i * len(metrics)}-{1 + (i + 1) * len(metrics)}}}' for i in range(len(DATASETS))))
    L.append(' & ' + ('& ' if venue else '') + ' & '.join(HEAD[k] for _ in DATASETS for k in metrics) + ' \\\\')
    for g, ms in present:
        if not ms:
            continue
        L.append('\\midrule')
        L.append(f'\\multicolumn{{{ncol + 1 + ven}}}{{l}}{{\\textit{{{g}}}}} \\\\')
        for m in ms:
            name = DISPLAY.get(m, m) + ('' if venue or not citations else (f'~\\cite{{{CITE[m]}}}' if m in CITE else ''))
            vcell = (VENUE[m] + ' & ') if venue else ''
            cells = []
            for d, _ in DATASETS:
                for k in metrics:
                    x = v.get((m, d, k))
                    if x is None or x != x:
                        cells.append('--'); continue
                    s = FMT[k].format(x)
                    r = rank[(m, d, k)]
                    s = f'\\best{{{s}}}' if r == 1 else (f'\\second{{{s}}}' if r == 2 else s)
                    cells.append(s)
            L.append(name + ' & ' + vcell + ' & '.join(cells) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}}', f'\\end{{{env}}}']
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    v = load()
    CAP = ('Full-reference results (BT.601 luma, full image). Best/second best: pink/blue. '
           'Pairwise methods use numeric focal order. --: not evaluated (ZMFF/ReDiffuse cover MD and Middlebury only). '
           'Deployment details and additional metrics are in the supplementary material.')
    open(os.path.join(HERE, 'tab_main_fr.tex'), 'w').write(table(
        v, ['SSIM', 'PSNR'], CAP, 'tab:main_fr', font='\\footnotesize', stretch='0.76', rules=True, citations=True))
    open(os.path.join(HERE, 'tab_supp_vif_nmi.tex'), 'w').write(table(
        v, ['VIF', 'NMI'],
        'VIF and NMI for the same runs as Table~1 of the main paper. Best and second-best values are '
        'highlighted in pink and blue; --: not evaluated.', 'tab:supp_vif_nmi', font='\\footnotesize',
        stretch='0.74', rules=True))
    open(os.path.join(HERE, 'tab_main_nr.tex'), 'w').write(table(
        v, ['Qabf_N', 'MEF_SSIM'],
        'No-reference fusion metrics computed against the input stack. Full method references are listed with the supplementary deployment protocol. Best and second-best unrounded means are highlighted in pink and blue. --: not evaluated.',
        'tab:main_nr', env='table'))
    open(os.path.join(HERE, 'tab_supp_extra.tex'), 'w').write(table(
        v, ['SSIM_sk7', 'MI'],
        'Additional metrics. SSIM$_{7}$: SSIM with the default settings of scikit-image ($7\\times7$ uniform window), '
        'the convention of earlier StackMFF tables; MI: mutual information with the reference (bits). Best and second-best values are highlighted in pink and blue, respectively. --: not evaluated.',
        'tab:supp_extra', font='\\footnotesize', stretch='0.66', rules=True))
    print('written', os.listdir(HERE))
