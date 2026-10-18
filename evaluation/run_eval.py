"""Resumable evaluation runner.

  python3 run_eval.py run --config config.json [--workers 2] [--only substring] [--dataset NAME]
  python3 run_eval.py summarize --config config.json
  (or give --gt_root, --stack_root, --results_root, --out_dir [--methods ...] instead of / in addition to --config)

run: per scene (dataset/scene), loads GT and the source stack once, then for every method
     computes raw and aligned metrics; one JSON cache file per (method, dataset, scene).
     Existing cache files are skipped, so the run can be restarted.
summarize: builds per_image.csv, summary.csv, summary.md, alignment_log.md from the caches.
"""
import os, sys, re, json, time, math, argparse, traceback, csv
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import metrics_lib as M
import common as C
from common import natural_key


def cache_root():
    return os.path.join(C.EVAL, 'cache')


METRICS = ['SSIM', 'SSIM_sk7', 'PSNR', 'VIF', 'MI', 'NMI', 'Qabf_N', 'MEF_SSIM']
SHIFT_THRESH = 2.0


def mdir(method):
    return re.sub(r'[^A-Za-z0-9]+', '_', method).strip('_')


def cache_file(method, ds, scene):
    return os.path.join(cache_root(), mdir(method), ds.replace(' ', '_'), scene + '.json')


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def scene_list(only=None, dataset=None):
    out = []
    for ds in C.DATASETS:
        if dataset and ds != dataset:
            continue
        gi = C.gt_index(ds)
        for key in sorted(gi, key=natural_key):
            gtfile = os.path.basename(gi[key])
            scene = os.path.splitext(gtfile)[0]
            if only and only not in f'{ds}:{scene}':
                continue
            out.append((ds, key, gtfile))
    return out


def work_scene(args):
    ds, key, gtfile = args
    scene = os.path.splitext(gtfile)[0]
    t0 = time.time()
    todo = []
    for method in C.METHODS:
        if not C.supports(method, ds):
            continue
        path = cache_file(method, ds, scene)
        try:
            with open(path) as f:
                cached = json.load(f)
            valid = 'raw' in cached and 'error' not in cached
        except (OSError, ValueError):
            valid = False
        if not valid:
            todo.append(method)
    if not todo:
        return (ds, scene, 0, 0.0, 'cached')
    msgs = []
    try:
        gt = M.load_gray(C.gt_index(ds)[key])
        si = C.stack_index(ds)
        if key not in si:
            raise RuntimeError('no source stack for ' + scene)
        files = C.stack_files(si[key])
        layers, stack_resized = [], 0
        for f in files:
            x, rs = M.resize_to(M.load_gray(f), gt.shape)
            stack_resized += int(rs)
            layers.append(x)
        L = np.stack(layers)
        src_grads = [M.grad_orient(x) for x in layers]
        Rf = M.mef_precompute(L)
        gkp = M.gt_keypoints(gt)
    except Exception as e:
        return (ds, scene, 0, time.time() - t0, 'SCENE ERROR: ' + repr(e))
    done = 0
    for meth in todo:
        rec = {'method': meth, 'dataset': ds, 'scene': scene, 'gt_file': gtfile,
               'gt_size': list(gt.shape), 'n_layers': len(layers), 'stack_resized_layers': stack_resized}
        try:
            ri = C.result_index(meth, ds)
            if key not in ri:
                rec['error'] = 'missing result'
                jdump(rec, cache_file(meth, ds, scene))
                msgs.append(f'{meth}: missing result')
                continue
            rec['result_file'] = os.path.relpath(ri[key], C.RES)
            res0 = M.load_gray(ri[key])
            rec['result_size'] = list(res0.shape)
            rec['result_resized'] = bool(res0.shape != gt.shape)
            res, _ = M.resize_to(res0, gt.shape)
            dy, dx = M.shift_phase(gt, res)
            rec['shift'] = [dy, dx]
            rec['shift_mag'] = math.hypot(dy, dx)
            raw = M.all_metrics(res, gt, src_grads, Rf)
            if (meth in C.COMMERCIAL) or (rec['shift_mag'] > SHIFT_THRESH):
                # homography is estimated from the ORIGINAL-size result (no prior resize), so a crop/scale
                # difference is undone by a single interpolation
                warped, info = M.align_to_gt(res0, gt, gkp, (dy, dx))
                aligned = M.all_metrics(warped, gt, src_grads, Rf)
            else:
                # non-commercial method with shift <= 2 px: aligned variant not needed (see alignment_log.md)
                info = {'method': 'not computed', 'matches': 0, 'inliers': 0, 'scale': None,
                        'fallback': False, 'valid_frac': None}
                aligned = None
            rec['align'] = info
            rec['raw'] = raw
            rec['aligned'] = aligned
        except Exception as e:
            rec['error'] = repr(e)
            msgs.append(f'{meth}: ERROR {e!r}')
            traceback.print_exc()
        jdump(rec, cache_file(meth, ds, scene))
        done += 1
    return (ds, scene, done, time.time() - t0, '; '.join(msgs))


def cmd_run(a):
    scenes = scene_list(a.only, a.dataset)
    print(f'{len(scenes)} scenes', flush=True)
    T = time.time()
    n = 0
    with Pool(a.workers, initializer=C.configure, initargs=(a.cfg,)) as pool:
        for ds, scene, done, dt, msg in pool.imap_unordered(work_scene, scenes):
            n += 1
            print(f'[{n}/{len(scenes)}] {ds}/{scene}: computed {done} methods in {dt:.1f}s {msg}', flush=True)
    print(f'DONE in {time.time()-T:.0f}s', flush=True)


# ------------------------------------------------------------------ summarize
def load_records():
    recs = []
    for root, _, files in os.walk(cache_root()):
        for f in files:
            if f.endswith('.json'):
                with open(os.path.join(root, f)) as fh:
                    record = json.load(fh)
                    if (record.get('method') in C.METHODS and record.get('dataset') in C.DATASETS
                            and C.supports(record['method'], record['dataset'])):
                        recs.append(record)
    return recs


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ''
    if isinstance(v, float) and math.isinf(v):
        return 'inf'
    return repr(float(v)) if isinstance(v, float) else str(v)


def cmd_summarize(a):
    from scipy.stats import rankdata
    recs = [r for r in load_records() if 'raw' in r]
    errs = [r for r in load_records() if 'error' in r]
    by = {}
    for r in recs:
        by.setdefault((r['method'], r['dataset']), []).append(r)
    # alignment decision per (method, dataset)
    flag = {}
    for (m, ds), rs in by.items():
        shifts = [r['shift_mag'] for r in rs]
        med = float(np.median(shifts))
        all_aligned = all(r.get('aligned') is not None for r in rs)
        flag[(m, ds)] = (m in C.COMMERCIAL) or (med > SHIFT_THRESH and all_aligned)
    # per-image CSV
    cols = ['method', 'dataset', 'scene', 'variant', 'primary', 'result_size', 'gt_size', 'result_resized',
            'n_layers', 'stack_resized_layers', 'shift_dy', 'shift_dx', 'shift_mag', 'align_method',
            'align_matches', 'align_inliers', 'align_scale', 'align_fallback', 'align_valid_frac'] + METRICS
    rows = []
    for r in sorted(recs, key=lambda r: (C.METHODS.index(r['method']), C.DATASETS.index(r['dataset']), natural_key(r['scene']))):
        f = flag[(r['method'], r['dataset'])]
        al = r['align']
        base = {'method': r['method'], 'dataset': r['dataset'], 'scene': r['scene'],
                'result_size': 'x'.join(map(str, r['result_size'])), 'gt_size': 'x'.join(map(str, r['gt_size'])),
                'result_resized': r['result_resized'], 'n_layers': r['n_layers'],
                'stack_resized_layers': r['stack_resized_layers'], 'shift_dy': r['shift'][0],
                'shift_dx': r['shift'][1], 'shift_mag': r['shift_mag'], 'align_method': al['method'],
                'align_matches': al['matches'], 'align_inliers': al['inliers'], 'align_scale': al['scale'],
                'align_fallback': al['fallback'], 'align_valid_frac': al['valid_frac']}
        for variant in ('raw', 'aligned'):
            if r[variant] is None:
                continue
            row = dict(base)
            row['variant'] = variant
            row['primary'] = (variant == 'aligned') == f
            for k in METRICS:
                row[k] = r[variant][k]
            rows.append(row)
    with open(os.path.join(C.EVAL, 'per_image.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(row.get(k)) for k in cols})
    # summary stats (primary variant) and raw alternatives
    def stats(sel, k):
        v = np.array([row[k] for row in sel], dtype=float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float('nan'), int(v.size)
    summ = {}   # (method, dataset) -> {metric: (mean, n)} primary
    rawsumm = {}
    for (m, ds) in by:
        prim = [row for row in rows if row['method'] == m and row['dataset'] == ds and row['primary']]
        rw = [row for row in rows if row['method'] == m and row['dataset'] == ds and row['variant'] == 'raw']
        summ[(m, ds)] = {k: stats(prim, k) for k in METRICS}
        rawsumm[(m, ds)] = {k: stats(rw, k) for k in METRICS}
    ranks = {}
    for ds in C.DATASETS:
        for k in METRICS:
            ms = [m for m in C.METHODS if (m, ds) in summ]
            vals = np.array([summ[(m, ds)][k][0] for m in ms])
            rk = rankdata(-vals, method='min') if len(ms) else []
            for m, rr in zip(ms, rk):
                ranks[(m, ds, k)] = int(rr)
    with open(os.path.join(C.EVAL, 'summary.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['method', 'dataset', 'variant_used', 'metric', 'mean', 'n_images', 'rank_in_dataset', 'n_methods'])
        for ds in C.DATASETS:
            for m in C.METHODS:
                if (m, ds) not in summ:
                    continue
                vu = 'aligned' if flag[(m, ds)] else 'raw'
                for k in METRICS:
                    mu, n = summ[(m, ds)][k]
                    w.writerow([m, ds, vu, k, fmt(mu), n, ranks[(m, ds, k)], sum(1 for x in C.METHODS if (x, ds) in summ)])
    # summary.md
    L = []
    L.append('# StackMFF fusion evaluation: summary')
    L.append('')
    L.append('Mean over images per dataset. Higher is better for every metric. Best per dataset and metric in **bold**. '
             'Methods marked with * use the aligned (ORB+RANSAC homography) result (Helicon Focus, and any method whose median '
             'phase-correlation shift is > 2 px on that dataset); all other rows are raw. Gray = BT.601 luma, full image.')
    L.append('')
    L.append('Metrics: SSIM = Wang 2004 (11x11 Gaussian, sigma 1.5) vs GT; SSIM_sk7 = skimage default (7x7 uniform) vs GT; '
             'PSNR vs GT (dB); VIF = VIFp vs GT; MI / NMI (bits, 256 bins) vs GT; Qabf_N = Q^{AB/F} over all N source layers; '
             'MEF_SSIM = mef_ssim.m (single scale) over all N source layers.')
    L.append('')
    for ds in C.DATASETS:
        ms = [m for m in C.METHODS if (m, ds) in summ]
        if not ms:
            continue
        L.append(f'## {ds}')
        L.append('')
        L.append('| Method | ' + ' | '.join(METRICS) + ' | n |')
        L.append('|---|' + '---|' * (len(METRICS) + 1))
        best = {k: max(summ[(m, ds)][k][0] for m in ms if np.isfinite(summ[(m, ds)][k][0])) for k in METRICS}
        for m in ms:
            cells = []
            for k in METRICS:
                v = summ[(m, ds)][k][0]
                s = f'{v:.4f}' if k not in ('PSNR',) else f'{v:.3f}'
                if k == 'PSNR' and not np.isfinite(v):
                    s = 'inf'
                if np.isfinite(v) and abs(v - best[k]) < 1e-12:
                    s = f'**{s}**'
                cells.append(s)
            star = '*' if flag[(m, ds)] else ''
            L.append(f'| {m}{star} | ' + ' | '.join(cells) + f' | {summ[(m, ds)]["SSIM"][1]} |')
        L.append('')
    # raw alternative for aligned methods
    L.append('## Raw (unaligned) values for methods that use the aligned variant')
    L.append('')
    L.append('| Dataset | Method | ' + ' | '.join(METRICS) + ' | aligned minus raw (primary - raw) for SSIM |')
    L.append('|---|---|' + '---|' * (len(METRICS) + 1))
    for ds in C.DATASETS:
        for m in C.METHODS:
            if (m, ds) in summ and flag[(m, ds)]:
                cells = [f'{rawsumm[(m, ds)][k][0]:.4f}' if k != 'PSNR' else f'{rawsumm[(m, ds)][k][0]:.3f}' for k in METRICS]
                d = summ[(m, ds)]['SSIM'][0] - rawsumm[(m, ds)]['SSIM'][0]
                L.append(f'| {ds} | {m} | ' + ' | '.join(cells) + f' | {d:+.4f} |')
    L.append('')
    # optional: comparison with the values of an earlier table (internal consistency check of the paper; --old_table)
    disc = []
    if a.old_table:
        # comparison with old table
        L.append('## Comparison with the old table (stackmff_tables.xlsx, sheet stackmffv4, SSIM and PSNR)')
        L.append('')
        import openpyxl
        wb = openpyxl.load_workbook(a.old_table, data_only=True)
        ws = wb['stackmffv4']
        old = {}
        for row in ws.iter_rows(min_row=3, max_row=28, values_only=True):  # first table only (rows below hold model-size info)
            name = row[0]
            if isinstance(name, str) and name.strip() in C.METHODS:
                old[name.strip()] = row[1:9]  # MD SSIM,PSNR; Mid SSIM,PSNR; FT SSIM,PSNR; Road SSIM,PSNR
        oldmap = {ds: (2 * i, 2 * i + 1) for i, ds in enumerate(['Mobile Depth', 'Middlebury', 'FlyingThings3D', 'Road-MF'])}
        L.append('Differences flagged when |SSIM diff| > 0.002 or |PSNR diff| > 0.1 dB. Ours uses SSIM_sk7 (skimage 7x7) to match the old convention; '
                 'raw = unaligned, primary = as used in the summary.')
        L.append('')
        L.append('| Dataset | Method | old SSIM | ours SSIM_sk7 raw | ours SSIM_sk7 primary | diff (raw) | old PSNR | ours PSNR raw | ours PSNR primary | diff (raw) | flag |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for ds in C.DATASETS:
            for m in C.METHODS:
                if m not in old or (m, ds) not in summ:
                    continue
                oS, oP = old[m][oldmap[ds][0]], old[m][oldmap[ds][1]]
                rS = rawsumm[(m, ds)]['SSIM_sk7'][0]; pS = summ[(m, ds)]['SSIM_sk7'][0]
                rP = rawsumm[(m, ds)]['PSNR'][0]; pP = summ[(m, ds)]['PSNR'][0]
                dS = rS - oS; dP = rP - oP
                f = (abs(dS) > 0.002) or (abs(dP) > 0.1)
                if f:
                    disc.append((ds, m, oS, rS, pS, dS, oP, rP, pP, dP))
                L.append(f'| {ds} | {m} | {oS:.4f} | {rS:.4f} | {pS:.4f} | {dS:+.4f} | {oP:.3f} | {rP:.3f} | {pP:.3f} | {dP:+.3f} | {"DIFF" if f else ""} |')
        L.append('')
        n_rows = sum(1 for ds in C.DATASETS for m in C.METHODS if m in old and (m, ds) in summ)
        L.append(f'{len(disc)} of {n_rows} method/dataset rows are flagged (DIFF).')
    open(os.path.join(C.EVAL, 'summary.md'), 'w').write('\n'.join(L) + '\n')
    # alignment log
    A = ['# Alignment and size log', '',
         'Phase correlation (skimage, 1/10 px) gives the translation that registers each result onto GT. '
         'ORB (2000 features) + RANSAC homography (cv2.findHomography, reprojection threshold 5 px) gives the aligned '
         'variant; if the homography has < 10 inliers or an implausible scale it falls back to the translation. '
         'valid_frac = fraction of the GT canvas covered by the warped result (uncovered pixels are filled by edge replication; metrics are on the full GT canvas).', '',
         '| Method | Dataset | n | median |shift| px | max |shift| px | frac > 2 px | aligned used | homography scale median (min-max) | inliers median | fallback count | valid_frac median | resized results |',
         '|---|---|---|---|---|---|---|---|---|---|---|']
    for ds in C.DATASETS:
        for m in C.METHODS:
            if (m, ds) not in by:
                continue
            rs = by[(m, ds)]
            sh = np.array([r['shift_mag'] for r in rs])
            sc = np.array([r['align']['scale'] for r in rs if r['align']['scale'] is not None])
            sc = sc if sc.size else np.array([float('nan')])
            inl = np.array([r['align']['inliers'] for r in rs])
            fb = sum(1 for r in rs if r['align']['fallback'])
            vf = np.array([r['align']['valid_frac'] for r in rs if r['align']['valid_frac'] is not None])
            vf = vf if vf.size else np.array([float('nan')])
            rz = sum(1 for r in rs if r['result_resized'])
            na = sum(1 for r in rs if r['aligned'] is None)
            A.append(f'| {m} | {ds} | {len(rs)} | {np.median(sh):.2f} | {sh.max():.2f} | {np.mean(sh>SHIFT_THRESH):.2f} | '
                     f'{"yes" if flag[(m,ds)] else "no"} (aligned computed for {len(rs)-na}/{len(rs)}) | {np.median(sc):.4f} ({sc.min():.4f}-{sc.max():.4f}) | {np.median(inl):.0f} | {fb} | {np.median(vf):.3f} | {rz} |')
    A.append('')
    A.append('## Results whose size differs from GT (resized bicubic to GT size)')
    A.append('')
    rz = [r for r in recs if r['result_resized']]
    if not rz:
        A.append('None.')
    for r in sorted(rz, key=lambda r: (r['method'], r['dataset'], natural_key(r['scene']))):
        A.append(f"- {r['method']} / {r['dataset']} / {r['scene']}: result {tuple(r['result_size'])} vs GT {tuple(r['gt_size'])}")
    A.append('')
    A.append('## Stack layers resized to GT size')
    A.append('')
    srz = sorted({(r['dataset'], r['scene'], r['stack_resized_layers']) for r in recs if r['stack_resized_layers']})
    A.append('None.' if not srz else '\n'.join(f'- {d} / {s}: {k} layers' for d, s, k in srz))
    A.append('')
    A.append('## Errors and missing results')
    A.append('')
    A.append('None.' if not errs else '\n'.join(f"- {r['method']} / {r['dataset']} / {r['scene']}: {r['error']}" for r in errs))
    open(os.path.join(C.EVAL, 'alignment_log.md'), 'w').write('\n'.join(A) + '\n')
    print(f'records={len(recs)} errors={len(errs)} rows={len(rows)} discrepancies={len(disc)}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Unified evaluation of fused images against references (see common.py).')
    ap.add_argument('cmd', choices=['run', 'summarize'])
    ap.add_argument('--config', default=os.environ.get('STACKMFF_EVAL_CONFIG'), help='JSON configuration (config_example.json)')
    ap.add_argument('--gt_root'); ap.add_argument('--gt_subdir'); ap.add_argument('--stack_root')
    ap.add_argument('--results_root', nargs='+', dest='results_roots')
    ap.add_argument('--out_dir'); ap.add_argument('--methods', nargs='+'); ap.add_argument('--datasets', nargs='+')
    ap.add_argument('--aligned_methods', nargs='+')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--only', default=None, help='only scenes whose "<dataset>:<scene>" contains this substring')
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--old_table', default='', help='summarize: compare with an earlier table (xlsx, sheet stackmffv4)')
    a = ap.parse_args()
    a.cfg = C.configure(a.config, gt_root=a.gt_root, gt_subdir=a.gt_subdir, stack_root=a.stack_root, results_roots=a.results_roots,
                        out_dir=a.out_dir, methods=a.methods, datasets=a.datasets, aligned_methods=a.aligned_methods)
    a.cfg.update(methods=C.METHODS, out_dir=C.EVAL)
    print(f'{len(C.METHODS)} methods, datasets {C.DATASETS}, output {C.EVAL}', flush=True)
    if a.cmd == 'run':
        cmd_run(a)
    else:
        cmd_summarize(a)
