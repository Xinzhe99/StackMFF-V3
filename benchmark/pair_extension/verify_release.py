"""Reproduce every published table from the released records.

This script is the public contract for the two-input extension: it regenerates the
six LaTeX tables from `summary_source_metrics.csv` and
`summary_reference_metrics.csv` alone, then compares them byte for byte against the
tables shipped in `benchmark/updates/2026-10-20-pair-extension/tables/`.

A reader who runs this either sees every table reproduce exactly, or sees precisely
which file differs. No manuscript source and no experiment workspace is required.

Usage:
    python verify_release.py [--updates DIR]

Default layout assumes this file sits in benchmark/pair_extension/ and the update
directory is its sibling, which is how the repository is published.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_UPDATES = HERE.parent / 'updates' / '2026-10-20-pair-extension'
METRICS = ['NMI', 'FMI_w', 'Qabf', 'QP', 'QY', 'QCB', 'QCV', 'VIFF']
EXPECTED_TABLES = ['tab_pair_Lytro.tex', 'tab_pair_MFFW.tex', 'tab_pair_MFI-WHU.tex',
                   'tab_pair_RealMFF.tex', 'tab_pair_ref_MFI-WHU.tex', 'tab_pair_ref_RealMFF.tex']


def prepare(updates, work):
    """Stage the generator inputs in a temporary tree."""
    for name in ('aggregate_scores.py', 'aggregate_verified_references.py',
                 'make_pair_tables.py', 'make_reference_tables.py',
                 'method_registry.py', 'methods.json', 'manifest.json', 'metrics_lib.py',
                 'reference_protocol.py'):
        shutil.copy2(HERE / name, work / name)
    shutil.copytree(HERE / 'metrics', work / 'metrics')
    shutil.copy2(updates / 'summary_source_metrics.csv', work / 'summary_complete_only.csv')
    shutil.copy2(updates / 'summary_reference_metrics.csv', work / 'reference_summary_complete_only.csv')
    shutil.copy2(updates / 'coverage_source_metrics.csv', work / 'coverage.csv')
    shutil.copy2(updates / 'coverage_reference_metrics.csv', work / 'reference_coverage.csv')


def check_records(updates):
    """The released per-image table must have one row per pair and method."""
    per = updates / 'per_image_source_metrics.csv'
    rows = list(csv.DictReader(per.open()))
    manifest = json.loads((HERE / 'manifest.json').read_text())
    pairs = len(manifest['scenes'])
    methods = len(json.loads((HERE / 'methods.json').read_text())['methods'])
    ok = len(rows) == pairs * methods
    return {'per_image_rows': len(rows), 'expected': pairs * methods, 'ok': ok}


def check_summaries(updates):
    """Recompute each dataset mean from the per-image rows and compare."""
    per = list(csv.DictReader((updates / 'per_image_source_metrics.csv').open()))
    summ = list(csv.DictReader((updates / 'summary_source_metrics.csv').open()))
    acc = {}
    for r in per:
        acc.setdefault((r['method'], r['dataset']), []).append(r)
    bad = 0
    for s in summ:
        rs = acc.get((s['method'], s['dataset']), [])
        if len(rs) != int(s['n']):
            bad += 1
            continue
        for m in METRICS:
            mean = sum(float(x[m]) for x in rs) / len(rs)
            if abs(mean - float(s[m])) > 1e-9:
                bad += 1
    return {'summary_rows': len(summ), 'mismatches': bad, 'ok': bad == 0}


def regenerate(updates):
    """Run the generators in a temporary tree and diff against released tables."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        prepare(updates, work)
        for script in ('make_pair_tables.py', 'make_reference_tables.py'):
            p = subprocess.run([sys.executable, script], cwd=work, capture_output=True, text=True)
            if p.returncode != 0:
                return {'ok': False, 'error': script, 'detail': p.stderr[-500:]}
        produced = work / 'publication_draft' / 'tables'
        results = {}
        for name in EXPECTED_TABLES:
            a, b = produced / name, updates / 'tables' / name
            results[name] = a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()
        return {'ok': all(results.values()), 'tables': results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--updates', default=str(DEFAULT_UPDATES))
    args = ap.parse_args()
    updates = Path(args.updates)
    if not updates.is_dir():
        raise SystemExit('update directory not found: %s' % updates)

    report = {
        'records': check_records(updates),
        'summary': check_summaries(updates),
        'tables': regenerate(updates),
    }
    print(json.dumps(report, indent=1))
    ok = report['records']['ok'] and report['summary']['ok'] and report['tables']['ok']
    print('REPRODUCED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
