"""Publishable means require every expected pair and all eight finite metrics.

Output verification has two accepted evidences, both hash-based:
  * a local output file whose SHA256 matches the score record, or
  * an entry in the server output inventory produced by
    build_output_inventory.sh, which lists the SHA256 of every scored PNG that
    lives on the host. The inventory is transferred once and covers the methods
    that are scored in place instead of being downloaded.

The rule is unchanged: no record contributes without byte-level binding to an
output. Only the location of the bytes differs.
"""
from pathlib import Path
import argparse, csv, json, math, hashlib, statistics
from method_registry import METHODS, PAUSED_METHOD_IDS

HERE = Path(__file__).resolve().parent
METRICS = ['NMI', 'FMI_w', 'Qabf', 'QP', 'QY', 'QCB', 'QCV', 'VIFF']


def load_inventory(paths):
    """Return {(method, dataset, filename): {sha256, ...}} from inventory TSVs.

    A set of digests is kept per key because the same scene may exist on more than
    one host (shard overlap, leftover smoke tests, or a re-render). A score record
    is valid if its bound output hash matches ANY copy that was actually produced;
    the record is bound to the bytes that were scored, and those bytes exist.
    """
    inv = {}
    for path in paths:
        p = Path(path)
        if not p.is_file():
            continue
        for line in p.read_text(encoding='utf8').splitlines():
            parts = line.split('\t')
            if len(parts) != 5:
                continue
            _root, method, dataset, name, digest = parts
            inv.setdefault((method, dataset, name), set()).add(digest)
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score-roots', nargs='+', default=[str(HERE / 'scores'), str(HERE / 'remote_scores')])
    ap.add_argument('--inventory', nargs='*', default=[str(HERE / 'output_inventory.tsv')])
    a = ap.parse_args()
    manifest = json.loads((HERE / 'manifest.json').read_text())
    inputs = {(r['dataset'], r['scene']): r for r in manifest['scenes']}
    inventory = load_inventory(a.inventory)
    records = {}
    conflicts = []
    implementation = hashlib.sha256((HERE / 'metrics' / 'provenance.json').read_bytes()).hexdigest()
    for root in a.score_roots:
        for p in Path(root).glob('*/*/*.json'):
            r = json.loads(p.read_text())
            k = r['method'], r['dataset'], r['scene']
            if k[0] in PAUSED_METHOD_IDS:
                continue
            expected = inputs[k[1:]]
            assert r['input_a_sha256'] == expected['a']['sha256'], p
            assert r['input_b_sha256'] == expected['b']['sha256'], p
            assert all(math.isfinite(r['metrics'][n]) for n in METRICS), p
            assert r['implementation_manifest_sha256'] == implementation, p
            fn = k[2] + ('_rgb.png' if k[0] == 'SwinFusion' else '.png')
            local = [root / k[0] / k[1] / fn for root in [HERE / 'results', HERE / 'remote_results']]
            local = [q for q in local if q.exists()]
            local_ok = bool(local) and all(hashlib.sha256(q.read_bytes()).hexdigest() == r['output_sha256'] for q in local)
            if local:
                assert local_ok, p
            else:
                assert r['output_sha256'] in inventory.get((k[0], k[1], fn), set()), ('no output evidence', p)
            if k in records and records[k] != r:
                conflicts.append(k)
            else:
                records[k] = r
    assert not conflicts, ('Duplicate score records disagree; do not choose by performance', conflicts)
    methods = [m['method'] for m in METHODS]
    assert {k[0] for k in records} <= set(methods)
    means = []
    coverage = []
    rows = []
    for k, r in sorted(records.items()):
        rows.append(dict(method=k[0], dataset=k[1], scene=k[2],
                         **{n: r['metrics'][n] for n in METRICS},
                         output_sha256=r['output_sha256']))
    for method in methods:
        for ds, meta in manifest['datasets'].items():
            ids = {r['scene'] for r in manifest['scenes'] if r['dataset'] == ds}
            rs = [r for (mm, dd, ss), r in records.items() if mm == method and dd == ds]
            complete = {r['scene'] for r in rs} == ids
            coverage.append({'method': method, 'dataset': ds, 'scored': len(rs),
                             'expected': len(ids), 'complete': complete})
            if complete:
                means.append(dict(method=method, dataset=ds, n=len(rs),
                                  **{n: statistics.mean(r['metrics'][n] for r in rs) for n in METRICS}))
    for name, data in [('per_image', rows), ('summary_complete_only', means), ('coverage', coverage)]:
        if not data:
            continue
        with (HERE / (name + '.csv')).open('w', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    print(json.dumps({'scored_outputs': len(rows), 'complete_method_datasets': len(means),
                      'metrics': METRICS, 'QCV_direction': 'lower',
                      'inventory_entries': len(inventory)}))


if __name__ == '__main__':
    main()
