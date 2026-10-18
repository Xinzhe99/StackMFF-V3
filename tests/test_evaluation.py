"""Regression checks for explicit coverage, retrying failures and subset summaries."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
import common as C
import run_eval as E


def configure(tmp_path):
    C.configure({'datasets': ['MD', 'Road'], 'methods': ['Full', 'Partial'],
                 'out_dir': str(tmp_path), 'results_roots': [], 'aligned_methods': [],
                 'overrides': {}, 'method_datasets': {'Partial': ['MD']}})


def test_explicit_partial_coverage_skips_loading_unreported_cell(tmp_path, monkeypatch):
    configure(tmp_path)
    C.METHODS = ['Partial']
    def forbidden(*args):
        raise AssertionError('An unsupported cell must not read any image')
    monkeypatch.setattr(E.M, 'load_gray', forbidden)
    assert E.work_scene(('Road', 'scene', 'scene.png'))[2] == 0
    assert not list(tmp_path.rglob('*.json'))
    assert C.supports('Full', 'Road')


def test_failed_cache_is_retried_but_successful_cache_is_reused(tmp_path, monkeypatch):
    configure(tmp_path)
    C.METHODS = ['Full']
    cache = Path(E.cache_file('Full', 'MD', 'scene'))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({'error': 'file absent before download'}))
    reads = []
    monkeypatch.setattr(C, 'gt_index', lambda ds: {'scene': 'gt.png'})
    def observed(path):
        reads.append(path)
        raise RuntimeError('sentinel: attempted recomputation')
    monkeypatch.setattr(E.M, 'load_gray', observed)
    result = E.work_scene(('MD', 'scene', 'scene.png'))
    assert 'sentinel: attempted recomputation' in result[-1]
    assert reads == ['gt.png']
    cache.write_text(json.dumps({'raw': {'SSIM': 1.0}}))
    reads.clear()
    assert E.work_scene(('MD', 'scene', 'scene.png'))[-1] == 'cached'
    assert reads == []


def test_summary_load_excludes_old_methods_and_unreported_cells(tmp_path):
    configure(tmp_path)
    records = [{'method': 'Full', 'dataset': 'Road'},
               {'method': 'Partial', 'dataset': 'Road'},
               {'method': 'Obsolete', 'dataset': 'MD'},
               {'method': 'Full', 'dataset': 'OldDataset'}]
    cache = Path(E.cache_root());cache.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(records):
        (cache / f'{i}.json').write_text(json.dumps(rec))
    assert E.load_records() == records[:1]
