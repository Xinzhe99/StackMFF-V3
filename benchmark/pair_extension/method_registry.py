"""Method registry for the two-input fusion extension.

Self-contained replacement for the private generator-derived registry used during
the experiment. The public data file `methods.json` records the 37 active
configurations with their display labels, venue citations and groups, so the
aggregation and table scripts run without access to the manuscript sources.
"""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
_DATA = json.loads((HERE / 'methods.json').read_text(encoding='utf8'))
METHODS = _DATA['methods']
PAUSED_METHOD_IDS = set(_DATA.get('paused', []))
assert len(METHODS) == _DATA['count'], 'methods.json is internally inconsistent'
