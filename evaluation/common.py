"""Paths, method list and file matching of the evaluation toolkit.

The configuration comes from a JSON file (run_eval.py --config, or the environment variable STACKMFF_EVAL_CONFIG for
validate.py and xval/export_cases.py) and/or command-line options; see config_example.json:

  gt_root        <gt_root>/<Dataset>[/<gt_subdir>]/<scene>.<ext>        reference (all-in-focus) images
  gt_subdir      optional, e.g. "AiF" when gt_root is the benchmark root (same files as the benchmark's Ground Truth)
  stack_root     <stack_root>/<Dataset>/image stack/<scene>/<images>     source stacks (for Q^{AB/F} and MEF-SSIM)
  results_roots  list of roots, <root>/<Method>/<Dataset>/<scene>.<ext>  fused results (first root holding the method)
  out_dir        cache/, per_image.csv, summary.csv, summary.md, alignment_log.md, stats.*
  datasets       dataset folder names
  methods        method folder names (default: every folder found in the results roots)
  aligned_methods  methods always evaluated after homography alignment to the reference (the paper: Helicon Focus,
                   whose output is cropped/rescaled); any other method is aligned only if its median shift is > 2 px
  overrides      {"<Method>|<Dataset>": "<folder>"}: results of one method/dataset stored elsewhere

Scene matching ignores case, extension and non-alphanumeric characters (norm()).
"""
import json, os, re

DATASETS = ['Mobile Depth', 'Middlebury', 'FlyingThings3D', 'Road-MF']
GTDIR = ''
GT_SUB = ''
STK = ''
RES_ROOTS = []
RES = ''
EVAL = os.path.abspath('eval_out')
METHODS = []
COMMERCIAL = set()
OVERRIDE = {}
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def configure(cfg=None, **kw):
    """cfg: dict or path of a JSON file; keyword arguments override it (None values are ignored)."""
    global DATASETS, GTDIR, GT_SUB, STK, RES_ROOTS, RES, EVAL, METHODS, COMMERCIAL, OVERRIDE
    if isinstance(cfg, str):
        with open(cfg) as f:
            cfg = json.load(f)
    c = dict(cfg or {})
    c.update({k: v for k, v in kw.items() if v is not None})
    DATASETS = list(c.get('datasets', DATASETS))
    GTDIR = c.get('gt_root', GTDIR)
    GT_SUB = c.get('gt_subdir', GT_SUB) or ''
    STK = c.get('stack_root', STK)
    roots = c.get('results_roots', c.get('results_root', RES_ROOTS))
    RES_ROOTS = [roots] if isinstance(roots, str) else list(roots)
    RES = RES_ROOTS[0] if RES_ROOTS else ''
    EVAL = os.path.abspath(c.get('out_dir', EVAL))
    COMMERCIAL = set(c.get('aligned_methods', COMMERCIAL))
    OVERRIDE = {tuple(k.split('|', 1)): v for k, v in c.get('overrides', {}).items()} if 'overrides' in c else OVERRIDE
    METHODS = list(c.get('methods') or METHODS or _discover_methods())
    return c


def _discover_methods():
    found = []
    for r in RES_ROOTS:
        if os.path.isdir(r):
            found += [m for m in sorted(os.listdir(r), key=natural_key)
                      if os.path.isdir(os.path.join(r, m)) and m not in found and m != 'Ground Truth']
    return found + [m for m, _ in OVERRIDE if m not in found]


def norm(s):
    s = os.path.splitext(s)[0].lower()
    return re.sub(r'[^a-z0-9]', '', s)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def gt_path(dataset, scene_file):
    return os.path.join(GTDIR, dataset, GT_SUB, scene_file)


def method_dir(method, dataset):
    if (method, dataset) in OVERRIDE:
        return OVERRIDE[(method, dataset)]
    for r in RES_ROOTS:
        if os.path.isdir(os.path.join(r, method)):
            return os.path.join(r, method, dataset)
    return os.path.join(RES, method, dataset)


def result_index(method, dataset):
    """Map normalized scene key -> file path for a method/dataset."""
    d = method_dir(method, dataset)
    return {norm(f): os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT)}


def gt_index(dataset):
    d = os.path.join(GTDIR, dataset, GT_SUB)
    return {norm(f): os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT)}


def stack_index(dataset):
    d = os.path.join(STK, dataset, 'image stack')
    return {norm(s): os.path.join(d, s) for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))}


def stack_files(scene_dir):
    fs = [f for f in os.listdir(scene_dir) if f.lower().endswith(IMG_EXT)]
    return [os.path.join(scene_dir, f) for f in sorted(fs, key=natural_key)]


if os.environ.get('STACKMFF_EVAL_CONFIG'):
    configure(os.environ['STACKMFF_EVAL_CONFIG'])
