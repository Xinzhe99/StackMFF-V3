import os, re
BASE = '/mnt/f/Working'
RES = '/mnt/e/StackMFF Series Benchmark/results_of_different_methods'
GTDIR = os.path.join(RES, 'Ground Truth')
STK = '/mnt/e/ImagefusionDatasets/StackMFF'
EVAL = os.path.join(BASE, 'StackMFFV3V4', 'work', 'eval')
DATASETS = ['Mobile Depth', 'Middlebury', 'FlyingThings3D', 'Road-MF']
METHODS = ['CVT', 'DWT', 'DCT', 'DTCWT', 'NSCT', 'IFCNN-MAX', 'U2Fusion', 'SDNet', 'MFF-GAN',
           'SwinFusion', 'MUFusion', 'SwinMFF', 'DDBFusion', 'CCSR-Net', 'MCCSR-Net',
           'Zerene Stacker - DMap', 'Zerene Stacker - PMax',
           'Helicon Focus 8 - A', 'Helicon Focus 8 - B', 'Helicon Focus 8 - C',
           'StackMFF', 'StackMFF V2', 'StackMFF V4']
# new baselines deployed in 2026-10 (work/new_methods); results live outside the benchmark tree
NEW_RES = '/mnt/f/Working/StackMFFV3V4/work/new_methods/server_copy/results'
NEW_METHODS = ['GIFNet', 'DeFusion', 'CSNet', 'MFIF-GAN', 'VAEEDOF', 'StackMFF_release', 'Ours_H_soft', 'Ours_H_hard', 'Ours_Hms_soft', 'Ours_Hms_hard', 'Ours_Hms_soft_s0']
# The stored DWT results for these datasets are NSCT outputs (byte-identical to NSCT and reproduced with
# nsct_fuse); DWT was regenerated with the original protocol (regenerate_mst/regen_mst.m), see README section 7.
OVERRIDE = {('DWT', ds): os.path.join(EVAL, 'regenerate_mst', 'DWT', ds) for ds in ('Mobile Depth', 'Middlebury', 'FlyingThings3D')}
METHODS = METHODS + NEW_METHODS
COMMERCIAL = {'Helicon Focus 8 - A', 'Helicon Focus 8 - B', 'Helicon Focus 8 - C'}
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def norm(s):
    s = os.path.splitext(s)[0].lower()
    return re.sub(r'[^a-z0-9]', '', s)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]


def gt_path(dataset, scene_file):
    return os.path.join(GTDIR, dataset, scene_file)


def result_index(method, dataset):
    """Map normalized scene key -> file path for a method/dataset."""
    d = OVERRIDE.get((method, dataset)) or os.path.join(NEW_RES if method in NEW_METHODS else RES, method, dataset)
    return {norm(f): os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT)}


def gt_index(dataset):
    d = os.path.join(GTDIR, dataset)
    return {norm(f): os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT)}


def stack_index(dataset):
    d = os.path.join(STK, dataset, 'image stack')
    return {norm(s): os.path.join(d, s) for s in os.listdir(d) if os.path.isdir(os.path.join(d, s))}


def stack_files(scene_dir):
    fs = [f for f in os.listdir(scene_dir) if f.lower().endswith(IMG_EXT)]
    return [os.path.join(scene_dir, f) for f in sorted(fs, key=natural_key)]
