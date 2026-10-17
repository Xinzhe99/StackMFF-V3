# -*- coding: utf-8 -*-
"""Fuse one focal stack with StackMFF V3.

  python fuse.py --input path/to/stack --output fused.png            # probability-weighted fusion (default)
  python fuse.py --input path/to/stack --output fused.png --hard     # hard selection: one source value per pixel

--input is a folder of images (one stack) or a list of image files. The images may be in any order and of any number
(>= 2); they must be registered. Images whose size differs from the first one are resized to it. 8-bit and 16-bit
images are supported. The network sees the luma (BT.601) of every image; the predicted probabilities weight (or, with
--hard, select) the colour values, so the output keeps the colours of the input. Weights are downloaded from the GitHub
release on first use if --weights is not given and weights/stackmff_v3_final.pth does not exist.
"""
import argparse
import hashlib
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stackmff_v3.model_io import DEFAULT_WEIGHTS, load_checkpoint  # noqa: E402
from stackmff_v3.network import smooth_probs  # noqa: E402

WEIGHTS_URL = 'https://github.com/Xinzhe99/StackMFF-V3/releases/download/v1.0/stackmff_v3_final.pth'
WEIGHTS_SHA256 = '39b2ea3738e8fea1676c379238d6be81a87988fc9952dbd6ac3825d5d7249f46'
EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def list_images(inputs):
    if len(inputs) == 1 and os.path.isdir(inputs[0]):
        d = inputs[0]
        files = sorted((f for f in os.listdir(d) if f.lower().endswith(EXTS)), key=natural_key)
        return [os.path.join(d, f) for f in files]
    return list(inputs)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_weights(path):
    if os.path.exists(path):
        return path
    if os.path.abspath(path) != os.path.abspath(DEFAULT_WEIGHTS):
        raise FileNotFoundError(f'weights not found: {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'Downloading the weights from {WEIGHTS_URL}')
    tmp = path + '.part'
    try:
        torch.hub.download_url_to_file(WEIGHTS_URL, tmp, progress=True)
    except Exception as e:
        raise RuntimeError(f'Download failed ({e}). Download stackmff_v3_final.pth from the release page of '
                           f'https://github.com/Xinzhe99/StackMFF-V3 and pass it with --weights.') from e
    if sha256(tmp) != WEIGHTS_SHA256:
        os.remove(tmp)
        raise RuntimeError('checksum mismatch of the downloaded weights')
    os.replace(tmp, path)
    return path


def read_stack(paths):
    """List of HxWx3 BGR arrays (uint8 or uint16) of equal size."""
    imgs = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
        if im is None:
            raise IOError(f'cannot read {p}')
        imgs.append(im)
    dtypes = {im.dtype for im in imgs}
    if len(dtypes) > 1 or not dtypes <= {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise ValueError(f'all images must be 8-bit or all 16-bit, got {sorted(map(str, dtypes))}')
    H, W = imgs[0].shape[:2]
    return [im if im.shape[:2] == (H, W) else cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC) for im in imgs]


def pad_to_64(x):
    """Reflect-pad (bottom/right) to a multiple of 64 pixels."""
    H, W = x.shape[-2:]
    ph, pw = (-H) % 64, (-W) % 64
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode='reflect' if (ph < H and pw < W) else 'replicate')
    return x


@torch.inference_mode()
def probabilities(model, imgs, device, sigma):
    """P [N, H, W] (float32 numpy) from the luma of the stack; sigma > 0 smooths every map (then renormalised)."""
    maxv = 255.0 if imgs[0].dtype == np.uint8 else 65535.0
    gray = np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]).astype(np.float32) / maxv
    H, W = gray.shape[1:]
    x = torch.from_numpy(gray).unsqueeze(0).to(device)
    P = model.forward_probs(pad_to_64(x))[:, :, :H, :W].float()
    if sigma > 0:
        P = smooth_probs(P, sigma)
    return P[0].cpu().numpy()


def fuse(P, imgs, hard, rows=256):
    """Hard selection (argmax) or probability-weighted sum of the colour values, rounded to the input bit depth."""
    st = np.stack(imgs)                                              # N, H, W, 3
    if hard:
        return np.take_along_axis(st, P.argmax(0)[None, :, :, None], 0)[0]
    maxv = np.iinfo(st.dtype).max
    out = np.empty(st.shape[1:], st.dtype)
    for r in range(0, st.shape[1], rows):                            # row blocks bound the memory of large stacks
        f = (P[:, r:r + rows, :, None] * st[:, r:r + rows].astype(np.float32)).sum(0)
        out[r:r + rows] = np.clip(np.round(f), 0, maxv).astype(st.dtype)
    return out


def main():
    ap = argparse.ArgumentParser(description='StackMFF V3: fuse one focal stack into an all-in-focus image.')
    ap.add_argument('--input', '-i', nargs='+', required=True, help='folder with the images of one stack, or image files')
    ap.add_argument('--output', '-o', required=True, help='fused image (e.g. fused.png)')
    ap.add_argument('--hard', action='store_true', help='hard selection (argmax) instead of probability-weighted fusion')
    ap.add_argument('--weights', default=DEFAULT_WEIGHTS, help='checkpoint (default: weights/stackmff_v3_final.pth, '
                                                                'downloaded on first use)')
    ap.add_argument('--device', default='auto', help="'auto' (CUDA if available), 'cpu', 'cuda', 'cuda:1', ...")
    ap.add_argument('--sigma', type=float, default=0.5,
                    help='Gaussian smoothing (pixels) of the probabilities for weighted fusion; not used with --hard')
    ap.add_argument('--scales', default='1,0.75', help="input scales whose probabilities are averaged ('1' = one pass)")
    ap.add_argument('--index_map', default='', help='optional: save the index of the most probable image per pixel '
                                                   '(0-based, in the natural sort order of the file names)')
    ap.add_argument('--threads', type=int, default=0, help='CPU threads for PyTorch (0: PyTorch default)')
    a = ap.parse_args()

    if a.threads > 0:
        torch.set_num_threads(a.threads)
    device = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available()) else
                          ('cpu' if a.device == 'auto' else a.device))
    paths = list_images(a.input)
    if len(paths) < 2:
        sys.exit(f'need at least two images, found {len(paths)} in {a.input}')
    model = load_checkpoint(ensure_weights(a.weights), device, scales=[float(s) for s in a.scales.split(',')])

    imgs = read_stack(paths)
    t0 = time.time()
    P = probabilities(model, imgs, device, 0.0 if a.hard else a.sigma)
    fused = fuse(P, imgs, a.hard)
    dt = time.time() - t0

    out_dir = os.path.dirname(os.path.abspath(a.output))
    os.makedirs(out_dir, exist_ok=True)
    if fused.dtype == np.uint16 and not a.output.lower().endswith(('.png', '.tif', '.tiff')):
        print('note: 16-bit input, but the output format is 8-bit; values are scaled to 8 bits')
        fused = (fused.astype(np.float32) / 257.0).round().astype(np.uint8)
    if not cv2.imwrite(a.output, fused):
        sys.exit(f'cannot write {a.output}')
    if a.index_map:
        idx = P.argmax(0)
        cv2.imwrite(a.index_map, idx.astype(np.uint8 if len(paths) <= 256 else np.uint16))
    H, W = fused.shape[:2]
    print(f'{len(paths)} images, {W}x{H}, {"hard selection" if a.hard else "weighted fusion"} on {device}: '
          f'{dt:.2f} s -> {a.output}')


if __name__ == '__main__':
    main()
