# -*- coding: utf-8 -*-
"""On-the-fly synthesis of training stacks at training resolution.

The original data were synthesised at the native resolution of each source image (kernel 2*d+1 px for depth
distance d) and then resized to 256, so the blur difference between neighbouring layers depended on the source
resolution and was below one pixel for large images. Here the stack is synthesised after resizing:
  q(p) = depth level in {0..N-1} (uniform bins, as before); layer i at p = B_{|q(p)-i|}(p),
  B_d = AiF blurred with radius r_d = s * d (s ~ U[smin, smax] px per level and per sample),
  kernel = disk (defocus, probability p_disk) or Gaussian with sigma = r_d / 2;
then per-layer gain jitter and Gaussian noise; layers are shuffled and the label permuted accordingly.
"""
import math, os, random
import numpy as np
import torch
import torch.nn.functional as F


class PairDataset(torch.utils.data.Dataset):
    """(AiF, depth) pairs from make_cache_aif.py, random crop `size` + flips. Each item carries a stack size N
    drawn once per item (uniform in [nmin, nmax]) so that the grouped batch sampler can batch equal N."""

    def __init__(self, files, size=256, nmin=2, nmax=24, seed=0, zoom=False):
        self.files, self.size, self.zoom = files, size, zoom
        rng = random.Random(seed)
        self.stack_sizes = [rng.randint(nmin, nmax) for _ in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i]); a, z = d['aif'], d['depth']
        S = a.shape[0]
        c = random.randint(int(self.size * 0.6), S) if self.zoom else self.size     # texture-scale augmentation
        t = random.randint(0, S - c); l = random.randint(0, S - c)
        a = a[t:t + c, l:l + c]; z = z[t:t + c, l:l + c]
        if c != self.size:
            import cv2
            a = cv2.resize(np.ascontiguousarray(a), (self.size, self.size), interpolation=cv2.INTER_AREA if c > self.size else cv2.INTER_LINEAR)
            z = cv2.resize(np.ascontiguousarray(z), (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if random.random() < 0.5:
            a, z = a[:, ::-1], z[:, ::-1]
        if random.random() < 0.5:
            a, z = a[::-1], z[::-1]
        x = torch.from_numpy(np.stack([np.ascontiguousarray(a), np.ascontiguousarray(z)])).float()   # 2,S,S
        far = any(f'/{n}/' in self.files[i] for n in FAR_IS_LARGE)
        return x, torch.tensor([int(far)], dtype=torch.long), self.stack_sizes[i]


def pair_files(root, names, split, fraction, seed=0):
    files = []
    for n in names:
        d = os.path.join(root, n, split)
        if os.path.isdir(d):
            fs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz'))
            if fraction < 1.0:
                fs = random.Random(seed).sample(fs, max(1, int(len(fs) * fraction)))
            files += fs
    return files


SMALL_FIX = False   # set by synth_batch(fix_small=True): disk with r < 1 -> Gaussian, sigma floor 0.1


def _kernel(r, disk, device):
    if SMALL_FIX:
        if r < 0.05:
            return None
        if disk and r < 1:
            disk = False
    elif r < 0.25:
        return None
    if disk:
        R = int(math.ceil(r)); g = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
        dist = torch.sqrt(g[None] ** 2 + g[:, None] ** 2)
        k = (r + 0.5 - dist).clamp(0, 1)
    else:
        sig = max(r / 2.0, 0.1 if SMALL_FIX else 0.3); R = max(1, int(math.ceil(3 * sig))); g = torch.arange(-R, R + 1, device=device, dtype=torch.float32)
        k1 = torch.exp(-g ** 2 / (2 * sig * sig)); k = k1[None] * k1[:, None]
    return k / k.sum()


@torch.no_grad()
def synth_batch(xp, N, smin=0.4, smax=1.6, p_disk=0.5, rmax=20.0, gain=0.03, noise=0.01, return_aif=False,
                p_hdr=0.0, slog=False, fix_small=False, far_large=None, disp=False):
    """xp: [B,2,S,S] (AiF 0..255, depth 0..255) on GPU -> stack [B,N,S,S] in [0,1], label [B,S,S] long.
    Per sample: all N blur levels in one convolution (kernels zero-padded to a common size), layers gathered."""
    B, _, S, _ = xp.shape
    dev = xp.device
    global SMALL_FIX
    SMALL_FIX = fix_small
    aif = xp[:, :1] / 255.0
    dep = xp[:, 1].clone()
    if disp and far_large is not None:   # metric depth (far = large): quantize in disparity, like a focal sweep
        for b_ in range(B):
            if int(far_large[b_]):
                c = random.uniform(8.0, 64.0)
                v = 1.0 / (dep[b_] + c)
                dep[b_] = 255.0 * (v - v.min()) / (v.max() - v.min() + 1e-12)
    q = (dep * N / 256.0).floor().long().clamp_(0, N - 1)                            # B,S,S
    aif_t = aif.clone()
    out = torch.empty(B, N, S, S, device=dev)
    lab = torch.empty(B, S, S, dtype=torch.long, device=dev)
    lev = torch.arange(N, device=dev).view(N, 1, 1)
    for b in range(B):
        s = math.exp(random.uniform(math.log(smin), math.log(smax))) if slog else random.uniform(smin, smax)
        disk = random.random() < p_disk
        ks = [_kernel(min(s * d, rmax), disk, dev) for d in range(N)]
        R = max([k.shape[0] // 2 for k in ks if k is not None] + [0])
        W = torch.zeros(N, 1, 2 * R + 1, 2 * R + 1, device=dev)
        for d, k in enumerate(ks):
            if k is None:
                W[d, 0, R, R] = 1.0
            else:
                r = k.shape[0] // 2
                W[d, 0, R - r:R + r + 1, R - r:R + r + 1] = k
        src = aif[b:b + 1]
        hdr = random.random() < p_hdr
        if hdr:   # saturated highlights: boost bright regions and add specular points in linear light, clip after blur
            lin = src.clamp(0, 1) ** 2.2
            t = random.uniform(0.75, 0.95); g = math.exp(random.uniform(math.log(4.0), math.log(200.0)))
            lin = lin * (1 + (g - 1) * ((src - t) / (1 - t)).clamp(0, 1) ** 2)
            k = random.randint(0, 30)
            if k:
                pts = torch.zeros_like(lin); yx = torch.randint(0, S, (2, k), device=dev)
                pts[0, 0, yx[0], yx[1]] = torch.empty(k, device=dev).uniform_(0.69, 4.6).exp()
                kp = _kernel(random.uniform(0.5, 1.2) * 2, False, dev); rp = kp.shape[0] // 2
                lin = lin + F.conv2d(F.pad(pts, (rp, rp, rp, rp)), kp[None, None])
            src = lin
        Bd = F.conv2d(F.pad(src, (R, R, R, R), mode='reflect'), W)[0]                # N(levels),S,S
        if hdr:
            Bd = Bd.clamp(0, 1) ** (1 / 2.2)
            aif_t[b, 0] = Bd[0]
        dist = (q[b][None] - lev).abs()                                               # N(layers),S,S
        layers = torch.gather(Bd, 0, dist)
        g = 1 + (torch.rand(N, 1, 1, device=dev) * 2 - 1) * gain
        layers = (layers * g + torch.randn_like(layers) * random.uniform(0, noise)).clamp_(0, 1)
        perm = torch.randperm(N, device=dev)                                          # level i -> position perm[i]
        out[b, perm] = layers
        lab[b] = perm[q[b]]
    if return_aif:
        return out, lab, aif_t[:, 0]
    return out, lab


# ---------------------------------------------------------------------------------------------------------------
# v2: occlusion-aware layered compositing (far to near). For focal level i, every depth level l contributes its
# masked image I*M_l and its mask M_l, both blurred with radius s*|l-i|; levels are composited from far to near,
#   C <- C * (1 - blur(M_l)) + blur(I * M_l),
# so that a defocused foreground spreads over the background at depth edges, as with a real lens. All blurs of a
# sample are done in the Fourier domain (one FFT per masked level and per kernel).
def _kernel_bank(N, s, disk, S, dev, rmax=20.0):
    """N kernels (radius s*d, d=0..N-1) zero-padded and centred on an SxS grid, as their 2-D FFTs [N, S, S//2+1]."""
    ks = torch.zeros(N, S, S, device=dev)
    for d in range(N):
        k = _kernel(min(s * d, rmax), disk, dev)
        if k is None:
            ks[d, 0, 0] = 1.0
            continue
        R = k.shape[0] // 2
        idx = torch.arange(-R, R + 1, device=dev) % S
        ks[d][idx[:, None], idx[None, :]] = k
    return torch.fft.rfft2(ks)


FAR_IS_LARGE = ('NYU-V2', 'DIODE')   # measured depth (far = large); other sources: Depth Anything (near = large)


@torch.no_grad()
def synth_batch_occ(xp, N, smin=0.4, smax=1.6, p_disk=0.5, gain=0.03, noise=0.01, pad=24, far_large=None):
    """Same interface as synth_batch, with occlusion-aware compositing (vectorised over focal planes and levels).
    far_large: bool tensor [B], True where the depth map stores far as large values."""
    B, _, S, _ = xp.shape
    dev = xp.device
    aif = xp[:, 0] / 255.0
    q = (xp[:, 1] * N / 256.0).floor().long().clamp_(0, N - 1)                       # B,S,S (depth levels)
    out = torch.empty(B, N, S, S, device=dev)
    lab = torch.empty(B, S, S, dtype=torch.long, device=dev)
    P = S + 2 * pad
    lev = torch.arange(N, device=dev)
    dist = (lev[:, None] - lev[None, :]).abs()                                        # focal i x level l
    for b in range(B):
        s = math.exp(random.uniform(math.log(smin), math.log(smax))) if slog else random.uniform(smin, smax)
        disk = random.random() < p_disk
        Kf = _kernel_bank(N, s, disk, P, dev)                                         # N(radii),P,P/2+1
        M = (q[b][None] == lev.view(N, 1, 1)).float()                                 # N(levels),S,S
        both = F.pad(torch.cat([aif[b][None] * M, M], 0)[None], (pad, pad, pad, pad), mode='reflect')[0]
        Bf = torch.fft.rfft2(both)                                                    # 2N,P,P/2+1
        K = Kf[dist]                                                                  # N(i),N(l),P,P/2+1
        bl = torch.fft.irfft2(torch.stack([Bf[:N][None] * K, Bf[N:][None] * K], 0), s=(P, P))[..., pad:pad + S, pad:pad + S]
        bi, bm = bl[0], bl[1].clamp(0, 1)                                             # N(i),N(l),S,S
        near_first = bool(far_large[b]) if far_large is not None else False
        # order levels from far to near; transmission of everything nearer than level l
        order = lev if not near_first else lev.flip(0)                                 # far ... near
        bi, bm = bi[:, order], bm[:, order]
        T = torch.flip(torch.cumprod(torch.flip(1 - bm, [1]), 1), [1])                # prod_{l' >= l}(1-bm)
        T = torch.cat([T[:, 1:], torch.ones_like(T[:, :1])], 1)                       # prod_{l' > l}
        cover = (bm * T).sum(1)                                                       # total coverage
        layers = (bi * T).sum(1) / cover.clamp_min(1e-3)                              # normalised (no dark seams)
        g = 1 + (torch.rand(N, 1, 1, device=dev) * 2 - 1) * gain
        layers = (layers * g + torch.randn_like(layers) * random.uniform(0, noise)).clamp_(0, 1)
        perm = torch.randperm(N, device=dev)
        out[b, perm] = layers
        lab[b] = perm[q[b]]
    return out, lab


@torch.no_grad()
def jitter_layers(st, max_shift=0.75, max_scale=0.004):
    """Per-image random sub-pixel misalignment (translation and magnification, as after imperfect registration or
    with focus breathing). st: [B, N, S, S]; labels are left unchanged (the shifts are below one pixel)."""
    B, N, S, _ = st.shape
    dev = st.device
    sc = 1 + (torch.rand(B * N, device=dev) * 2 - 1) * max_scale
    tx = (torch.rand(B * N, device=dev) * 2 - 1) * max_shift * 2 / S
    ty = (torch.rand(B * N, device=dev) * 2 - 1) * max_shift * 2 / S
    theta = torch.zeros(B * N, 2, 3, device=dev)
    theta[:, 0, 0] = sc; theta[:, 1, 1] = sc; theta[:, 0, 2] = tx; theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (B * N, 1, S, S), align_corners=False)
    out = F.grid_sample(st.reshape(B * N, 1, S, S), grid, mode='bilinear', padding_mode='border', align_corners=False)
    return out.reshape(B, N, S, S)
