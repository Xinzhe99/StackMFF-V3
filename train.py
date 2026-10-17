# -*- coding: utf-8 -*-
"""
Training script for StackMFF V3 (pixel-wise classification over the images of a focal stack).

The defaults reproduce the final model of the paper: StackMFFLoop with two attention units at 1/4 resolution and a
linear head (C=48, 3 heads, window 8, cross-image attention on 2x2-pooled tokens), contrast-weighted cross-entropy,
stacks synthesized on the fly from (all-in-focus image, depth) pairs (highlight synthesis for half of them),
EMA of the weights (0.999), 6 epochs, batch 8 per GPU on two GPUs. scripts/train_final.sh gives the exact command.

Loss: pixel-wise cross-entropy over the N images (full resolution), weighted by the stack contrast (--w_spread 1).
Optim: AdamW(lr=5e-4, wd=0.05, no decay on norm/bias/relative-position tables),
       1 epoch linear warm-up then cosine to 1e-6, grad clip 1.0, bf16 autocast.
Multi-GPU: DistributedDataParallel, launch with
    torchrun --nproc_per_node=2 train.py --out_dir ... (batch_size is per GPU).
Each epoch: train loss, validation (TE subsets, pixel accuracy + CE, rank 0),
            checkpoint epoch_{k}.pth (EMA weights when --ema > 0), best_val.pth (by val accuracy),
            benchmark monitoring on the 4 test sets (hard selection, grey; bench_log.csv, best_bench.pth).
The checkpoint reported in the paper was selected afterwards with tools/eval_epochs.py (mean SSIM of the weighted
fusion over the four benchmarks; see the paper). Options that are off by default were tried during development and
are not used by the final model.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stackmff_v3.Dataloader import FocusStackDataset, CombinedDataset  # noqa: E402
from stackmff_v3.network import StackMFFV3, StackMFFLoop  # noqa: E402
from predict_benchmark import run_benchmark, DEFAULT_DATASETS  # noqa: E402

DATA_ROOT = os.environ.get('STACKMFF_TRAIN_ROOT', 'data/stackmffv3_training_datasets')   # <ds>/<TR|TE>/{AiF,depth,focus_stack,focus_index_gt}
BENCH_ROOT = os.environ.get('STACKMFF_BENCH_ROOT', 'data/benchmarks')   # <dataset>/{image stack,AiF}
ALL_DATASETS = ['NYU-V2', 'DUTS', 'DIODE', 'Cityscapes', 'ADE']


def parse_args():
    p = argparse.ArgumentParser(description='StackMFF-V3 training')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--datasets_root', default=DATA_ROOT)
    p.add_argument('--train_datasets', nargs='+', default=ALL_DATASETS)
    p.add_argument('--val_datasets', nargs='+', default=ALL_DATASETS)
    p.add_argument('--subset_fraction_train', type=float, default=1.0)
    p.add_argument('--subset_fraction_val', type=float, default=0.05)
    p.add_argument('--val_seed', type=int, default=0)
    p.add_argument('--training_image_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_epochs', type=int, default=6)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--warmup_epochs', type=float, default=1.0)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--num_workers', type=int, default=10)
    p.add_argument('--val_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--C1', type=int, default=48)
    p.add_argument('--C2', type=int, default=96, help='channels of the 1/8-resolution unit (not used with --single_scale)')
    p.add_argument('--rounds', type=int, nargs=3, default=[1, 2, 1])
    p.add_argument('--window', type=int, default=8)
    p.add_argument('--mlp_ratio', type=float, default=2.0)
    p.add_argument('--inter_pool', type=int, default=2, help='spatial aggregation r of cross-image attention (SACA)')
    p.add_argument('--bench_root', default=BENCH_ROOT)
    p.add_argument('--bench_datasets', nargs='+', default=DEFAULT_DATASETS)
    p.add_argument('--no_bench', action='store_true')
    p.add_argument('--resume', action='store_true', help='resume from out_dir/last.pth')
    p.add_argument('--cache_root', default='data/cache256', help='pre-resized .npz stacks from tools/make_cache.py (validation; training too if --synth_root is empty)')
    p.add_argument('--scale_aug', type=float, default=1.0,
                   help='>1: per batch, upsample by s ~ U[1, scale_aug] (bilinear, labels nearest) and random-crop '
                        'back to the input size, so that the network sees larger defocus blur (real stacks)')
    p.add_argument('--heads', type=int, nargs=2, default=[3, 6])
    p.add_argument('--intra', choices=['attn', 'conv'], default='attn', help='ablation: intra-layer operator')
    p.add_argument('--layer_pos', action='store_true', help='ablation: learned positional encoding along N')
    p.add_argument('--loss', choices=['ce', 'reg'], default='ce',
                   help='ablation: ce = layer classification; reg = L1 on the soft-argmax index (V2-style regression)')
    p.add_argument('--arch', choices=['h', 'loop'], default='loop', help='loop: StackMFFLoop (final); h: StackMFFV3 (earlier multi-stage variant, rounds per stage)')
    p.add_argument('--kmax', type=int, default=1, help='loop: K ~ U{1..kmax} during training (final: 1, a single pass)')
    p.add_argument('--k_eval', type=int, default=1, help='loop: K used for validation / benchmark monitoring')
    p.add_argument('--share_fine', action='store_true', help='loop: same 1/4-resolution unit before and after the loop')
    p.add_argument('--single_scale', action='store_true', default=True,
                   help='loop: no 1/8 stage, units U1 and U2 at 1/4 resolution (final; U2 applied K times)')
    p.add_argument('--two_scale', dest='single_scale', action='store_false',
                   help='loop ablation: U1 -> [down -> U8 x K -> up] + skip -> U2 (second resolution)')
    p.add_argument('--margin', type=float, default=0.0, help='additive logit margin m: CE(z - m*onehot(y))')
    p.add_argument('--synth_root', default='data/cache_aif320', help="on-the-fly synthesis from (AiF, depth) pairs (tools/make_cache_aif.py); '' = pre-synthesized stacks")
    p.add_argument('--synth_smin', type=float, default=0.4)
    p.add_argument('--synth_smax', type=float, default=1.6)
    p.add_argument('--synth_pdisk', type=float, default=0.5)
    p.add_argument('--synth_occ', action='store_true', help='occlusion-aware layered compositing')
    p.add_argument('--synth_zoom', action='store_true', help='random crop size (texture-scale augmentation)')
    p.add_argument('--synth_jitter', action='store_true', help='per-image sub-pixel misalignment')
    p.add_argument('--synth_hdr', type=float, default=0.5, help='probability of saturated-highlight synthesis per sample')
    p.add_argument('--synth_slog', action='store_true', help='log-uniform blur step')
    p.add_argument('--synth_fix_small', action='store_true', help='Gaussian kernel for small radii (distinct small blurs)')
    p.add_argument('--synth_disp', action='store_true', help='quantize metric depth (NYU, DIODE) in disparity')
    p.add_argument('--synth_nmax', type=int, default=24)
    p.add_argument('--ema', type=float, default=0.999, help='EMA decay of the weights (0: off); checkpoints, validation and benchmark use the EMA weights')
    p.add_argument('--fine_head', action='store_true', help='full-resolution guided head')
    p.add_argument('--rec_w', type=float, default=0.0, help='weight of the L1 loss between the weighted fusion and the AiF image')
    p.add_argument('--synth_plin', type=float, default=0.0, help='probability of blurring in linear light (besides HDR samples)')
    p.add_argument('--synth_ljit', type=float, default=0.0, help='std of the per-level log-normal blur-radius jitter')
    p.add_argument('--synth_cont', type=int, default=0, help='continuous-depth defocus (blur follows the unquantized depth)')
    p.add_argument('--synth_ordered', type=int, default=0, help='keep the focal order of the synthesized images (no shuffle)')
    p.add_argument('--train_raw', type=int, default=0, help='train on the original-synthesis stacks read from --datasets_root (file order) even if --cache_root is given (the cache is used for validation only)')
    p.add_argument('--subset_seed', type=int, default=-1, help='seed of the training subset (-1: --seed); fixes the 35%% subset across training seeds')
    p.add_argument('--synth_pblur', type=float, default=0.0, help='max sigma of an independent per-image Gaussian blur (non-focus sharpness variation)')
    p.add_argument('--synth_qbins', type=int, default=0, help='equal-population depth bins')
    p.add_argument('--w_sigma', type=float, default=0.0)
    p.add_argument('--w_floor', type=float, default=0.0)
    p.add_argument('--w_cap', type=float, default=4.0)
    p.add_argument('--w_gamma', type=float, default=1.0)
    p.add_argument('--soft_tau', type=float, default=0.0, help='>0: intensity-aware soft targets softmax(-d/tau)')
    p.add_argument('--lam_exp', type=float, default=0.0, help='>0: adds lam * E_P[d], the fused-error bound')
    p.add_argument('--sharpen', type=float, default=0.0, help='max unsharp amount on synthesised images')
    p.add_argument('--w_spread', type=int, default=1, help='weight the CE of each pixel by the stack contrast there (contrast-weighted CE)')
    p.add_argument('--label_smooth', type=float, default=0.0, help='CE label smoothing over the images')
    p.add_argument('--log_every', type=int, default=1000)
    p.add_argument('--dry_run_steps', type=int, default=0,
                   help='>0: run this many training iterations, report speed, exit')
    return p.parse_args()


def worker_init(_):
    """Single-threaded torch/cv2 inside DataLoader workers (avoids 64-thread oversubscription)."""
    import cv2
    torch.set_num_threads(1)
    cv2.setNumThreads(0)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EpochGroupedBatchSampler(Sampler):
    """Batches of indices with equal stack size; re-shuffled every epoch (set_epoch)."""

    def __init__(self, stack_sizes, batch_size, seed=0, rank=0, world=1):
        self.groups = defaultdict(list)
        for i, s in enumerate(stack_sizes):
            self.groups[s].append(i)
        self.bs = batch_size
        self.seed = seed
        self.epoch = 0
        self.rank, self.world = rank, world
        n_batches = sum((len(v) + batch_size - 1) // batch_size for v in self.groups.values())
        # every rank gets the same number of batches (the last n_batches % world are dropped)
        self._len = n_batches // world

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        batches = []
        for size in sorted(self.groups):
            idx = list(self.groups[size])
            rng.shuffle(idx)
            batches += [idx[i:i + self.bs] for i in range(0, len(idx), self.bs)]
        rng.shuffle(batches)  # same seed on every rank -> identical list, disjoint shards below
        batches = batches[:self._len * self.world]
        return iter(batches[self.rank::self.world])

    def __len__(self):
        return self._len


class CachedStackDataset(torch.utils.data.Dataset):
    """Stacks pre-resized by make_cache.py (<cache>/<dataset>/<split>/<stack>.npz, x uint8 (N,S,S), y uint8 (S,S)).
    Same output as FocusStackDataset: (stack float [N,S,S] in [0,1], focus index long [S,S], N); random h/v flips."""

    def __init__(self, files, augment):
        import zipfile
        from concurrent.futures import ThreadPoolExecutor
        self.files, self.augment = files, augment

        def n_layers(f):
            with zipfile.ZipFile(f) as z, z.open('x.npy') as fh:
                v = np.lib.format.read_magic(fh)
                shape = np.lib.format._read_array_header(fh, v)[0]
            return shape[0]
        with ThreadPoolExecutor(32) as ex:
            self.stack_sizes = list(ex.map(n_layers, files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i])
        x, y = d['x'], d['y']
        if self.augment:
            if random.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if random.random() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
        x = torch.from_numpy(np.ascontiguousarray(x)).float().div_(255.0)
        y = torch.from_numpy(np.ascontiguousarray(y)).long()
        return x, y, x.shape[0]


def cached_dataset(cache_root, names, split, fraction, augment, seed=0):
    files = []
    for n in names:
        d = os.path.join(cache_root, n, split)
        if os.path.isdir(d):
            fs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz') and '.tmp' not in f)
            if fraction < 1.0:
                fs = random.Random(seed).sample(fs, max(1, int(len(fs) * fraction)))
            files += fs
    return CachedStackDataset(files, augment)


def build_dataset(params, size, augment):
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    dsets = [FocusStackDataset(root_dir=p['root_dir'], focus_index_dir=p['focus_index_gt'],
                               transform=tf, augment=augment, subset_fraction=p['subset_fraction'])
             for p in params]
    return CombinedDataset(dsets)


def dataset_params(root, name, split, subset):
    base = os.path.join(root, name, split)
    return {'root_dir': os.path.join(base, 'focus_stack'),
            'focus_index_gt': os.path.join(base, 'focus_index_gt'),
            'subset_fraction': subset}


def lr_at(step, total, warm, base, min_lr):
    if step < warm:
        return base * (step + 1) / max(1, warm)
    p = min(1.0, (step - warm) / max(1, total - warm))
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * p))


def param_groups(model, wd):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or 'rel_pos_bias_table' in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]


def train_epoch(model, loader, optimizer, device, args, epoch, steps_per_epoch, total_steps,
                log, max_steps=0):
    model.train()
    warm = int(args.warmup_epochs * steps_per_epoch)
    loss_sum = torch.zeros((), device=device)
    n_done = 0
    t_data = 0.0
    t_epoch0 = time.time()
    t_wait = time.time()
    for it, (x, y, _n) in enumerate(loader):
        t_data += time.time() - t_wait
        step = epoch * steps_per_epoch + it
        lr = lr_at(step, total_steps, warm, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g['lr'] = lr
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        aif = None
        if args.synth_root:
            from synth import synth_batch, synth_batch_occ
            if args.synth_occ:
                x, y = synth_batch_occ(x, int(_n[0]), args.synth_smin, args.synth_smax, args.synth_pdisk, far_large=y[:, 0])
            else:
                x, y, aif = synth_batch(x, int(_n[0]), args.synth_smin, args.synth_smax, args.synth_pdisk, return_aif=True,
                                        p_hdr=args.synth_hdr, slog=args.synth_slog, fix_small=args.synth_fix_small,
                                        far_large=y[:, 0], disp=args.synth_disp,
                                        p_linear=args.synth_plin, ljit=args.synth_ljit, qbins=bool(args.synth_qbins), cont=bool(args.synth_cont), ordered=bool(args.synth_ordered), pblur=args.synth_pblur, sharpen=args.sharpen)
            if args.synth_jitter:
                from synth import jitter_layers
                x = jitter_layers(x)
        if args.scale_aug > 1.0:
            s_ = random.uniform(1.0, args.scale_aug)
            if s_ > 1.02:
                B_, N_, H_, W_ = x.shape
                h2, w2 = int(round(H_ * s_)), int(round(W_ * s_))
                x = F.interpolate(x, size=(h2, w2), mode='bilinear', align_corners=False)
                y = F.interpolate(y[:, None].float(), size=(h2, w2), mode='nearest')[:, 0].long()
                t0_, l0_ = random.randint(0, h2 - H_), random.randint(0, w2 - W_)
                x = x[:, :, t0_:t0_ + H_, l0_:l0_ + W_].contiguous()
                y = y[:, t0_:t0_ + H_, l0_:l0_ + W_].contiguous()
                if aif is not None:
                    aif = F.interpolate(aif[:, None], size=(h2, w2), mode='bilinear', align_corners=False)[:, 0]
                    aif = aif[:, t0_:t0_ + H_, l0_:l0_ + W_].contiguous()
        core = model.module if hasattr(model, 'module') else model
        if args.arch == 'loop':   # same K on every rank: drawn from the global step
            core.cur_K = random.Random(args.seed * 100003 + epoch * 1000003 + it).randint(1, args.kmax)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(x)
        if args.arch == 'loop':
            core.cur_K = args.k_eval
        if args.loss == 'reg':
            pr = torch.softmax(logits.float(), 1)
            n_idx = torch.arange(pr.shape[1], device=pr.device, dtype=pr.dtype).view(1, -1, 1, 1)
            loss = F.l1_loss((pr * n_idx).sum(1), y.float())
        elif args.margin > 0:
            z = logits.float() - args.margin * F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2).float()
            loss = F.cross_entropy(z, y)
        elif args.w_spread or args.soft_tau > 0 or args.lam_exp > 0:   # contrast-weighted / soft-target CE (+ fused-error bound)
            from synth import contrast_weight, intensity_dist
            logp = F.log_softmax(logits.float(), dim=1)
            d = intensity_dist(x, y) if (args.soft_tau > 0 or args.lam_exp > 0) else None
            if args.soft_tau > 0:
                ce = -(torch.softmax(-d / args.soft_tau, dim=1) * logp).sum(1)
            else:
                ce = F.nll_loss(logp, y, reduction='none')
            w = contrast_weight(x, args.w_sigma, args.w_floor, args.w_cap, args.w_gamma) if args.w_spread else 1.0
            loss = (ce * w).mean()
            if args.lam_exp > 0:
                loss = loss + args.lam_exp * (logp.exp() * d).sum(1).mean()
        else:
            loss = F.cross_entropy(logits.float(), y, label_smoothing=args.label_smooth)
        if args.rec_w > 0 and aif is not None:   # fused image of Eq. (3) vs the all-in-focus image
            fused = (torch.softmax(logits.float(), 1) * x).sum(1)
            loss = loss + args.rec_w * (fused - aif).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if getattr(args, '_ema', None) is not None:
            with torch.no_grad():
                for pe, pm in zip(args._ema.parameters(), core.parameters()):
                    pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
                for be, bm in zip(args._ema.buffers(), core.buffers()):
                    be.copy_(bm)
        loss_sum += loss.detach()
        n_done += 1
        if (it + 1) % args.log_every == 0:
            avg = loss_sum.item() / n_done
            el = time.time() - t_epoch0
            log(f'  ep {epoch} it {it + 1}/{steps_per_epoch} loss {loss.item():.4f} avg {avg:.4f} '
                f'lr {lr:.2e} {(it + 1) / el:.2f} it/s data_wait {t_data / (it + 1):.3f}s/it')
        if max_steps and n_done >= max_steps:
            break
        t_wait = time.time()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    el = time.time() - t_epoch0
    return loss_sum.item() / max(1, n_done), {'iters': n_done, 'sec': el,
                                               'it_per_s': n_done / el, 'data_wait_per_it': t_data / max(1, n_done)}


@torch.no_grad()
def validate(model, val_loaders, device):
    model.eval()
    per = {}
    tot_c = tot_p = 0
    tot_ce = 0.0
    for name, loader in val_loaders:
        c = p = 0
        ce = 0.0
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(x, return_logits=True)
            logits = logits.float()
            ce += F.cross_entropy(logits, y, reduction='sum').item()
            c += (logits.argmax(1) == y).sum().item()
            p += y.numel()
        per[name] = {'ce': ce / max(1, p), 'acc': c / max(1, p), 'pixels': p}
        tot_c += c
        tot_p += p
        tot_ce += ce
    overall = {'ce': tot_ce / max(1, tot_p), 'acc': tot_c / max(1, tot_p)}
    return overall, per


def main():
    args = parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        import datetime
        # rank 0 validates and runs benchmark monitoring while the other ranks wait at a barrier
        dist.init_process_group('nccl', timeout=datetime.timedelta(hours=3))
    torch.cuda.set_device(local_rank)
    main_proc = rank == 0
    model_dir = os.path.join(args.out_dir, 'model_save')
    if main_proc:
        os.makedirs(model_dir, exist_ok=True)
        logf = open(os.path.join(args.out_dir, 'train_events.txt'), 'a')

    def log(msg):
        if not main_proc:
            return
        print(msg, flush=True)
        logf.write(msg + '\n')
        logf.flush()

    log(f'=== StackMFF-V3 training {time.strftime("%Y-%m-%d %H:%M:%S")} ===')
    log('args: ' + json.dumps(vars(args)))
    set_seed(args.seed + rank)
    device = torch.device('cuda', local_rank)
    log(f'world size {world}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False  # variable N -> avoid per-shape autotuning

    # Validation sets: fixed 5% subsets (seed 0) per TE split.
    val_loaders = []
    for name in (args.val_datasets if main_proc else []):
        random.seed(args.val_seed)
        if args.cache_root:
            ds = cached_dataset(args.cache_root, [name], 'TE', args.subset_fraction_val, False, args.val_seed)
        else:
            ds = build_dataset([dataset_params(args.datasets_root, name, 'TE', args.subset_fraction_val)],
                               args.training_image_size, augment=False)
        if len(ds) == 0:
            continue
        ld = DataLoader(ds, batch_sampler=EpochGroupedBatchSampler(ds.stack_sizes, args.batch_size, args.val_seed),
                        num_workers=args.val_workers, pin_memory=True, worker_init_fn=worker_init,
                        prefetch_factor=4)
        val_loaders.append((name, ld))
        log(f'val {name}: {len(ds)} stacks')

    # Training set: all TR splits, full data.
    random.seed(args.seed)
    tr_params = [dataset_params(args.datasets_root, n, 'TR', args.subset_fraction_train)
                 for n in args.train_datasets]
    if args.synth_root:
        from synth import PairDataset, pair_files
        train_ds = PairDataset(pair_files(args.synth_root, args.train_datasets, 'TR', args.subset_fraction_train, args.subset_seed if args.subset_seed >= 0 else args.seed),
                               size=args.training_image_size, seed=args.seed, zoom=args.synth_zoom, nmax=args.synth_nmax)
    elif args.cache_root and not args.train_raw:
        train_ds = cached_dataset(args.cache_root, args.train_datasets, 'TR', args.subset_fraction_train, True, args.subset_seed if args.subset_seed >= 0 else args.seed)
    else:
        train_ds = build_dataset(tr_params, args.training_image_size, augment=True)
    train_sampler = EpochGroupedBatchSampler(train_ds.stack_sizes, args.batch_size, args.seed, rank, world)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=args.num_workers > 0,
                              worker_init_fn=worker_init, prefetch_factor=4)
    steps_per_epoch = len(train_sampler)
    total_steps = steps_per_epoch * args.num_epochs
    log(f'train stacks {len(train_ds)} | steps/epoch per GPU {steps_per_epoch} | total steps {total_steps} | '
        f'global batch {args.batch_size * world}')

    if args.arch == 'loop':
        model = StackMFFLoop(C1=args.C1, C2=args.C2, heads=tuple(args.heads), window=args.window,
                             mlp_ratio=args.mlp_ratio, inter_pool=args.inter_pool, kmax=args.kmax,
                             share_fine=args.share_fine, intra=args.intra, layer_pos=args.layer_pos,
                             regression=(args.loss == 'reg'), single_scale=args.single_scale, fine_head=args.fine_head).to(device)
        model.cur_K = args.k_eval
    else:
        model = StackMFFV3(C1=args.C1, C2=args.C2, rounds=tuple(args.rounds), window=args.window,
                           mlp_ratio=args.mlp_ratio, inter_pool=args.inter_pool, heads=tuple(args.heads),
                           intra=args.intra, layer_pos=args.layer_pos, regression=(args.loss == 'reg')).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'model params {n_params} ({n_params / 1e6:.3f} M)')
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)

    start_epoch, best_acc, best_epoch = 0, -1.0, -1
    best_bench, best_bench_epoch = -1.0, -1
    last_path = os.path.join(args.out_dir, 'last.pth')
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location='cpu')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch, best_acc, best_epoch = ck['epoch'] + 1, ck['best_acc'], ck['best_epoch']
        log(f'resumed from {last_path} at epoch {start_epoch}')

    n_gpu = world
    model_dp = DDP(model, device_ids=[local_rank]) if world > 1 else model
    train_model = model
    if args.ema > 0:
        import copy
        args._ema = copy.deepcopy(model).eval()
        for p_ in args._ema.parameters():
            p_.requires_grad_(False)

    if args.dry_run_steps > 0:
        log(f'--- dry run: {args.dry_run_steps} iterations ---')
        t0 = time.time()
        _, st = train_epoch(model_dp, train_loader, optimizer, device, args, 0, steps_per_epoch,
                            total_steps, log, max_steps=args.dry_run_steps)
        log(f"dry run: {st['iters']} iters in {st['sec']:.1f}s -> {st['it_per_s']:.2f} it/s, "
            f"data wait {st['data_wait_per_it']:.3f}s/it, est. epoch {steps_per_epoch / st['it_per_s'] / 60:.1f} min "
            f"({steps_per_epoch} iters)")
        log(f'peak mem on rank 0 (GiB): {torch.cuda.max_memory_allocated(device) / 2 ** 30:.2f}')
        if world > 1:
            dist.destroy_process_group()
        return

    bench_path = os.path.join(args.out_dir, 'bench_log.csv')
    val_path = os.path.join(args.out_dir, 'val_log.csv')
    for epoch in range(start_epoch, args.num_epochs):
        t_ep = time.time()
        train_sampler.set_epoch(epoch)
        tr_loss, st = train_epoch(model_dp, train_loader, optimizer, device, args, epoch,
                                  steps_per_epoch, total_steps, log)
        t_train = time.time() - t_ep
        if not main_proc:
            dist.barrier()  # wait for rank 0 (validation, checkpoints, benchmark monitoring)
            continue

        if args.ema > 0:
            model = args._ema
        overall, per = validate(model, val_loaders, device)
        log(f'[epoch {epoch}] train_loss {tr_loss:.4f} | val CE {overall["ce"]:.4f} acc {overall["acc"]:.4f} | ' +
            ' '.join(f'{k} {v["acc"]:.4f}' for k, v in per.items()) +
            f' | train {t_train / 60:.1f} min')

        # checkpoints
        state = model.state_dict()
        torch.save(state, os.path.join(model_dir, f'epoch_{epoch}.pth'))
        improved = overall['acc'] > best_acc
        if improved:
            best_acc, best_epoch = overall['acc'], epoch
            torch.save(state, os.path.join(model_dir, 'best_val.pth'))
        torch.save({'model': state, 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'best_acc': best_acc, 'best_epoch': best_epoch}, last_path)

        # val log
        row = {'epoch': epoch, 'lr_end': lr_at((epoch + 1) * steps_per_epoch - 1, total_steps,
                                               int(args.warmup_epochs * steps_per_epoch), args.lr, args.min_lr),
               'train_loss': tr_loss, 'val_ce': overall['ce'], 'val_acc': overall['acc'],
               'best_val_acc': best_acc, 'improved': int(improved)}
        for k, v in per.items():
            row[f'val_{k}_acc'] = v['acc']
            row[f'val_{k}_ce'] = v['ce']
        new = not os.path.exists(val_path)
        with open(val_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)

        # benchmark monitoring (not used for checkpoint selection)
        if not args.no_bench:
            t_b = time.time()
            summary, _ = run_benchmark(model, args.bench_root, args.bench_datasets, device,
                                       save_root=None, log=log, input_mode='pad')
            new = not os.path.exists(bench_path)
            with open(bench_path, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['epoch', 'dataset', 'n', 'psnr', 'ssim', 'sec_per_stack'])
                for ds, v in summary.items():
                    w.writerow([epoch, ds, v['n'], round(v['psnr'], 4), round(v['ssim'], 5),
                                round(v['sec_per_stack'], 4)])
            # model selection on the test benchmarks (user decision 2026-10-14): mean SSIM over the 4 sets
            m_ssim = float(np.mean([v['ssim'] for v in summary.values()]))
            m_psnr = float(np.mean([v['psnr'] for v in summary.values()]))
            log(f'  bench mean SSIM {m_ssim:.5f} PSNR {m_psnr:.3f} | ' +
                ' '.join(f'{k}: {v["ssim"]:.4f}/{v["psnr"]:.2f}' for k, v in summary.items()))
            if m_ssim > best_bench:
                best_bench, best_bench_epoch = m_ssim, epoch
                torch.save(state, os.path.join(model_dir, 'best_bench.pth'))
                log(f'  new best_bench.pth (epoch {epoch})')
            log(f'  bench took {(time.time() - t_b) / 60:.1f} min')

        log(f'[epoch {epoch}] done in {(time.time() - t_ep) / 60:.1f} min | '
            f'best val acc {best_acc:.4f} @ epoch {best_epoch}')
        if world > 1:
            dist.barrier()

    log(f'=== finished. best epoch {best_epoch} (val acc {best_acc:.4f}) -> {os.path.join(model_dir, "best_val.pth")}')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
