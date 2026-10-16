# -*- coding: utf-8 -*-
"""Rebuild a trained model from the args line of its training log (<run_dir>.out) and load a checkpoint.
Works for both StackMFFV3 (arch 'h') and StackMFFLoop (arch 'loop')."""
import json, os
import torch
from stackmff_v3.network import StackMFFV3, StackMFFLoop


def run_args(run_dir):
    run_dir = run_dir.rstrip('/')
    for cand in (run_dir + '.out', os.path.join(run_dir, 'train_events.txt')):
        if os.path.exists(cand):
            for l in open(cand):
                if l.startswith('args: '):
                    return json.loads(l[6:])
    raise FileNotFoundError(f'no args line for {run_dir}')


MS_SCALES = (1.0, 0.75)   # default inference: probabilities averaged over the input at full and 0.75 scale


def build(args):
    if args.get('arch', 'h') == 'loop':
        m = StackMFFLoop(C1=args['C1'], C2=args['C2'], heads=tuple(args['heads']), window=args['window'],
                         mlp_ratio=args['mlp_ratio'], inter_pool=args['inter_pool'], kmax=args.get('kmax', 4),
                         share_fine=args.get('share_fine', False), intra=args.get('intra', 'attn'),
                         layer_pos=args.get('layer_pos', False), regression=(args.get('loss', 'ce') == 'reg'),
                         single_scale=args.get('single_scale', False), fine_head=args.get('fine_head', False))
        m.cur_K = args.get('k_eval', 2)
        m.ms_scales = MS_SCALES
        return m
    return StackMFFV3(C1=args['C1'], C2=args['C2'], rounds=tuple(args['rounds']), window=args['window'],
                      mlp_ratio=args['mlp_ratio'], inter_pool=args['inter_pool'], heads=tuple(args['heads']),
                      intra=args.get('intra', 'attn'), layer_pos=args.get('layer_pos', False),
                      regression=(args.get('loss', 'ce') == 'reg'))


def load(run_dir, ckpt, device='cuda', K=None):
    m = build(run_args(run_dir))
    path = ckpt if os.path.isabs(ckpt) else os.path.join(run_dir, 'model_save', ckpt)
    sd = torch.load(path, map_location='cpu'); m.load_state_dict(sd.get('model', sd))
    if K is not None and hasattr(m, 'cur_K'):
        m.cur_K = K
    return m.to(device).eval()
