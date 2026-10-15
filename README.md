# StackMFF V3

**Order-invariant multi-focus stack fusion by pixel-wise classification over the input images.**

> Status: under review. Code, trained weights, the evaluation toolkit and all fused results will be released here
> with the paper. Numbers marked *TBD* are filled in at release.

StackMFF V3 fuses a focal stack of any length into one all-in-focus image. For every pixel the network predicts a
probability distribution over the N input images (which image is in focus there), and the fused value is the
probability-weighted combination of the values observed at that pixel; a hard-selection output that copies exactly
one observed value per pixel is available from the same forward pass.

- **Order-invariant by construction.** Every operation either acts on each image with shared weights or attends
  across images without any encoding of their order, so shuffling the input permutes the probabilities and leaves the
  fused image unchanged.
- **Any number of images.** One model handles stacks of 2 to (at least) 33 images.
- **Simple design.** One kind of attention unit (intra-layer window attention followed by inter-layer attention),
  used at 1/4 resolution and looped K times with shared weights at 1/8 resolution. *TBD* M parameters.
- **Observed values only.** The output is a convex combination of the input pixels; no intensities are synthesized.

## Repository layout

| Path | Content |
|---|---|
| `stackmff_v3/network.py` | `StackMFFLoop` (final model) and `StackMFFV3` (earlier multi-stage variant used in ablations) |
| `synth.py` | on-the-fly synthesis of training stacks from (all-in-focus image, depth) pairs at training resolution |
| `train.py` | training (single GPU or DDP), benchmark monitoring per epoch |
| `tools/run_ours_roots.py` | fuse folders of stacks (`<root>/<dataset>/image stack/<scene>/<images>`) |
| `tools/eval_epochs.py`, `tools/eval_ablation.py`, `tools/eval_loop_k.py` | per-epoch, ablation and loop-iteration evaluation |
| `evaluation/` | unified evaluation toolkit (SSIM, PSNR, VIF, MI, NMI, Q^{X/F}, MEF-SSIM), statistics, MATLAB cross-validation |

## Inference

```bash
python tools/run_ours_roots.py --run_dir <run_dir> --ckpt <checkpoint.pth> \
    --root /path/to/stacks --out /path/to/results --fusion soft --sigma 1
```

`--fusion hard --sigma 0` gives the hard-selection output. Images inside a stack may be in any order and of any number.

## Training

```bash
# 1. cache (grey all-in-focus image, depth) pairs at 320x320
python tools/make_cache_aif.py --out /path/to/cache_aif320 --workers 24
# 2. train with on-the-fly synthesis (two GPUs)
torchrun --nproc_per_node=2 train.py --out_dir runs/final --arch loop --C1 48 --C2 96 --heads 3 6 \
    --inter_pool 2 --kmax 4 --k_eval 2 --synth_root /path/to/cache_aif320 --cache_root /path/to/cache256
```

Training sources: NYU Depth V2, DUTS, DIODE, Cityscapes and ADE20K images with measured or estimated depth
(Depth Anything V2). Stacks are synthesized on the GPU after resizing: depth quantized into N in {2..24} levels, blur
radius s·|level − i| with a random step s and a disk or Gaussian kernel, per-image gain and noise, random image order.

## Evaluation

```bash
python evaluation/run_eval.py run --workers 6
python evaluation/run_eval.py summarize
python evaluation/stats.py --ref <our method name>
```

All metrics were cross-checked against the reference MATLAB implementations (differences below 1e-13). Pairwise
baselines fuse stacks image by image in the numeric order of the file names (the focal order of the benchmarks).

## Results

*TBD at release* (Mobile Depth, Middlebury, FlyingThings3D, Road-MF; robustness to shuffled, incomplete and redundant
stacks; real microscopy and camera stacks).

## Related repositories

- [StackMFF](https://github.com/Xinzhe99/StackMFF), [StackMFF V2](https://github.com/Xinzhe99/StackMFF-V2)
- [StackMFF-V3-legacy](https://github.com/Xinzhe99/StackMFF-V3-legacy) and [StackMFF-V4](https://github.com/Xinzhe99/StackMFF-V4) (deprecated; superseded by this repository)
- [OpenFocus](https://github.com/Xinzhe99/OpenFocus): open-source focus stacking software

## License

See `LICENSE`.
