# StackMFF V3

Work in progress; code and weights will be released with the paper.

StackMFF V3 is a multi-focus image stack fusion network. It classifies every pixel of the input stack into one of the N input images (the layer that is in focus at that pixel), and the fused image copies the observed pixel values from the selected layer. The network is an all-attention model that alternates intra-layer window attention (each layer processed separately) with inter-layer attention (across the N layers at every spatial position), with optional spatial aggregation between stages. Inter-layer attention has no positional encoding and no mask, so the network is permutation-equivariant over the layer axis and accepts any number of layers N >= 2.

## Installation

```bash
pip install -r requirements.txt
```

Install PyTorch first with the wheel matching your CUDA version if the default wheel does not fit your system (see https://pytorch.org).

## Data

Training stacks are synthesized from NYU Depth V2, DUTS, DIODE, Cityscapes and ADE20K images using layered defocus synthesis. The synthesis code is not part of this repository; the training script reads the resulting stacks from `--datasets_root`, or from the pre-resized cache created by `tools/make_cache.py`.

Benchmarks: Mobile Depth, Middlebury, FlyingThings3D and Road-MF. The benchmark root is set with `--bench_root` (used by `train.py` and `predict_benchmark.py`).

## Training

Single GPU:

```bash
# Optional: pre-resize the synthetic stacks into a uint8 cache (one .npz per stack)
python tools/make_cache.py --size 256 --out /path/to/cache256 --workers 48

python train.py --out_dir runs/exp1 --cache_root /path/to/cache256 \
    --C1 32 --C2 64 --heads 2 4 --rounds 1 2 1 --num_epochs 16 --batch_size 8
```

Two GPUs with DistributedDataParallel (`--batch_size` is per GPU):

```bash
torchrun --nproc_per_node=2 train.py --out_dir runs/exp1 --cache_root /path/to/cache256 \
    --C1 32 --C2 64 --heads 2 4 --rounds 1 2 1 --num_epochs 16 --batch_size 8
```

Each epoch writes `epoch_{k}.pth`, `best_val.pth` (selected by validation accuracy) and `bench_log.csv` (benchmark monitoring only, not used for model selection). Use `--resume` to continue from `out_dir/last.pth`. `scripts/launch_night1.sh` shows a two-run launch on two GPUs with the server paths used during development.

Unit tests for shapes, permutation equivariance and memory:

```bash
python tests/test_network.py
```

## Evaluation

`evaluation/` contains the unified evaluation toolkit used for all methods in the paper:

```bash
python evaluation/run_eval.py run --workers 2 [--dataset NAME] [--only SUBSTRING]
python evaluation/run_eval.py summarize
```

`run` writes one resumable JSON cache per (method, dataset, scene). `summarize` builds `per_image.csv`, `summary.csv`, `summary.md` and `alignment_log.md`.

Metrics (higher is better): SSIM (Wang et al., and the 7x7 variant), PSNR, VIF, mutual information (MI), normalized mutual information (NMI), Qabf and MEF-SSIM. Raw and aligned variants are computed; alignment uses phase correlation against the ground truth.

The metric implementations were verified against the reference MATLAB files in `evaluation/xval/matlab/` (Octave runs in `evaluation/xval/octave/`); the cross-check tables are `evaluation/xval/xval_table.md` and `evaluation/xval/xval_table_matlab.md`. `evaluation/xval/export_cases.py` and `py_side.py` produce the Python side of that comparison.

`evaluation/common.py` contains the local dataset and result paths of the author's machine. Edit these paths before running the toolkit elsewhere.

## Notes on paths

Several defaults contain absolute paths from the development server (for example `train.py`, `predict_benchmark.py`, `tools/make_cache.py`, `tools/size_candidates.py`). Override them with command line arguments or edit them. `tools/size_candidates.py` also optionally imports the StackMFF V4 ablation model from an external directory; this is only needed for the size comparison.

## Previous versions

- StackMFF: https://github.com/Xinzhe99/StackMFF
- StackMFF V2: https://github.com/Xinzhe99/StackMFF-V2

The earlier StackMFF-V3 and StackMFF-V4 repositories are deprecated and superseded by this repository.

## Citation

Citation information will be added with the paper.

```bibtex
@article{xie_stackmff_v3,
  title   = {TBD},
  author  = {Xie, Xinzhe and TBD},
  journal = {TBD},
  year    = {2026}
}
```
