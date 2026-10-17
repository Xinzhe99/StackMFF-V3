<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg">
  <img src="assets/logo.svg" alt="StackMFF V3" width="560">
</picture>

**Multi-focus image stack fusion by pixel-wise classification over the input images**

[![Paper](https://img.shields.io/badge/Paper-coming%20soon-lightgrey.svg)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Weights](https://img.shields.io/badge/Weights-v1.0-244A68.svg)](https://github.com/Xinzhe99/StackMFF-V3/releases/tag/v1.0)
[![GitHub stars](https://img.shields.io/github/stars/Xinzhe99/StackMFF-V3?style=social)](https://github.com/Xinzhe99/StackMFF-V3/stargazers)

Official PyTorch implementation of *StackMFF V3: Pixel-Wise Image Classification with Cross-Image Attention for Multi-Focus Stack Fusion* (submitted to Information Fusion).

[Quick start](#quick-start) | [Weights](#pretrained-weights) | [Results](#results) | [Reproducing the paper](#reproducing-the-paper) | [StackMFF series](#related-projects) | [Citation](#citation)

</div>

## News

- **2026.10**: Code, pretrained weights and the evaluation toolkit of StackMFF V3 are released. The paper (*StackMFF V3: Pixel-Wise Image Classification with Cross-Image Attention for Multi-Focus Stack Fusion*) is under review at Information Fusion.
- **Coming soon**: fused results of all four benchmarks as release assets, and StackMFF V3 in [OpenFocus](https://github.com/Xinzhe99/OpenFocus).
- This repository replaces the earlier StackMFF V3 and StackMFF V4 code bases, now archived as [StackMFF-V3-legacy](https://github.com/Xinzhe99/StackMFF-V3-legacy) and [StackMFF-V4](https://github.com/Xinzhe99/StackMFF-V4).

<div align="center">
<img src="assets/fig_order.png" width="900" alt="Failure modes of earlier stack fusion methods and our result on a shuffled stack">
</div>

*Failure modes of earlier learning-based stack fusion on Middlebury Adirondack (15 images). (a) A pairwise network applied image by image (DeFusion) accumulates blur. (b) Regressing the output intensities (StackMFF) produces values that no image recorded. (c) Regressing a focal index along the ordered stack (StackMFF V2) fails on a shuffled stack. (d) StackMFF V3 selects observed values per pixel; its result on the shuffled stack equals that on the ordered stack. Bottom: absolute error with respect to the reference; SSIM / PSNR of the full image.*

## Highlights

- **Fusion as classification.** For every pixel, the network predicts a probability distribution over the N input images. The fused value is the probability-weighted combination of the values observed at that pixel (default) or, with hard selection, the value of the most probable image. The output never leaves the range of the observations.
- **Any number of images, any order.** Cross-image attention has no positional encoding along the image axis, and every other operation acts on each image separately, so the network accepts any N and its output does not depend on the order of the images.
- **Small and fast.** 0.156 M parameters; 3.03 GFLOPs for two 256x256 images with the default two-scale inference (1.94 GFLOPs with one scale); 0.16 s per Mobile Depth stack (12 to 33 images) on an RTX A6000.
- **Training data and loss.** Stacks are synthesized on the fly at the training resolution from all-in-focus images with depth (including saturated highlights), and a contrast-weighted cross-entropy concentrates the training signal on pixels where the choice of the image changes the fused value.
- **Re-evaluated benchmark.** On Mobile Depth, Middlebury, FlyingThings3D and Road-MF, the model has the highest SSIM and PSNR among the re-evaluated methods (one toolkit for all methods, cross-checked against MATLAB reference implementations).

## Method

<div align="center">
<img src="assets/fig_architecture.png" width="900" alt="Architecture of StackMFF V3">
</div>

Each image of the stack is embedded separately into tokens at 1/4 resolution (4x4 convolution with stride 4). Two attention units with separate weights follow; each unit alternates **intra-image window attention** (8x8 windows, shifted in the second pass) with **cross-image attention**, in which the N tokens at the same position attend to each other after 2x2 spatial aggregation (SACA). A linear layer and a pixel shuffle give one logit map per image at full resolution; the softmax over the images gives the probabilities P_n, which weight the observed pixel values X_n. Colour stacks are processed on their luma (BT.601), and the same probabilities are applied to the three colour channels. At inference, the probabilities are averaged over the input at full and 0.75 scale and smoothed with a Gaussian of 0.5 pixel before the weighted sum.

<details>
<summary>Formulations of the StackMFF series and the step-by-step roadmap (click to expand)</summary>

<div align="center">
<img src="assets/fig_formulations.png" width="900" alt="Formulations of StackMFF, StackMFF V2 and StackMFF V3">
</div>

*From StackMFF to this work: (a) StackMFF regresses the fused intensities with a 3D network; (b) StackMFF V2 regresses a continuous focal index along the ordered stack and samples source pixels at the rounded index; (c) StackMFF V3 classifies for every pixel which image is in focus and combines the observed values with the predicted probabilities. All maps are actual outputs on the Mobile Depth "balls" scene.*

<div align="center">
<img src="assets/fig_roadmap.png" width="760" alt="Roadmap from a StackMFF V2-style network to StackMFF V3">
</div>

*Mean SSIM over the four benchmarks along the roadmap from a StackMFF V2-style network to the final model, one cumulative change per row, with the cost of each configuration (GFLOPs for two 256x256 images). Blue: a gain above the seed variation of the short training protocol; light blue: a change within it; red: full training of the final model.*

</details>

## Results

Full-reference comparison on the four benchmarks (unified evaluation: BT.601 luma, full image, Gaussian SSIM). The table lists the strongest methods of each group; the paper reports all re-evaluated methods together with VIF, NMI, no-reference metrics and statistical tests. Pairwise methods fuse the stack image by image in focal order. Best values in **bold**.

| Method | Type | MD SSIM | MD PSNR | Mid. SSIM | Mid. PSNR | FT3D SSIM | FT3D PSNR | Road-MF SSIM | Road-MF PSNR |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| DTCWT | transform domain | 0.9402 | 32.81 | 0.8914 | 29.26 | 0.9214 | 29.32 | 0.9772 | 34.69 |
| MFIF-GAN | pairwise, adapted | 0.9507 | 34.54 | 0.9448 | 31.50 | 0.9603 | 32.79 | 0.9802 | 34.20 |
| CSNet | pairwise, adapted | 0.9355 | 33.62 | 0.9430 | 31.82 | 0.9620 | 33.47 | 0.9850 | 36.17 |
| Helicon Focus (A) | commercial | 0.9592 | 33.97 | 0.9197 | 30.14 | 0.9299 | 31.46 | 0.9416 | 31.16 |
| Helicon Focus (B) | commercial | 0.9528 | 34.48 | 0.9277 | 30.68 | 0.9275 | 31.31 | 0.9523 | 32.27 |
| StackMFF | native stack | 0.9470 | 32.13 | 0.9336 | 30.90 | 0.9568 | 32.99 | 0.9715 | 33.19 |
| StackMFF V2 | native stack | 0.9510 | 35.21 | 0.9453 | 32.18 | 0.9531 | 32.75 | 0.9811 | 36.27 |
| **StackMFF V3** (hard selection) | native stack | 0.9769 | 38.36 | 0.9595 | 33.59 | 0.9692 | 34.79 | **0.9984** | **46.12** |
| **StackMFF V3** (weighted, default) | native stack | **0.9855** | **40.16** | **0.9622** | **34.28** | **0.9722** | **35.35** | 0.9979 | 44.53 |

MD: Mobile Depth (11 stacks), Mid.: Middlebury (15), FT3D: FlyingThings3D (100), Road-MF (80). Following the protocol of the earlier StackMFF papers, the checkpoint, the smoothing sigma and the second inference scale were selected on these benchmarks; over the last four epochs and sigma in {0, 0.5, 1}, the mean SSIM (7x7 window) varies only between 0.9784 and 0.9789 (see the paper).

**Efficiency** (one fusion call on two 256x256 images; time: mean GPU time per Mobile Depth stack, 12 to 33 images, one RTX A6000, excluding disk I/O and model loading; pairwise methods are applied iteratively):

| Method | Params (M) | GFLOPs | Time (s) |
|---|:-:|:-:|:-:|
| MFIF-GAN | 13.70 | 588.48 | 4.06 |
| CSNet | 1.89 | 25.32 | 0.86 |
| StackMFF | 6.08 | 37.37 | 0.33 |
| StackMFF V2 | 0.05 | 5.50 | 0.10 |
| StackMFF V3, one inference scale | 0.16 | 1.94 | 0.10 (fp16: 0.05) |
| StackMFF V3, two inference scales (default) | 0.16 | 3.03 | 0.16 (fp16: 0.08) |

## Installation

```bash
git clone https://github.com/Xinzhe99/StackMFF-V3.git
cd StackMFF-V3
conda create -n stackmffv3 python=3.10 -y && conda activate stackmffv3
pip install -r requirements.txt          # inference only: pip install torch numpy opencv-python
```

Install PyTorch for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) if the default wheel does not match. Training used Python 3.8, PyTorch 2.1.2 and CUDA 12.1; `fuse.py` was also tested with PyTorch 2.14 on the CPU (`requirements.txt` lists all tested versions).

## Quick start

Fuse one stack (a folder with the images of one scene, registered, in any order):

```bash
python fuse.py --input path/to/stack --output fused.png            # probability-weighted fusion (default)
python fuse.py --input path/to/stack --output fused_hard.png --hard # hard selection: one source pixel per output pixel
```

The weights are downloaded to `weights/stackmff_v3_final.pth` on first use. `fuse.py` runs on the GPU when one is available and on the CPU otherwise.

| Option | Default | Meaning |
|---|---|---|
| `--input` | | folder of one stack, or a list of image files (2 or more; 8-bit or 16-bit; colour or grey) |
| `--output` | | fused image (PNG/TIFF keep 16-bit input) |
| `--hard` | off | hard selection (argmax) instead of the weighted fusion |
| `--weights` | `weights/stackmff_v3_final.pth` | checkpoint |
| `--device` | `auto` | `cpu`, `cuda`, `cuda:1`, ... |
| `--sigma` | `0.5` | Gaussian smoothing (pixels) of the probabilities for the weighted fusion |
| `--scales` | `1,0.75` | input scales whose probabilities are averaged (`1`: one pass, about 40% faster) |
| `--index_map` | | also save the index of the most probable image per pixel (0-based, natural order of the file names) |
| `--threads` | PyTorch default | CPU threads |

Images of a different size are resized to the first image; the stack must be registered (for hand-held bursts, align the images first, for example with [OpenFocus](https://github.com/Xinzhe99/OpenFocus)). To fuse many stacks at once, use `tools/run_ours_roots.py` (see below).

In Python:

```python
import torch
from fuse import read_stack, probabilities, fuse
from stackmff_v3.model_io import load_checkpoint

model = load_checkpoint('weights/stackmff_v3_final.pth', 'cuda' if torch.cuda.is_available() else 'cpu')
imgs = read_stack(['img_01.png', 'img_02.png', 'img_03.png'])        # list of HxWx3 BGR arrays (OpenCV)
P = probabilities(model, imgs, next(model.parameters()).device, sigma=0.5)   # [N, H, W], sums to 1 over N
fused = fuse(P, imgs, hard=False)                                     # HxWx3, same dtype as the input
```

## Pretrained weights

| File | Description | Size | SHA-256 |
|---|---|---|---|
| [stackmff_v3_final.pth](https://github.com/Xinzhe99/StackMFF-V3/releases/download/v1.0/stackmff_v3_final.pth) | final model of the paper (EMA weights after the third of six epochs); architecture arguments included | 0.66 MB | `39b2ea3738e8fea1676c379238d6be81a87988fc9952dbd6ac3825d5d7249f46` |

The checkpoint is a dictionary `{'model': state_dict, 'args': architecture}` (fp32, no optimizer state) and is loaded with `stackmff_v3.model_io.load_checkpoint`. Running `fuse.py` with these weights on the CPU reproduces the paper's fused images exactly or within one grey level at a few pixels (GPU vs. CPU rounding).

## Reproducing the paper

### 1. Data

**Benchmarks** (testing only): Mobile Depth, Middlebury, FlyingThings3D and Road-MF as distributed with [StackMFF V2](https://github.com/Xinzhe99/StackMFF-V2#-downloads), organized as

```
benchmarks/
  Mobile Depth/   image stack/<scene>/<images>   AiF/<scene>.png
  Middlebury/     image stack/<scene>/...        AiF/...
  FlyingThings3D/ image stack/<scene>/...        AiF/...
  Road-MF/        image stack/<scene>/...        AiF/...
```

**Training sources**: NYU Depth V2, DUTS, DIODE, Cityscapes and ADE20K, prepared with the `make_datasets` scripts of [StackMFF V2](https://github.com/Xinzhe99/StackMFF-V2) (depth from Depth Anything V2 except for NYU Depth V2). Layout:

```
stackmffv3_training_datasets/<NYU-V2|DUTS|DIODE|Cityscapes|ADE>/<TR|TE>/
  AiF/<name>.<ext>          all-in-focus image
  depth/<name>.png          8-bit depth map
  focus_stack/<name>/       pre-synthesized stack (used for validation)
  focus_index_gt/<name>.npy its focus-index map
```

Build the two caches used by `train.py`:

```bash
python tools/make_cache_aif.py --data_root /path/to/stackmffv3_training_datasets --out data/cache_aif320 --workers 24   # (AiF, depth) pairs, 320x320
python tools/make_cache.py --data_root /path/to/stackmffv3_training_datasets --size 256 --out data/cache256 --workers 48 # validation stacks, 256x256
```

### 2. Training

```bash
DATA=/path/to/stackmffv3_training_datasets CACHE=data/cache256 PAIRS=data/cache_aif320 BENCH=/path/to/benchmarks \
RUN=runs/final bash scripts/train_final.sh
```

This is the command of the reported run (two GPUs, batch 8 per GPU, 6 epochs, about 14 min per epoch on two RTX A6000); the defaults of `train.py` are the same. Every epoch, `train.py` writes the EMA weights to `runs/final/model_save/epoch_<k>.pth`, validates on the held-out synthetic stacks and monitors the four benchmarks (`bench_log.csv`). Options that are off by default were tried during development and are not used by the final model.

### 3. Fused results

```bash
RUN=runs/final BENCH=/path/to/benchmarks OUT=results bash scripts/eval_final.sh
```

evaluates every epoch, selects the one with the best mean SSIM of the weighted fusion and writes the colour results of that epoch (`results/final_soft`, `results/final_hard`). The results of the released weights are produced with

```bash
python tools/run_ours_roots.py --root /path/to/benchmarks --out results/Ours_soft                 # weighted fusion, sigma 0.5
python tools/run_ours_roots.py --root /path/to/benchmarks --out results/Ours_hard --fusion hard   # hard selection
```

`tools/run_ours_roots.py` fuses any folder tree `<root>/<dataset>/image stack/<scene>/` and was also used for the robustness variants (shuffled, incomplete and redundant stacks) and the real-data experiments. The fused results of the paper will be available as one archive per benchmark in the [v1.0 release](https://github.com/Xinzhe99/StackMFF-V3/releases/tag/v1.0) (coming soon).

### 4. Evaluation

`evaluation/` is the toolkit used for all numbers of the paper: SSIM (Gaussian 11x11 and 7x7), PSNR, VIF, MI, NMI against the reference, and Q<sup>AB/F</sup> and MEF-SSIM against the source stack, with optional homography alignment for results that are cropped or rescaled. All metrics were cross-checked against the reference MATLAB implementations (`evaluation/xval/`, `evaluation/validate.py`). To evaluate a folder of results `<results>/<method>/<dataset>/<scene>.png`:

```bash
python evaluation/run_eval.py run --gt_root /path/to/benchmarks --gt_subdir AiF --stack_root /path/to/benchmarks \
    --results_root results --methods Ours_soft Ours_hard --out_dir eval_out --workers 6
python evaluation/run_eval.py summarize --gt_root /path/to/benchmarks --gt_subdir AiF --stack_root /path/to/benchmarks \
    --results_root results --methods Ours_soft Ours_hard --out_dir eval_out
python evaluation/stats.py --dir eval_out --ref Ours_soft      # Friedman / Nemenyi / Wilcoxon-Holm (needs >= 3 methods)
```

`run` caches one JSON file per method and scene (restartable); `summarize` writes `per_image.csv`, `summary.csv`, `summary.md` and `alignment_log.md`. The same options can be given in a JSON file (`--config`, see `evaluation/config_example.json`), which also maps methods stored in other folders (`overrides`) and lists methods that are always aligned (`aligned_methods`; Helicon Focus in the paper). Scene names are matched ignoring case, extension and punctuation.

### 5. Efficiency and other tools

| Script | Purpose |
|---|---|
| `tools/count_cost.py` | parameters and GFLOPs (0.156 M; 1.94 / 3.03 GFLOPs for two 256x256 images) |
| `tools/time_final.py` | inference time with the protocol of the efficiency table (GPU) |
| `tools/time_vs_n.py` | time and memory versus the number of images (GPU) |
| `tools/eval_epochs.py` | per-epoch benchmark evaluation and colour results of the selected epoch |
| `tools/eval_ablation.py` | per-epoch evaluation of ablation runs |
| `tools/eval_smooth.py` | effect of the smoothing sigma |
| `tools/save_conf.py`, `tools/save_probs_crop.py` | probability and confidence maps for the figures |
| `predict_benchmark.py` | hard selection on grey images (the monitoring protocol of `train.py`) |
| `tests/test_network.py` | shapes, permutation equivariance (float32 and float64), order invariance, parameter count |

## Repository layout

```
fuse.py                   fuse one stack (entry point)
train.py                  training (single GPU or DDP); defaults = final model
synth.py                  on-the-fly synthesis of training stacks on the GPU, contrast weight of the loss
predict_benchmark.py      benchmark monitoring (hard selection, grey)
stackmff_v3/
  network.py              StackMFFLoop (final model: single_scale=True, linear head) and the earlier StackMFFV3 variant
  model_io.py             final configuration, checkpoint loading
  Dataloader.py           pre-synthesized stacks (validation)
scripts/                  train_final.sh, eval_final.sh
tools/                    data caches, evaluation of epochs and ablations, cost and timing, figure data
evaluation/               evaluation toolkit (metrics, runner, statistics, MATLAB/Octave cross-validation)
tests/                    unit tests
assets/                   logo and figures
```

`stackmff_v3/network.py` also contains the components that were evaluated in the ablations and are not part of the final model (second resolution, full-resolution image branch in the head, convolutional intra-image block, image-index embedding, regression head); they are off by default.

## Related projects

StackMFF V3 is part of the **StackMFF series** on multi-focus image stack fusion ([overview](https://github.com/Xinzhe99/StackMFF-Series)):

| Project | Description |
|---|---|
| [StackMFF](https://github.com/Xinzhe99/StackMFF) | end-to-end stack fusion network that regresses the fused image with 3D convolutions (Applied Intelligence, 2025) |
| [StackMFF V2](https://github.com/Xinzhe99/StackMFF-V2) | one-shot stack fusion via focal depth regression (Engineering Applications of Artificial Intelligence, 2025); test datasets and benchmark results |
| **StackMFF V3** (this repository) | pixel-wise classification over the input images with intra- and cross-image attention |
| [OpenFocus](https://github.com/Xinzhe99/OpenFocus) | free, open-source focus stacking software for Windows and macOS with a command-line interface: image registration, classical and learned fusion methods, batch processing of stack folders. StackMFF V3 will be included (coming soon). |

If this work or any project of the series is useful to you, please consider giving a star to the [StackMFF series](https://github.com/Xinzhe99/StackMFF-Series) and to [OpenFocus](https://github.com/Xinzhe99/OpenFocus). It helps us a lot, thank you.

## Citation

The paper is under review; the entry below will be updated with the final title and bibliographic data.

```bibtex
@article{xie2026stackmffv3,
  title   = {StackMFF V3: Pixel-Wise Image Classification with Cross-Image Attention for Multi-Focus Stack Fusion},
  author  = {Xie, Xinzhe and Guo, Buyu and Li, Bolin and He, Shuangyan and Gu, Yanzhen and Jiang, Qingyan and Liu, Tao and Li, Peiliang},
  journal = {Information Fusion},
  note    = {under review},
  year    = {2026}
}
```

Earlier work of the series:

```bibtex
@article{xie2025stackmff,
  title     = {StackMFF: end-to-end multi-focus image stack fusion network},
  author    = {Xie, Xinzhe and Qingyan, Jiang and Chen, Dong and Guo, Buyu and Li, Peiliang and Zhou, Sangjun},
  journal   = {Applied Intelligence},
  volume    = {55},
  number    = {6},
  pages     = {503},
  year      = {2025},
  publisher = {Springer}
}

@article{xie2025stackmffv2,
  title     = {One-shot multi-focus image stack fusion via focal depth regression},
  author    = {Xie, Xinzhe and Guo, Buyu and He, Shuangyan and Gu, Yanzhen and Li, Yanjun and Li, Peiliang},
  journal   = {Engineering Applications of Artificial Intelligence},
  volume    = {162},
  pages     = {112667},
  year      = {2025},
  publisher = {Elsevier}
}
```

## License

This project is released under the [MIT License](LICENSE). The datasets used for training and evaluation are subject to their own licenses.

## Acknowledgements

- Training sources: [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [DUTS](http://saliencydetection.net/duts/), [DIODE](https://diode-dataset.org/), [Cityscapes](https://www.cityscapes-dataset.com/) and [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/); depth maps estimated with [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
- Benchmarks: Mobile Depth (Suwajanakorn et al., CVPR 2015), Middlebury (Scharstein et al., GCPR 2014), FlyingThings3D (Mayer et al., CVPR 2016) and Road-MF, in the focal-stack versions of the StackMFF V2 benchmark.
- The intra-image window attention follows [Swin Transformer](https://github.com/microsoft/Swin-Transformer). The baselines were run with the official code and weights of their authors; the metrics follow the reference implementations credited in `evaluation/metrics_lib.py`.
- The wordmark of the logo uses glyph outlines of Noto Sans (SIL Open Font License 1.1).
