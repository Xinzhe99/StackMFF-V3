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
[![GitHub stars](https://img.shields.io/github/stars/Xinzhe99/StackMFF-V3?cacheSeconds=3600&style=social)](https://github.com/Xinzhe99/StackMFF-V3/stargazers)

Official PyTorch implementation of *StackMFF V3: Order-Invariant Focus Stacking via Pixel-Wise Source Classification*.

[Quick start](#quick-start) | [Weights](#pretrained-weights) | [Results](#results) | [Reproducing the paper](#reproducing-the-paper) | [StackMFF series](#related-projects) | [Citation](#citation)

</div>

## News

- **2026.10**: Code, pretrained weights and the evaluation toolkit of StackMFF V3 are released. The manuscript (*StackMFF V3: Order-Invariant Focus Stacking via Pixel-Wise Source Classification*) is being prepared for submission.
- **2026.10**: All **7,262 evaluated benchmark outputs** (including 412 of ours) and **2,170 registered real-stack outputs** are archived with SHA-256 manifests. See the [archive index and downloader](benchmark/README.md). Data assets remain in the v1.0 release; audited code is versioned separately.
- **Coming soon**: StackMFF V3 in [OpenFocus](https://github.com/Xinzhe99/OpenFocus).
- This repository develops the deterministic fusion formulation of earlier StackMFF V3/V4 studies; historical materials remain in [StackMFF-V3-legacy](https://github.com/Xinzhe99/StackMFF-V3-legacy) and [StackMFF-V4-legacy](https://github.com/Xinzhe99/StackMFF-V4-legacy).

<div align="center">
<img src="assets/fig_order.png" width="900" alt="Failure modes of earlier stack fusion methods and our result on a shuffled stack">
</div>

*Failure modes of earlier learning-based stack fusion on Middlebury Adirondack (15 images). (a) A pairwise network applied image by image (DeFusion) accumulates blur. (b) Regressing the output intensities (StackMFF) produces values that no image recorded. (c) Regressing a focal index along the ordered stack (StackMFF V2) fails on a shuffled stack. (d) StackMFF V3 selects observed values per pixel; its output is invariant to shuffling up to floating-point error (near-tied hard decisions can differ). Bottom: absolute error with respect to the reference; SSIM / PSNR of the full image.*

## Highlights

- **Fusion as classification.** For every pixel, the network predicts a probability distribution over the N input images. The fused value is the probability-weighted combination of the values observed at that pixel (default) or, with hard selection, the value of the most probable image. The output never leaves the range of the observations.
- **Any number of images, any order.** Cross-image attention has no positional encoding along the image axis, and the remaining image-axis operations are symmetric or shared across images. One trained network handles variable stack lengths, and weighted fusion is order-invariant up to floating-point error; near-tied hard decisions can differ.
- **Small and fast.** 0.156 M parameters; 3.03 GFLOPs for two 256x256 images with the default two-scale inference (1.94 GFLOPs with one scale); 0.16 s per Mobile Depth stack (12 to 33 images) on an RTX A6000.
- **Training data and loss.** Stacks are synthesized on the fly at the training resolution from all-in-focus images with depth (including saturated highlights), and a contrast-weighted cross-entropy concentrates the training signal on pixels where the choice of the image changes the fused value.
- **Re-evaluated benchmark.** Weighted fusion has the highest mean SSIM and PSNR on Mobile Depth, Middlebury and FlyingThings3D; hard selection leads on Road-MF. The Mobile Depth PSNR margin over GFF is not statistically significant after correction.

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

*Mean SSIM over the four benchmarks along the roadmap from a StackMFF V2-style network to the final model, one cumulative change per row, with the cost of each configuration (GFLOPs for two 256x256 images). Blue: a gain above the seed variation of the short training protocol; light blue: a change within it; red: full training of the final model. Displayed Δ values subtract adjacent four-decimal means; bar lengths use unrounded scores.*

</details>

## Results

Full-reference comparison on the four benchmarks (unified evaluation: BT.601 luma, full image, Gaussian SSIM). This selected comparison includes the strongest spatial baselines and earlier StackMFF models. The [complete tables and per-image records](benchmark/README.md) cover the current complete cohort of 32 baseline families (35 settings) plus our two outputs. ReDiffuse now covers all 206 scenes; its [validated update](benchmark/updates/2026-10-20-rediffuse/README.md) is included. ZMFF also covers all 206 scenes in its [validated update](benchmark/updates/2026-10-20-zmff/README.md); its separate full RGB archive is still transferring. VIF, NMI, no-reference metrics and statistical tests are also provided. Pairwise methods fuse the stack image by image in focal order. Best values in **bold**.

| Method | Type | MD SSIM | MD PSNR | Mid. SSIM | Mid. PSNR | FT3D SSIM | FT3D PSNR | Road-MF SSIM | Road-MF PSNR |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GFF | spatial domain | 0.9829 | 39.50 | 0.9557 | 32.94 | 0.9682 | 34.44 | 0.9921 | 38.58 |
| GFDF | spatial domain | 0.9819 | 39.49 | 0.9557 | 32.89 | 0.9681 | 34.33 | 0.9905 | 37.82 |
| DTCWT | transform domain | 0.9402 | 32.81 | 0.8914 | 29.26 | 0.9214 | 29.32 | 0.9772 | 34.69 |
| MFIF-GAN | pairwise, adapted | 0.9507 | 34.54 | 0.9448 | 31.50 | 0.9603 | 32.79 | 0.9802 | 34.20 |
| CSNet | pairwise, adapted | 0.9355 | 33.62 | 0.9430 | 31.82 | 0.9620 | 33.47 | 0.9850 | 36.17 |
| Helicon Focus (A) | commercial | 0.9592 | 33.97 | 0.9197 | 30.14 | 0.9299 | 31.46 | 0.9416 | 31.16 |
| Helicon Focus (B) | commercial | 0.9528 | 34.48 | 0.9277 | 30.68 | 0.9275 | 31.31 | 0.9523 | 32.27 |
| StackMFF | native stack | 0.9470 | 32.13 | 0.9336 | 30.90 | 0.9568 | 32.99 | 0.9715 | 33.19 |
| StackMFF V2 | native stack | 0.9510 | 35.21 | 0.9453 | 32.18 | 0.9531 | 32.75 | 0.9811 | 36.27 |
| **StackMFF V3** (hard selection) | native stack | 0.9769 | 38.36 | 0.9595 | 33.59 | 0.9692 | 34.79 | **0.9984** | **46.12** |
| **StackMFF V3** (weighted, default) | native stack | **0.9855** | **40.16** | **0.9622** | **34.28** | **0.9722** | **35.35** | 0.9979 | 44.53 |

MD: Mobile Depth (11 stacks), Mid.: Middlebury (15), FT3D: FlyingThings3D (100), Road-MF (80). The checkpoint, smoothing sigma and inference scales were selected on these benchmarks, so these scores are not an independently held-out estimate of model-selection generalization. In the sensitivity records, over the last four epochs and sigma in {0, 0.5, 1}, the mean SSIM (7x7 window) varies only between 0.9784 and 0.9789 (see the paper).

**Efficiency** (one fusion call on two 256x256 images; time: mean GPU time per Mobile Depth stack, 12 to 33 images, one RTX A6000, excluding disk I/O and model loading; pairwise methods are applied iteratively):

| Method | Params (M) | GFLOPs | Time (s) |
|---|:-:|:-:|:-:|
| MFIF-GAN | 13.70 | 588.48 | 4.06 |
| CSNet | 1.89 | 25.32 | 0.86 |
| StackMFF | 6.08 | 37.37 | 0.33 |
| StackMFF V2 | 0.05 | 5.50 | 0.10 |
| StackMFF V3, one inference scale | 0.16 | 1.94 | 0.10 (fp16: 0.05) |
| StackMFF V3, two inference scales (default) | 0.16 | 3.03 | 0.16 (fp16: 0.08) |

<!-- pair-extension:start -->

### Two-input multi-focus fusion (863 pairs)

The main tables evaluate variable-length focal stacks. This section evaluates the
same methods on ordinary two-image inputs from four public benchmarks: Lytro (20
pairs), MFFW2 (13, the complete two-source subset), MFI-WHU (120) and Real-MFF
(710). Every method receives the same two sources at native resolution, with no
fine-tuning or model selection on these data.

Eight source-based metrics are reported; `QCV` is minimized and the rest are
maximized. A dataset appears here only when all 37 configurations are complete
for it, so no row is ever ranked on a successful subset. Per-image records,
coverage tables and the LaTeX tables are in
[benchmark/updates/2026-10-20-pair-extension](benchmark/updates/2026-10-20-pair-extension/README.md).

**Lytro**

| Method | NMI↑ | FMI_w↑ | Qabf↑ | QP↑ | QY↑ | QCB↑ | QCV↓ | VIFF↑ |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Ours (weighted) | 1.1834 | 0.5983 | **0.7627** | 0.8498 | **0.9895** | 0.8093 | 15.85 | 0.9478 |
| BFMF | 1.1847 | **0.6003** | 0.7570 | 0.8478 | 0.9895 | 0.8091 | 21.48 | 0.9322 |
| Ours (hard selection) | **1.1916** | 0.5978 | 0.7625 | 0.8484 | 0.9893 | 0.8101 | 15.93 | 0.9477 |
| QB | 1.1886 | 0.5988 | 0.7610 | 0.8452 | 0.9892 | 0.8097 | 16.29 | 0.9450 |
| GFDF | 1.1643 | 0.5975 | 0.7618 | **0.8507** | 0.9886 | **0.8114** | 15.89 | 0.9459 |
| MFIF-GAN | 1.1868 | 0.5895 | 0.7618 | 0.8464 | 0.9881 | 0.8077 | 16.08 | 0.9466 |
| MCCSR-Net | 1.1858 | 0.5887 | 0.7615 | 0.8455 | 0.9881 | 0.8076 | 16.86 | 0.9450 |
| CSNet | 1.1862 | 0.5871 | 0.7583 | 0.8375 | 0.9872 | 0.8037 | 18.25 | 0.9361 |
| CCSR-Net | 1.1829 | 0.5828 | 0.7578 | 0.8389 | 0.9869 | 0.8057 | 18.59 | 0.9420 |
| StackMFF V2 | 1.1864 | 0.5849 | 0.7582 | 0.8377 | 0.9857 | 0.8048 | 16.03 | 0.9436 |
| GFF | 1.0980 | 0.5727 | 0.7599 | 0.8485 | 0.9824 | 0.7976 | 16.02 | 0.9442 |
| IFM | 1.1419 | 0.5786 | 0.7531 | 0.8220 | 0.9796 | 0.7952 | 20.32 | 0.9395 |
| Zerene (DMap) | 1.1185 | 0.5783 | 0.7510 | 0.8410 | 0.9780 | 0.7891 | 15.55 | 0.9384 |
| Helicon Focus (C) | 1.1107 | 0.5179 | 0.7561 | 0.8454 | 0.9770 | 0.7885 | 15.87 | 0.9455 |
| Helicon Focus (B) | 1.0806 | 0.4968 | 0.7522 | 0.8434 | 0.9718 | 0.7789 | 15.81 | 0.9439 |
| ReDiffuse | 0.9819 | 0.5159 | 0.7494 | 0.8401 | 0.9712 | 0.7554 | 18.11 | 0.9572 |
| DTCWT | 0.9401 | 0.5160 | 0.7449 | 0.8330 | 0.9659 | 0.7508 | 20.52 | 0.9287 |
| NSCT | 0.9526 | 0.5090 | 0.7455 | 0.8345 | 0.9624 | 0.7478 | 21.94 | 0.9437 |
| Helicon Focus (A) | 1.0592 | 0.4749 | 0.7491 | 0.8347 | 0.9563 | 0.7632 | **15.52** | 0.9373 |
| IFCNN (MAX) | 0.9376 | 0.4567 | 0.7296 | 0.8178 | 0.9518 | 0.7292 | 19.96 | 0.9421 |
| CVT | 0.9041 | 0.4856 | 0.7311 | 0.8135 | 0.9506 | 0.7277 | 20.54 | 0.9272 |
| Zerene (PMax) | 0.9055 | 0.4814 | 0.7346 | 0.8247 | 0.9481 | 0.7367 | 30.21 | 0.9822 |
| ZMFF | 0.8813 | 0.3533 | 0.7028 | 0.7843 | 0.9331 | 0.7419 | 22.77 | 0.9321 |
| DWT | 0.8712 | 0.4142 | 0.7077 | 0.7792 | 0.9234 | 0.7192 | 29.70 | 0.9171 |
| DCT | 1.1097 | 0.4972 | 0.7279 | 0.8169 | 0.9155 | 0.6698 | 56.44 | 1.0720 |
| SDNet | 0.8677 | 0.4560 | 0.6870 | 0.7732 | 0.9105 | 0.6418 | 55.37 | 0.8414 |
| MFF-GAN | 0.8424 | 0.4426 | 0.6763 | 0.7564 | 0.9080 | 0.6580 | 35.09 | 0.8951 |
| MUFusion | 0.8087 | 0.4038 | 0.6727 | 0.7456 | 0.8999 | 0.6823 | 34.80 | 1.0361 |
| SwinFusion | 0.8664 | 0.4265 | 0.7116 | 0.7639 | 0.8981 | 0.6689 | 40.46 | 0.9083 |
| MGFF | 0.8068 | 0.4230 | 0.6774 | 0.7454 | 0.8902 | 0.6724 | 377.52 | 0.9842 |
| SwinMFF | 0.7724 | 0.4476 | 0.7046 | 0.7811 | 0.8850 | 0.6419 | 64.59 | **1.1031** |
| U2Fusion | 0.8236 | 0.3845 | 0.6483 | 0.7549 | 0.8672 | 0.6689 | 43.59 | 0.7173 |
| StackMFF | 0.8111 | 0.3559 | 0.6281 | 0.6874 | 0.8506 | 0.6502 | 59.34 | 0.8830 |
| DDBFusion | 0.8407 | 0.3907 | 0.5341 | 0.6733 | 0.8148 | 0.6089 | 90.83 | 0.7970 |
| GIFNet | 0.7551 | 0.2953 | 0.5282 | 0.5598 | 0.7909 | 0.6050 | 150.72 | 1.0649 |
| DeFusion | 0.8390 | 0.3098 | 0.5318 | 0.6599 | 0.7901 | 0.6158 | 93.77 | 0.7518 |
| VAEEDOF | 0.7177 | 0.2308 | 0.5064 | 0.4111 | 0.7479 | 0.6779 | 25.70 | 0.8917 |

**MFFW2**

| Method | NMI↑ | FMI_w↑ | Qabf↑ | QP↑ | QY↑ | QCB↑ | QCV↓ | VIFF↑ |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BFMF | 1.1642 | **0.5817** | 0.7311 | **0.7631** | **0.9899** | **0.7596** | 404.96 | 0.8287 |
| MFIF-GAN | 1.1748 | 0.5657 | **0.7420** | 0.7550 | 0.9812 | 0.7569 | 108.39 | 0.8442 |
| QB | 1.1597 | 0.5712 | 0.7373 | 0.7546 | 0.9807 | 0.7566 | 397.39 | 0.8395 |
| CSNet | **1.1858** | 0.5678 | 0.7366 | 0.7513 | 0.9801 | 0.7561 | 109.64 | 0.8399 |
| MCCSR-Net | 1.1767 | 0.5672 | 0.7357 | 0.7562 | 0.9787 | 0.7497 | 399.73 | 0.8392 |
| CCSR-Net | 1.1751 | 0.5582 | 0.7321 | 0.7372 | 0.9781 | 0.7489 | 114.42 | 0.8345 |
| Ours (hard selection) | 1.1647 | 0.5548 | 0.7358 | 0.7409 | 0.9738 | 0.7540 | 196.01 | 0.8474 |
| StackMFF V2 | 1.1764 | 0.5528 | 0.7372 | 0.7345 | 0.9737 | 0.7511 | 106.91 | 0.8442 |
| GFDF | 1.0978 | 0.5629 | 0.7393 | 0.7630 | 0.9701 | 0.7526 | 105.46 | 0.8465 |
| IFM | 1.0906 | 0.5429 | 0.7292 | 0.7345 | 0.9603 | 0.7378 | 118.65 | 0.8395 |
| Ours (weighted) | 1.0937 | 0.5433 | 0.7340 | 0.7436 | 0.9596 | 0.7433 | 182.92 | 0.8487 |
| Zerene (DMap) | 0.9947 | 0.5284 | 0.7252 | 0.7448 | 0.9504 | 0.7261 | **99.69** | 0.8427 |
| GFF | 0.9555 | 0.5090 | 0.7336 | 0.7528 | 0.9413 | 0.7245 | 105.56 | 0.8487 |
| Helicon Focus (B) | 1.0349 | 0.4606 | 0.7215 | 0.7441 | 0.9335 | 0.7247 | 113.63 | 0.8683 |
| Helicon Focus (C) | 1.0565 | 0.4667 | 0.7246 | 0.7411 | 0.9324 | 0.7249 | 114.08 | 0.8677 |
| NSCT | 0.8359 | 0.4618 | 0.7114 | 0.7071 | 0.9155 | 0.6795 | 107.26 | 0.8624 |
| DTCWT | 0.8186 | 0.4601 | 0.7049 | 0.6998 | 0.9046 | 0.6695 | 106.19 | 0.8309 |
| ReDiffuse | 0.8627 | 0.4720 | 0.7091 | 0.7288 | 0.8922 | 0.6366 | 108.33 | 0.8859 |
| Zerene (PMax) | 0.8137 | 0.4355 | 0.7016 | 0.6946 | 0.8916 | 0.6702 | 109.79 | 0.9142 |
| DCT | 1.0760 | 0.4772 | 0.6996 | 0.7058 | 0.8888 | 0.6361 | 237.07 | 0.9582 |
| Helicon Focus (A) | 0.9876 | 0.4256 | 0.7166 | 0.6966 | 0.8833 | 0.6924 | 112.49 | 0.8623 |
| IFCNN (MAX) | 0.8206 | 0.4101 | 0.6765 | 0.6680 | 0.8789 | 0.6421 | 122.93 | 0.8545 |
| CVT | 0.7790 | 0.4311 | 0.6849 | 0.6640 | 0.8721 | 0.6455 | 121.44 | 0.8323 |
| SDNet | 0.8103 | 0.4231 | 0.6385 | 0.6437 | 0.8656 | 0.5847 | 185.83 | 0.7740 |
| ZMFF | 0.7754 | 0.3228 | 0.6670 | 0.6487 | 0.8576 | 0.6788 | 138.18 | 0.8298 |
| DWT | 0.7831 | 0.3877 | 0.6625 | 0.6096 | 0.8550 | 0.6435 | 118.38 | 0.8280 |
| MFF-GAN | 0.7802 | 0.4133 | 0.6257 | 0.6245 | 0.8483 | 0.5846 | 203.88 | 0.8008 |
| SwinFusion | 0.8078 | 0.4021 | 0.6800 | 0.6460 | 0.8457 | 0.6105 | 140.66 | 0.8198 |
| MGFF | 0.6816 | 0.4071 | 0.6318 | 0.5996 | 0.8300 | 0.6015 | 205.10 | 1.0011 |
| MUFusion | 0.7547 | 0.3694 | 0.6062 | 0.5921 | 0.8247 | 0.6207 | 243.35 | 0.9043 |
| SwinMFF | 0.7329 | 0.3846 | 0.6381 | 0.6447 | 0.8221 | 0.6065 | 230.13 | **1.0014** |
| U2Fusion | 0.7642 | 0.3574 | 0.5902 | 0.6182 | 0.7958 | 0.5996 | 180.33 | 0.6490 |
| StackMFF | 0.7485 | 0.3587 | 0.5815 | 0.5630 | 0.7829 | 0.5842 | 194.86 | 0.8035 |
| DDBFusion | 0.7792 | 0.3581 | 0.5360 | 0.5544 | 0.7602 | 0.5662 | 212.01 | 0.7843 |
| DeFusion | 0.7942 | 0.2806 | 0.4921 | 0.5183 | 0.7224 | 0.5648 | 164.61 | 0.6830 |
| GIFNet | 0.7210 | 0.2737 | 0.4762 | 0.4472 | 0.7128 | 0.5500 | 265.92 | 0.9460 |
| VAEEDOF | 0.6638 | 0.2105 | 0.4539 | 0.3076 | 0.6491 | 0.6266 | 125.88 | 0.7824 |

**MFI-WHU**

| Method | NMI↑ | FMI_w↑ | Qabf↑ | QP↑ | QY↑ | QCB↑ | QCV↓ | VIFF↑ |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| BFMF | 1.2171 | **0.6318** | 0.7256 | 0.7596 | **0.9901** | 0.8271 | 27.86 | 0.9688 |
| MCCSR-Net | **1.2233** | 0.6286 | 0.7293 | 0.7585 | 0.9899 | 0.8234 | 23.09 | 0.9815 |
| QB | 1.2207 | 0.6304 | 0.7350 | 0.7583 | 0.9897 | **0.8286** | 31.15 | 0.9822 |
| MFIF-GAN | 1.2190 | 0.6246 | 0.7366 | 0.7576 | 0.9894 | 0.8260 | 35.06 | 0.9848 |
| Ours (hard selection) | 1.2216 | 0.6183 | **0.7381** | 0.7563 | 0.9891 | 0.8257 | 44.43 | 0.9860 |
| Ours (weighted) | 1.2015 | 0.6197 | 0.7375 | 0.7588 | 0.9890 | 0.8248 | 41.52 | 0.9860 |
| CSNet | 1.2215 | 0.6237 | 0.7326 | 0.7567 | 0.9887 | 0.8259 | 27.33 | 0.9820 |
| GFDF | 1.1887 | 0.6266 | 0.7357 | **0.7610** | 0.9885 | 0.8281 | 33.50 | 0.9828 |
| CCSR-Net | 1.2204 | 0.6233 | 0.7240 | 0.7541 | 0.9885 | 0.8204 | 24.43 | 0.9746 |
| StackMFF V2 | 1.2126 | 0.6166 | 0.7344 | 0.7529 | 0.9874 | 0.8228 | 34.11 | 0.9826 |
| Helicon Focus (C) | 1.1455 | 0.5739 | 0.7345 | 0.7576 | 0.9807 | 0.8072 | 46.66 | 0.9839 |
| IFM | 1.1773 | 0.6148 | 0.7185 | 0.7350 | 0.9790 | 0.8041 | 39.30 | 0.9735 |
| Zerene (DMap) | 1.1281 | 0.6079 | 0.7215 | 0.7565 | 0.9780 | 0.8140 | 30.81 | 0.9719 |
| GFF | 1.1242 | 0.6023 | 0.7297 | 0.7597 | 0.9754 | 0.8143 | 27.34 | 0.9813 |
| Helicon Focus (B) | 1.0923 | 0.5492 | 0.7303 | 0.7534 | 0.9715 | 0.7939 | 45.70 | 0.9797 |
| ReDiffuse | 1.0637 | 0.5599 | 0.7249 | 0.7580 | 0.9694 | 0.7730 | 32.10 | 1.0231 |
| NSCT | 0.9964 | 0.5538 | 0.7163 | 0.7511 | 0.9647 | 0.7991 | 38.17 | 0.9715 |
| Zerene (PMax) | 0.9517 | 0.5383 | 0.7162 | 0.7450 | 0.9626 | 0.7637 | 40.38 | 1.0047 |
| Helicon Focus (A) | 1.0839 | 0.5272 | 0.7288 | 0.7543 | 0.9621 | 0.7914 | 49.50 | 0.9776 |
| DTCWT | 0.9668 | 0.5537 | 0.7138 | 0.7440 | 0.9589 | 0.7874 | 38.71 | 0.9653 |
| SDNet | 0.9241 | 0.5215 | 0.6968 | 0.7412 | 0.9538 | 0.7657 | 22.98 | 0.9354 |
| IFCNN (MAX) | 0.8977 | 0.4917 | 0.6922 | 0.7455 | 0.9471 | 0.7364 | **20.14** | 0.9974 |
| CVT | 0.9464 | 0.5265 | 0.7006 | 0.7343 | 0.9468 | 0.7707 | 29.50 | 0.9630 |
| MFF-GAN | 0.8323 | 0.4876 | 0.6576 | 0.7085 | 0.9282 | 0.6847 | 40.86 | 0.9811 |
| DWT | 0.8539 | 0.4882 | 0.6747 | 0.7029 | 0.9257 | 0.7646 | 47.85 | 0.9324 |
| DCT | 1.1391 | 0.5419 | 0.6901 | 0.7106 | 0.9198 | 0.6736 | 110.53 | 1.1426 |
| MGFF | 0.8643 | 0.4871 | 0.6594 | 0.7098 | 0.9180 | 0.7446 | 59.92 | 0.9378 |
| SwinFusion | 0.8375 | 0.4837 | 0.6778 | 0.7187 | 0.9069 | 0.7263 | 29.22 | 0.9379 |
| SwinMFF | 0.7685 | 0.4810 | 0.6788 | 0.7143 | 0.8965 | 0.6545 | 75.90 | 1.1645 |
| MUFusion | 0.7448 | 0.4384 | 0.6107 | 0.6590 | 0.8686 | 0.6482 | 43.04 | 1.0525 |
| U2Fusion | 0.7417 | 0.4206 | 0.5941 | 0.6665 | 0.8528 | 0.6295 | 40.46 | 0.7473 |
| StackMFF | 0.7762 | 0.4151 | 0.5975 | 0.6449 | 0.8481 | 0.6783 | 36.73 | 0.9359 |
| ZMFF | 0.7000 | 0.3686 | 0.6262 | 0.6294 | 0.8434 | 0.6715 | 188.25 | 0.9464 |
| DDBFusion | 0.8131 | 0.4350 | 0.5480 | 0.6471 | 0.8267 | 0.7072 | 27.96 | 0.8728 |
| DeFusion | 0.8206 | 0.3389 | 0.5152 | 0.5895 | 0.7780 | 0.7088 | 34.94 | 0.7826 |
| GIFNet | 0.6787 | 0.3229 | 0.4452 | 0.4800 | 0.7363 | 0.5457 | 202.48 | **1.1782** |
| VAEEDOF | 0.6181 | 0.2369 | 0.3971 | 0.2911 | 0.6400 | 0.6577 | 33.89 | 0.8452 |

**MFI-WHU, against verified all-in-focus references**

| Method | PSNR↑ | SSIM↑ |
|---|:-:|:-:|
| MCCSR-Net | **40.95** | 0.9968 |
| CCSR-Net | 38.52 | 0.9923 |
| CSNet | 38.40 | 0.9960 |
| QB | 38.04 | 0.9955 |
| GFF | 38.02 | 0.9957 |
| GFDF | 37.56 | 0.9963 |
| IFCNN (MAX) | 37.13 | 0.9910 |
| Zerene (DMap) | 37.12 | 0.9861 |
| MFIF-GAN | 37.01 | 0.9972 |
| BFMF | 36.88 | 0.9889 |
| IFM | 36.60 | 0.9905 |
| StackMFF V2 | 36.36 | 0.9951 |
| Ours (weighted) | 35.52 | **0.9978** |
| CVT | 35.41 | 0.9877 |
| SDNet | 35.34 | 0.9886 |
| Zerene (PMax) | 34.93 | 0.9931 |
| Ours (hard selection) | 34.90 | 0.9975 |
| Helicon Focus (B) | 34.73 | 0.9955 |
| Helicon Focus (C) | 34.73 | 0.9962 |
| NSCT | 34.72 | 0.9910 |
| DTCWT | 34.53 | 0.9897 |
| Helicon Focus (A) | 34.11 | 0.9946 |
| MFF-GAN | 31.76 | 0.9771 |
| DWT | 31.76 | 0.9704 |
| SwinFusion | 31.58 | 0.9647 |
| ReDiffuse | 31.22 | 0.9911 |
| MGFF | 31.04 | 0.9637 |
| StackMFF | 29.42 | 0.9404 |
| DDBFusion | 28.45 | 0.9144 |
| DeFusion | 28.10 | 0.8785 |
| DCT | 27.52 | 0.9519 |
| MUFusion | 26.96 | 0.9488 |
| VAEEDOF | 25.95 | 0.7964 |
| SwinMFF | 25.80 | 0.9511 |
| ZMFF | 25.31 | 0.9380 |
| U2Fusion | 24.18 | 0.9248 |
| GIFNet | 21.18 | 0.8648 |

**Real-MFF, against verified all-in-focus references**

| Method | PSNR↑ | SSIM↑ |
|---|:-:|:-:|
| IFCNN (MAX) | **41.44** | **0.9892** |
| Ours (weighted) | 41.07 | 0.9883 |
| Ours (hard selection) | 40.90 | 0.9878 |
| GFF | 40.54 | 0.9876 |
| Helicon Focus (C) | 40.51 | 0.9874 |
| Helicon Focus (A) | 40.42 | 0.9871 |
| GFDF | 40.36 | 0.9867 |
| Helicon Focus (B) | 40.36 | 0.9868 |
| Zerene (DMap) | 40.24 | 0.9843 |
| MFIF-GAN | 40.16 | 0.9864 |
| MCCSR-Net | 39.89 | 0.9859 |
| QB | 39.87 | 0.9854 |
| Zerene (PMax) | 39.72 | 0.9879 |
| CSNet | 39.47 | 0.9843 |
| BFMF | 39.32 | 0.9836 |
| StackMFF V2 | 38.73 | 0.9814 |
| NSCT | 38.70 | 0.9823 |
| CVT | 38.50 | 0.9809 |
| DTCWT | 38.50 | 0.9814 |
| CCSR-Net | 38.39 | 0.9798 |
| SDNet | 37.11 | 0.9812 |
| DWT | 36.74 | 0.9723 |
| MGFF | 36.72 | 0.9776 |
| IFM | 36.42 | 0.9780 |
| ZMFF | 36.28 | 0.9756 |
| SwinFusion | 34.92 | 0.9773 |
| MFF-GAN | 34.56 | 0.9707 |
| DDBFusion | 34.00 | 0.9692 |
| DeFusion | 33.78 | 0.9498 |
| StackMFF | 32.58 | 0.9690 |
| VAEEDOF | 31.19 | 0.8987 |
| ReDiffuse | 30.73 | 0.9697 |
| MUFusion | 30.31 | 0.9577 |
| DCT | 28.28 | 0.9168 |
| U2Fusion | 28.06 | 0.9612 |
| SwinMFF | 26.67 | 0.8935 |
| GIFNet | 23.49 | 0.8951 |

Reference metrics use 8-bit BT.601 luma, PSNR range 255 and an 11x11 Gaussian
SSIM window with sigma 1.5, without resizing or reference-dependent alignment.
Real-MFF is evaluated on a JPEG quality-100 mirror of the author release,
reproduced byte for byte from the complete author archive; the results therefore
characterize that shared frozen version for all methods alike.

<!-- pair-extension:end -->
## Installation

```bash
git clone https://github.com/Xinzhe99/StackMFF-V3.git
cd StackMFF-V3
conda create -n stackmffv3 python=3.10 -y && conda activate stackmffv3
pip install -r requirements.txt          # inference only: pip install torch numpy opencv-python
```

Install PyTorch for your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/) if the default wheel does not match. Training used Python 3.8, PyTorch 2.1.2 and CUDA 12.1; `fuse.py` was also tested with PyTorch 2.14.1 on the CPU (`requirements.txt` lists all tested versions).

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
| `--sigma` | `0.5` | Gaussian smoothing (pixels) of weighted-fusion probabilities; ignored by `--hard` |
| `--scales` | `1,0.75` | input scales whose probabilities are averaged (`1`: one pass, about 40% faster) |
| `--index_map` | | also save the index of the most probable image per pixel (0-based, natural order of the file names) |
| `--threads` | PyTorch default | CPU threads |

A stack must use one dtype throughout (all 8-bit or all 16-bit); grayscale inputs are loaded into three channels, and the saved fusion is three-channel. Images of a different size are resized to the first image; the stack must be registered (for hand-held bursts, align the images first, for example with [OpenFocus](https://github.com/Xinzhe99/OpenFocus)). To fuse many stacks at once, use `tools/run_ours_roots.py` (see below).

In Python (pass `scales=(1, 0.75)` explicitly to `load_checkpoint` to fix the paper configuration; an explicit `scales` argument takes precedence over `MS_SCALES`, whose value is read when the module is imported):

```python
import torch
from fuse import read_stack, probabilities, fuse
from stackmff_v3.model_io import load_checkpoint

model = load_checkpoint('weights/stackmff_v3_final.pth', 'cuda' if torch.cuda.is_available() else 'cpu', scales=(1, 0.75))
imgs = read_stack(['img_01.png', 'img_02.png', 'img_03.png'])        # list of HxWx3 BGR arrays (OpenCV)
P = probabilities(model, imgs, next(model.parameters()).device, sigma=0.5)   # [N, H, W], sums to 1 over N
fused = fuse(P, imgs, hard=False)                                     # HxWx3, same dtype as the input

# Hard selection uses the same checkpoint/scales and unsmoothed probabilities.
P_hard = probabilities(model, imgs, next(model.parameters()).device, sigma=0)
fused_hard = fuse(P_hard, imgs, hard=True)
```

## Pretrained weights

| File | Description | Size | SHA-256 |
|---|---|---|---|
| [stackmff_v3_final.pth](https://github.com/Xinzhe99/StackMFF-V3/releases/download/v1.0/stackmff_v3_final.pth) | final model of the paper (EMA weights after the third of six epochs); architecture arguments included | 0.66 MB | `39b2ea3738e8fea1676c379238d6be81a87988fc9952dbd6ac3825d5d7249f46` |

The checkpoint is a dictionary `{'model': state_dict, 'args': architecture}` (fp32, no optimizer state) and is loaded with `stackmff_v3.model_io.load_checkpoint`. Running `fuse.py` with these weights reproduces the reported configuration. Floating-point differences can affect a few rounded weighted values or change hard selection at near-tied probabilities; the archived outputs are the canonical inputs for recomputing the reported metrics.

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

`tools/run_ours_roots.py` fuses any folder tree `<root>/<dataset>/image stack/<scene>/` and was also used for the robustness variants (shuffled, incomplete and redundant stacks) and the real-data experiments. The fused images of our model (weighted and hard selection) on all four benchmarks are released as `stackmff_v3_fused_results.tar` (412 PNGs) in the [v1.0 release](https://github.com/Xinzhe99/StackMFF-V3/releases/tag/v1.0); the full-reference and no-reference scores for these outputs can be recomputed with `evaluation/run_eval.py`. The [archive index](benchmark/archives.json) provides 7,416 non-ZMFF benchmark outputs, 26 historical ZMFF outputs and 2,170 registered real-stack outputs. The full current ZMFF RGB archive remains pending; its complete numerical records and hashes are already available. The complete ReDiffuse image update is available as the `rediffuse_complete_20261020` download group; original archives remain available. Use `python tools/download_results.py --list` to inspect groups and download selected groups with SHA-256 verification. The [reproduction bundle](reproduction/README.md) provides the numerical source records and generators for the paper tables, including robustness and ablations; fused images for every training/perturbation variant are not part of the image archives.

### 4. Evaluation

`evaluation/` computes the primary reference-based and stack-based metrics in the paper: SSIM (Gaussian 11x11 and 7x7), PSNR, VIF, MI, NMI against the reference, and Q<sup>AB/F</sup> and MEF-SSIM against the source stack, with optional homography alignment for results that are cropped or rescaled. All metrics were cross-checked against the reference MATLAB implementations (`evaluation/xval/`, `evaluation/validate.py`). To evaluate a folder of results `<results>/<method>/<dataset>/<scene>.png`:

```bash
python evaluation/run_eval.py run --gt_root /path/to/benchmarks --gt_subdir AiF --stack_root /path/to/benchmarks \
    --results_root results --methods Ours_soft Ours_hard --out_dir eval_out --workers 6
python evaluation/run_eval.py summarize --gt_root /path/to/benchmarks --gt_subdir AiF --stack_root /path/to/benchmarks \
    --results_root results --methods Ours_soft Ours_hard --out_dir eval_out
python evaluation/stats.py --dir eval_out --ref Ours_soft      # Friedman / Nemenyi / Wilcoxon-Holm (needs >= 3 methods)
```

`run` caches one JSON file per method and scene (restartable); `summarize` writes `per_image.csv`, `summary.csv`, `summary.md` and `alignment_log.md`. The same options can be given in a JSON file (`--config`, see `evaluation/config_example.json`), which also maps methods stored in other folders (`overrides`) and lists methods that are always aligned (`aligned_methods`; Helicon Focus in the paper). Scene names are matched ignoring case, extension and punctuation.

### 5. Two-input fusion extension

The 863-pair experiment on Lytro, MFFW2, MFI-WHU and Real-MFF uses a separate
toolkit in [`benchmark/pair_extension/`](benchmark/pair_extension/README.md):
eight source-based metrics (MATLAB, with a toolbox-free Linux path) and
reference PSNR/SSIM, plus the aggregation and table generators. The frozen
protocol, per-pair manifest and metric provenance are included there, and every
per-image record binds input, implementation and output hashes. Results and the
typeset tables are in
[`benchmark/updates/2026-10-20-pair-extension/`](benchmark/updates/2026-10-20-pair-extension/README.md).

### 6. Efficiency and other tools

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

The entry below will be updated with the final bibliographic data once the paper is published.

```bibtex
@article{xie2026stackmffv3,
  title   = {StackMFF V3: Order-Invariant Focus Stacking via Pixel-Wise Source Classification},
  author  = {Xie, Xinzhe and Guo, Buyu and Li, Bolin and He, Shuangyan and Gu, Yanzhen and Jiang, Qingyan and Li, Peiliang},
  note    = {Manuscript in preparation},
  year    = {2026}
}
```

Earlier work of the series:

```bibtex
@article{xie2025stackmff,
  title     = {StackMFF: end-to-end multi-focus image stack fusion network},
  author    = {Xie, Xinzhe and Jiang, Qingyan and Chen, Dong and Guo, Buyu and Li, Peiliang and Zhou, Sangjun},
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

Our code is released under the [MIT License](LICENSE). Benchmark images, fused outputs and third-party components retain the rights and conditions of their underlying sources; they are not relicensed as MIT. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Acknowledgements

- Training sources: [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), [DUTS](http://saliencydetection.net/duts/), [DIODE](https://diode-dataset.org/), [Cityscapes](https://www.cityscapes-dataset.com/) and [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/); depth maps estimated with [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
- Benchmarks: Mobile Depth (Suwajanakorn et al., CVPR 2015), Middlebury (Scharstein et al., GCPR 2014), FlyingThings3D (Mayer et al., CVPR 2016) and Road-MF, in the focal-stack versions of the StackMFF V2 benchmark.
- The intra-image window attention follows [Swin Transformer](https://github.com/microsoft/Swin-Transformer). The baselines were run with the official code and weights of their authors; the metrics follow the reference implementations credited in `evaluation/metrics_lib.py`.
- The wordmark of the logo uses glyph outlines of Noto Sans (SIL Open Font License 1.1).
