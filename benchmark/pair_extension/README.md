# Two-input multi-focus fusion extension

This directory contains the complete evaluation toolkit and results for the
863-pair two-input experiment reported in the supplement.

Data: Lytro (20 pairs), MFFW2 (13), MFI-WHU (120), Real-MFF (710). MFFW2 is the
complete two-source subset of MFFW.

## Layout

```
pair_extension/
  README.md                  this file
  methods.json               the 37 active configurations (labels, venues, groups)
  method_registry.py         loads methods.json for the scripts below
  metrics_lib.py             reference metrics: PSNR, Wang SSIM (manuscript code)
  metrics/                   the eight source-based metrics (MATLAB)
    evaluate_pair_metrics.m        Windows entry point (compiled FMI helper)
    evaluate_pair_metrics_linux.m  Linux entry point (pure-MATLAB FMI)
    run_metric_worker.m            resumable lane, Windows
    run_metric_worker_linux.m      resumable lane, Linux
    validate_linux_metrics.m       equivalence harness against archived scores
    metricsFMI_wavelet*.m          wavelet FMI (reference, compiled, vectorized)
    metricsQabf_corrected.m        edge preservation with documented repair
    metricsQp_cached.m             phase congruency with source-feature cache
    metricsQy_stable.m             Yang structural similarity, stable variances
    vendor/                        upstream implementations (see provenance.json)
    dmey.mat                       Meyer wavelet filter table
    {dwt2,fspecial,mean2,im2double,blkproc,fflush}.m
                                   compatibility layer for base MATLAB
  aggregate_scores.py        per-image -> complete-dataset means, hash-verified
  aggregate_verified_references.py  PSNR/SSIM aggregation with protocol binding
  make_pair_tables.py        LaTeX tables from the source-metric summary
  make_reference_tables.py   LaTeX tables from the reference-metric summary
  score_verified_references.py  reference scoring against verified references
  prepare_manifest.py        builds the frozen pair manifest
```

Results live in `../updates/2026-10-20-pair-extension/`.

## Reproducing the tables

The aggregation scripts are paths-relative to the directory holding `scores/`,
`results/`, `manifest.json` and the summaries. From a checkout where those exist:

```bash
python aggregate_scores.py --inventory output_inventory.tsv
python aggregate_verified_references.py
python make_pair_tables.py
python make_reference_tables.py
```

`aggregate_scores.py` enforces, per record: input hashes equal the frozen
manifest, the metric implementation hash equals the provenance file, the output
bytes hash equals the record, and no two records for the same scene disagree. A
method-dataset combination enters a summary only when every expected pair is
scored, so no table is ever ranked on a successful subset.

## The eight source metrics

| Metric | Meaning | Direction |
|---|---|---|
| NMI | normalized mutual information | higher |
| FMI_w | wavelet feature mutual information (dmey) | higher |
| Qabf | gradient-based edge preservation ($Q^{AB/F}$) | higher |
| QP | phase congruency | higher |
| QY | Yang structural similarity | higher |
| QCB | Chen-Blum perceptual quality | higher |
| QCV | Chen-Varshney perceptual distortion | **lower** |
| VIFF | fusion-specific visual information fidelity | higher |

Grayscale conversion is a single BT.601 luma pass applied to every saved output.

## Running the metrics without a MATLAB toolbox licence

The production values on Windows used a compiled FMI helper
(`fmi_windows_mex.mexw64`, built by `build_fmi_mex.m`). On a host with base
MATLAB only, use `evaluate_pair_metrics_linux.m`, which evaluates the same dmey
features and Nelsen formulas through `metricsFMI_wavelet_vectorized.m`.

Six small files reproduce toolbox functions that the base installation lacks. Each
supports only the call forms this pipeline uses and raises an error otherwise:

| File | Toolbox origin | Restriction |
|---|---|---|
| `dwt2.m` | Wavelet | only `dwt2(x,'dmey')`, default mode |
| `fspecial.m` | Image Processing | only `('gaussian', n, sigma)`, odd n |
| `mean2.m` | Image Processing | numeric/logical arrays |
| `im2double.m` | Image Processing | uint8/uint16/single/double/logical |
| `blkproc.m` | Image Processing | zero border, scalar block output |
| `fflush.m` | (Octave guard) | no-op; the vendored `version(1)<'5'` test misfires on R2025a |

Equivalence was verified on real data: a 20-record cohort drawn from the archived
Windows scores was recomputed on Linux and all eight metrics agreed to within
2.7e-13 (see `metrics/mex_validation.json` and the experiment audit record).

## Reference metrics

`score_verified_references.py` computes PSNR and SSIM against the all-in-focus
references using `metrics_lib.py`: Pillow uint8 `L` conversion (BT.601 luma), PSNR
data range 255, and Wang SSIM with an 11x11 Gaussian window (sigma 1.5) over valid
windows. No resizing and no reference-dependent alignment.

MFI-WHU references match the author release byte for byte. Real-MFF is evaluated
on a JPEG quality-100 mirror of the author release; all 2,130 mirror files were
reproduced byte for byte from the complete author archive with a deterministic
encoder.

## Provenance summary

* `manifest.json` fixes every pair's A/B/reference paths, dimensions and SHA256.
* `metrics/provenance.json` records the implementation revisions, the vendor
  sources and the numeric repairs (Qabf equality branch, QY variance stability,
  QP source-feature caching), with the validation cases for each.
* Every per-image score record stores the metric implementation SHA256, both input
  hashes and the output SHA256, so a record cannot be separated from the bytes it
  describes.

Inputs, references and third-party weights are not redistributed here; obtain them
from the original dataset and method releases listed in the manuscript.
