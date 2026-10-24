# Benchmark data and how to use it

This directory holds every number behind the comparisons in the paper, plus the
images they were computed from. You do not need to re-run anything to check a figure
or a table: start with the CSV files below.

## Two experiments

The paper reports two separate experiments. They use different inputs and different
metrics, so their numbers are never mixed.

| | **Image-stack fusion** | **Two-input pair fusion** |
|---|---|---|
| Input | a stack of 11–100 images per scene | exactly two images |
| Scenes | 206 | 863 |
| Datasets | Mobile Depth, Middlebury, FlyingThings3D, Road-MF | Lytro, MFFW2, MFI-WHU, Real-MFF |
| Methods | 37 configurations | the same 37 configurations |
| Metrics | SSIM, PSNR against an all-in-focus reference; VIF, MI, NMI, $\mathrm{Q}^{AB/F}$, MEF-SSIM against the inputs | 8 source-based metrics; PSNR and SSIM where a reference exists |
| Start here | `current/TABLES.md` | `../pair_extension/RESULTS.md` |

## The image-stack experiment

Everything lives under `current/`.

| File | What it is |
|---|---|
| `TABLES.md` | the full 37-row comparison, one row per method, four benchmark columns |
| `per_image.csv` | one row per method and scene: the metric vector for that scene |
| `summary.csv` | per method and dataset: the mean of each metric and its rank |
| `stats.md`, `stats.csv` | paired Wilcoxon tests with Holm correction, Friedman ranks, Nemenyi intervals |
| `methods.json` | method identifiers, display names and publication venues |
| `scene_ssim_gains.csv` | per-scene SSIM difference against the fixed external comparator |
| `qualitative_method_selection.json` | which methods appear in the qualitative figures, and the ranking rule that picked them |

A method appears in a summary only when every scene of that dataset is scored, so no
row is ever ranked on a successful subset. `per_image.csv` uses `primary=True` to mark
the evaluation the paper reports where a method has more than one legitimate reading
(Helicon Focus, whose output is cropped and needs alignment).

## The two-input experiment

`../pair_extension/` holds the toolkit and `../pair_extension/RESULTS.md` holds the
tables. Each of the 31,931 outputs has a record binding three hashes: the input
files, the metric implementation, and the output bytes. `python
../pair_extension/verify_release.py` rebuilds every table from those records and
compares byte for byte; it prints `REPRODUCED` when they agree.

## The images

All released outputs and the checkpoint are in the single
[v2.0 release](https://github.com/Xinzhe99/StackMFF-V3/releases/tag/v2.0).
`archives.json` lists each archive with its size, SHA-256 and image count, grouped by
experiment. The release's own `release_index.json` carries the same information.

```bash
python tools/download_results.py --list                 # show the 18 groups
python tools/download_results.py --groups ours zmff   # fetch and verify a selection
```

The downloader resumes interrupted transfers and checks every part against the
recorded hashes before extraction.

## Reproducing the evaluation

To recompute a metric rather than read a stored value, get the source benchmarks and
run the evaluator from the repository root:

```bash
python evaluation/run_eval.py run --config benchmark/config_paper.json \
  --gt_root /path/to/benchmarks --gt_subdir AiF \
  --stack_root /path/to/benchmarks --out_dir eval_out --workers 2
python evaluation/run_eval.py summarize --config benchmark/config_paper.json --out_dir eval_out
```

`config_paper.json` selects the 37 complete-cohort configurations;
`config_archive.json` keeps the older partial-coverage set for provenance. Use a fresh
`--out_dir` whenever the result images, metric code or library versions change, since
cached records are reused within a run and are not invalidated automatically.

## Scope and caveats

* The checkpoint, smoothing sigma and inference scales were chosen using these four
  benchmarks, so the scores are not an independently held-out estimate of
  generalization.
* GFF, DTCWT and the other classical baselines run in MATLAB, so their cost is not
  comparable with the GPU timings; the efficiency table reports GPU methods only and
  says so.
* ZMFF is not bit-reproducible on this GPU stack: two runs of one scene with the same
  seed differ in 63.7% of pixels. Its scores describe the released images.
* Real-MFF is evaluated on a JPEG quality-100 mirror of the author release, reproduced
  byte for byte from the author archive; all methods see the same frozen version.

`historical/` keeps earlier snapshots for provenance. Their links point at releases
that have since been consolidated and should not be used.
