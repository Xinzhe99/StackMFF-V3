# Recomputing the reported comparisons

The **current complete cohort** contains **30 baseline families / 33 settings, plus two proposed read-outs**: 35 methods/settings on all 206 scenes (7,210 primary images, 140 method/dataset cells).

- [Current SSIM/PSNR table](current/TABLES.md), [metric means and cohort-specific ranks](current/summary.csv), [per-scene records](current/per_image.csv), [method identifiers and venues](current/methods.json).
- [All-scene SSIM differences](current/scene_ssim_gains.png), with [206 scene records](current/scene_ssim_gains.csv) and [selection/provenance metadata](current/scene_ssim_gains.json). The external comparator is fixed per dataset by its highest unrounded mean SSIM; every scene is included. Weighted fusion improves SSIM on 204 of 206 scenes against these fixed comparators.
- [Current statistical comparisons](current/stats.md): paired Wilcoxon tests with Holm correction, Friedman ranks and Nemenyi critical differences. The Mobile Depth PSNR margin over GFF is not statistically significant.
- [Baseline protocols](baseline_protocols.tex) document the source implementations, stack adaptations and runtime accommodations.
- [Qualitative comparator selection](current/qualitative_method_selection.json) records the equal-dataset mean SSIM ranking for each selected category. The displayed set is GFF, DTCWT, MFIF-GAN, IFCNN (MAX), StackMFF V1/V2 and the proposed weighted output. CSNet has higher mean PSNR than MFIF-GAN; selecting by SSIM does not imply superiority on both metrics. IFCNN here uses the recorded sequential adaptation, although its generic implementation also accepts multiple inputs; a joint variant was not evaluated in these records.
- `auxiliary/` retains corrected real-stack metrics, registration matrices, range checks, robustness records and exploratory downstream results. The downstream probe does not establish an advantage of the proposed model.
- The current [confidence diagnostic](auxiliary/confidence_main_analysis.json) uses Motorcycle and keyboard; [export provenance](auxiliary/confidence_main_provenance.json) records the released weights and input/reference hashes. `auxiliary/confidence_analysis.json` retains the previous example set. Export current arrays with `python tools/save_conf.py --scenes "Middlebury/Motorcycle" "Mobile Depth/keyboard" --bench_root /path/to/benchmarks --out conf_maps.npz --device cpu`. The helper preserves float32 confidence and uses luma decoded from the rounded RGB output; these scene-specific correlations do not establish uncertainty calibration.

The original [37-row table](TABLES.md), `summary.csv`, `per_image.csv` and `stats.*` remain as the **historical archive**. It includes 52 partial ZMFF/ReDiffuse outputs on Mobile Depth and Middlebury in addition to the 7,210 complete-cohort images. These partial methods are excluded from the current comparison until all four cohorts are complete. Archive image manifests and hashes are unchanged.


## Image archives

The [v1.0 data release](https://github.com/Xinzhe99/StackMFF-V3/releases/tag/v1.0) contains **all 7,262 benchmark outputs** and **2,170 registered real-stack outputs**, alongside the checkpoint. [archives.json](archives.json) records complete-archive SHA-256 values and, for large archives, verified 128 MiB part hashes. [image_manifest.csv](image_manifest.csv) and [real_image_manifest.csv](real_image_manifest.csv) identify every published image. Original inputs/references and third-party model weights are not included. Data conditions are in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

From the repository root:

```bash
python tools/download_results.py --list
python tools/download_results.py --help
```

Select only the groups needed for a comparison. The downloader resumes interrupted transfers, verifies parts and the assembled archive, and checks extraction paths. The [table reproduction bundle](../reproduction/README.md) also contains the numerical inputs and generators for robustness and ablations; images from every perturbation/training variant are outside the image-archive scope.

## Protocol

The evaluator loads 8-bit grayscale through Pillow's BT.601 conversion, evaluates the full image, and computes Gaussian-window SSIM, PSNR, VIF, MI and NMI, plus the alternate 7×7 SSIM convention and two stack-based no-reference metrics. The six full-reference columns represent five metric families; two SSIM conventions are retained for comparison with earlier StackMFF tables. Every metric's reference and boundary rules are documented in `evaluation/metrics_lib.py`; MATLAB/Octave cross-validation is in `evaluation/xval/`.

Pairwise methods fuse in natural numeric focal order. Helicon's cropped/rescaled outputs are aligned to the reference using the recorded ORB–RANSAC homography protocol; both raw and aligned rows are retained in `per_image.csv`, and `primary=True` identifies the paper's evaluation. Other comparison images are evaluated without registration to the reference. Real optical stacks use a separate input-only ECC affine registration, shared by all methods; the corrected composition is `W @ T_neighbor`. Their range audit uses a common per-channel envelope over Pillow and OpenCV decodings, tolerance two, with single-decoder sensitivity columns retained.

The checkpoint, smoothing and inference scales were selected using these four public benchmarks. These results therefore do not constitute an independently held-out estimate of model-selection generalization.

`config_paper.json` selects the 35 complete-cohort settings. `config_archive.json` retains the historical partial-coverage configuration. In either configuration, an absent image in an expected cell is still an error. Failed cached records are retried, and summaries ignore obsolete methods outside the selected configuration. Use a fresh output directory when changing result images, metric code or library versions: successful cache entries are reused within a run and are not automatically invalidated by changes to source files.

## Run

Obtain the focal stacks and AiF references from the [StackMFF V2 benchmark](https://github.com/Xinzhe99/StackMFF-V2#-downloads), respecting the original [data conditions](../THIRD_PARTY_NOTICES.md). Extract the verified output archives into this repository, producing `results/<method>/<dataset>/<scene>.<ext>`. Run from the repository root:

```bash
python evaluation/run_eval.py run --config benchmark/config_paper.json \
  --gt_root /path/to/benchmarks --gt_subdir AiF \
  --stack_root /path/to/benchmarks --out_dir eval_out --workers 2
python evaluation/run_eval.py summarize --config benchmark/config_paper.json --out_dir eval_out
python evaluation/stats.py --dir eval_out --ref Ours_final_soft
```

The frozen records in this folder are supplied so every mean and statistical comparison can be examined without rerunning an image metric. Exact last digits may depend on the library versions; archive image hashes identify the actual evaluated files. `current/summary.csv` ranks are recalculated over the current 35-row cohort. Rebuild it from the immutable historical measurements with `python tools/build_current_benchmark.py`, then run `python evaluation/stats.py --dir benchmark/current --ref Ours_final_soft`.
