# Recomputing the reported comparisons

This directory records the exact paper subset: **32 baseline families / 35 settings, plus our two fusion rules**. MD has 11 stacks, Middlebury 15, FlyingThings3D 100 and Road-MF 80. ZMFF and ReDiffuse currently cover MD and Middlebury only: this is 7,262 primary result images across 144 method/dataset cells. Missing coverage is explicit; partial results from the ongoing runs are not mixed into this snapshot.

- [Complete SSIM/PSNR table](TABLES.md), [all metric means and ranks](summary.csv), [per-image metrics](per_image.csv), [method identifiers](methods.json).
- [Statistical comparisons](stats.md): two-sided Wilcoxon signed-rank tests against `Ours_final_soft`, Holm correction within each dataset/metric family; Friedman ranks and Nemenyi critical differences. Direction and non-significant comparisons matter. The 0.66 dB MD mean margin over GFF is not statistically significant.
- [Baseline protocols](baseline_protocols.tex): evaluated source references, runtime accommodations, pairwise ordering and method-specific details, as included in the manuscript supplement.
- `auxiliary/`: corrected real-stack metrics, decoder sensitivity, registration matrices, robustness scores, and the exploratory downstream probe's OOF predictions, splits and repeat-level statistics. Downstream results do not establish an advantage of the proposed model.

## Image archives

The existing [v1.0 release](https://github.com/Xinzhe99/StackMFF-V3/releases/tag/v1.0) contains our 412 fused PNGs and the checkpoint. The complete comparison archives are being assembled and verified; they are not yet claimed as publicly available here. Canonical archive paths and SHA256 checksums will be added after upload verification. Original inputs/references and third-party model weights are not included.

## Protocol

The evaluator loads 8-bit grayscale through Pillow's BT.601 conversion, evaluates the full image, and computes Gaussian-window SSIM, PSNR, VIF, MI and NMI, plus the alternate 7×7 SSIM convention and two stack-based no-reference metrics. The six full-reference columns represent five metric families; two SSIM conventions are retained for comparison with earlier StackMFF tables. Every metric's reference and boundary rules are documented in `evaluation/metrics_lib.py`; MATLAB/Octave cross-validation is in `evaluation/xval/`.

Pairwise methods fuse in natural numeric focal order. Helicon's cropped/rescaled outputs are aligned to the reference using the recorded ORB–RANSAC homography protocol; both raw and aligned rows are retained in `per_image.csv`, and `primary=True` identifies the paper's evaluation. Other comparison images are evaluated without registration to the reference. Real optical stacks use a separate input-only ECC affine registration, shared by all methods; the corrected composition is `W @ T_neighbor`. Their range audit uses a common per-channel envelope over Pillow and OpenCV decodings, tolerance two, with single-decoder sensitivity columns retained.

The checkpoint, smoothing and inference scales were selected using these four public benchmarks. These results therefore do not constitute an independently held-out estimate of model-selection generalization.

`config_paper.json` explicitly declares the two partial-coverage methods. The runner skips only those unreported cells; an absent image in an expected cell is still an error. Failed cached records are retried, and summaries ignore obsolete methods outside the selected configuration. Use a fresh output directory when changing result images, metric code or library versions: successful cache entries are reused within a run and are not automatically invalidated by changes to source files.

## Run

Obtain the focal stacks and AiF references from the [StackMFF V2 benchmark](https://github.com/Xinzhe99/StackMFF-V2#-downloads), respecting the original [data conditions](../THIRD_PARTY_NOTICES.md). Extract the verified output archives into this repository, producing `results/<method>/<dataset>/<scene>.<ext>`. Run from the repository root:

```bash
python evaluation/run_eval.py run --config benchmark/config_paper.json \
  --gt_root /path/to/benchmarks --gt_subdir AiF \
  --stack_root /path/to/benchmarks --out_dir eval_out --workers 2
python evaluation/run_eval.py summarize --config benchmark/config_paper.json --out_dir eval_out
python evaluation/stats.py --dir eval_out --ref Ours_final_soft
```

The frozen records in this folder are supplied so every mean and statistical comparison can be examined without rerunning an image metric. Exact last digits may depend on the library versions; archive image hashes identify the actual evaluated files. `summary.csv` ranks are recalculated over the paper's method subset, rather than obsolete experimental variants.
