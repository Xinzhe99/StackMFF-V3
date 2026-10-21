# Two-input multi-focus fusion extension, 20 October 2026

863 image pairs from four public benchmarks: Lytro 20, MFFW2 13, MFI-WHU 120,
Real-MFF 710. MFFW2 is the complete two-source subset of MFFW.

## Contents

| File | Content |
|---|---|
| `per_image_source_metrics.csv` | one row per scored output, eight source metrics plus output SHA256 |
| `summary_source_metrics.csv` | dataset means, emitted only for complete 37-configuration datasets |
| `coverage_source_metrics.csv` | scored versus expected pairs for every method and dataset |
| `summary_reference_metrics.csv` | PSNR / SSIM on the verified MFI-WHU and Real-MFF references |
| `coverage_reference_metrics.csv` | reference-metric coverage |
| `tables/` | the LaTeX tables exactly as they appear in the supplement |
| `evidence.json` | manifest, provenance and coverage bindings |

## Protocol summary

Eight source-based metrics: NMI, wavelet FMI, $Q^{AB/F}$, $Q_P$, $Q_Y$, $Q_{CB}$,
$Q_{CV}$ (minimized) and VIFF. Reference metrics: PSNR and Wang SSIM on 8-bit
BT.601 luma, data range 255, 11x11 Gaussian window with sigma 1.5, no resizing
and no reference-dependent alignment. Every reported mean uses the complete
dataset; a method-dataset combination is never ranked on a successful subset.

37 configurations: 28 non-commercial baselines, StackMFF V3 hard and weighted,
ZMFF, ReDiffuse, and Helicon Focus A/B/C plus Zerene Stacker PMax/DMap.

## Verification

Every per-image record binds the input hashes from the frozen manifest, the
metric implementation SHA256, and the output SHA256. `release_gate.py` in the
project workspace re-derives dataset means from the per-image rows, recomputes
reference metrics from raw PNGs, and compares every number printed in the LaTeX
tables against this summary. The export refuses to run unless that gate passes.

## Known qualifications

* Real-MFF is evaluated on a JPEG quality-100 mirror of the author release.
  All 2,130 files were reproduced byte for byte from the complete author
  archive with a deterministic encoder, which verifies identity and provenance
  but not pixel equality with the author PNGs.
* The released SwinFusion training instructions refer to MFI-WHU data; the exact
  membership of the evaluated checkpoint is not fully documented.
* Source-based agreement with the inputs is not reference fidelity; the two
  metric families can disagree.

Inputs, references and third-party weights are not redistributed here.
