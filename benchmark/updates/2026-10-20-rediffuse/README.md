# Validated ReDiffuse update, 20 October 2026

This update contains all 206 expected scenes (Mobile Depth 11, Middlebury 15,
FlyingThings3D 100 and Road-MF 80), evaluated with the same metric source as the
other methods. The original 26-scene record remains in the historical archive;
all eight scores on those scenes reproduce to a maximum difference of
5.6e-16. No result is selected by its metric value.

The frozen selection retains the original Mobile Depth/Middlebury outputs.
FlyingThings3D uses primary cohort A on even int(scene)/10 indices and validation
cohort B on odd indices. Scenes 0000010, 0000210, 0000870 and 0000890 instead use
the documented corrected 15-layer runs. Road-MF uses cohort A on odd IDs and
cohort B on even IDs, with previously assigned assistance outputs for IDs
69, 71, 73, 75, 77 and 79. Invalid short-stack runs and duplicate input copies
are excluded. Input, reference and output hashes identify the evaluated files.

ReDiffuse uses the released 2,000-step sampler, numeric focal order, reflect
padding to multiples of eight, official chroma weighting and saved 8-bit color
intermediates between pairwise steps. Its compiled convolution module is used
as released; two unavailable unused imports are removed. Repaired runners keep
the same inference function bodies. Outputs are scored without reference-guided
alignment. Shared-GPU inference logs are not reported as idle-GPU timings.

Metric evaluation retains the original Linux JPEG decoding. A local Windows
libjpeg-9 cross-check changed source-based scores, so that decoder was not used
for this update. See metric_crosscheck.json and runtime_fingerprints.json.

The complete 206-image RGB archive is now available in the [v1.0 release](https://github.com/Xinzhe99/StackMFF-V3/releases/download/v1.0/comparison_rediffuse_complete_20261020_outputs.tar). Every image was decoded and checked against the frozen input/output evidence and score record; the archive was checked again after packaging. The [image manifest](image_manifest.json) lists all per-file hashes and [archive metadata](image_archive.json) records the download and SHA-256. Download it with `python tools/download_results.py --groups rediffuse_complete_20261020`. The original archive contains 26 grayscale ReDiffuse PNGs. Their decoded grayscale pixels equal those from this RGB update, as checked in [the equivalence record](historical_rgb_equivalence.json); their file hashes differ. The downloader extracts the update under `image_updates/2026-10-20-rediffuse/`, and `benchmark/config_paper.json` selects that complete cohort explicitly. Existing historical files are preserved. Original inputs and third-party weights are not redistributed.
