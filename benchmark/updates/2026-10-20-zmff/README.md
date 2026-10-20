# Validated ZMFF update, 20 October 2026

All 206 scenes are included: Mobile Depth 11, Middlebury 15, FlyingThings3D 100
and Road-MF 80. Scores use the same metric source and Linux decoding as the
established comparison. The original 26 scenes reproduce all eight measurements
within 8.9e-16. Each file is bound to input, reference and output hashes.

The fixed selection was recorded before scoring. It retains the original
Mobile Depth/Middlebury runs and the registered primary/assistance assignments.
The four corrected FT3D scenes use 15 inputs; invalid duplicate-input results
are excluded. No metric chooses between duplicate outputs. See selection.json.

ZMFF is zero-shot per-pair optimization, with no pretrained checkpoint. The
released 1,300 iterations, losses, Adam schedule and seed 17 are retained. The
stack adapter uses numeric focal order, quantizes luminance between pairs and
retains the first source's chroma. GPU nondeterminism prevents bitwise agreement
between repeated official runs; the wrapper follows the original inference
functions and settings. This limitation is disclosed in the manuscript.
Outputs are scored without resizing or reference-guided alignment. Shared-GPU
timings are not reported as comparable idle-GPU runtimes. The reported operation
count is forward-only and excludes the optimization backward pass.

Per-scene metrics and hashes are complete. The separate full RGB image archive
is still transferring; the historical release contains only 26 ZMFF images.
The current evaluation config reserves image_updates/2026-10-20-zmff/ for the
complete RGB cohort, so it cannot silently fall back to those historical files.
Do not interpret this score release as publication of the full image archive.
Original inputs, references and third-party weights are not redistributed.
