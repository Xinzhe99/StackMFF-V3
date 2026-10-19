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

The per-scene scores and image hashes are released here. Transfer of the new
full RGB image archive is still in progress; the v1.0 image release remains the
historical 26-scene subset for this method. The original inputs and third-party
weights are not redistributed.
