"""Reference-dataset declaration for the two-input extension.

The private experiment workspace binds each reference file to a proof record and
an approval hash before PSNR/SSIM may be computed. Those proofs reference local
filesystem paths and are not redistributed.

This public module declares only which datasets supply verified all-in-focus
references. Every released reference-metric record already carries its own
`reference_sha256`, `reference_proof_sha256` (and, for Real-MFF, the approval
hash), so a reader can verify an individual record without this module; see
benchmark/updates/2026-10-20-pair-extension/summary_reference_metrics.csv and the
README section "Reference metrics".
"""

DATASETS = ("MFI-WHU", "RealMFF")


def lock_filename(dataset):
    """Name of the per-dataset scoring lock used during the experiment."""
    assert dataset in DATASETS, dataset
    return "reference_score_%s.lock" % dataset
