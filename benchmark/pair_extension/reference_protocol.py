"""Provenance gates for the frozen image-pair reference protocol."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = ("MFI-WHU", "RealMFF")
ROLES = ("a", "b", "reference_candidate")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reference_protocol(dataset, base=HERE):
    """Fail closed unless all input/reference roles have verified provenance."""
    base = Path(base)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    rows = [r for r in manifest["scenes"] if r["dataset"] == dataset]
    expected = {(str(r["scene"]), role): r[role] for r in rows for role in ROLES}
    assert dataset in DATASETS, dataset
    assert len(rows) == (120 if dataset == "MFI-WHU" else 710)
    assert len(expected) == 3 * len(rows)
    provenance = base / "reference_provenance"
    if dataset == "MFI-WHU":
        proof_path = provenance / "mfi_reference_validation.json"
        proof = json.loads(proof_path.read_text(encoding="utf8"))
        assert proof["status"] == "verified" and not proof["missing"]
        actual = [(r["local"], r["role"]) for r in proof["matched"]]
        assert len(actual) == len(expected)
        assert set(actual) == {(r["path"], role) for (_, role), r in expected.items()}
        return {"proof_sha256": sha(proof_path), "encoding": "Author-release JPEG files, byte-identical"}

    proof_path = provenance / "real_reference_validation.json"
    approval_path = provenance / "real_reference_scoring_approval.json"
    proof = json.loads(proof_path.read_text(encoding="utf8"))
    approval = json.loads(approval_path.read_text(encoding="utf8"))
    assert proof["status"] == "verified_author_jpeg_derivative"
    assert proof["expected_files"] == proof["matched_files"] == 2130
    assert proof["mismatched_files"] == 0
    assert proof["manifest_sha256"] == sha(manifest_path)
    assert approval["status"] == "approved" and approval["dataset"] == dataset
    assert approval["proof_sha256"] == sha(proof_path)
    assert approval["manifest_sha256"] == sha(manifest_path)
    assert approval["independently_rehashed_files"] == 2130
    assert approval["role_counts"] == {role: 710 for role in ROLES}
    records = proof["records"]
    assert len(records) == len(expected)
    assert {(str(r["scene"]), r["role"]) for r in records} == set(expected)
    for record in records:
        info = expected[str(record["scene"]), record["role"]]
        assert record["byte_exact_jpeg_q100_derivative"] is True
        assert record["local"] == info["path"]
        assert record["local_sha256"] == record["reencoded_sha256"] == info["sha256"]
    return {
        "proof_sha256": sha(proof_path),
        "approval_sha256": sha(approval_path),
        "encoding": "Frozen JPEG quality-100 mirror of author PNG sources and references",
    }


def lock_filename(dataset):
    # Keep compatibility with the already-running MFI-WHU scorer.
    assert dataset in DATASETS
    return "reference_score.lock" if dataset == "MFI-WHU" else "reference_score_RealMFF.lock"
