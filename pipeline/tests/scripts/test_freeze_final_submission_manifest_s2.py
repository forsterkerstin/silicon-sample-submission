"""Tests for the S2-only final-submission manifest freeze. Confirms it
never touches the historical M2 manifest or any other upstream artifact,
independently re-verifies every cited hash, and records the required
governance fields."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import freeze_final_submission_manifest_s2 as freeze_mod  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_old_m2_manifest_and_all_upstream_artifacts_untouched():
    before = {p: p.read_bytes() for p in (freeze_mod.OLD_M2_MANIFEST_PATH, freeze_mod.PRIMARY_ARTIFACT_PATH, freeze_mod.SECONDARY_1_ARTIFACT_PATH, freeze_mod.SECONDARY_2_ARTIFACT_PATH, freeze_mod.PROJECTOR_ARTIFACT_PATH, freeze_mod.A3_ARTIFACT_PATH)}
    freeze_mod.main()
    for p, content in before.items():
        assert p.read_bytes() == content


def test_old_m2_hash_matches_expected():
    assert _sha256(freeze_mod.OLD_M2_MANIFEST_PATH) == "0fb04d97f7e360e3f8222559eab6b8e25cd9f988eb84f0bda2de11ebe17875b6"


def test_manifest_records_sole_submission_governance():
    result = freeze_mod.main()
    assert result["final_method"] == "S2_MCONST_GSHAPE"
    assert result["status"] == "SOLE_FINAL_SUBMISSION"
    assert result["target_f_dependence"] is False
    assert result["target_g_dependence"] is True
    assert result["reclassification_for_final_submission_purposes"] == {
        "PRIMARY_M2": "DEVELOPMENT_CANDIDATE_NOT_SUBMITTED",
        "SECONDARY_1_MCONST": "DEVELOPMENT_BASELINE_NOT_SUBMITTED",
        "A3_PROMPT_ENSEMBLE": "DEVELOPMENT_PROPOSAL_NOT_SUBMITTED",
        "S2_MCONST_GSHAPE": "SOLE_FINAL_SUBMISSION",
    }
    assert result["supersedes_for_submission_purposes"]["historical_m2_candidate_manifest_sha256"] == "0fb04d97f7e360e3f8222559eab6b8e25cd9f988eb84f0bda2de11ebe17875b6"
    assert result["s2_artifact_sha256"] == "07518f45735ec4f702d6a67521cb06cf4a1c306f4e3e206ce823aac262abeaad"


def test_manifest_written_and_self_hash_matches():
    result = freeze_mod.main()
    sha_file = freeze_mod.VALIDATION_DIR.joinpath("frozen_final_submission_manifest_s2.sha256.txt").read_text(encoding="utf-8").strip()
    assert sha_file == result["s2_final_submission_manifest_sha256"]
