"""Tests for the shared-projection prospective freeze script. Confirms the
frozen artifact records the required disclosure fields, matches
independently re-computed upstream provenance hashes, and that freezing
never touches the primary/Secondary-1/Secondary-2 artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import freeze_target_projection_method as freeze_mod  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_upstream_artifacts_untouched():
    before = {p: p.read_bytes() for p in (freeze_mod.PRIMARY_ARTIFACT_PATH, freeze_mod.SECONDARY_1_ARTIFACT_PATH, freeze_mod.SECONDARY_2_ARTIFACT_PATH)}
    freeze_mod.main()
    for p, content in before.items():
        assert p.read_bytes() == content


def test_upstream_provenance_hashes_independently_reproduced():
    result = freeze_mod.main()
    pointers = result["upstream_provenance_pointers"]
    assert pointers["primary_artifact_sha256"] == _sha256(freeze_mod.PRIMARY_ARTIFACT_PATH)
    assert pointers["secondary_1_artifact_sha256"] == _sha256(freeze_mod.SECONDARY_1_ARTIFACT_PATH)
    assert pointers["secondary_1_artifact_sha256"] == "c0391906784f54c17f1503fccc91963f22af85dd81dc090bdc69ec38e75684e0"
    assert pointers["secondary_2_artifact_sha256"] == _sha256(freeze_mod.SECONDARY_2_ARTIFACT_PATH)


def test_required_disclosure_fields_present():
    result = freeze_mod.main()
    assert result["method_name"] == "SHARED_COMMON_SHIFT_SUPPORT_PROJECTION"
    assert result["status"] == "PROSPECTIVELY_FROZEN_BEFORE_TARGET_G_OUTPUT_INSPECTION"
    assert result["target_g_outputs_inspected_before_freeze"] is False
    assert result["target_human_outcomes_used"] is False
    assert set(result["shared_across_methods"]) == {"PRIMARY_M2", "SECONDARY_1_MCONST", "SECONDARY_2_MCONST_GSHAPE"}
    assert result["real_benchmark_shared_raw_items"] == {}
    assert result["not_executed_on_real_target_output_in_this_freeze"] is True


def test_artifact_written_only_under_secondary_diagnostic_dir_and_self_hash_matches():
    result = freeze_mod.main()
    artifact_path = freeze_mod.OUT_DIR / "target_projection_method.json"
    on_disk = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert on_disk["method_name"] == "SHARED_COMMON_SHIFT_SUPPORT_PROJECTION"
    sha_file = (freeze_mod.OUT_DIR / "target_projection_method.sha256.txt").read_text(encoding="utf-8").strip()
    assert sha_file == result["projection_artifact_sha256"]
    assert freeze_mod.OUT_DIR.name == "secondary_calibration_diagnostic"
