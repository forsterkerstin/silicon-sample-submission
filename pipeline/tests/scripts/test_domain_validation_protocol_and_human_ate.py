"""Tests for the domain-validation protocol freeze and the real Orchinik
human ATE surface computation. Confirms hashes match the OSF-reported
values (independently re-verified, not trusted blindly), the Howe blocker
is recorded honestly, and neither script touches MU_EXTERNAL/S2/target
data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import compute_orchinik_human_ate_surface as ate_mod  # noqa: E402
import freeze_domain_validation_protocol as protocol_mod  # noqa: E402

EXPECTED_HASHES = {
    "howe/Analysis_Howe.R": "22912d07b6c9a73b15f7bc070a240002fce3b690301590f4fa09357a6e3379bc",
    "howe/Data_Howe.csv": "d22176cae94a9efa560f9942a4b363eb4f1fce55326d334f101f08fc97c5618e",
    "orchinik/Bovitz_qualtrics.docx": "fd3843244d28d3f6d64f809e982ccd5ebed8e347ea585ce9a4e4807abadfd44f",
    "orchinik/Bovitz_qualtrics.qsf": "7a6528b5a13febb8f8eb30ee98c1df43100890df0b075800924ef4bc2ed52026",
    "orchinik/final_bovitz_raw.csv": "cbae3d4a7faf1027e0434d1af62b527dc82a8bf6b9b7935a469fb058a0ea65f6",
    "orchinik/bovitz_data_clean.R": "b871da4832a631dbdfe765b6de2591639d8b7afb459f5daf557c3b3176cfd8e3",
    "orchinik/final_clean.csv": "6d9168d87c5ff471b4bcb94fe1542a80816c6dd2ff23eb7fa4f6aec5cac2bf06",
}


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_downloaded_source_hashes_match_osf_reported_values():
    for rel, expected in EXPECTED_HASHES.items():
        assert _sha256(protocol_mod.DOMAIN_DATA_DIR / rel) == expected


def test_orchinik_human_ate_surface_has_exactly_50_real_cells():
    result = ate_mod.main()
    assert result["n_cells"] == 50
    assert result["eligible_n"] == 2545
    assert result["arm_counts_eligible"] == {"control": 847, "skill": 837, "trust": 861}
    for c in result["cells"]:
        assert c["treat_n"] > 0 and c["control_n"] > 0


def test_orchinik_eligible_n_mismatch_fails_closed(monkeypatch):
    monkeypatch.setattr(ate_mod, "EXPECTED_ELIGIBLE_N", 999999)
    import pytest

    with pytest.raises(ValueError, match="eligible N mismatch"):
        ate_mod.main()


def test_protocol_records_howe_blocker_and_orchinik_status():
    result = protocol_mod.main()
    assert result["howe"]["status"] == "BLOCKED"
    assert "trustmed2" in result["howe"]["blocker"]
    assert result["orchinik"]["eligible_n"] == 2545
    assert result["orchinik"]["condition_label_mapping"]["History"] == "skill (Skill Intervention block)"
    assert result["orchinik"]["condition_label_mapping"]["Institutions"] == "trust (Trust Intervention block)"
    assert result["orchinik"]["manifest_construction_status"].startswith("NOT BUILT")


def test_protocol_never_touches_mu_external_or_s2():
    result = protocol_mod.main()
    assert result["mu_external_reestimated"] is False
    assert result["s2_gamma_changed"] is False
    assert result["s2_frozen_unchanged"] == {"mu_external": 1.9558595458395387, "gamma_g_shape": 1.0}
    assert result["target_g_scientific_outputs_used"] is False
    assert result["target_human_outcomes_used"] is False


def test_protocol_written_and_self_hash_matches():
    result = protocol_mod.main()
    out_path = protocol_mod.OUT_DIR / "frozen_domain_validation_protocol.json"
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk["status"] == "FROZEN_BEFORE_MODEL_PREDICTIONS_OBSERVED"
    sha_file = (protocol_mod.OUT_DIR / "frozen_domain_validation_protocol.sha256.txt").read_text(encoding="utf-8").strip()
    assert sha_file == result["protocol_sha256"]
