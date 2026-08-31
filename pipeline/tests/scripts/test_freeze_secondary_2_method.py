"""Tests for the Secondary-2 MCONST_GSHAPE prospective freeze script.
Confirms the frozen artifact records the exact required fields/provenance,
matches independently-verified hashes (Secondary-1 artifact, frozen CES
source), and that freezing never touches any primary/Secondary-1/target
artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from secondary_calibration_diagnostic import PRIMARY_ARTIFACT_PATH  # noqa: E402

import freeze_secondary_2_method as freeze_mod  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_primary_and_secondary_1_artifacts_untouched():
    primary_before = PRIMARY_ARTIFACT_PATH.read_bytes()
    s1_before = freeze_mod.SECONDARY_1_ARTIFACT_PATH.read_bytes()
    freeze_mod.main()
    assert PRIMARY_ARTIFACT_PATH.read_bytes() == primary_before
    assert freeze_mod.SECONDARY_1_ARTIFACT_PATH.read_bytes() == s1_before


def test_ces_source_sha256_matches_config_frozen_amendment():
    cfg = yaml.safe_load(freeze_mod.POPULATION_CONFIG_PATH.read_text(encoding="utf-8"))
    expected = None
    for amendment in cfg["population_source_amendments"]:
        if amendment.get("amendment_type") == "TARGET_DONOR_SOURCE_ACS_PUMS_TO_CES_2024_FROZEN":
            expected = amendment["frozen_roster"]["sha256"]
    assert expected == "73cb40367efd0f92b8593cb9555660f68bdc95d8c1aad38f41aa13ab089ca43d"
    result = freeze_mod.main()
    assert result["ces_source_sha256"] == expected


def test_secondary_1_sha256_matches_the_committed_secondary_1_artifact():
    result = freeze_mod.main()
    assert result["secondary_1_artifact_sha256"] == _sha256(freeze_mod.SECONDARY_1_ARTIFACT_PATH)
    assert result["secondary_1_artifact_sha256"] == "c0391906784f54c17f1503fccc91963f22af85dd81dc090bdc69ec38e75684e0"


def test_frozen_artifact_has_required_disclosure_and_equation_fields():
    result = freeze_mod.main()
    assert result["method_name"] == "MCONST_GSHAPE"
    assert result["status"] == "PROSPECTIVELY_FROZEN_BEFORE_TARGET_G_OUTPUT_INSPECTION"
    assert result["post_primary_external_development"] is True
    assert result["target_human_outcome_blind"] is True
    assert result["target_g_outputs_inspected_before_freeze"] is False
    assert result["external_mu"] == 1.9558595458395387
    assert result["gamma_g_shape"] == 1.0
    assert result["centering_scope"] == "all_208_normalized_target_ATE_cells"
    assert result["cell_weighting"] == "equal"
    assert result["target_f_dependence"] is False
    assert result["target_g_dependence"] is True
    assert result["g_star_model"] == "google/gemma-4-31B-it"
    assert result["not_executed_in_this_freeze"] is True


def test_artifact_written_only_under_secondary_diagnostic_dir_and_self_hash_matches():
    result = freeze_mod.main()
    artifact_path = freeze_mod.OUT_DIR / "secondary_2_mconst_gshape_method.json"
    assert artifact_path.exists()
    on_disk = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert on_disk["method_name"] == "MCONST_GSHAPE"
    sha_file = (freeze_mod.OUT_DIR / "secondary_2_mconst_gshape_method.sha256.txt").read_text(encoding="utf-8").strip()
    assert sha_file == result["secondary_2_artifact_sha256"]
    assert freeze_mod.OUT_DIR.name == "secondary_calibration_diagnostic"
