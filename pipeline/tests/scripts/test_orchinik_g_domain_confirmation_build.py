"""Tests for the Orchinik G-vs-DeepSeek domain-confirmation manifest builder
and protocol freeze. Confirms real manifests (2,545 requests per model,
zero collision with every other manifest in the repo) and that freezing
never touches the prior (superseded-role) protocol artifact or any
primary/secondary calibration artifact."""

from __future__ import annotations

import csv
import glob
import hashlib
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_orchinik_g_domain_confirmation_manifest as build_mod  # noqa: E402
import freeze_orchinik_g_domain_confirmation_protocol as freeze_mod  # noqa: E402

pytestmark = pytest.mark.skipif(not build_mod.OUT_ROOT.exists(), reason="Orchinik domain-confirmation manifests not built in this environment")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ids(path: Path) -> set[str]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return {row["custom_id"] for row in r} if "custom_id" in (r.fieldnames or []) else set()


def test_real_manifests_have_2545_requests_each():
    for model in build_mod.MODELS:
        dir_name = model.replace("/", "_")
        ids = _load_ids(build_mod.OUT_ROOT / dir_name / "request_manifest.csv")
        assert len(ids) == 2545


def test_zero_collision_with_every_other_manifest_in_outputs():
    gemma_ids = _load_ids(build_mod.OUT_ROOT / "google_gemma-4-31B-it" / "request_manifest.csv")
    deepseek_ids = _load_ids(build_mod.OUT_ROOT / "deepseek-ai_DeepSeek-V4-Pro-0813" / "request_manifest.csv")
    assert gemma_ids & deepseek_ids == set()

    prior: set[str] = set()
    for path in glob.glob(str(PIPELINE_ROOT / "outputs" / "**" / "*.csv"), recursive=True):
        if "orchinik_g_domain_confirmation" in path:
            continue
        prior |= _load_ids(Path(path))
    assert len(prior) > 250_000  # threshold lowered after submission-cleanup removed several deprecated development manifests; the collision invariant below is unaffected
    assert gemma_ids & prior == set()
    assert deepseek_ids & prior == set()


def test_protocol_freeze_preserves_prior_artifact_and_all_primary_secondary_artifacts(tmp_path):
    prior_path = freeze_mod.PRIOR_PROTOCOL_PATH
    before = prior_path.read_bytes()
    primary_path = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
    primary_before = primary_path.read_bytes()
    frozen_before = (freeze_mod.OUT_DIR / "frozen_orchinik_g_domain_confirmation.json").read_bytes()
    freeze_mod.main(out_dir=tmp_path)
    assert prior_path.read_bytes() == before
    assert primary_path.read_bytes() == primary_before
    # freezing (even re-running the freeze logic) must never overwrite the
    # real committed frozen artifact -- --out-dir-equivalent redirection to
    # tmp_path is what prevents that; this is the same test-hygiene class of
    # bug already fixed once this session in validate_g_v2_smoke.py.
    assert (freeze_mod.OUT_DIR / "frozen_orchinik_g_domain_confirmation.json").read_bytes() == frozen_before


def test_protocol_records_superseded_role_and_governance_flags(tmp_path):
    result = freeze_mod.main(out_dir=tmp_path)
    assert result["prior_role_relationship"]["prior_role"] == "ORCHINIK_GAMMA_VALIDATION"
    assert result["prior_role_relationship"]["prior_role_status"] == "SUPERSEDED_FOR_CURRENT_METHOD_DECISION"
    assert result["prior_role_relationship"]["prior_artifacts_preserved"] is True
    assert result["orchinik_used_to_estimate_mu_external"] is False
    assert result["orchinik_used_to_estimate_gamma_g"] is False
    assert result["orchinik_used_to_select_s1_vs_s2"] is False
    assert result["current_s2_unchanged"] == {"mu_external": 1.9558595458395387, "gamma_g": 1.0}
    assert result["eligible_donors"] == 2545
    assert result["primary_cells"] == 75


def test_protocol_self_hash_matches_written_file(tmp_path):
    result = freeze_mod.main(out_dir=tmp_path)
    out_path = tmp_path / "frozen_orchinik_g_domain_confirmation.json"
    sha_file = (tmp_path / "frozen_orchinik_g_domain_confirmation.sha256.txt").read_text(encoding="utf-8").strip()
    assert sha_file == result["protocol_sha256"]
    assert _sha256(out_path) == sha_file
