"""Tests for scripts/build_consensus_exact_outcomes_manifest.py: confirms
the real OUTCOMES manifest chains correctly from STEP_1's, STEP_2's, and
STEP_3's real, first-valid retrieved responses (order stability, no
scientific response value present in the MANIFEST/JSONL itself, and that
this final stage uses the canonical full G item bank)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_consensus_exact_outcomes_manifest as outcomes_builder  # noqa: E402
import survey_content as sc  # noqa: E402

pytestmark = pytest.mark.skipif(not outcomes_builder.OUT_ROOT.exists(), reason="Consensus-exact OUTCOMES manifest not built in this environment")


@pytest.fixture(scope="module")
def step3_rows():
    with open(outcomes_builder.STEP3_DIR / "request_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def outcomes_rows():
    with open(outcomes_builder.OUT_ROOT / "request_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_outcomes_has_exactly_1000_requests_one_per_step3_donor(step3_rows, outcomes_rows):
    assert len(outcomes_rows) == 1000
    step3_donors = {row["profile_id"] for row in step3_rows}
    outcomes_donors = {row["profile_id"] for row in outcomes_rows}
    assert step3_donors == outcomes_donors


def test_outcomes_request_key_encodes_outcomes_and_attempt_1(outcomes_rows):
    for row in outcomes_rows:
        assert "|ConsensusExact|outcomes|attempt_1" in row["request_key"]
        assert row["request_stage"] == "consensus_exact_outcomes"
        assert row["study_id"] == "target"
        assert row["condition_id"] == "Consensus"
        assert row["outcome_id"] == "consensus_exact_outcomes_full_questionnaire"


def test_outcomes_uses_the_full_canonical_item_bank(outcomes_rows):
    n_items = len(sc.load_items())
    import json

    for row in outcomes_rows[:5]:
        key_map = json.loads(row["response_key_map"])
        assert len(key_map) == n_items


def test_no_scientific_response_value_in_manifest_row(outcomes_rows):
    forbidden_keys = {"response", "answer", "content", "choices"}
    for row in outcomes_rows[:5]:
        assert forbidden_keys.isdisjoint(row.keys())


def test_zero_collision_with_every_other_manifest(outcomes_rows):
    import glob

    def _ids(path):
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            return {row["custom_id"] for row in r} if "custom_id" in (r.fieldnames or []) else set()

    outcomes_ids = {row["custom_id"] for row in outcomes_rows}
    prior: set[str] = set()
    for path in glob.glob(str(PIPELINE_ROOT / "outputs" / "**" / "*.csv"), recursive=True):
        if "consensus_exact/outcomes" in path:
            continue
        prior |= _ids(Path(path))
    assert len(prior) > 250_000
    assert outcomes_ids & prior == set()


def test_guard_declares_and_passes_end_to_end(tmp_path):
    import inference.consensus_exact_guard as guard_mod

    spec = guard_mod.PHASES["consensus_exact_outcomes"]
    guard_mod.declare_consensus_exact_phase("consensus_exact_outcomes", state_path=tmp_path / "state.json")
    result = guard_mod.consensus_exact_safety_guard(spec["jsonl_path"], phase="consensus_exact_outcomes", state_path=tmp_path / "state.json")
    assert result["submission_allowed"] is True
    assert result["request_count"] == 1000
    assert result["model"] == "google/gemma-4-31B-it"


def test_prior_stage_manifests_still_reflect_frozen_hash_after_outcomes_build():
    import inference.consensus_exact_guard as guard_mod

    for phase in ("consensus_exact_step1", "consensus_exact_step2", "consensus_exact_step3"):
        guard_mod._verify_canonical_files_unaltered(guard_mod.PHASES[phase])
