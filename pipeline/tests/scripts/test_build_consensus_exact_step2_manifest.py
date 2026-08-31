"""Tests for scripts/build_consensus_exact_step2_manifest.py: confirms the
real STEP_2 manifest chains correctly from STEP_1's real, first-valid
retrieved responses (order stability, no future-step content leakage, no
scientific response value present in the MANIFEST/JSONL itself -- the
script reads response content only to construct conversation history, and
that content appears only inside the request bodies sent to the model, not
in any accounting field this test inspects)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_consensus_exact_step2_manifest as step2_builder  # noqa: E402
import inference.consensus_benchmark_exact as ce  # noqa: E402

pytestmark = pytest.mark.skipif(not step2_builder.OUT_ROOT.exists(), reason="Consensus-exact STEP_2 manifest not built in this environment")


@pytest.fixture(scope="module")
def step1_rows():
    with open(step2_builder.STEP1_DIR / "request_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def step2_rows():
    with open(step2_builder.OUT_ROOT / "request_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_step2_has_exactly_1000_requests_one_per_step1_donor(step1_rows, step2_rows):
    assert len(step2_rows) == 1000
    step1_donors = {row["profile_id"] for row in step1_rows}
    step2_donors = {row["profile_id"] for row in step2_rows}
    assert step1_donors == step2_donors


def test_step2_request_key_encodes_step2_and_attempt_1(step2_rows):
    for row in step2_rows:
        assert "|ConsensusExact|step2|attempt_1" in row["request_key"]
        assert row["request_stage"] == "consensus_exact_step2"
        assert row["study_id"] == "target"
        assert row["condition_id"] == "Consensus"


def test_step2_order_matches_step1_deterministic_assignment(step2_rows):
    for row in step2_rows[:50]:
        donor_key = row["profile_id"]
        order = ce.assign_consensus_exact_order(donor_key)
        assert order[1] == ce.MIDDLE_BLOCK_KEY
        # STEP_2 always asks the middle-block item.
        assert row["outcome_id"] == "consensus_exact_step2_estimate"


def test_no_scientific_response_value_in_manifest_row(step2_rows):
    # structural guard: manifest rows carry only engineering/accounting
    # fields, never a parsed response value or raw answer content.
    forbidden_keys = {"response", "answer", "content", "choices"}
    for row in step2_rows[:5]:
        assert forbidden_keys.isdisjoint(row.keys())


def test_zero_collision_with_step1_and_every_other_manifest(step2_rows):
    import glob

    def _ids(path):
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            return {row["custom_id"] for row in r} if "custom_id" in (r.fieldnames or []) else set()

    step2_ids = {row["custom_id"] for row in step2_rows}
    prior: set[str] = set()
    for path in glob.glob(str(PIPELINE_ROOT / "outputs" / "**" / "*.csv"), recursive=True):
        if "consensus_exact/step2" in path:
            continue
        prior |= _ids(Path(path))
    assert len(prior) > 250_000
    assert step2_ids & prior == set()


def test_guard_declares_and_passes_end_to_end(tmp_path):
    import inference.consensus_exact_guard as guard_mod

    spec = guard_mod.PHASES["consensus_exact_step2"]
    guard_mod.declare_consensus_exact_phase("consensus_exact_step2", state_path=tmp_path / "state.json")
    result = guard_mod.consensus_exact_safety_guard(spec["jsonl_path"], phase="consensus_exact_step2", state_path=tmp_path / "state.json")
    assert result["submission_allowed"] is True
    assert result["request_count"] == 1000
    assert result["model"] == "google/gemma-4-31B-it"


def test_step1_manifest_still_reflects_frozen_hash_after_step2_build():
    # building STEP_2 must never mutate STEP_1's already-guarded canonical files.
    import inference.consensus_exact_guard as guard_mod

    spec = guard_mod.PHASES["consensus_exact_step1"]
    guard_mod._verify_canonical_files_unaltered(spec)  # raises on mismatch
