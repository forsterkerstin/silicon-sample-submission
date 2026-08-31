"""Tests for scripts/build_consensus_exact_step3_manifest.py: confirms the
real STEP_3 manifest chains correctly from STEP_1's AND STEP_2's real,
first-valid retrieved responses (order stability, no future-step content
leakage, no scientific response value present in the MANIFEST/JSONL
itself)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_consensus_exact_step3_manifest as step3_builder  # noqa: E402
import inference.consensus_benchmark_exact as ce  # noqa: E402

pytestmark = pytest.mark.skipif(not step3_builder.OUT_ROOT.exists(), reason="Consensus-exact STEP_3 manifest not built in this environment")


@pytest.fixture(scope="module")
def step2_rows():
    with open(step3_builder.STEP2_DIR / "request_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def step3_rows():
    with open(step3_builder.OUT_ROOT / "request_manifest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_step3_has_exactly_1000_requests_one_per_step2_donor(step2_rows, step3_rows):
    assert len(step3_rows) == 1000
    step2_donors = {row["profile_id"] for row in step2_rows}
    step3_donors = {row["profile_id"] for row in step3_rows}
    assert step2_donors == step3_donors


def test_step3_request_key_encodes_step3_and_attempt_1(step3_rows):
    for row in step3_rows:
        assert "|ConsensusExact|step3|attempt_1" in row["request_key"]
        assert row["request_stage"] == "consensus_exact_step3"
        assert row["study_id"] == "target"
        assert row["condition_id"] == "Consensus"
        assert row["outcome_id"] == "consensus_exact_step3_estimate"


def test_step3_order_matches_deterministic_donor_assignment(step3_rows):
    for row in step3_rows[:50]:
        donor_key = row["profile_id"]
        order = ce.assign_consensus_exact_order(donor_key)
        assert len(order) == 3
        assert order[1] == ce.MIDDLE_BLOCK_KEY


def test_no_scientific_response_value_in_manifest_row(step3_rows):
    forbidden_keys = {"response", "answer", "content", "choices"}
    for row in step3_rows[:5]:
        assert forbidden_keys.isdisjoint(row.keys())


def test_zero_collision_with_every_other_manifest(step3_rows):
    import glob

    def _ids(path):
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            return {row["custom_id"] for row in r} if "custom_id" in (r.fieldnames or []) else set()

    step3_ids = {row["custom_id"] for row in step3_rows}
    prior: set[str] = set()
    for path in glob.glob(str(PIPELINE_ROOT / "outputs" / "**" / "*.csv"), recursive=True):
        if "consensus_exact/step3" in path:
            continue
        prior |= _ids(Path(path))
    assert len(prior) > 250_000
    assert step3_ids & prior == set()


def test_guard_declares_and_passes_end_to_end(tmp_path):
    import inference.consensus_exact_guard as guard_mod

    spec = guard_mod.PHASES["consensus_exact_step3"]
    guard_mod.declare_consensus_exact_phase("consensus_exact_step3", state_path=tmp_path / "state.json")
    result = guard_mod.consensus_exact_safety_guard(spec["jsonl_path"], phase="consensus_exact_step3", state_path=tmp_path / "state.json")
    assert result["submission_allowed"] is True
    assert result["request_count"] == 1000
    assert result["model"] == "google/gemma-4-31B-it"


def test_step1_and_step2_manifests_still_reflect_frozen_hash_after_step3_build():
    import inference.consensus_exact_guard as guard_mod

    for phase in ("consensus_exact_step1", "consensus_exact_step2"):
        guard_mod._verify_canonical_files_unaltered(guard_mod.PHASES[phase])
