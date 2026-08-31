"""Tests for the benchmark-exact Consensus correction: item order
(1-3-2/2-3-1 only, item #3 always middle, stable across retries), the
four-stage interleaved chaining (no future feedback leakage, correct
feedback-before-next-estimate ordering), the per-stage bounded retry
ledger, and that old Consensus artifacts/phases are correctly disabled
while standard target-G data is untouched."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inference.consensus_benchmark_exact as ce  # noqa: E402
import inference.consensus_exact_retry_engine as re_engine  # noqa: E402
import survey_content as sc  # noqa: E402
from inference.together_batch import _profile_dict  # noqa: E402

G_MASTER_PATH = PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv"
pytestmark = pytest.mark.skipif(not G_MASTER_PATH.exists(), reason="G donor population not built in this environment")


@pytest.fixture(scope="module")
def donors():
    return pd.read_csv(G_MASTER_PATH).set_index("donor_key", drop=False)


@pytest.fixture(scope="module")
def items():
    return sc.load_items()


def _profile(donors, donor_key):
    return _profile_dict(donors.loc[donor_key])


def _chain(donors, items, donor_key, *, answers=(50, 60, 70), attempt_id=1):
    profile = _profile(donors, donor_key)
    r1 = ce.build_step1_prompt_render(profile, donor_key=donor_key, attempt_id=attempt_id)
    s1 = ce.step_record(r1, {"Q001": answers[0]}, donor_key=donor_key, attempt_id=attempt_id)
    r2 = ce.build_step2_prompt_render(profile, donor_key=donor_key, step1_record=s1, attempt_id=attempt_id)
    s2 = ce.step_record(r2, {"Q001": answers[1]}, donor_key=donor_key, attempt_id=attempt_id)
    r3 = ce.build_step3_prompt_render(profile, donor_key=donor_key, step1_record=s1, step2_record=s2, attempt_id=attempt_id)
    s3 = ce.step_record(r3, {"Q001": answers[2]}, donor_key=donor_key, attempt_id=attempt_id)
    r4 = ce.build_outcomes_prompt_render(profile, items, donor_key=donor_key, step1_record=s1, step2_record=s2, step3_record=s3, attempt_id=attempt_id)
    return r1, s1, r2, s2, r3, s3, r4


# 1 & 2. only 1-3-2 or 2-3-1 can occur; item #3 always second.
def test_only_legal_orders_occur_across_all_donors(donors):
    for donor_key in donors["donor_key"].head(200):
        order = ce.assign_consensus_exact_order(donor_key)
        assert order in ce.LEGAL_ORDERS
        assert order[1] == ce.MIDDLE_BLOCK_KEY


# 3. retry number never changes order.
def test_retry_attempt_id_never_changes_order(donors):
    donor_key = donors["donor_key"].iloc[0]
    orders = {ce.assign_consensus_exact_order(donor_key) for _ in range(5)}  # deterministic, but call repeatedly
    assert len(orders) == 1
    profile = _profile(donors, donor_key)
    r1a = ce.build_step1_prompt_render(profile, donor_key=donor_key, attempt_id=1)
    r1b = ce.build_step1_prompt_render(profile, donor_key=donor_key, attempt_id=2)
    assert r1a.provenance["order"] == r1b.provenance["order"]
    assert r1a.request_key != r1b.request_key


# 4/5/6 & 7. feedback ordering and no future leakage, verified across the
# full chain for both legal-order donors (one of each orientation).
@pytest.mark.parametrize("donor_index", [0, 1, 2, 3, 4, 5])
def test_feedback_ordering_and_no_future_leakage(donors, items, donor_index):
    donor_key = donors["donor_key"].iloc[donor_index]
    order = ce.assign_consensus_exact_order(donor_key)
    r1, s1, r2, s2, r3, s3, r4 = _chain(donors, items, donor_key)

    fb0 = sc.get_consensus_single_item_feedback_text(order[0])
    fb1 = sc.get_consensus_single_item_feedback_text(order[1])
    fb2 = sc.get_consensus_single_item_feedback_text(order[2])

    # STEP_1: no feedback at all.
    assert fb0 not in r1.user_prompt and fb1 not in r1.user_prompt and fb2 not in r1.user_prompt

    # STEP_2: item[0]'s feedback visible (appears after estimate 1); item[1]
    # (=item #3)'s own feedback NOT visible yet (not before its own estimate).
    assert fb0 in r2.user_prompt
    assert fb1 not in r2.user_prompt
    assert fb2 not in r2.user_prompt

    # STEP_3: item[1] (item #3)'s feedback now visible (after estimate 2,
    # before estimate 3); item[2]'s own feedback NOT visible yet.
    assert fb1 in r3.user_prompt
    assert fb2 not in r3.user_prompt

    # OUTCOMES: item[2]'s feedback (the final one) now visible, delivered
    # after estimate 3 and before the outcomes questionnaire.
    assert fb2 in r4.user_prompt


# 8. previously valid stage is not resampled after downstream failure --
# structural guarantee: build_step3_prompt_render/build_outcomes_prompt_render
# take the EARLIER steps' records as fixed INPUT; a downstream retry (new
# attempt_id for step3) reuses the identical step1_record/step2_record.
def test_downstream_retry_does_not_resample_upstream_stage(donors, items):
    donor_key = donors["donor_key"].iloc[0]
    r1, s1, r2, s2, r3a, s3a, _ = _chain(donors, items, donor_key)
    profile = _profile(donors, donor_key)
    # simulate a step3 retry (attempt_id=2) using the SAME locked s1/s2.
    r3b = ce.build_step3_prompt_render(profile, donor_key=donor_key, step1_record=s1, step2_record=s2, attempt_id=2)
    assert r3b.conversation_history == r3a.conversation_history  # step1+step2 content identical
    assert r3b.request_key != r3a.request_key  # fresh attempt, fresh seed


# 9, 10, 11. first-valid wins, max 3 attempts per stage, attempt 4 impossible.
def test_first_valid_wins_max_3_attempts_no_attempt_4():
    universe = {"d1"}
    r1 = {"attempt_number": 1, "donor_status": {"d1": re_engine.SCHEMA_INVALID}}
    r2 = {"attempt_number": 2, "donor_status": {"d1": re_engine.SCHEMA_VALID}}
    ledger = re_engine.build_stage_ledger(universe, [r1, r2])
    assert ledger["d1"]["resolved"] is True
    assert ledger["d1"]["resolved_attempt"] == 2
    assert re_engine.pending_donors(ledger) == []

    r3 = {"attempt_number": 2, "donor_status": {"d1": re_engine.SCHEMA_VALID}}  # would-be attempt 3 result, but donor already resolved
    all_invalid = re_engine.build_stage_ledger(universe, [{"attempt_number": 1, "donor_status": {"d1": re_engine.SCHEMA_INVALID}}, {"attempt_number": 2, "donor_status": {"d1": re_engine.PROVIDER_ERROR}}, {"attempt_number": 3, "donor_status": {"d1": re_engine.SCHEMA_INVALID}}])
    assert all_invalid["d1"]["attempt_count"] == 3
    assert re_engine.pending_donors(all_invalid) == []  # exhausted, not eligible for a 4th

    with pytest.raises(re_engine.ConsensusExactLedgerError, match="exceeds MAX_PRODUCTION_ATTEMPTS_PER_STAGE"):
        re_engine.build_stage_ledger(universe, [{"attempt_number": 1, "donor_status": {"d1": re_engine.SCHEMA_INVALID}}, {"attempt_number": 2, "donor_status": {"d1": re_engine.SCHEMA_INVALID}}, {"attempt_number": 3, "donor_status": {"d1": re_engine.SCHEMA_INVALID}}, {"attempt_number": 4, "donor_status": {"d1": re_engine.SCHEMA_INVALID}}])


# final row eligibility requires all four stages resolved.
def test_final_row_eligible_requires_all_four_stages_resolved():
    step1 = re_engine.build_stage_ledger({"d1", "d2"}, [{"attempt_number": 1, "donor_status": {"d1": re_engine.SCHEMA_VALID, "d2": re_engine.SCHEMA_VALID}}])
    step2 = re_engine.build_stage_ledger({"d1", "d2"}, [{"attempt_number": 1, "donor_status": {"d1": re_engine.SCHEMA_VALID, "d2": re_engine.SCHEMA_VALID}}])
    step3 = re_engine.build_stage_ledger({"d1", "d2"}, [{"attempt_number": 1, "donor_status": {"d1": re_engine.SCHEMA_VALID, "d2": re_engine.SCHEMA_INVALID}}])
    outcomes = re_engine.build_stage_ledger({"d1"}, [{"attempt_number": 1, "donor_status": {"d1": re_engine.SCHEMA_VALID}}])
    eligible = re_engine.final_row_eligible_donors(step1, step2, step3, outcomes)
    assert eligible == {"d1"}  # d2 never reached outcomes (step3 invalid)


# 12 & 13. old Consensus outputs cannot enter final assembly; old 82-request
# completion manifest cannot be submitted.
def test_legacy_consensus_completion_manifest_permanently_disabled():
    import inference.target_g_completion_guard as completion_guard

    assert "target_g_wave1_completion_consensus_a" not in completion_guard.PHASES
    with pytest.raises(completion_guard.TargetGCompletionNotAuthorized):
        completion_guard.declare_target_g_completion_phase("target_g_wave1_completion_consensus_a", state_path=Path("/tmp/unused_consensus_a_state.json"))


def test_legacy_consensus_marker_records_unused_status():
    import json

    marker_path = PIPELINE_ROOT / "outputs" / "target_production" / "legacy_consensus_outputs_unused.json"
    if not marker_path.exists():
        pytest.skip("legacy consensus marker not built in this environment")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["status"] == "SCIENTIFICALLY_UNUSED_FOR_FINAL_SUBMISSION"
    assert marker["may_never_enter_final_17000_row_dataset"] is True


# 14. standard target-G outputs remain untouched.
def test_standard_target_g_completion_guard_unaffected():
    import tempfile

    import inference.target_g_completion_guard as completion_guard

    assert "target_g_wave1_completion_standard" in completion_guard.PHASES
    tmp = Path(tempfile.mkdtemp())
    completion_guard.declare_target_g_completion_phase("target_g_wave1_completion_standard", state_path=tmp / "state.json")
    spec = completion_guard.PHASES["target_g_wave1_completion_standard"]
    result = completion_guard.target_g_completion_safety_guard(spec["jsonl_path"], phase="target_g_wave1_completion_standard", state_path=tmp / "state.json")
    assert result["submission_allowed"] is True
    assert result["request_count"] == 1401
