"""Tests for the replacement F* R1 partition-validity guard's fan-out state
machine (part1 gate -> parts 2/3/4 concurrent). No submission occurs; these
are pure state-machine tests against a synthetic tmp_path state file."""

from __future__ import annotations

import pytest

from inference.f_reliability_r1_replacement_guard import (
    ALL_PARTITIONS,
    FANOUT_PARTITIONS,
    GATE_PARTITION,
    INVALID_RATE_THRESHOLD,
    PartitionSequenceError,
    assert_partition_ready_to_submit,
    guard_state,
    record_partition_serving_validity,
)


def test_threshold_reused_not_reinvented():
    assert INVALID_RATE_THRESHOLD == 0.005


def test_partition_names():
    assert GATE_PARTITION == "part1"
    assert FANOUT_PARTITIONS == ("part2", "part3", "part4")
    assert ALL_PARTITIONS == ("part1", "part2", "part3", "part4")


# ---- STATE_0: only part1 permitted ----


def test_state0_only_part1_eligible(tmp_path):
    sp = tmp_path / "state.json"
    assert guard_state(state_path=sp) == "STATE_0"
    assert_partition_ready_to_submit("part1", state_path=sp)  # does not raise
    for p in FANOUT_PARTITIONS:
        with pytest.raises(PartitionSequenceError, match="cannot be submitted before part1"):
            assert_partition_ready_to_submit(p, state_path=sp)


# ---- STATE_2_PASS: fan-out, any order, concurrent ----


def test_state2_pass_all_three_fanout_partitions_independently_eligible(tmp_path):
    sp = tmp_path / "state.json"
    record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.001, state_path=sp)
    assert guard_state(state_path=sp) == "STATE_2_PASS"
    # order-independence: check part3 before part2 before part4 -- none depends on the others
    assert_partition_ready_to_submit("part3", state_path=sp)
    assert_partition_ready_to_submit("part2", state_path=sp)
    assert_partition_ready_to_submit("part4", state_path=sp)


def test_state2_pass_part3_eligible_without_part2_ever_being_recorded(tmp_path):
    """The key fan-out property: part3 does not wait on part2's outcome at
    all, only on part1."""
    sp = tmp_path / "state.json"
    record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)
    assert_partition_ready_to_submit("part3", state_path=sp)  # does not raise, part2 was never touched


def test_state2_pass_recording_part4_does_not_require_part2_or_part3_recorded(tmp_path):
    sp = tmp_path / "state.json"
    record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)
    record_partition_serving_validity("part4", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.003, state_path=sp)
    # part2/part3 remain independently eligible -- recording part4 doesn't gate them
    assert_partition_ready_to_submit("part2", state_path=sp)
    assert_partition_ready_to_submit("part3", state_path=sp)


# ---- STATE_2_FAIL: parts 2/3/4 permanently blocked ----


def test_state2_fail_blocks_all_three_fanout_partitions(tmp_path):
    sp = tmp_path / "state.json"
    record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.5, state_path=sp)
    assert guard_state(state_path=sp) == "STATE_2_FAIL"
    for p in FANOUT_PARTITIONS:
        with pytest.raises(PartitionSequenceError, match="not 'PASS'|permanently blocked"):
            assert_partition_ready_to_submit(p, state_path=sp)


# ---- threshold edge cases ----


def test_invalid_rate_above_threshold_recorded_as_fail(tmp_path):
    sp = tmp_path / "state.json"
    entry = record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.006, state_path=sp)
    assert entry["status"] == "FAIL"


def test_invalid_rate_at_exact_threshold_passes(tmp_path):
    sp = tmp_path / "state.json"
    entry = record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.005, state_path=sp)
    assert entry["status"] == "PASS"


def test_reconciliation_failure_fails_even_with_zero_invalid_rate(tmp_path):
    sp = tmp_path / "state.json"
    entry = record_partition_serving_validity("part1", expected_requests=6000, missing_ids=1, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)
    assert entry["status"] == "FAIL"


# ---- resubmission / double-recording protection ----


def test_resubmitting_already_recorded_partition_blocked(tmp_path):
    sp = tmp_path / "state.json"
    record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)
    with pytest.raises(PartitionSequenceError):
        assert_partition_ready_to_submit("part1", state_path=sp)


def test_recording_same_partition_twice_blocked(tmp_path):
    sp = tmp_path / "state.json"
    record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)
    with pytest.raises(PartitionSequenceError):
        record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)


def test_each_fanout_partition_recordable_at_most_once(tmp_path):
    sp = tmp_path / "state.json"
    record_partition_serving_validity("part1", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)
    record_partition_serving_validity("part2", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)
    with pytest.raises(PartitionSequenceError):
        assert_partition_ready_to_submit("part2", state_path=sp)
    with pytest.raises(PartitionSequenceError):
        record_partition_serving_validity("part2", expected_requests=6000, missing_ids=0, unexpected_ids=0, duplicate_ids=0, invalid_rate=0.0, state_path=sp)


def test_unknown_partition_name_rejected(tmp_path):
    with pytest.raises(PartitionSequenceError):
        assert_partition_ready_to_submit("part5", state_path=tmp_path / "state.json")


def test_record_signature_has_no_ate_or_scientific_performance_argument():
    """The function signature itself must make it impossible to record an
    ATE/effect/correlation/sign-agreement value here -- serving-validity
    recording and scientific scoring are structurally separate paths."""
    import inspect

    params = set(inspect.signature(record_partition_serving_validity).parameters)
    forbidden = {"ate", "rmse", "correlation", "pearson", "spearman", "sign_agreement", "effect", "draw3_ate_pp", "draw4_ate_pp", "draw5_ate_pp", "draw6_ate_pp"}
    assert not (params & forbidden)
