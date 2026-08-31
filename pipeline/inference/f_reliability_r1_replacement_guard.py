"""Partition-validity guard for the replacement F* R1 submission.

Layered ON TOP of (never instead of) inference.scientific_bakeoff_guard's
existing phase-declaration/allowlist/cost-cap/duplicate machinery, which is
reused unmodified for the single declared phase f_reliability_r1_replacement
(exact 24,000-id canonical allowlist).

Frozen operational plan (per the fan-out amendment, made before any
replacement-R1 output was observed): part1 alone is a serving-qualification
gate. Parts 2/3/4 are permanently blocked until part1 has a RECORDED PASS
against the reused (not reinvented) 0.005 invalid-rate threshold; once part1
passes, parts 2/3/4 become eligible for submission in any order or
concurrently -- they do NOT depend on each other, only on part1. A part1
FAIL blocks parts 2/3/4 permanently for this replacement attempt (no
retries). "Serving validity" here means ONLY reconciliation counts and the
invalid-response rate -- never ATE/effect/correlation/sign statistics,
which record_partition_serving_validity's own signature makes impossible
to pass in by construction.

State machine:
    STATE_0        -- nothing recorded; only part1 may be submitted.
    STATE_1        -- part1 submitted, not yet retrieved/scored; parts 2/3/4
                       still forbidden.
    STATE_2_PASS   -- part1 recorded PASS; parts 2/3/4 all eligible (any
                       order / concurrently).
    STATE_2_FAIL   -- part1 recorded FAIL; parts 2/3/4 permanently blocked
                       for this replacement attempt.

Each partition may be submitted (recorded) at most once. No new numeric
threshold is invented -- the 0.005 invalid-rate gate is imported from the
existing R1 module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ate.r_f_decision import R_F_PASS_THRESHOLDS

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPLACEMENT_R1_ROOT = PIPELINE_ROOT / "outputs" / "scientific_bakeoff" / "f_reliability_r1_replacement"
PARTITION_VALIDITY_STATE_PATH = REPLACEMENT_R1_ROOT / "partition_validity_state.json"
GATE_PARTITION = "part1"
FANOUT_PARTITIONS = ("part2", "part3", "part4")
ALL_PARTITIONS = (GATE_PARTITION, *FANOUT_PARTITIONS)
INVALID_RATE_THRESHOLD = R_F_PASS_THRESHOLDS["max_invalid_response_rate"]  # reused, not reinvented -- 0.005


class PartitionSequenceError(RuntimeError):
    """A partition was attempted before the gate partition's recorded PASS,
    or was resubmitted after already being recorded."""


def _load_state(state_path: Path = PARTITION_VALIDITY_STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {"schema_version": "f_reliability_r1_replacement_partition_validity_v1", "partitions": {}}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "f_reliability_r1_replacement_partition_validity_v1":
        raise RuntimeError(f"ambiguous existing partition-validity state: {state_path}")
    return state


def guard_state(state_path: Path = PARTITION_VALIDITY_STATE_PATH) -> str:
    """Returns one of STATE_0 / STATE_1 / STATE_2_PASS / STATE_2_FAIL."""
    recorded = _load_state(state_path).get("partitions", {})
    if GATE_PARTITION not in recorded:
        return "STATE_0"
    return "STATE_2_PASS" if recorded[GATE_PARTITION]["status"] == "PASS" else "STATE_2_FAIL"


def assert_partition_ready_to_submit(partition: str, *, state_path: Path = PARTITION_VALIDITY_STATE_PATH) -> None:
    """Raises PartitionSequenceError unless `partition` is eligible right
    now: part1 is eligible iff not already recorded; any of parts 2/3/4 is
    eligible iff part1 is recorded PASS and that partition itself has not
    already been recorded -- independent of whether its siblings have been
    submitted or recorded."""
    if partition not in ALL_PARTITIONS:
        raise PartitionSequenceError(f"unknown partition {partition!r}; must be one of {ALL_PARTITIONS}")
    state = _load_state(state_path)
    recorded = state.get("partitions", {})
    if partition in recorded:
        raise PartitionSequenceError(f"{partition} already has a recorded outcome ({recorded[partition]['status']!r}) -- resubmission is refused")
    if partition == GATE_PARTITION:
        return
    gate_entry = recorded.get(GATE_PARTITION)
    if gate_entry is None:
        raise PartitionSequenceError(f"{partition} cannot be submitted before {GATE_PARTITION} has been submitted and scored")
    if gate_entry["status"] != "PASS":
        raise PartitionSequenceError(f"{partition} cannot be submitted: {GATE_PARTITION} is recorded as {gate_entry['status']!r}, not PASS -- parts 2/3/4 are permanently blocked for this replacement attempt")


def record_partition_serving_validity(
    partition: str,
    *,
    expected_requests: int,
    missing_ids: int,
    unexpected_ids: int,
    duplicate_ids: int,
    invalid_rate: float,
    system_fingerprint_distribution: dict[str, int] | None = None,
    state_path: Path = PARTITION_VALIDITY_STATE_PATH,
) -> dict[str, Any]:
    """Record ONLY serving-validity facts for one retrieved partition --
    reconciliation counts, invalid rate, and (for root-cause continuity)
    the observed system_fingerprint distribution. Deliberately accepts NO
    ATE/effect/correlation/sign-agreement argument: this function's
    signature itself enforces that no scientific-performance statistic can
    be recorded here, matching the instruction that operational partition
    checks must never inspect treatment-effect results.

    status is PASS iff reconciliation is exact (no missing/unexpected/
    duplicate ids) AND invalid_rate <= INVALID_RATE_THRESHOLD (the same
    0.005 already frozen for R1 -- not a new number)."""
    if partition not in ALL_PARTITIONS:
        raise PartitionSequenceError(f"unknown partition {partition!r}")
    reconciliation_ok = missing_ids == 0 and unexpected_ids == 0 and duplicate_ids == 0
    passed = reconciliation_ok and invalid_rate <= INVALID_RATE_THRESHOLD
    state = _load_state(state_path)
    partitions = state.setdefault("partitions", {})
    if partition in partitions:
        raise PartitionSequenceError(f"{partition} already has a recorded outcome -- this function may be called at most once per partition")
    partitions[partition] = {
        "status": "PASS" if passed else "FAIL",
        "expected_requests": expected_requests,
        "missing_ids": missing_ids,
        "unexpected_ids": unexpected_ids,
        "duplicate_ids": duplicate_ids,
        "invalid_rate": invalid_rate,
        "invalid_rate_threshold": INVALID_RATE_THRESHOLD,
        "system_fingerprint_distribution": system_fingerprint_distribution or {},
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return partitions[partition]
