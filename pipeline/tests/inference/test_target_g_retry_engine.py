"""Tests for the bounded 3-attempt target G Wave-1 production retry engine.

Most tests (1-9, 13, 19) construct SYNTHETIC ledger dicts directly and
exercise identities_pending_next_attempt/assemble_first_valid as pure
functions -- no I/O, no dependency on real retrieved batch files. A single
module-scoped fixture computes the REAL attempt-1 ledger once (expensive:
reads ~24MB of retrieved output across 6 parts) and is reused by the tests
that need real-data reconciliation (14, 15) and the frozen-artifact
mutation check (20)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inference.target_g_retry_engine as e  # noqa: E402

WAVE1_V2_BY_STAGE_STANDARD = e.WAVE1_V2_BY_STAGE_ROOT / "standard" / "request_manifest.csv"

pytestmark = pytest.mark.skipif(not WAVE1_V2_BY_STAGE_STANDARD.exists(), reason="target G Wave-1 v2 replacement manifests not built in this environment")


def _entry(*, request_stage="standard", attempts, resolved, resolved_attempt, attempt_count, next_attempt_number):
    return {
        "request_stage": request_stage,
        "profile_id": "LP0001",
        "condition_id": "control",
        "v2_request_key": "G|LP0001|control|replicate_1",
        "v2_custom_id": "G-fake",
        "attempts": attempts,
        "resolved": resolved,
        "resolved_attempt": resolved_attempt,
        "attempt_count": attempt_count,
        "next_attempt_number": next_attempt_number,
    }


def _attempt(n, status, cid=None):
    return {"attempt_number": n, "custom_id": cid or f"cid-attempt-{n}", "status": status, "system_fingerprint": None, "provider_batch_source": f"synthetic/attempt_{n}"}


# 1. attempt-1 valid -> never retry.
def test_attempt1_valid_never_retried():
    ledger = {"id1": _entry(attempts=[_attempt(1, e.SCHEMA_VALID)], resolved=True, resolved_attempt=1, attempt_count=1, next_attempt_number=None)}
    pending = e.identities_pending_next_attempt(ledger)
    assert pending["standard"] == []


# 2. attempt-1 invalid -> attempt-2 eligible.
def test_attempt1_invalid_attempt2_eligible():
    ledger = {"id1": _entry(attempts=[_attempt(1, e.SCHEMA_INVALID)], resolved=False, resolved_attempt=None, attempt_count=1, next_attempt_number=2)}
    pending = e.identities_pending_next_attempt(ledger)
    assert pending["standard"] == ["id1"]
    assert ledger["id1"]["next_attempt_number"] == 2


# 3. attempt-1 invalid + attempt-2 valid -> attempt-2 locked (never retried again).
def test_attempt2_valid_locks():
    ledger = {"id1": _entry(attempts=[_attempt(1, e.SCHEMA_INVALID), _attempt(2, e.SCHEMA_VALID)], resolved=True, resolved_attempt=2, attempt_count=2, next_attempt_number=None)}
    pending = e.identities_pending_next_attempt(ledger)
    assert pending["standard"] == []
    provenance = e.assemble_first_valid(ledger)
    assert provenance["id1"]["resolved"] is True
    assert provenance["id1"]["selected_attempt"] == 2


# 4. attempts 1 and 2 invalid -> attempt-3 eligible.
def test_attempts_1_2_invalid_attempt3_eligible():
    ledger = {"id1": _entry(attempts=[_attempt(1, e.SCHEMA_INVALID), _attempt(2, e.PROVIDER_ERROR)], resolved=False, resolved_attempt=None, attempt_count=2, next_attempt_number=3)}
    pending = e.identities_pending_next_attempt(ledger)
    assert pending["standard"] == ["id1"]
    assert ledger["id1"]["next_attempt_number"] == 3


# 5. attempt-3 valid -> selected.
def test_attempt3_valid_selected():
    ledger = {"id1": _entry(attempts=[_attempt(1, e.SCHEMA_INVALID), _attempt(2, e.PROVIDER_ERROR), _attempt(3, e.SCHEMA_VALID)], resolved=True, resolved_attempt=3, attempt_count=3, next_attempt_number=None)}
    provenance = e.assemble_first_valid(ledger)
    assert provenance["id1"]["selected_attempt"] == 3


# 6. all three invalid -> unresolved STOP (no attempt 4 eligibility).
def test_all_three_invalid_stops():
    ledger = {"id1": _entry(attempts=[_attempt(1, e.SCHEMA_INVALID), _attempt(2, e.PROVIDER_ERROR), _attempt(3, e.SCHEMA_INVALID)], resolved=False, resolved_attempt=None, attempt_count=3, next_attempt_number=None)}
    pending = e.identities_pending_next_attempt(ledger)
    assert pending["standard"] == []
    provenance = e.assemble_first_valid(ledger)
    assert provenance["id1"]["resolved"] is False


# 7. attempt 4 cannot be generated -- build_attempt_ledger refuses a 4th round.
def test_attempt_4_refused_by_ledger_builder(monkeypatch):
    universe = {"id1": {"request_stage": "standard", "profile_id": "LP0001", "condition_id": "control", "v2_request_key": "k", "v2_custom_id": "c"}}
    monkeypatch.setattr(e, "load_intended_universe", lambda: universe)
    monkeypatch.setattr(e, "classify_attempt1_responses", lambda: {"cid-1": {"identity": "id1", "stage": "standard", "part": "part1", "status": e.SCHEMA_INVALID, "system_fingerprint": None}})
    monkeypatch.setattr(e, "verify_attempt1_classification_matches_committed_report", lambda classification: None)
    round2 = {"attempt_number": 2, "custom_id_to_identity": {"cid-2": "id1"}, "custom_id_status": {"cid-2": e.SCHEMA_INVALID}}
    round3 = {"attempt_number": 3, "custom_id_to_identity": {"cid-3": "id1"}, "custom_id_status": {"cid-3": e.SCHEMA_INVALID}}
    round4 = {"attempt_number": 4, "custom_id_to_identity": {"cid-4": "id1"}, "custom_id_status": {"cid-4": e.SCHEMA_INVALID}}
    with pytest.raises(RuntimeError, match="exceeds MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY"):
        e.build_attempt_ledger([round2, round3, round4])


# 8. a later valid response cannot replace an earlier valid one -- more than
# one valid attempt is a provenance-ambiguity error, never silently resolved
# by picking the latest.
def test_later_valid_cannot_replace_earlier_valid(monkeypatch):
    universe = {"id1": {"request_stage": "standard", "profile_id": "LP0001", "condition_id": "control", "v2_request_key": "k", "v2_custom_id": "c"}}
    monkeypatch.setattr(e, "load_intended_universe", lambda: universe)
    monkeypatch.setattr(e, "classify_attempt1_responses", lambda: {"cid-1": {"identity": "id1", "stage": "standard", "part": "part1", "status": e.SCHEMA_VALID, "system_fingerprint": None}})
    monkeypatch.setattr(e, "verify_attempt1_classification_matches_committed_report", lambda classification: None)
    round2 = {"attempt_number": 2, "custom_id_to_identity": {"cid-2": "id1"}, "custom_id_status": {"cid-2": e.SCHEMA_VALID}}
    with pytest.raises(RuntimeError, match="more than one schema-valid production response"):
        e.build_attempt_ledger([round2])


# 9. scientific answer values cannot affect retry membership -- the ledger
# and pending-computation machinery never accept or reference response
# content, only the status enum.
def test_scientific_values_never_influence_membership():
    import inspect

    sig = inspect.signature(e.identities_pending_next_attempt)
    assert list(sig.parameters) == ["ledger"]
    sig2 = inspect.signature(e.build_completion_requests)
    assert "response" not in sig2.parameters and "answer" not in sig2.parameters


# 10. provider error is retry-eligible.
def test_provider_error_retry_eligible():
    ledger = {"id1": _entry(attempts=[_attempt(1, e.PROVIDER_ERROR)], resolved=False, resolved_attempt=None, attempt_count=1, next_attempt_number=2)}
    pending = e.identities_pending_next_attempt(ledger)
    assert pending["standard"] == ["id1"]


# 11. malformed fenced JSON stays invalid -- covered directly by the real
# ate.f_screen_validation.validate_response (no fence-stripping), reused
# unmodified by classify_attempt1_responses; spot-check the classification
# constant naming matches the frozen validator's vocabulary.
def test_no_fence_stripping_reused_from_frozen_validator():
    from ate.f_screen_validation import validate_response

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    rec = {"custom_id": "c", "response": {"body": {"choices": [{"message": {"content": "```json\n{\"x\": 1}\n```"}}]}}}
    v = validate_response(rec, schema)
    assert v["valid"] is False
    assert v["reason"].startswith("malformed_json")


# 12. smoke response can never become production -- classify_attempt1_responses
# only reads outputs/target_production/wave1_g_v2_replacement/submission
# (the real production submission), never outputs/target_production/
# g_v2_engineering_smoke.
def test_classification_never_reads_smoke_directory():
    smoke_root = PIPELINE_ROOT / "outputs" / "target_production" / "g_v2_engineering_smoke"
    for stage, part in e.attempt1_scorer.PARTS:
        part_dir = e.attempt1_scorer.SUBMISSION_ROOT / stage / part
        assert not str(part_dir).startswith(str(smoke_root))


# 13. smoke-only identity starts at production attempt 1.
def test_smoke_only_identity_starts_at_attempt_1(monkeypatch):
    universe = {"id_smoke_only": {"request_stage": "standard", "profile_id": "LP0001", "condition_id": "control", "v2_request_key": "k", "v2_custom_id": "c"}}
    monkeypatch.setattr(e, "load_intended_universe", lambda: universe)
    monkeypatch.setattr(e, "classify_attempt1_responses", lambda: {})  # no real attempt-1 row -- smoke-only
    monkeypatch.setattr(e, "verify_attempt1_classification_matches_committed_report", lambda classification: None)
    ledger = e.build_attempt_ledger()
    assert ledger["id_smoke_only"]["attempt_count"] == 0
    assert ledger["id_smoke_only"]["next_attempt_number"] == 1
    assert ledger["id_smoke_only"]["resolved"] is False


@pytest.fixture(scope="module")
def real_ledger():
    return e.build_attempt_ledger()


REAL_FROZEN_PATHS = [
    PIPELINE_ROOT / "outputs" / "target_production" / "g_wave1_v1_format_failure_amendment.json",
    PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "g_wave1_v2_replacement_validation_report.json",
    e.WAVE1_V2_BY_STAGE_ROOT / "standard" / "request_manifest.csv",
    e.WAVE1_V2_BY_STAGE_ROOT / "consensus_stage_a" / "request_manifest.csv",
]


@pytest.fixture(scope="module", autouse=True)
def _verify_no_frozen_artifact_mutation():
    before = {p: (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None) for p in REAL_FROZEN_PATHS}
    yield
    for p in REAL_FROZEN_PATHS:
        after = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        assert after == before[p], f"frozen artifact mutated by the retry-engine test suite: {p}"


# 14. completion manifest excludes every currently valid production identity.
def test_completion_excludes_every_valid_identity(real_ledger):
    pending = e.identities_pending_next_attempt(real_ledger)
    pending_ids = set(pending["standard"]) | set(pending["consensus_stage_a"])
    valid_ids = {identity for identity, entry in real_ledger.items() if entry["resolved"]}
    assert pending_ids & valid_ids == set()
    assert len(valid_ids) == 15517


# 15. completion membership exactly reconciles to the intended 17,000 universe.
def test_completion_membership_reconciles_to_intended_universe(real_ledger):
    pending = e.identities_pending_next_attempt(real_ledger)
    total_pending = len(pending["standard"]) + len(pending["consensus_stage_a"])
    resolved = sum(1 for entry in real_ledger.values() if entry["resolved"])
    assert len(real_ledger) == e.EXPECTED_UNIVERSE_SIZE
    assert total_pending + resolved == e.EXPECTED_UNIVERSE_SIZE
    assert total_pending == 1483
    assert len(pending["standard"]) == 1401
    assert len(pending["consensus_stage_a"]) == 82


# 20 (partial -- the guard test file covers the rest): no frozen scientific
# G/S2 artifact is modified by any of this module's real-data calls.
def test_no_frozen_artifact_modified_by_real_ledger_build(real_ledger):
    assert real_ledger is not None  # the module-scoped autouse fixture above does the actual assertion


# Attempt-3 fallback builder, exercised NOW with SYNTHETIC attempt-2 data
# (real attempt-2 does not exist yet) -- proves the SAME machinery that
# will build the current completion manifest can later build an Attempt-3
# manifest, and that it never submits anything itself.
def test_attempt_3_fallback_builder_works_prospectively_with_synthetic_round2(real_ledger, monkeypatch):
    pending_standard = e.identities_pending_next_attempt(real_ledger)["standard"]
    # must already have a real attempt 1 (attempt_count==1, i.e. previously
    # failed, not smoke-only) so a synthetic "round 2" is attempt-order-valid.
    two_pending_standard_ids = [i for i in pending_standard if real_ledger[i]["attempt_count"] == 1][:2]
    assert len(two_pending_standard_ids) == 2

    synthetic_round2 = {
        "attempt_number": 2,
        "custom_id_to_identity": {f"synthetic-r2-{i}": identity for i, identity in enumerate(two_pending_standard_ids)},
        "custom_id_status": {f"synthetic-r2-{i}": e.SCHEMA_INVALID for i in range(2)},
        "source_label": "synthetic_test_round2",
    }
    ledger_with_round2 = e.build_attempt_ledger([synthetic_round2])
    for identity in two_pending_standard_ids:
        assert ledger_with_round2[identity]["attempt_count"] == 2
        assert ledger_with_round2[identity]["next_attempt_number"] == 3

    attempt3_pending = e.identities_pending_next_attempt(ledger_with_round2)
    for identity in two_pending_standard_ids:
        assert identity in attempt3_pending["standard"]

    g_star = "google/gemma-4-31B-it"
    attempt3_pending_subset = {"standard": [i for i in attempt3_pending["standard"] if i in two_pending_standard_ids], "consensus_stage_a": []}
    requests = e.build_completion_requests(ledger_with_round2, attempt3_pending_subset, requested_model=g_star)
    assert len(requests["standard"]) == 2
    for req in requests["standard"]:
        assert req.replicate_id == 3
        assert req.request_key.endswith("|fmt_v2")
        assert "|replicate_3|" in req.request_key
    # the builder is a pure function -- constructing requests never submits
    # or writes anything; nothing here calls submit_batch or touches Together.
