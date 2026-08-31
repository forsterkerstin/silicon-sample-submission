"""Regression tests for the frozen ATP G-screen invalid-response handling
rule. Synthetic data only -- never run against real DeepSeek/Gemma
scientific output.

Confirms: (1) the generic reconciliation/validation machinery is reused
unmodified from ate.f_screen_validation, (2) NO numeric invalid-rate
threshold is invented or auto-borrowed from F's 0.005 gate, (3) any nonzero
invalid rate on any item blocks automatic scoring (fail-closed), and only an
exactly-zero invalid rate proceeds without a further explicit decision."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ate.g_atp_screen import ATP1_SUBSTANTIVE_CHOICES, ATP2_SUBSTANTIVE_CHOICES, item_w1_pp
from ate.g_atp_screen_validation import (
    EXPECTED_REQUESTS_PER_MODEL_ATP,
    IntegrityFailure,
    build_g_atp_ledger,
    enforce_reconciliation,
    g_fail_closed_decision,
    g_invalid_response_report,
    invalid_rate,
    reconciliation_report,
    score_g_atp_candidate_from_raw,
    validate_response,
)

SCHEMA = {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["response"]}


def _raw(cid: str, content: str, status: int = 200) -> dict:
    return {"custom_id": cid, "response": {"status_code": status, "body": {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}}}


def _manifest_row(cid: str, source_id: str, respondent_id: str) -> dict:
    return {"custom_id": cid, "study_id": source_id, "profile_id": respondent_id}


def test_expected_requests_per_model_is_1226_not_fs_6000():
    assert EXPECTED_REQUESTS_PER_MODEL_ATP == 650 + 576


def test_reconciliation_and_validate_response_are_the_same_f_screen_functions():
    # These are imported, not reimplemented -- generic machinery reused as-is.
    from ate.f_screen_validation import reconciliation_report as f_recon
    from ate.f_screen_validation import validate_response as f_validate

    assert reconciliation_report is f_recon
    assert validate_response is f_validate


def test_no_gate_decision_function_exists():
    import ate.g_atp_screen_validation as mod

    assert not hasattr(mod, "gate_decision")
    assert not hasattr(mod, "INVALID_RATE_GATE")


def _clean_ledger(n_atp1=2, n_atp2=2) -> pd.DataFrame:
    manifest_rows = [_manifest_row(f"a1_{i}", "ATP1", str(i)) for i in range(n_atp1)] + [
        _manifest_row(f"a2_{i}", "ATP2", str(i)) for i in range(n_atp2)
    ]
    manifest = pd.DataFrame(manifest_rows)
    raw = {r["custom_id"]: _raw(r["custom_id"], '{"response": 2}') for r in manifest_rows}
    schemas = {r["custom_id"]: SCHEMA for r in manifest_rows}
    return build_g_atp_ledger(manifest, raw, schemas)


def test_build_ledger_all_valid():
    ledger = _clean_ledger()
    assert len(ledger) == 4
    assert ledger["valid"].all()


def test_build_ledger_missing_response_classified_missing():
    manifest = pd.DataFrame([_manifest_row("a1_0", "ATP1", "0")])
    ledger = build_g_atp_ledger(manifest, {}, {"a1_0": SCHEMA})
    assert ledger.iloc[0]["valid"] == False  # noqa: E712 -- numpy.bool_, not a Python bool literal
    assert ledger.iloc[0]["reason"] == "missing"
    assert invalid_rate(ledger, expected_total=1) == 1.0


def test_build_ledger_schema_invalid_out_of_range():
    manifest = pd.DataFrame([_manifest_row("a1_0", "ATP1", "0")])
    raw = {"a1_0": _raw("a1_0", '{"response": 99}')}
    ledger = build_g_atp_ledger(manifest, raw, {"a1_0": SCHEMA})
    assert ledger.iloc[0]["valid"] == False  # noqa: E712 -- numpy.bool_, not a Python bool literal
    assert "schema_invalid" in ledger.iloc[0]["reason"]
    assert invalid_rate(ledger, expected_total=1) == 1.0


def test_invalid_rate_requires_full_reconciled_ledger():
    manifest = pd.DataFrame([_manifest_row("a1_0", "ATP1", "0")])
    ledger = build_g_atp_ledger(manifest, {"a1_0": _raw("a1_0", '{"response": 2}')}, {"a1_0": SCHEMA})
    with pytest.raises(ValueError):
        invalid_rate(ledger, expected_total=2)


def test_report_is_per_item_not_pooled():
    manifest_rows = [_manifest_row("a1_0", "ATP1", "0"), _manifest_row("a1_1", "ATP1", "1"), _manifest_row("a2_0", "ATP2", "0")]
    manifest = pd.DataFrame(manifest_rows)
    raw = {
        "a1_0": _raw("a1_0", '{"response": 2}'),
        "a1_1": _raw("a1_1", '{"response": 99}'),  # invalid: out of range
        "a2_0": _raw("a2_0", '{"response": 1}'),
    }
    schemas = {r["custom_id"]: SCHEMA for r in manifest_rows}
    ledger = build_g_atp_ledger(manifest, raw, schemas)
    report = g_invalid_response_report(ledger, expected_total=3)
    assert report["per_item"]["ATP1"]["invalid_n"] == 1
    assert report["per_item"]["ATP1"]["valid_n"] == 1
    assert report["per_item"]["ATP2"]["invalid_n"] == 0
    assert "a1_1" in report["per_item"]["ATP1"]["invalid_custom_ids"]


def test_fail_closed_clean_model_needs_no_decision():
    ledger = _clean_ledger()
    report = g_invalid_response_report(ledger, expected_total=4)
    decision = g_fail_closed_decision({"modelA": report})
    assert decision["status_by_model"]["modelA"]["status"] == "CLEAN_NO_DECISION_NEEDED"
    assert decision["decision"] == "ALL_CANDIDATES_CLEAN_PROCEED_TO_SCORE"


def test_fail_closed_single_invalid_response_blocks_that_model():
    manifest_rows = [_manifest_row("a1_0", "ATP1", "0"), _manifest_row("a2_0", "ATP2", "0")]
    manifest = pd.DataFrame(manifest_rows)
    raw = {"a1_0": _raw("a1_0", '{"response": 99}'), "a2_0": _raw("a2_0", '{"response": 1}')}
    schemas = {r["custom_id"]: SCHEMA for r in manifest_rows}
    ledger = build_g_atp_ledger(manifest, raw, schemas)
    report = g_invalid_response_report(ledger, expected_total=2)
    decision = g_fail_closed_decision({"modelA": report})
    assert decision["status_by_model"]["modelA"]["status"] == "STOP_REQUIRE_EXPLICIT_METHODOLOGICAL_DECISION"
    assert decision["status_by_model"]["modelA"]["dirty_items"] == ["ATP1"]
    assert decision["decision"] == "STOP_REQUIRE_EXPLICIT_METHODOLOGICAL_DECISION"


def test_fail_closed_one_clean_one_dirty_model_still_stops_overall():
    clean = g_invalid_response_report(_clean_ledger(), expected_total=4)
    manifest = pd.DataFrame([_manifest_row("a1_0", "ATP1", "0")])
    dirty_ledger = build_g_atp_ledger(manifest, {"a1_0": _raw("a1_0", '{"response": 99}')}, {"a1_0": SCHEMA})
    dirty = g_invalid_response_report(dirty_ledger, expected_total=1)
    decision = g_fail_closed_decision({"clean_model": clean, "dirty_model": dirty})
    assert decision["status_by_model"]["clean_model"]["status"] == "CLEAN_NO_DECISION_NEEDED"
    assert decision["status_by_model"]["dirty_model"]["status"] == "STOP_REQUIRE_EXPLICIT_METHODOLOGICAL_DECISION"
    assert decision["decision"] == "STOP_REQUIRE_EXPLICIT_METHODOLOGICAL_DECISION"


def test_unexpected_custom_id_raises_integrity_failure():
    expected = {"a1_0"}
    report = reconciliation_report(expected, [{"custom_id": "a1_0"}, {"custom_id": "unexpected_id"}])
    with pytest.raises(IntegrityFailure):
        enforce_reconciliation(report)


def test_duplicate_custom_id_raises_integrity_failure():
    expected = {"a1_0"}
    report = reconciliation_report(expected, [{"custom_id": "a1_0"}, {"custom_id": "a1_0"}])
    with pytest.raises(IntegrityFailure):
        enforce_reconciliation(report)


# ---- score_g_atp_candidate_from_raw: the production entry point wiring
# reconciliation + validation + valid-subset W1 + fail-closed finality ----

_SCHEMA_ATP1 = {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["response"]}
_SCHEMA_ATP2 = {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": 1, "maximum": 4}}, "required": ["response"]}
_CHOICES_BY_SOURCE = {"ATP1": ATP1_SUBSTANTIVE_CHOICES, "ATP2": ATP2_SUBSTANTIVE_CHOICES}


def _scoring_fixture(atp1_responses: list[int], atp2_responses: list[int]):
    manifest_rows = [_manifest_row(f"a1_{i}", "ATP1", str(i)) for i in range(len(atp1_responses))] + [
        _manifest_row(f"a2_{i}", "ATP2", str(i)) for i in range(len(atp2_responses))
    ]
    manifest = pd.DataFrame(manifest_rows)
    raw = {}
    schemas = {}
    for i, resp in enumerate(atp1_responses):
        raw[f"a1_{i}"] = _raw(f"a1_{i}", json.dumps({"response": resp}))
        schemas[f"a1_{i}"] = _SCHEMA_ATP1
    for i, resp in enumerate(atp2_responses):
        raw[f"a2_{i}"] = _raw(f"a2_{i}", json.dumps({"response": resp}))
        schemas[f"a2_{i}"] = _SCHEMA_ATP2
    return manifest, raw, schemas


def test_score_g_atp_candidate_matches_direct_item_w1_pp_computation_when_clean():
    # Human reference: ATP1 all at position 0.0 (choice 1), ATP2 all at 1.0 (choice 4).
    # Model: ATP1 answers choice 1 (position 0.0) for everyone, ATP2 answers choice 4 (position 1.0).
    manifest, raw, schemas = _scoring_fixture(atp1_responses=[1, 1, 1], atp2_responses=[4, 4])
    human_ref = {"ATP1": [0.0, 0.0, 0.0], "ATP2": [1.0, 1.0]}
    result = score_g_atp_candidate_from_raw(
        model="modelA", manifest=manifest, raw_by_custom_id=raw, schema_by_custom_id=schemas,
        human_reference_positions=human_ref, choices_by_source=_CHOICES_BY_SOURCE, expected_total=5,
    )
    assert result["w1_pp_atp1"] == pytest.approx(0.0)
    assert result["w1_pp_atp2"] == pytest.approx(0.0)
    assert result["g_atp_loss"] == pytest.approx(0.0)
    assert result["is_final"] is True
    assert result["decision_status"] == "CLEAN_NO_DECISION_NEEDED"


def test_score_g_atp_candidate_reproduces_manual_w1_when_human_and_model_diverge():
    # Human ATP1 all at position 1.0 (choice 5), model answers choice 1 (position 0.0) -> W1 = 100pp.
    manifest, raw, schemas = _scoring_fixture(atp1_responses=[1, 1], atp2_responses=[4, 4])
    human_ref = {"ATP1": [1.0, 1.0], "ATP2": [1.0, 1.0]}
    result = score_g_atp_candidate_from_raw(
        model="modelA", manifest=manifest, raw_by_custom_id=raw, schema_by_custom_id=schemas,
        human_reference_positions=human_ref, choices_by_source=_CHOICES_BY_SOURCE, expected_total=4,
    )
    expected_atp1 = item_w1_pp([1.0, 1.0], [0.0, 0.0])
    assert result["w1_pp_atp1"] == pytest.approx(expected_atp1)
    assert result["g_atp_loss"] == pytest.approx((expected_atp1 + 0.0) / 2)


def test_score_g_atp_candidate_invalid_response_is_not_final_but_still_scored():
    manifest, raw, schemas = _scoring_fixture(atp1_responses=[1, 1], atp2_responses=[4, 4])
    raw["a1_0"] = _raw("a1_0", '{"response": 99}')  # schema-invalid: out of range
    human_ref = {"ATP1": [0.0, 0.0], "ATP2": [1.0, 1.0]}
    result = score_g_atp_candidate_from_raw(
        model="modelA", manifest=manifest, raw_by_custom_id=raw, schema_by_custom_id=schemas,
        human_reference_positions=human_ref, choices_by_source=_CHOICES_BY_SOURCE, expected_total=4,
    )
    assert result["is_final"] is False
    assert result["decision_status"] == "STOP_REQUIRE_EXPLICIT_METHODOLOGICAL_DECISION"
    assert result["invalid_report"]["per_item"]["ATP1"]["invalid_n"] == 1
    # still computed transparently on the valid subset, not withheld entirely
    assert result["g_atp_loss"] is not None


def test_score_g_atp_candidate_raises_on_reconciliation_failure():
    manifest, raw, schemas = _scoring_fixture(atp1_responses=[1], atp2_responses=[4])
    raw["unexpected_id"] = _raw("unexpected_id", '{"response": 1}')
    with pytest.raises(IntegrityFailure):
        score_g_atp_candidate_from_raw(
            model="modelA", manifest=manifest, raw_by_custom_id=raw, schema_by_custom_id=schemas,
            human_reference_positions={"ATP1": [0.0], "ATP2": [1.0]}, choices_by_source=_CHOICES_BY_SOURCE, expected_total=2,
        )


def test_score_g_atp_candidate_raises_on_zero_valid_responses_for_an_item():
    manifest, raw, schemas = _scoring_fixture(atp1_responses=[1], atp2_responses=[4])
    raw["a1_0"] = _raw("a1_0", '{"response": 99}')  # the only ATP1 response, invalid
    with pytest.raises(ValueError):
        score_g_atp_candidate_from_raw(
            model="modelA", manifest=manifest, raw_by_custom_id=raw, schema_by_custom_id=schemas,
            human_reference_positions={"ATP1": [0.0], "ATP2": [1.0]}, choices_by_source=_CHOICES_BY_SOURCE, expected_total=2,
        )


def test_score_g_atp_candidate_never_lets_dirty_score_reach_select_g_star_automatically():
    """End-to-end proof that G_INVALID_RULE_WIRED_TO_SELECTION holds: a
    caller that only feeds is_final=True candidates into select_g_star can
    never automatically finalize a dirty candidate."""
    from ate.g_atp_screen import select_g_star

    manifest, raw, schemas = _scoring_fixture(atp1_responses=[1, 1], atp2_responses=[4, 4])
    raw["a1_0"] = _raw("a1_0", '{"response": 99}')
    human_ref = {"ATP1": [0.0, 0.0], "ATP2": [1.0, 1.0]}
    dirty = score_g_atp_candidate_from_raw(
        model="dirty_model", manifest=manifest, raw_by_custom_id=raw, schema_by_custom_id=schemas,
        human_reference_positions=human_ref, choices_by_source=_CHOICES_BY_SOURCE, expected_total=4,
    )
    manifest2, raw2, schemas2 = _scoring_fixture(atp1_responses=[1, 1], atp2_responses=[4, 4])
    clean = score_g_atp_candidate_from_raw(
        model="clean_model", manifest=manifest2, raw_by_custom_id=raw2, schema_by_custom_id=schemas2,
        human_reference_positions=human_ref, choices_by_source=_CHOICES_BY_SOURCE, expected_total=4,
    )
    assert dirty["is_final"] is False and clean["is_final"] is True
    finalizable = {r["model"]: r["g_atp_loss"] for r in [dirty, clean] if r["is_final"]}
    assert set(finalizable) == {"clean_model"}
    with pytest.raises(ValueError):
        # select_g_star requires matching model sets across all three dicts --
        # a caller cannot silently pass the dirty model through without also
        # supplying its (nonexistent, not-yet-decided) invalid_response_rate/cost.
        select_g_star(finalizable, invalid_response_rate={"clean_model": 0.0, "dirty_model": 0.0}, realized_cost_usd={"clean_model": 1.0, "dirty_model": 1.0})
