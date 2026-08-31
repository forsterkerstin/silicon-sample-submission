"""Regression tests for the frozen F-screen invalid-response handling rule.
Synthetic data only -- never run against real DeepSeek/Gemma scientific
output."""

from __future__ import annotations

import pandas as pd
import pytest

from ate.f_screen_validation import (
    EXPECTED_REQUESTS_PER_MODEL,
    INVALID_RATE_GATE,
    IntegrityFailure,
    build_ledger,
    enforce_reconciliation,
    gate_decision,
    invalid_rate,
    paired_complete_case_effect,
    reconciliation_report,
    validate_response,
)

SCHEMA = {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["response"]}


def _raw(cid: str, content: str, status: int = 200) -> dict:
    return {"custom_id": cid, "response": {"status_code": status, "body": {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}}}


def _manifest_row(cid, study, profile, condition, effect):
    return {"custom_id": cid, "study_id": study, "profile_id": profile, "condition_id": condition, "outcome_id": effect}


# ---- validate_response scenarios ----

def test_all_responses_valid():
    r = validate_response(_raw("c1", '{"response": 3}'), SCHEMA)
    assert r["valid"] is True
    assert r["parsed"] == {"response": 3}


def test_malformed_json():
    r = validate_response(_raw("c1", '{"response": 3'), SCHEMA)  # truncated
    assert r["valid"] is False
    assert "malformed_json" in r["reason"]


def test_missing_entirely_id():
    r = validate_response(None, SCHEMA)
    assert r["valid"] is False
    assert r["reason"] == "missing"


def test_out_of_range_numeric_response():
    r = validate_response(_raw("c1", '{"response": 999}'), SCHEMA)
    assert r["valid"] is False
    assert "schema_invalid" in r["reason"]


def test_wrong_type():
    r = validate_response(_raw("c1", '{"response": "high"}'), SCHEMA)
    assert r["valid"] is False
    assert "schema_invalid" in r["reason"]


def test_terminal_provider_failure():
    r = validate_response(_raw("c1", "", status=500), SCHEMA)
    assert r["valid"] is False
    assert r["reason"] == "terminal_provider_failure"


def test_no_coercion_of_out_of_range_value():
    r = validate_response(_raw("c1", '{"response": 999}'), SCHEMA)
    assert r["parsed"] is None  # never clipped/repaired to a usable value


# ---- reconciliation scenarios ----

def test_duplicate_id_is_integrity_failure():
    expected = {"c1", "c2"}
    raw = [_raw("c1", '{"response": 1}'), _raw("c1", '{"response": 2}'), _raw("c2", '{"response": 1}')]
    report = reconciliation_report(expected, raw)
    assert report["duplicate"] == ["c1"]
    assert report["integrity_ok"] is False
    with pytest.raises(IntegrityFailure):
        enforce_reconciliation(report)


def test_unexpected_id_is_integrity_failure():
    expected = {"c1"}
    raw = [_raw("c1", '{"response": 1}'), _raw("c_rogue", '{"response": 1}')]
    report = reconciliation_report(expected, raw)
    assert report["unexpected"] == ["c_rogue"]
    assert report["integrity_ok"] is False
    with pytest.raises(IntegrityFailure):
        enforce_reconciliation(report)


def test_missing_entirely_does_not_alone_trigger_integrity_failure():
    # missing responses are a validity problem (feed the invalid rate), not
    # a reconciliation integrity failure -- only unexpected/duplicate stop.
    expected = {"c1", "c2"}
    raw = [_raw("c1", '{"response": 1}')]
    report = reconciliation_report(expected, raw)
    assert report["missing_entirely"] == ["c2"]
    assert report["integrity_ok"] is True
    enforce_reconciliation(report)  # must not raise


# ---- paired-complete-case scenarios ----

def _ledger_two_profiles(control_valid, treatment_valid, control_val=2, treatment_val=4):
    rows = [
        {"custom_id": "c1", "study_id": "S", "profile_id": "P1", "condition_id": "control", "outcome_id": "E1", "valid": control_valid, "reason": "", "parsed": {"response": control_val} if control_valid else None},
        {"custom_id": "c2", "study_id": "S", "profile_id": "P1", "condition_id": "treatment", "outcome_id": "E1", "valid": treatment_valid, "reason": "", "parsed": {"response": treatment_val} if treatment_valid else None},
    ]
    return pd.DataFrame(rows)


def test_control_invalid_treatment_valid_excluded_from_pair():
    ledger = _ledger_two_profiles(control_valid=False, treatment_valid=True)
    detail = paired_complete_case_effect(ledger, "E1", response_field="response", scale_low=1, scale_high=5)
    assert detail["valid_paired_n"] == 0
    assert detail["invalid_control_n"] == 1
    assert detail["invalid_treatment_n"] == 0
    assert detail["invalid_both_n"] == 0
    assert detail["theta_l_pp"] is None


def test_treatment_invalid_control_valid_excluded_from_pair():
    ledger = _ledger_two_profiles(control_valid=True, treatment_valid=False)
    detail = paired_complete_case_effect(ledger, "E1", response_field="response", scale_low=1, scale_high=5)
    assert detail["valid_paired_n"] == 0
    assert detail["invalid_control_n"] == 0
    assert detail["invalid_treatment_n"] == 1


def test_both_invalid_for_same_profile():
    ledger = _ledger_two_profiles(control_valid=False, treatment_valid=False)
    detail = paired_complete_case_effect(ledger, "E1", response_field="response", scale_low=1, scale_high=5)
    assert detail["valid_paired_n"] == 0
    assert detail["invalid_both_n"] == 1


def test_paired_complete_case_not_unpaired_arm_means():
    # 2 profiles: P1 fully valid (control=2,treatment=4), P2 has invalid treatment.
    # An UNPAIRED arm-means estimator would use both control values (2,2) and
    # only the one treatment value (4) it has -> ate = 4 - 2 = 2.
    # The frozen PAIRED estimator must use only P1's complete pair -> ate = 4-2 = 2 too
    # in this case coincidentally, so use asymmetric values to distinguish them.
    rows = [
        {"custom_id": "c1", "study_id": "S", "profile_id": "P1", "condition_id": "control", "outcome_id": "E1", "valid": True, "reason": "", "parsed": {"response": 1}},
        {"custom_id": "c2", "study_id": "S", "profile_id": "P1", "condition_id": "treatment", "outcome_id": "E1", "valid": True, "reason": "", "parsed": {"response": 5}},
        {"custom_id": "c3", "study_id": "S", "profile_id": "P2", "condition_id": "control", "outcome_id": "E1", "valid": True, "reason": "", "parsed": {"response": 3}},
        {"custom_id": "c4", "study_id": "S", "profile_id": "P2", "condition_id": "treatment", "outcome_id": "E1", "valid": False, "reason": "missing", "parsed": None},
    ]
    ledger = pd.DataFrame(rows)
    detail = paired_complete_case_effect(ledger, "E1", response_field="response", scale_low=1, scale_high=5)
    # Unpaired arm-means would give mean(treatment=[5]) - mean(control=[1,3]) = 5-2 = 3 -> 75pp
    # Paired-complete-case must use ONLY P1: 5-1 = 4 -> 100pp
    assert detail["valid_paired_n"] == 1
    assert detail["invalid_treatment_n"] == 1
    assert detail["theta_l_pp"] == pytest.approx(100.0)


# ---- invalid rate + gate scenarios ----

def test_invalid_rate_exactly_at_gate_is_eligible():
    n = 1000
    ledger = pd.DataFrame([{"custom_id": f"c{i}", "valid": i >= 5} for i in range(n)])  # 5 invalid / 1000 = 0.005
    rate = invalid_rate(ledger, expected_total=n)
    assert rate == pytest.approx(0.005)
    decision = gate_decision({"m": rate})
    assert decision["eligible_models"] == ["m"]


def test_invalid_rate_immediately_above_gate_is_ineligible():
    n = 1000
    ledger = pd.DataFrame([{"custom_id": f"c{i}", "valid": i >= 6} for i in range(n)])  # 6 invalid / 1000 = 0.006
    rate = invalid_rate(ledger, expected_total=n)
    assert rate > INVALID_RATE_GATE
    decision = gate_decision({"m": rate})
    assert decision["eligible_models"] == []


def test_only_deepseek_eligible():
    decision = gate_decision({"deepseek-ai/DeepSeek-V4-Pro-0813": 0.001, "google/gemma-4-31B-it": 0.01})
    assert decision["decision"] == "SINGLE_ELIGIBLE_CANDIDATE_IS_F_STAR"
    assert decision["f_star"] == "deepseek-ai/DeepSeek-V4-Pro-0813"


def test_only_gemma_eligible():
    decision = gate_decision({"deepseek-ai/DeepSeek-V4-Pro-0813": 0.01, "google/gemma-4-31B-it": 0.001})
    assert decision["decision"] == "SINGLE_ELIGIBLE_CANDIDATE_IS_F_STAR"
    assert decision["f_star"] == "google/gemma-4-31B-it"


def test_neither_eligible_stops():
    decision = gate_decision({"deepseek-ai/DeepSeek-V4-Pro-0813": 0.01, "google/gemma-4-31B-it": 0.02})
    assert decision["decision"] == "STOP_REQUIRE_NEW_EXPLICIT_DECISION"
    assert decision["f_star"] is None


def test_both_eligible_scores_both():
    decision = gate_decision({"deepseek-ai/DeepSeek-V4-Pro-0813": 0.001, "google/gemma-4-31B-it": 0.002})
    assert decision["decision"] == "SCORE_BOTH_ELIGIBLE_CANDIDATES"
    assert decision["eligible_models"] == ["deepseek-ai/DeepSeek-V4-Pro-0813", "google/gemma-4-31B-it"]


def test_invalid_rate_requires_exact_expected_total():
    ledger = pd.DataFrame([{"custom_id": "c1", "valid": True}])
    with pytest.raises(ValueError, match="expected exactly"):
        invalid_rate(ledger, expected_total=EXPECTED_REQUESTS_PER_MODEL)


def test_build_ledger_keeps_missing_ids_as_rows_not_dropped():
    manifest = pd.DataFrame(
        [
            _manifest_row("c1", "S", "P1", "control", "E1"),
            _manifest_row("c2", "S", "P1", "treatment", "E1"),
        ]
    )
    raw_by_id = {"c1": _raw("c1", '{"response": 3}'), "c2": None}
    schema_by_id = {"c1": SCHEMA, "c2": SCHEMA}
    ledger = build_ledger(manifest, raw_by_id, schema_by_id)
    assert len(ledger) == 2
    assert ledger.set_index("custom_id").loc["c2", "valid"] == False  # noqa: E712
    assert ledger.set_index("custom_id").loc["c2", "reason"] == "missing"
