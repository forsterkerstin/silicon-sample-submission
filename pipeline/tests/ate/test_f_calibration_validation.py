"""Regression tests for the frozen F* external-calibration production
raw-output scoring path. Synthetic data only -- never run against real
calibration-production output (none exists yet; no inference has been
submitted)."""

from __future__ import annotations

import pandas as pd
import pytest

from ate.f_calibration_validation import (
    IntegrityFailure,
    build_r1_ledger,
    enforce_reconciliation,
    invalid_rate,
    paired_complete_case_effect_native,
    reconciliation_report,
    score_f_calibration_production_from_raw,
    validate_response,
)

SCHEMA = {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": 1, "maximum": 7}}, "required": ["response"]}


def _raw(cid: str, content: str, status: int = 200) -> dict:
    return {"custom_id": cid, "response": {"status_code": status, "body": {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}}}


def _manifest_row(cid, study, profile, condition, effect, replicate=7):
    return {"custom_id": cid, "study_id": study, "profile_id": profile, "condition_id": condition, "outcome_id": effect, "replicate_id": replicate}


# ---- paired_complete_case_effect_native ----


def test_paired_complete_case_effect_native_computes_raw_and_pp():
    rows, raw = [], {}
    for i in range(4):
        for cond, val in [("control", 2), ("treatment", 5)]:
            cid = f"{cond}_{i}"
            rows.append(_manifest_row(cid, "S1", f"P{i}", cond, "E1"))
            raw[cid] = _raw(cid, f'{{"response": {val}}}')
    manifest = pd.DataFrame(rows)
    schemas = {r["custom_id"]: SCHEMA for r in rows}
    ledger = build_r1_ledger(manifest, raw, schemas)
    draw_df, accounting = paired_complete_case_effect_native(
        ledger, replicate_id=7, effect_native_bounds={"E1": (1, 7)}, effect_response_field={"E1": "response"}, study_id_by_effect={"E1": "S1"}
    )
    assert draw_df.iloc[0]["z_se_native"] == pytest.approx(3.0)
    assert draw_df.iloc[0]["theta_l_pp"] == pytest.approx(100 * 3.0 / 6.0)
    assert draw_df.iloc[0]["paired_n"] == 4
    assert accounting["E1"]["valid_paired_n"] == 4
    assert accounting["E1"]["invalid_control_n"] == 0
    assert accounting["E1"]["invalid_treatment_n"] == 0


def test_paired_complete_case_effect_native_excludes_unpaired_arms():
    rows, raw = [], {}
    # profile 0: both valid; profile 1: control invalid, treatment valid -- must be excluded from the paired mean
    rows.append(_manifest_row("c0", "S1", "P0", "control", "E1"))
    raw["c0"] = _raw("c0", '{"response": 2}')
    rows.append(_manifest_row("t0", "S1", "P0", "treatment", "E1"))
    raw["t0"] = _raw("t0", '{"response": 6}')
    rows.append(_manifest_row("c1", "S1", "P1", "control", "E1"))
    raw["c1"] = _raw("c1", '{"response": 99}')  # invalid (out of schema range)
    rows.append(_manifest_row("t1", "S1", "P1", "treatment", "E1"))
    raw["t1"] = _raw("t1", '{"response": 6}')
    manifest = pd.DataFrame(rows)
    schemas = {r["custom_id"]: SCHEMA for r in rows}
    ledger = build_r1_ledger(manifest, raw, schemas)
    draw_df, accounting = paired_complete_case_effect_native(
        ledger, replicate_id=7, effect_native_bounds={"E1": (1, 7)}, effect_response_field={"E1": "response"}, study_id_by_effect={"E1": "S1"}
    )
    assert draw_df.iloc[0]["paired_n"] == 1
    assert draw_df.iloc[0]["z_se_native"] == pytest.approx(4.0)  # only profile 0's (6-2)
    assert accounting["E1"]["invalid_control_n"] == 1


def test_paired_complete_case_effect_native_raises_on_zero_valid_pairs():
    manifest = pd.DataFrame([_manifest_row("c1", "S1", "P1", "control", "E1")])
    raw = {"c1": _raw("c1", '{"response": 99}')}  # invalid, and no treatment row at all
    ledger = build_r1_ledger(manifest, raw, {"c1": SCHEMA})
    with pytest.raises(ValueError):
        paired_complete_case_effect_native(ledger, replicate_id=7, effect_native_bounds={"E1": (1, 7)}, effect_response_field={"E1": "response"}, study_id_by_effect={"E1": "S1"})


def test_each_effect_uses_its_own_native_bounds():
    """Unlike the reliability path's shared benchmark-composite scale, every
    calibration effect must be normalized by its OWN outcome range. Each
    effect's own native bounds double as its own response schema range, just
    as a real calibration request's schema is derived from that effect's
    archived outcome_scale_min/outcome_scale_max."""
    rows, raw, schemas = [], {}, {}
    for effect_id, (low, high) in [("E1", (1, 7)), ("E2", (0, 100))]:
        schema = {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": low, "maximum": high}}, "required": ["response"]}
        for i in range(2):
            for cond, val in [("control", low), ("treatment", high)]:
                cid = f"{effect_id}_{cond}_{i}"
                rows.append(_manifest_row(cid, "S1", f"P{i}", cond, effect_id))
                raw[cid] = _raw(cid, f'{{"response": {val}}}')
                schemas[cid] = schema
    manifest = pd.DataFrame(rows)
    ledger = build_r1_ledger(manifest, raw, schemas)
    draw_df, _ = paired_complete_case_effect_native(
        ledger,
        replicate_id=7,
        effect_native_bounds={"E1": (1, 7), "E2": (0, 100)},
        effect_response_field={"E1": "response", "E2": "response"},
        study_id_by_effect={"E1": "S1", "E2": "S1"},
    )
    by_effect = draw_df.set_index("effect_id")
    # full-range swing on both effects -> theta_l_pp == 100 regardless of native scale
    assert by_effect.loc["E1", "theta_l_pp"] == pytest.approx(100.0)
    assert by_effect.loc["E2", "theta_l_pp"] == pytest.approx(100.0)
    assert by_effect.loc["E1", "z_se_native"] == pytest.approx(6.0)
    assert by_effect.loc["E2", "z_se_native"] == pytest.approx(100.0)


# ---- score_f_calibration_production_from_raw (end-to-end) ----


def _single_effect_manifest_and_raw(control_val=2, treatment_val=6, n_profiles=4):
    rows, raw = [], {}
    for i in range(n_profiles):
        for cond, val in [("control", control_val), ("treatment", treatment_val)]:
            cid = f"{cond}_{i}"
            rows.append(_manifest_row(cid, "S1", f"P{i}", cond, "E1"))
            raw[cid] = _raw(cid, f'{{"response": {val}}}')
    return pd.DataFrame(rows), raw


def test_end_to_end_single_draw_no_replicate_comparison(tmp_path):
    manifest, raw = _single_effect_manifest_and_raw()
    schemas = {r["custom_id"]: SCHEMA for r in manifest.to_dict("records")}
    result = score_f_calibration_production_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_native_bounds={"E1": (1, 7)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
        expected_total=8,
    )
    assert result["invalid_rate"] == 0.0
    assert result["replicate_id"] == 7
    assert "decision" not in result  # calibration scoring never makes an R_F freeze/escalate decision
    assert len(result["per_effect_ate"]) == 1
    row = result["per_effect_ate"][0]
    assert row["z_se_native"] == pytest.approx(4.0)
    assert row["theta_l_pp"] == pytest.approx(100 * 4.0 / 6.0)


def test_end_to_end_invalid_responses_reflected_in_rate():
    # profile P0 is a complete valid pair; profile P1's treatment response is
    # malformed, so P1 contributes to invalid_rate but is excluded from the
    # paired-complete-case mean (not silently coerced or dropped from accounting).
    rows, raw = [], {}
    for cid, profile, cond, content in [
        ("c0", "P0", "control", '{"response": 2}'),
        ("t0", "P0", "treatment", '{"response": 6}'),
        ("c1", "P1", "control", '{"response": 3}'),
        ("t1", "P1", "treatment", "not json"),
    ]:
        rows.append(_manifest_row(cid, "S1", profile, cond, "E1"))
        raw[cid] = _raw(cid, content)
    manifest = pd.DataFrame(rows)
    schemas = {r["custom_id"]: SCHEMA for r in rows}
    result = score_f_calibration_production_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_native_bounds={"E1": (1, 7)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
        expected_total=4,
    )
    assert result["invalid_rate"] == pytest.approx(0.25)
    assert len(result["per_effect_ate"]) == 1
    row = result["per_effect_ate"][0]
    assert row["paired_n"] == 1  # only P0; P1 excluded from the mean, not coerced
    assert row["z_se_native"] == pytest.approx(4.0)
    assert result["per_effect_accounting"]["E1"]["invalid_treatment_n"] == 1


def test_end_to_end_zero_valid_pairs_raises_not_silently_omits():
    manifest = pd.DataFrame([_manifest_row("c0", "S1", "P0", "control", "E1"), _manifest_row("t0", "S1", "P0", "treatment", "E1")])
    raw = {"c0": _raw("c0", "not json"), "t0": _raw("t0", "not json")}
    schemas = {"c0": SCHEMA, "t0": SCHEMA}
    with pytest.raises(ValueError):
        score_f_calibration_production_from_raw(
            manifest=manifest,
            raw_by_custom_id=raw,
            schema_by_custom_id=schemas,
            effect_native_bounds={"E1": (1, 7)},
            effect_response_field={"E1": "response"},
            study_id_by_effect={"E1": "S1"},
            expected_total=2,
        )


def test_missing_response_classified_invalid_not_an_integrity_failure():
    """A custom_id entirely absent from the retrieved batch output is
    classified 'missing'/invalid by validate_response, exactly like R1 --
    reconciliation (enforce_reconciliation) only guards duplicate/unexpected
    ids, not missingness, so this must NOT raise IntegrityFailure."""
    manifest = pd.DataFrame(
        [
            _manifest_row("c0", "S1", "P0", "control", "E1"),
            _manifest_row("t0", "S1", "P0", "treatment", "E1"),
            _manifest_row("c1", "S1", "P1", "control", "E1"),
            _manifest_row("t1", "S1", "P1", "treatment", "E1"),
        ]
    )
    raw = {"c0": _raw("c0", '{"response": 2}'), "t0": _raw("t0", '{"response": 6}'), "c1": _raw("c1", '{"response": 3}')}  # t1 entirely missing
    schemas = {cid: SCHEMA for cid in ["c0", "t0", "c1", "t1"]}
    result = score_f_calibration_production_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_native_bounds={"E1": (1, 7)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
        expected_total=4,
    )
    assert result["invalid_rate"] == pytest.approx(0.25)
    assert result["per_effect_ate"][0]["paired_n"] == 1
    assert result["per_effect_accounting"]["E1"]["invalid_treatment_n"] == 1


def test_no_coercion_of_out_of_range_or_malformed_responses():
    """Same frozen invalid policy as every other F scoring path: an
    out-of-schema-range or malformed response is classified invalid, never
    clipped/coerced/repaired into support."""
    manifest = pd.DataFrame([_manifest_row("c0", "S1", "P0", "control", "E1")])
    result = validate_response(_raw("c0", '{"response": 999}'), SCHEMA)
    assert result["valid"] is False
    assert result["parsed"] is None
    result2 = validate_response(_raw("c0", '{"response": "A"}'), SCHEMA)
    assert result2["valid"] is False
