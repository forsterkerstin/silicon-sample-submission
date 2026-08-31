"""Regression tests for the frozen F* R1 stochastic-reliability raw-output
scoring path. Synthetic data only -- never run against real R1 output."""

from __future__ import annotations

import pandas as pd
import pytest

from ate.f_reliability_validation import (
    EXPECTED_REQUESTS_R1,
    IntegrityFailure,
    build_r1_ledger,
    enforce_reconciliation,
    invalid_rate,
    paired_complete_case_draw,
    reconciliation_report,
    score_f_reliability_r1_from_raw,
    validate_response,
)

SCHEMA = {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["response"]}


def _raw(cid: str, content: str, status: int = 200) -> dict:
    return {"custom_id": cid, "response": {"status_code": status, "body": {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}}}


def _manifest_row(cid, study, profile, condition, effect, replicate):
    return {"custom_id": cid, "study_id": study, "profile_id": profile, "condition_id": condition, "outcome_id": effect, "replicate_id": replicate}


def test_expected_requests_r1_is_24000():
    assert EXPECTED_REQUESTS_R1 == 12 * 500 * 2 * 2


# ---- build_r1_ledger ----


def test_build_r1_ledger_carries_replicate_id():
    manifest = pd.DataFrame([_manifest_row("c1", "S1", "P1", "control", "E1", 3)])
    ledger = build_r1_ledger(manifest, {"c1": _raw("c1", '{"response": 3}')}, {"c1": SCHEMA})
    assert ledger.iloc[0]["replicate_id"] == 3
    assert ledger.iloc[0]["valid"] == True  # noqa: E712


def test_build_r1_ledger_missing_response_classified_missing():
    manifest = pd.DataFrame([_manifest_row("c1", "S1", "P1", "control", "E1", 3)])
    ledger = build_r1_ledger(manifest, {}, {"c1": SCHEMA})
    assert ledger.iloc[0]["valid"] == False  # noqa: E712
    assert ledger.iloc[0]["reason"] == "missing"


# ---- paired_complete_case_draw ----


def _clean_manifest_and_raw(n_profiles=4):
    rows, raw = [], {}
    for i in range(n_profiles):
        for cond, val in [("control", 2), ("treatment", 4)]:
            cid = f"{cond}_{i}"
            rows.append(_manifest_row(cid, "S1", f"P{i}", cond, "E1", 3))
            raw[cid] = _raw(cid, f'{{"response": {val}}}')
    return pd.DataFrame(rows), raw


def test_paired_complete_case_draw_computes_expected_z_pp():
    manifest, raw = _clean_manifest_and_raw()
    ledger = build_r1_ledger(manifest, raw, {r["custom_id"]: SCHEMA for r in manifest.to_dict("records")})
    draw_df, accounting = paired_complete_case_draw(
        ledger,
        replicate_id=3,
        effect_scale_bounds={"E1": (1, 5)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
    )
    # mean(treatment)=4, mean(control)=2, raw_ate=2, range=4 -> 100*2/4=50pp
    assert draw_df.iloc[0]["z_pp"] == pytest.approx(50.0)
    assert accounting["E1"]["valid_paired_n"] == 4
    assert accounting["E1"]["planned_pairs"] == 4


def test_paired_complete_case_draw_excludes_unpaired_invalid_profile():
    manifest, raw = _clean_manifest_and_raw(n_profiles=3)
    # profile 0's treatment response is invalid (out of schema range) -> excluded from pairing
    raw["treatment_0"] = _raw("treatment_0", '{"response": 99}')
    ledger = build_r1_ledger(manifest, raw, {r["custom_id"]: SCHEMA for r in manifest.to_dict("records")})
    draw_df, accounting = paired_complete_case_draw(
        ledger, replicate_id=3, effect_scale_bounds={"E1": (1, 5)}, effect_response_field={"E1": "response"}, study_id_by_effect={"E1": "S1"}
    )
    assert accounting["E1"]["valid_paired_n"] == 2
    assert accounting["E1"]["invalid_treatment_n"] == 1


def test_paired_complete_case_draw_only_uses_requested_replicate():
    manifest, raw = _clean_manifest_and_raw(n_profiles=2)
    # add replicate 4 rows with wildly different values -- must not leak into replicate 3's z_pp
    for i in range(2):
        for cond, val in [("control", 1), ("treatment", 1)]:
            cid = f"{cond}_{i}_r4"
            manifest = pd.concat([manifest, pd.DataFrame([_manifest_row(cid, "S1", f"P{i}", cond, "E1", 4)])], ignore_index=True)
            raw[cid] = _raw(cid, f'{{"response": {val}}}')
    ledger = build_r1_ledger(manifest, raw, {r["custom_id"]: SCHEMA for r in manifest.to_dict("records")})
    draw_df, _ = paired_complete_case_draw(
        ledger, replicate_id=3, effect_scale_bounds={"E1": (1, 5)}, effect_response_field={"E1": "response"}, study_id_by_effect={"E1": "S1"}
    )
    assert draw_df.iloc[0]["z_pp"] == pytest.approx(50.0)  # unchanged by replicate-4 rows


def test_paired_complete_case_draw_raises_on_zero_valid_pairs():
    manifest = pd.DataFrame([_manifest_row("c1", "S1", "P1", "control", "E1", 3)])
    raw = {"c1": _raw("c1", '{"response": 99}')}  # invalid, and no treatment row at all
    ledger = build_r1_ledger(manifest, raw, {"c1": SCHEMA})
    with pytest.raises(ValueError):
        paired_complete_case_draw(ledger, replicate_id=3, effect_scale_bounds={"E1": (1, 5)}, effect_response_field={"E1": "response"}, study_id_by_effect={"E1": "S1"})


# ---- score_f_reliability_r1_from_raw (end-to-end) ----


def _two_draw_manifest_and_raw(control_val=2, treatment_val=4, n_profiles=4):
    rows, raw = [], {}
    for replicate in (3, 4):
        for i in range(n_profiles):
            for cond, val in [("control", control_val), ("treatment", treatment_val)]:
                cid = f"{cond}_{i}_rep{replicate}"
                rows.append(_manifest_row(cid, "S1", f"P{i}", cond, "E1", replicate))
                raw[cid] = _raw(cid, f'{{"response": {val}}}')
    return pd.DataFrame(rows), raw


def test_end_to_end_identical_draws_pass_r1(tmp_path):
    manifest, raw = _two_draw_manifest_and_raw()
    schemas = {r["custom_id"]: SCHEMA for r in manifest.to_dict("records")}
    result = score_f_reliability_r1_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_scale_bounds={"E1": (1, 5)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
        outputs_dir=tmp_path,
        expected_total=16,
    )
    assert result["invalid_rate"] == 0.0
    assert result["decision"]["decision"] == "FREEZE_R_F"
    assert result["decision"]["r_f"] == 1
    assert result["decision"]["observed"]["replicate_rmse_pp"] == pytest.approx(0.0)


def test_end_to_end_replicate_ids_parameter_scores_non_default_draws(tmp_path):
    """The replacement-R1 provenance amendment uses fresh draws 5/6, not the
    historical 3/4. replicate_ids must let the frozen scorer read those ids
    without any change to the scoring rule itself."""
    rows, raw = [], {}
    for replicate in (5, 6):
        for i in range(4):
            for cond, val in [("control", 2), ("treatment", 4)]:
                cid = f"{cond}_{i}_rep{replicate}"
                rows.append(_manifest_row(cid, "S1", f"P{i}", cond, "E1", replicate))
                raw[cid] = _raw(cid, f'{{"response": {val}}}')
    manifest = pd.DataFrame(rows)
    schemas = {r["custom_id"]: SCHEMA for r in manifest.to_dict("records")}
    result = score_f_reliability_r1_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_scale_bounds={"E1": (1, 5)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
        outputs_dir=tmp_path,
        expected_total=16,
        replicate_ids=(5, 6),
    )
    assert result["replicate_ids"] == [5, 6]
    assert "draw_replicate_5" in result and "draw_replicate_6" in result
    assert "draw_replicate_3" not in result and "draw_replicate_4" not in result
    assert set(result["per_effect_accounting"]) == {"replicate_5", "replicate_6"}
    assert result["invalid_rate"] == 0.0
    assert result["decision"]["decision"] == "FREEZE_R_F"
    assert result["decision"]["r_f"] == 1
    assert result["decision"]["observed"]["replicate_rmse_pp"] == pytest.approx(0.0)


def test_end_to_end_default_replicate_ids_unchanged():
    """Omitting replicate_ids must still default to (3, 4) -- the historical
    R1 draw identity -- so every pre-existing caller is byte-for-byte
    unaffected by the parameterization."""
    import inspect

    default = inspect.signature(score_f_reliability_r1_from_raw).parameters["replicate_ids"].default
    assert default == (3, 4)


def test_end_to_end_wildly_different_draws_escalate_to_r2(tmp_path):
    rows, raw = [], {}
    for replicate, (c, t) in [(3, (1, 5)), (4, (5, 1))]:  # opposite-sign effects across draws
        for i in range(4):
            for cond, val in [("control", c), ("treatment", t)]:
                cid = f"{cond}_{i}_rep{replicate}"
                rows.append(_manifest_row(cid, "S1", f"P{i}", cond, "E1", replicate))
                raw[cid] = _raw(cid, f'{{"response": {val}}}')
    manifest = pd.DataFrame(rows)
    schemas = {r["custom_id"]: SCHEMA for r in rows}
    result = score_f_reliability_r1_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_scale_bounds={"E1": (1, 5)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
        outputs_dir="/tmp",
        expected_total=16,
    )
    assert result["decision"]["decision"] == "ESCALATE_TO_STAGE_R2"
    assert result["decision"]["r_f"] is None


def test_end_to_end_raises_on_reconciliation_failure(tmp_path):
    manifest, raw = _two_draw_manifest_and_raw(n_profiles=1)
    schemas = {r["custom_id"]: SCHEMA for r in manifest.to_dict("records")}
    raw["unexpected_id"] = _raw("unexpected_id", '{"response": 1}')
    with pytest.raises(IntegrityFailure):
        score_f_reliability_r1_from_raw(
            manifest=manifest,
            raw_by_custom_id=raw,
            schema_by_custom_id=schemas,
            effect_scale_bounds={"E1": (1, 5)},
            effect_response_field={"E1": "response"},
            study_id_by_effect={"E1": "S1"},
            outputs_dir=tmp_path,
        )


def test_end_to_end_invalid_responses_reflected_in_rate_and_accounting(tmp_path):
    manifest, raw = _two_draw_manifest_and_raw(n_profiles=4)
    schemas = {r["custom_id"]: SCHEMA for r in manifest.to_dict("records")}
    raw["control_0_rep3"] = _raw("control_0_rep3", '{"response": 99}')  # invalid
    result = score_f_reliability_r1_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_scale_bounds={"E1": (1, 5)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
        outputs_dir=tmp_path,
        expected_total=16,
    )
    assert result["invalid_rate"] == pytest.approx(1 / 16)
    assert result["per_effect_accounting"]["replicate_3"]["E1"]["valid_paired_n"] == 3
    assert result["per_effect_accounting"]["replicate_4"]["E1"]["valid_paired_n"] == 4


# ---- hand-calculable fixture: proves RMSE/max-abs-diff use the correct
# level (percentage points of range) and unit, not raw native units or a
# mis-scaled variant, by comparing against numbers computed by hand ----


def test_replicate_rmse_and_max_abs_diff_match_hand_calculation(tmp_path):
    # Three effects on three different native scales, one profile each (so
    # z_pp for a draw is just that single paired contrast, normalized).
    #   E1, range=100 (0-100): draw3 raw_ate=60-50=10  -> z_pp=100*10/100=10.0
    #                           draw4 raw_ate=55-50=5   -> z_pp=100*5/100=5.0    diff=+5.0
    #   E2, range=4   (1-5):   draw3 raw_ate=4-2=2      -> z_pp=100*2/4=50.0
    #                           draw4 raw_ate=3-2=1      -> z_pp=100*1/4=25.0    diff=+25.0
    #   E3, range=10  (0-10):  draw3 raw_ate=3-3=0      -> z_pp=100*0/10=0.0
    #                           draw4 raw_ate=5-3=2      -> z_pp=100*2/10=20.0   diff=-20.0
    # By hand: diffs=[5,25,-20]; RMSE=sqrt((25+625+400)/3)=sqrt(350)=18.70828693...
    # max_abs_diff=25.0; mean_abs_diff=50/3=16.6667; sign_agreement=2/3 (E3's
    # 0-vs-positive pair is the one disagreement).
    specs = {
        "E1": {"bounds": (0, 100), "control": 50, "draw3_t": 60, "draw4_t": 55},
        "E2": {"bounds": (1, 5), "control": 2, "draw3_t": 4, "draw4_t": 3},
        "E3": {"bounds": (0, 10), "control": 3, "draw3_t": 3, "draw4_t": 5},
    }
    rows, raw = [], {}
    for effect, spec in specs.items():
        for replicate, t_val in [(3, spec["draw3_t"]), (4, spec["draw4_t"])]:
            c_cid = f"{effect}_control_rep{replicate}"
            t_cid = f"{effect}_treatment_rep{replicate}"
            rows.append(_manifest_row(c_cid, "S1", "P1", "control", effect, replicate))
            rows.append(_manifest_row(t_cid, "S1", "P1", "treatment", effect, replicate))
            raw[c_cid] = _raw(c_cid, f'{{"response": {spec["control"]}}}')
            raw[t_cid] = _raw(t_cid, f'{{"response": {t_val}}}')
    manifest = pd.DataFrame(rows)
    schemas = {r["custom_id"]: {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["response"]} for r in rows}

    result = score_f_reliability_r1_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_scale_bounds={e: s["bounds"] for e, s in specs.items()},
        effect_response_field={e: "response" for e in specs},
        study_id_by_effect={e: "S1" for e in specs},
        outputs_dir=tmp_path,
        expected_total=12,
    )

    observed = result["decision"]["observed"]
    assert observed["replicate_rmse_pp"] == pytest.approx(350**0.5, abs=1e-9)
    assert observed["max_abs_diff_pp"] == pytest.approx(25.0, abs=1e-9)
    diagnostics = result["decision"]["diagnostics"]
    assert diagnostics["mad_mean_abs_diff_pp"] == pytest.approx(50 / 3, abs=1e-9)
    assert diagnostics["mad_median_abs_diff_pp"] == pytest.approx(20.0, abs=1e-9)
    assert diagnostics["sign_agreement"] == pytest.approx(2 / 3, abs=1e-9)
    # RMSE=18.708... exceeds the frozen 2.0pp threshold -> must escalate, not freeze.
    assert result["decision"]["decision"] == "ESCALATE_TO_STAGE_R2"


def test_z_pp_itself_matches_hand_calculation_before_aggregation(tmp_path):
    """Isolates the normalization step alone (not the RMSE aggregation) --
    a single effect, single profile, so z_pp is directly checkable."""
    manifest, raw = _two_draw_manifest_and_raw(control_val=20, treatment_val=32, n_profiles=1)
    schemas = {r["custom_id"]: SCHEMA_100 for r in manifest.to_dict("records")}
    result = score_f_reliability_r1_from_raw(
        manifest=manifest,
        raw_by_custom_id=raw,
        schema_by_custom_id=schemas,
        effect_scale_bounds={"E1": (0, 100)},
        effect_response_field={"E1": "response"},
        study_id_by_effect={"E1": "S1"},
        outputs_dir=tmp_path,
        expected_total=4,
    )
    # raw_ate = 32 - 20 = 12; range = 100; z_pp = 100 * 12 / 100 = 12.0 (hand-computed).
    assert result["draw_replicate_3"][0]["z_pp"] == pytest.approx(12.0, abs=1e-9)
    assert result["draw_replicate_4"][0]["z_pp"] == pytest.approx(12.0, abs=1e-9)


SCHEMA_100 = {"additionalProperties": False, "properties": {"response": {"type": "integer", "minimum": 0, "maximum": 100}}, "required": ["response"]}


def test_reconciliation_and_validate_response_are_the_same_f_screen_functions():
    from ate.f_screen_validation import reconciliation_report as f_recon
    from ate.f_screen_validation import validate_response as f_validate

    assert reconciliation_report is f_recon
    assert validate_response is f_validate
