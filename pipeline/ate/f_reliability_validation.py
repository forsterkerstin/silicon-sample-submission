"""F* R1 stochastic-reliability raw-output validation/scoring path.

Frozen BEFORE any R1 inference is submitted or observed. Mirrors
ate/f_screen_validation.py's structure exactly, reusing its generic
(non-F-screen-specific) reconciliation and per-response JSON-Schema
validation machinery unmodified. The one structural difference from
f_screen_validation is that R1 has TWO independent draws (replicate_id 3
and 4, per the provenance-separation amendment) rather than one, so the
ledger carries a replicate_id column and paired-complete-case ATEs are
computed separately per (effect, replicate) before handing the two
per-effect draw tables to the unmodified ate.r_f_decision.stage_r1_decision.

Pipeline: reconcile expected ids -> validate every response against its own
request-specific JSON Schema -> per-replicate paired-complete-case ATE per
effect -> ate.r_f_decision.stage_r1_decision. No coercion, clipping,
imputation, or automatic retries anywhere in this path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ate.f_screen import f_screen_theta_l_pp
from ate.f_screen_validation import (  # noqa: F401 -- re-exported for callers
    IntegrityFailure,
    enforce_reconciliation,
    reconciliation_report,
    validate_response,
)
from ate.r_f_decision import stage_r1_decision

EXPECTED_REQUESTS_R1 = 24000
CONDITIONS = ("control", "treatment")
REPLICATES = (3, 4)


def build_r1_ledger(
    manifest: pd.DataFrame,
    raw_by_custom_id: dict[str, dict[str, Any] | None],
    schema_by_custom_id: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """One row per manifest custom_id (all EXPECTED_REQUESTS_R1 of them).
    Unlike f_screen_validation.build_ledger, this carries replicate_id
    (R1 has two draws, not one) alongside study_id/profile_id/condition_id/
    outcome_id. A custom_id absent from raw_by_custom_id is passed through
    as None and classified 'missing', never silently dropped."""
    rows = []
    for _, r in manifest.iterrows():
        cid = r["custom_id"]
        v = validate_response(raw_by_custom_id.get(cid), schema_by_custom_id[cid])
        rows.append(
            {
                "custom_id": cid,
                "study_id": r["study_id"],
                "profile_id": r["profile_id"],
                "condition_id": r["condition_id"],
                "outcome_id": r["outcome_id"],
                "replicate_id": int(r["replicate_id"]),
                "valid": v["valid"],
                "reason": v["reason"],
                "parsed": v["parsed"],
            }
        )
    return pd.DataFrame(rows)


def invalid_rate(ledger: pd.DataFrame, expected_total: int = EXPECTED_REQUESTS_R1) -> float:
    if len(ledger) != expected_total:
        raise ValueError(f"ledger has {len(ledger)} rows, expected exactly {expected_total} (manifest reconciliation must run first)")
    return float((~ledger["valid"]).sum()) / expected_total


def paired_complete_case_draw(
    ledger: pd.DataFrame,
    *,
    replicate_id: int,
    effect_scale_bounds: dict[str, tuple[float, float]],
    effect_response_field: dict[str, str],
    study_id_by_effect: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One draw's per-effect z_pp DataFrame (study_id, effect_id, z_pp),
    computed ONLY from paired-complete profiles (both control and treatment
    valid) within that single replicate -- unpaired arm means are never
    used, same rule as f_screen_validation.paired_complete_case_effect.
    Returns (draw_df, per_effect_accounting)."""
    draw = ledger[ledger["replicate_id"] == replicate_id]
    rows = []
    accounting: dict[str, Any] = {}
    for effect_id, (low, high) in effect_scale_bounds.items():
        eff = draw[draw["outcome_id"] == effect_id]
        control = eff[eff["condition_id"] == "control"].set_index("profile_id")
        treatment = eff[eff["condition_id"] == "treatment"].set_index("profile_id")
        profiles = sorted(set(control.index) | set(treatment.index))
        field = effect_response_field[effect_id]
        control_native, treatment_native = [], []
        invalid_control_n = invalid_treatment_n = invalid_both_n = 0
        for pid in profiles:
            c_valid = pid in control.index and bool(control.loc[pid, "valid"])
            t_valid = pid in treatment.index and bool(treatment.loc[pid, "valid"])
            if c_valid and t_valid:
                control_native.append(control.loc[pid, "parsed"][field])
                treatment_native.append(treatment.loc[pid, "parsed"][field])
            elif (not c_valid) and t_valid:
                invalid_control_n += 1
            elif c_valid and (not t_valid):
                invalid_treatment_n += 1
            else:
                invalid_both_n += 1
        accounting[effect_id] = {
            "planned_pairs": len(profiles),
            "valid_paired_n": len(control_native),
            "invalid_control_n": invalid_control_n,
            "invalid_treatment_n": invalid_treatment_n,
            "invalid_both_n": invalid_both_n,
        }
        if not control_native:
            raise ValueError(f"replicate {replicate_id}, {effect_id}: zero valid paired profiles, cannot compute z_pp")
        z_pp = f_screen_theta_l_pp(control_native, treatment_native, low, high)
        rows.append({"study_id": study_id_by_effect[effect_id], "effect_id": effect_id, "z_pp": z_pp})
    return pd.DataFrame(rows), accounting


def score_f_reliability_r1_from_raw(
    *,
    manifest: pd.DataFrame,
    raw_by_custom_id: dict[str, dict[str, Any] | None],
    schema_by_custom_id: dict[str, dict[str, Any]],
    effect_scale_bounds: dict[str, tuple[float, float]],
    effect_response_field: dict[str, str],
    study_id_by_effect: dict[str, str],
    outputs_dir,
    expected_total: int = EXPECTED_REQUESTS_R1,
    replicate_ids: tuple[int, int] = (3, 4),
) -> dict[str, Any]:
    """Full R1 pipeline: reconcile -> validate -> per-replicate paired ATEs
    -> ate.r_f_decision.stage_r1_decision (unmodified). Raises
    IntegrityFailure if reconciliation fails; never coerces, clips,
    imputes, or auto-retries. expected_total defaults to the real 24,000 --
    callers with a partial/synthetic ledger (e.g. tests) must pass the
    matching length explicitly.

    replicate_ids defaults to (3, 4) -- the historical R1 draw identity --
    so every existing caller (including the already-scored, already-
    committed historical R1 result) is byte-for-byte unaffected unless it
    explicitly passes different ids. The replacement-R1 provenance
    amendment uses replicate_ids=(5, 6); ate.r_f_decision.stage_r1_decision
    itself was already verified replicate-label-agnostic (it only ever
    receives two per-effect z_pp DataFrames, positionally), so this
    parameterization is plumbing, not a change to the scoring rule."""
    id_a, id_b = replicate_ids
    expected_ids = set(manifest["custom_id"])
    raw_records = [v for v in raw_by_custom_id.values() if v is not None]
    report = reconciliation_report(expected_ids, raw_records)
    enforce_reconciliation(report)

    ledger = build_r1_ledger(manifest, raw_by_custom_id, schema_by_custom_id)
    rate = invalid_rate(ledger, expected_total=expected_total)

    draw_a, acc_a = paired_complete_case_draw(
        ledger, replicate_id=id_a, effect_scale_bounds=effect_scale_bounds, effect_response_field=effect_response_field, study_id_by_effect=study_id_by_effect
    )
    draw_b, acc_b = paired_complete_case_draw(
        ledger, replicate_id=id_b, effect_scale_bounds=effect_scale_bounds, effect_response_field=effect_response_field, study_id_by_effect=study_id_by_effect
    )

    decision = stage_r1_decision(draw_a, draw_b, invalid_response_rate=rate, outputs_dir=outputs_dir)

    return {
        "reconciliation": report,
        "invalid_rate": rate,
        "replicate_ids": [id_a, id_b],
        "per_effect_accounting": {f"replicate_{id_a}": acc_a, f"replicate_{id_b}": acc_b},
        f"draw_replicate_{id_a}": draw_a.to_dict("records"),
        f"draw_replicate_{id_b}": draw_b.to_dict("records"),
        "decision": decision,
    }
