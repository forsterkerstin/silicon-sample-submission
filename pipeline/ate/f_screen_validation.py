"""Canonical F-screen scientific-output validation/accounting path.

Frozen per model_config.yaml's scientific_plan_amendments entry
F_SCREEN_INVALID_RESPONSE_HANDLING_FROZEN (2026-08-25), BEFORE any
DeepSeek/Gemma F-screen scientific results were observed or retrieved. This
module is the ONLY path that may turn raw retrieved F-screen batch output
into scoreable per-effect data; it feeds exclusively into the unmodified
pipeline.ate.f_screen scoring functions (score_f_screen_candidate,
select_f_star) -- nothing here alters the 12 effects, N=250, profiles,
prompt content, model configuration, theta normalization, study-equal RMSE,
diagnostics, or tie-break order.

Pipeline: reconcile expected ids -> validate every response against its own
request-specific JSON Schema -> gate candidate eligibility on invalid rate ->
build paired-complete-case per-effect ATEs -> hand off to ate.f_screen.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import jsonschema
import numpy as np
import pandas as pd

from ate.normalize_effects import to_percent_of_range
from inference.together_batch import _message_content_from_output

INVALID_RATE_GATE = 0.005
EXPECTED_REQUESTS_PER_MODEL = 6000


class IntegrityFailure(Exception):
    """Retrieved batch output failed reconciliation (unexpected/duplicate
    custom_ids) -- stop rather than score. A re-download of the same
    already-submitted batch is not a scientific retry and never raises this
    on its own; only genuinely unexpected/duplicate ids do."""


def reconciliation_report(expected_custom_ids: set[str], raw_records: list[dict[str, Any]]) -> dict[str, Any]:
    """raw_records: one dict per line of the retrieved batch_output.jsonl,
    already json.loads()'d. A record with no 'custom_id' key at all is
    itself a malformed provider record, reported separately from id-set
    mismatches."""
    malformed_records = [r for r in raw_records if not isinstance(r, dict) or "custom_id" not in r]
    present = [r["custom_id"] for r in raw_records if isinstance(r, dict) and "custom_id" in r]
    counts = Counter(present)
    duplicate = sorted(cid for cid, n in counts.items() if n > 1)
    unexpected = sorted(set(present) - expected_custom_ids)
    missing_entirely = sorted(expected_custom_ids - set(present))
    return {
        "missing_entirely": missing_entirely,
        "unexpected": unexpected,
        "duplicate": duplicate,
        "malformed_records": malformed_records,
        "integrity_ok": not duplicate and not unexpected,
    }


def enforce_reconciliation(report: dict[str, Any]) -> None:
    if not report["integrity_ok"]:
        raise IntegrityFailure(
            f"F-screen batch failed reconciliation: unexpected={report['unexpected'][:5]}, "
            f"duplicate={report['duplicate'][:5]} -- stop rather than score"
        )


def validate_response(raw_record: dict[str, Any] | None, schema: dict[str, Any]) -> dict[str, Any]:
    """raw_record is None for a custom_id that is missing entirely from the
    retrieved batch output. Never coerces, clips, rounds, repairs, or
    imputes -- only classifies valid vs. invalid (with a reason)."""
    if raw_record is None:
        return {"valid": False, "reason": "missing", "parsed": None}
    response_obj = raw_record.get("response") if isinstance(raw_record.get("response"), dict) else None
    if "error" in raw_record and raw_record["error"]:
        return {"valid": False, "reason": "terminal_provider_failure", "parsed": None}
    if response_obj is not None and response_obj.get("status_code", 200) != 200:
        return {"valid": False, "reason": "terminal_provider_failure", "parsed": None}
    content = _message_content_from_output(raw_record)
    if content is None:
        return {"valid": False, "reason": "missing", "parsed": None}
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        return {"valid": False, "reason": f"malformed_json: {exc}", "parsed": None}
    try:
        jsonschema.validate(parsed, schema)
    except jsonschema.ValidationError as exc:
        return {"valid": False, "reason": f"schema_invalid: {exc.message}", "parsed": None}
    return {"valid": True, "reason": "", "parsed": parsed}


def build_ledger(
    manifest: pd.DataFrame,
    raw_by_custom_id: dict[str, dict[str, Any] | None],
    schema_by_custom_id: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """One row per manifest custom_id (all EXPECTED_REQUESTS_PER_MODEL of
    them) -- a custom_id absent from raw_by_custom_id is passed through as
    None and classified 'missing', never silently dropped from the ledger."""
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
                "valid": v["valid"],
                "reason": v["reason"],
                "parsed": v["parsed"],
            }
        )
    return pd.DataFrame(rows)


def invalid_rate(ledger: pd.DataFrame, expected_total: int = EXPECTED_REQUESTS_PER_MODEL) -> float:
    if len(ledger) != expected_total:
        raise ValueError(f"ledger has {len(ledger)} rows, expected exactly {expected_total} (manifest reconciliation must run first)")
    return float((~ledger["valid"]).sum()) / expected_total


def gate_decision(invalid_rate_by_model: dict[str, float]) -> dict[str, Any]:
    """Frozen decision rule: both eligible -> score both; exactly one
    eligible -> that one is F*; neither eligible -> stop. No additional
    invalid-rate threshold is used."""
    eligible = {m: r <= INVALID_RATE_GATE for m, r in invalid_rate_by_model.items()}
    n_eligible = sum(eligible.values())
    if n_eligible == 0:
        return {"decision": "STOP_REQUIRE_NEW_EXPLICIT_DECISION", "eligible_models": [], "f_star": None, "invalid_rate_by_model": invalid_rate_by_model}
    if n_eligible == 1:
        winner = next(m for m, ok in eligible.items() if ok)
        return {"decision": "SINGLE_ELIGIBLE_CANDIDATE_IS_F_STAR", "eligible_models": [winner], "f_star": winner, "invalid_rate_by_model": invalid_rate_by_model}
    return {
        "decision": "SCORE_BOTH_ELIGIBLE_CANDIDATES",
        "eligible_models": sorted(m for m, ok in eligible.items() if ok),
        "f_star": None,
        "invalid_rate_by_model": invalid_rate_by_model,
    }


def paired_complete_case_effect(
    ledger: pd.DataFrame,
    effect_id: str,
    *,
    response_field: str,
    scale_low: float,
    scale_high: float,
) -> dict[str, Any]:
    """Paired-complete-case ATE for one effect: a profile contributes to
    z_e,m = mean_i(Y_T - Y_C) only if BOTH its control and treatment
    responses are valid. Unpaired arm means (dropping a profile from only
    the arm where it's invalid) are explicitly not used when missingness
    differs between arms."""
    eff = ledger[ledger["outcome_id"] == effect_id]
    control = eff[eff["condition_id"] == "control"].set_index("profile_id")
    treatment = eff[eff["condition_id"] == "treatment"].set_index("profile_id")
    profiles = sorted(set(control.index) | set(treatment.index))
    planned_pairs = len(profiles)
    valid_paired_n = invalid_control_n = invalid_treatment_n = invalid_both_n = 0
    diffs: list[float] = []
    for pid in profiles:
        c_valid = pid in control.index and bool(control.loc[pid, "valid"])
        t_valid = pid in treatment.index and bool(treatment.loc[pid, "valid"])
        if c_valid and t_valid:
            valid_paired_n += 1
            diffs.append(treatment.loc[pid, "parsed"][response_field] - control.loc[pid, "parsed"][response_field])
        elif (not c_valid) and t_valid:
            invalid_control_n += 1
        elif c_valid and (not t_valid):
            invalid_treatment_n += 1
        else:
            invalid_both_n += 1
    theta_l_pp = to_percent_of_range(float(np.mean(diffs)), scale_low, scale_high) if diffs else None
    return {
        "effect_id": effect_id,
        "planned_pairs": planned_pairs,
        "valid_paired_n": valid_paired_n,
        "invalid_control_n": invalid_control_n,
        "invalid_treatment_n": invalid_treatment_n,
        "invalid_both_n": invalid_both_n,
        "theta_l_pp": theta_l_pp,
    }


def score_candidate_from_raw(
    *,
    model: str,
    manifest: pd.DataFrame,
    raw_by_custom_id: dict[str, dict[str, Any] | None],
    schema_by_custom_id: dict[str, dict[str, Any]],
    effect_scale_bounds: dict[str, tuple[float, float]],
    effect_response_field: dict[str, str],
    theta_h_pp_by_effect: dict[str, float],
    study_id_by_effect: dict[str, str],
) -> dict[str, Any]:
    """Full pipeline for one candidate: reconcile -> validate -> paired
    ATEs -> feed into the unmodified ate.f_screen scorer. Raises
    IntegrityFailure if reconciliation fails."""
    from ate.f_screen import score_f_screen_candidate

    expected_ids = set(manifest["custom_id"])
    raw_records = [v for v in raw_by_custom_id.values() if v is not None]
    report = reconciliation_report(expected_ids, raw_records)
    enforce_reconciliation(report)

    ledger = build_ledger(manifest, raw_by_custom_id, schema_by_custom_id)
    rate = invalid_rate(ledger)

    per_effect_accounting = {}
    score_rows = []
    for effect_id, (low, high) in effect_scale_bounds.items():
        detail = paired_complete_case_effect(
            ledger, effect_id, response_field=effect_response_field[effect_id], scale_low=low, scale_high=high
        )
        per_effect_accounting[effect_id] = detail
        if detail["theta_l_pp"] is not None:
            score_rows.append(
                {
                    "study_id": study_id_by_effect[effect_id],
                    "effect_id": effect_id,
                    "theta_l_pp": detail["theta_l_pp"],
                    "theta_h_pp": theta_h_pp_by_effect[effect_id],
                }
            )

    scoreable_effects = [r["effect_id"] for r in score_rows]
    unscoreable_effects = [e for e in effect_scale_bounds if e not in scoreable_effects]
    score = score_f_screen_candidate(score_rows) if score_rows else None

    return {
        "model": model,
        "reconciliation": report,
        "invalid_rate": rate,
        "gate_eligible": rate <= INVALID_RATE_GATE,
        "per_effect_accounting": per_effect_accounting,
        "scoreable_effects": scoreable_effects,
        "unscoreable_effects": unscoreable_effects,
        "score": score,
    }
