"""ATP G-screen invalid/missing-response handling.

Frozen BEFORE any DeepSeek/Gemma G-screen (ATP1/ATP2) request is submitted,
and before any G-screen scientific output is observed.

Audit finding (2026-08-25): no G-specific numeric invalid-rate threshold
exists anywhere in this repository. The only invalid-rate gate frozen so far
(ate.f_screen_validation.INVALID_RATE_GATE = 0.005, ate.r_f_decision) is
explicitly scoped to the R/F production selection and is not reused here
without a fresh, separate justification -- none exists for G, so none is
invented. The generic (non-F-specific) parts of ate.f_screen_validation --
reconciliation, per-response JSON-Schema validation -- ARE reused directly
via import, since they encode no F-specific assumption (no pairing, no
study/condition vocabulary). ATP has no control/treatment pairing, so
f_screen_validation.paired_complete_case_effect and its module-level
INVALID_RATE_GATE constant are deliberately NOT imported or reused here.

Fail-closed rule (the smallest defensible one that does not invent a
threshold): a candidate with ANY invalid/missing response is never silently
scored as if nothing happened. Only a candidate with invalid_rate == 0 for a
given item may be scored without a further explicit decision. Any nonzero
invalid rate blocks automatic G* selection for that item/model and requires
an explicit, separate, contemporaneous methodological decision (a new
scientific-plan amendment) on how to treat it -- e.g. score on the valid
subset only, discard the item, or re-audit the prompt/schema -- before that
candidate's g_atp_loss is treated as final. There is no automatic paid
retry: retries, if ever authorized, are a new explicit submission decided
after seeing WHY responses were invalid, never an automatic resubmission.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ate.f_screen_validation import (  # noqa: F401 -- re-exported for callers
    IntegrityFailure,
    enforce_reconciliation,
    reconciliation_report,
    validate_response,
)
from ate.g_atp_screen import g_atp_loss, item_w1_pp, model_response_to_unit_interval

# Not a numeric threshold: this is the number of requests per model
# (650 ATP1 + 576 ATP2 = 1226), used only to check the ledger is complete
# before an invalid rate is computed at all -- never used as a pass/fail cut.
EXPECTED_REQUESTS_PER_MODEL_ATP = 1226


def build_g_atp_ledger(
    manifest: pd.DataFrame,
    raw_by_custom_id: dict[str, dict[str, Any] | None],
    schema_by_custom_id: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """One row per manifest custom_id (all EXPECTED_REQUESTS_PER_MODEL_ATP of
    them). manifest must have custom_id, study_id (ATP1/ATP2), and
    profile_id (respondent id) columns -- the same field names
    inference.together_batch.BatchRequest and every other manifest in this
    pipeline already use, with study_id/profile_id carrying the ATP source
    and respondent id rather than F's archive study/profile id. A custom_id
    absent from raw_by_custom_id is passed through as None and classified
    'missing', never silently dropped."""
    rows = []
    for _, r in manifest.iterrows():
        cid = r["custom_id"]
        v = validate_response(raw_by_custom_id.get(cid), schema_by_custom_id[cid])
        rows.append(
            {
                "custom_id": cid,
                "study_id": r["study_id"],
                "profile_id": r["profile_id"],
                "valid": v["valid"],
                "reason": v["reason"],
                "parsed": v["parsed"],
            }
        )
    return pd.DataFrame(rows)


def invalid_rate(ledger: pd.DataFrame, expected_total: int = EXPECTED_REQUESTS_PER_MODEL_ATP) -> float:
    if len(ledger) != expected_total:
        raise ValueError(f"ledger has {len(ledger)} rows, expected exactly {expected_total} (manifest reconciliation must run first)")
    return float((~ledger["valid"]).sum()) / expected_total


def g_invalid_response_report(ledger: pd.DataFrame, expected_total: int = EXPECTED_REQUESTS_PER_MODEL_ATP) -> dict[str, Any]:
    """Full transparent accounting -- never gates or drops anything itself.
    Reports overall and per-item (ATP1/ATP2) invalid rate, effective (valid)
    N per item, a reason-frequency breakdown, and the exact list of invalid
    custom_ids for audit. expected_total is only a completeness check
    (defaults to the real 1226/model) -- callers with a partial/synthetic
    ledger (e.g. tests) must pass the matching length explicitly."""
    overall_rate = invalid_rate(ledger, expected_total=expected_total)
    per_item: dict[str, Any] = {}
    for source_id, group in ledger.groupby("study_id"):
        invalid_mask = ~group["valid"]
        per_item[source_id] = {
            "total": int(len(group)),
            "valid_n": int((~invalid_mask).sum()),
            "invalid_n": int(invalid_mask.sum()),
            "invalid_rate": float(invalid_mask.sum()) / len(group) if len(group) else 0.0,
            "invalid_reasons": {reason: int(n) for reason, n in group.loc[invalid_mask, "reason"].value_counts().items()},
            "invalid_custom_ids": sorted(group.loc[invalid_mask, "custom_id"].tolist()),
        }
    return {
        "total": int(len(ledger)),
        "overall_invalid_rate": overall_rate,
        "per_item": per_item,
    }


def g_fail_closed_decision(report_by_model: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """No invented numeric threshold: a model/item is CLEAN only at exactly
    zero invalid responses. Any nonzero invalid rate on any item blocks that
    model from automatic G* selection until an explicit, separate
    methodological decision documents how to treat it. Reconciliation
    failures (IntegrityFailure) must already have been raised/handled before
    this is called -- this function only classifies observed invalid rates,
    it never re-checks id-set integrity."""
    status_by_model: dict[str, Any] = {}
    for model, report in report_by_model.items():
        dirty_items = [source_id for source_id, item in report["per_item"].items() if item["invalid_n"] > 0]
        status_by_model[model] = {
            "status": "CLEAN_NO_DECISION_NEEDED" if not dirty_items else "STOP_REQUIRE_EXPLICIT_METHODOLOGICAL_DECISION",
            "dirty_items": sorted(dirty_items),
            "overall_invalid_rate": report["overall_invalid_rate"],
        }
    any_dirty = any(s["status"] != "CLEAN_NO_DECISION_NEEDED" for s in status_by_model.values())
    return {
        "decision": "STOP_REQUIRE_EXPLICIT_METHODOLOGICAL_DECISION" if any_dirty else "ALL_CANDIDATES_CLEAN_PROCEED_TO_SCORE",
        "status_by_model": status_by_model,
    }


def score_g_atp_candidate_from_raw(
    *,
    model: str,
    manifest: pd.DataFrame,
    raw_by_custom_id: dict[str, dict[str, Any] | None],
    schema_by_custom_id: dict[str, dict[str, Any]],
    human_reference_positions: dict[str, list[float]],
    choices_by_source: dict[str, tuple[str, ...]],
    response_field: str = "response",
    expected_total: int = EXPECTED_REQUESTS_PER_MODEL_ATP,
) -> dict[str, Any]:
    """The production entry point: raw retrieved batch output -> reconcile ->
    validate -> per-item normalized W1 on the VALID subset only -> g_atp_loss.

    human_reference_positions/choices_by_source are candidate-independent
    (computed once from the human ATP data via
    ate.g_atp_screen.atp{1,2}_human_reference_positions and
    ATP{1,2}_SUBSTANTIVE_CHOICES) and are never adjusted per candidate.
    Diagnostics (mean error, variance ratio, KS, subgroup error, response
    frequencies) are NOT computed here -- this function returns only what
    feeds the frozen primary metric, so nothing diagnostic can leak into
    selection by accident.

    g_atp_loss is always computed and returned (transparency), but is_final
    is False whenever this candidate has ANY invalid response on ANY item --
    per the frozen fail-closed rule, a non-final score must not be consumed
    by automatic G* selection until an explicit follow-up decision resolves
    it. Raises IntegrityFailure if reconciliation fails."""
    expected_ids = set(manifest["custom_id"])
    raw_records = [v for v in raw_by_custom_id.values() if v is not None]
    report = reconciliation_report(expected_ids, raw_records)
    enforce_reconciliation(report)

    ledger = build_g_atp_ledger(manifest, raw_by_custom_id, schema_by_custom_id)
    invalid_report = g_invalid_response_report(ledger, expected_total=expected_total)

    w1_pp_by_source: dict[str, float] = {}
    for source_id, choices in choices_by_source.items():
        valid_rows = ledger[(ledger["study_id"] == source_id) & ledger["valid"]]
        model_positions = [model_response_to_unit_interval(int(row["parsed"][response_field]), choices) for _, row in valid_rows.iterrows()]
        if not model_positions:
            raise ValueError(f"{model}/{source_id}: zero valid model responses, cannot compute W1")
        w1_pp_by_source[source_id] = item_w1_pp(human_reference_positions[source_id], model_positions)

    loss = g_atp_loss(w1_pp_by_source["ATP1"], w1_pp_by_source["ATP2"])
    decision = g_fail_closed_decision({model: invalid_report})
    is_final = decision["status_by_model"][model]["status"] == "CLEAN_NO_DECISION_NEEDED"

    return {
        "model": model,
        "reconciliation": report,
        "invalid_report": invalid_report,
        "w1_pp_atp1": w1_pp_by_source["ATP1"],
        "w1_pp_atp2": w1_pp_by_source["ATP2"],
        "g_atp_loss": loss,
        "is_final": is_final,
        "decision_status": decision["status_by_model"][model]["status"],
    }
