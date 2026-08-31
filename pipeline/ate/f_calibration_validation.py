"""F* external-calibration production raw-output validation/scoring path.

Frozen BEFORE any calibration inference is submitted or observed -- built and
synthetic-tested only, exactly like ate/f_reliability_validation.py was built
and synthetic-tested before R1 inference. Reuses that module's reconciliation,
per-response JSON-Schema validation, and ledger construction unmodified; the
only structural difference is that calibration is a SINGLE draw (R_F=1, no
replicate-vs-replicate comparison) computed per effect against that effect's
OWN native outcome scale (data/ate_archive.csv outcome_min/outcome_max), not
a shared benchmark-composite scale.

Pipeline: reconcile expected ids -> validate every response against its own
request-specific JSON Schema -> per-effect paired-complete-case raw ATE
(native units) and percent-of-range ATE, using the SAME formula
(ate.f_screen.f_screen_theta_l_pp: 100 * (mean(treatment) - mean(control)) /
range) already frozen and used by F-screen and R1. No coercion, clipping,
imputation, or automatic retries anywhere in this path. This module does NOT
fit M0/M1/M2 (ate/calibrate_lambda.py, unmodified, unchanged) -- it only
produces the per-effect synthetic ATE those models consume.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ate.f_reliability_validation import build_r1_ledger, invalid_rate  # noqa: F401 -- re-exported for callers
from ate.f_screen_validation import (  # noqa: F401 -- re-exported for callers
    IntegrityFailure,
    enforce_reconciliation,
    reconciliation_report,
    validate_response,
)
from ate.normalize_effects import to_percent_of_range


def paired_complete_case_effect_native(
    ledger: pd.DataFrame,
    *,
    replicate_id: int,
    effect_native_bounds: dict[str, tuple[float, float]],
    effect_response_field: dict[str, str],
    study_id_by_effect: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One effect's raw native-scale synthetic ATE (z_se_native) and its
    percent-of-range normalization (theta_l_pp), computed ONLY from
    paired-complete profiles (both control and treatment valid) -- unpaired
    arm means are never used, same rule as
    f_reliability_validation.paired_complete_case_draw and
    f_screen_validation.paired_complete_case_effect. Returns
    (per_effect_df[study_id, effect_id, z_se_native, theta_l_pp,
    paired_n], per_effect_accounting)."""
    draw = ledger[ledger["replicate_id"] == replicate_id]
    rows = []
    accounting: dict[str, Any] = {}
    for effect_id, (low, high) in effect_native_bounds.items():
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
            raise ValueError(f"replicate {replicate_id}, {effect_id}: zero valid paired profiles, cannot compute z_se")
        z_se_native = float(np.mean(treatment_native)) - float(np.mean(control_native))
        theta_l_pp = to_percent_of_range(z_se_native, low, high)
        rows.append(
            {
                "study_id": study_id_by_effect[effect_id],
                "effect_id": effect_id,
                "z_se_native": z_se_native,
                "theta_l_pp": theta_l_pp,
                "paired_n": len(control_native),
            }
        )
    return pd.DataFrame(rows), accounting


def score_f_calibration_production_from_raw(
    *,
    manifest: pd.DataFrame,
    raw_by_custom_id: dict[str, dict[str, Any] | None],
    schema_by_custom_id: dict[str, dict[str, Any]],
    effect_native_bounds: dict[str, tuple[float, float]],
    effect_response_field: dict[str, str],
    study_id_by_effect: dict[str, str],
    expected_total: int,
    replicate_id: int = 7,
) -> dict[str, Any]:
    """Full calibration-production scoring pipeline: reconcile -> validate ->
    per-effect paired-complete-case raw+pp ATE. Raises IntegrityFailure if
    reconciliation fails; never coerces, clips, imputes, or auto-retries.
    Does NOT call any calibration-model fitter (M0/M1/M2 live in
    ate/calibrate_lambda.py, downstream and unmodified) and does NOT write to
    data/ate_archive.csv -- this function only produces the per-effect
    synthetic-ATE table a future write-back step would consume."""
    expected_ids = set(manifest["custom_id"])
    raw_records = [v for v in raw_by_custom_id.values() if v is not None]
    report = reconciliation_report(expected_ids, raw_records)
    enforce_reconciliation(report)

    ledger = build_r1_ledger(manifest, raw_by_custom_id, schema_by_custom_id)
    rate = invalid_rate(ledger, expected_total=expected_total)

    per_effect_df, accounting = paired_complete_case_effect_native(
        ledger,
        replicate_id=replicate_id,
        effect_native_bounds=effect_native_bounds,
        effect_response_field=effect_response_field,
        study_id_by_effect=study_id_by_effect,
    )

    return {
        "reconciliation": report,
        "invalid_rate": rate,
        "replicate_id": replicate_id,
        "per_effect_accounting": accounting,
        "per_effect_ate": per_effect_df.to_dict("records"),
    }
