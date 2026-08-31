"""Validation for this pipeline's Tier-1 native-response output.

The benchmark's R validator checks deposited CSV structure. This module
adds pipeline-specific checks that are easier to do before writing:
integer support for raw native items, composite consistency, 16 x 13 ATE
coverage, state-aware stimulus resolvability, and target-vs-achieved arm
calibration diagnostics.
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import survey_content as sc
from ate.estimate_ates import estimate_raw_ates

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "config" / "benchmark_schema.yaml"


def validate_tier1(
    df: pd.DataFrame,
    *,
    calibration_diagnostics: pd.DataFrame | None = None,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Raise ValueError on invalid Tier-1 output; return a compact report on
    success. Extra columns are allowed because this pipeline intentionally
    carries raw items alongside the required submitted columns."""
    schema = _load_schema()
    items = sc.load_items()
    item_by_label = {item["target_label"]: item for item in items}
    trust_items = sc.OUTCOME_COMPOSITES["trust_multidimensional"][1]
    required = [
        "profile_id",
        "condition",
        *schema["moderators"].keys(),
        "trust_multidimensional",
        *trust_items,
        "trust_post",
        "distrust_post",
        "funding_perceptions",
        "policy_role_mean",
        "inst_trust_mean",
        "belief_post",
        "concern_mean",
        "policy_general",
        "policy_specific_mean",
        "behavior_mean",
        "donation_ams",
        "newsletter_signup",
    ]

    problems: list[str] = []
    _check_required_columns(df, required, problems)
    _check_no_missing(df, required, problems)
    _check_unique_profile_ids(df, problems)
    _check_conditions(df, schema["conditions"], problems)
    _check_demographics(df, schema["moderators"], problems)
    _check_raw_item_support(df, item_by_label, problems)
    composite_errors = _check_composites(df, problems, tolerance)
    outcome_ates = _check_ate_grid(df, schema["conditions"], problems)
    _check_state_specific_stimuli(df, problems)
    max_calibration_error = _check_calibration_diagnostics(calibration_diagnostics, problems)

    if problems:
        raise ValueError("Tier-1 validation failed:\n- " + "\n- ".join(problems))

    return {
        "n_rows": len(df),
        "n_conditions": df["condition"].nunique() if "condition" in df else 0,
        "n_outcome_ates": len(outcome_ates),
        "max_composite_error": composite_errors,
        "max_target_ate_error": max_calibration_error,
        "integer_item_checks": len([label for label in item_by_label if label in df.columns]),
    }


def _load_schema() -> dict[str, Any]:
    return yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))


def _check_required_columns(df: pd.DataFrame, required: list[str], problems: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        problems.append(f"missing required columns: {missing}")


def _check_no_missing(df: pd.DataFrame, columns: list[str], problems: list[str]) -> None:
    for col in columns:
        if col in df.columns and df[col].isna().any():
            problems.append(f"{col} has {int(df[col].isna().sum())} missing value(s)")


def _check_unique_profile_ids(df: pd.DataFrame, problems: list[str]) -> None:
    if "profile_id" in df and df["profile_id"].duplicated().any():
        problems.append(f"profile_id has {int(df['profile_id'].duplicated().sum())} duplicate(s)")


def _check_conditions(df: pd.DataFrame, conditions: list[str], problems: list[str]) -> None:
    if "condition" not in df:
        return
    observed = set(df["condition"].astype(str))
    expected = set(conditions)
    if observed - expected:
        problems.append(f"unknown condition labels: {sorted(observed - expected)}")
    if expected - observed:
        problems.append(f"missing condition labels: {sorted(expected - observed)}")
    if any(df.groupby("condition").size() == 0):
        problems.append("one or more conditions has zero rows")


def _check_demographics(df: pd.DataFrame, moderators: dict[str, list[str]], problems: list[str]) -> None:
    for col, allowed in moderators.items():
        if col not in df:
            continue
        bad = sorted(set(df[col].dropna().astype(str)) - set(allowed))
        if bad:
            problems.append(f"{col} has invalid level(s): {bad}")


def _check_raw_item_support(df: pd.DataFrame, item_by_label: dict[str, dict[str, str]], problems: list[str]) -> None:
    for label, item in item_by_label.items():
        if label not in df:
            problems.append(f"missing raw item column: {label}")
            continue
        values = df[label]
        if not values.map(lambda x: isinstance(x, Integral) and not isinstance(x, bool)).all():
            problems.append(f"{label} contains non-integer raw response(s)")
            continue
        if item["scale"] == sc.SCALE_SLIDER_0_100 and not values.between(0, 100).all():
            problems.append(f"{label} has value(s) outside [0, 100]")
        elif item["scale"] == sc.SCALE_DONATION_0_10 and not values.between(0, 10).all():
            problems.append(f"{label} has value(s) outside [0, 10]")
        elif item["scale"] == sc.SCALE_BINARY_0_1 and not values.isin([0, 1]).all():
            problems.append(f"{label} has non-binary value(s)")


def _check_composites(df: pd.DataFrame, problems: list[str], tolerance: float) -> float:
    max_error = 0.0
    missing_raw = [label for _, spec in sc.OUTCOME_COMPOSITES.values() for label in (spec if isinstance(spec, list) else [spec]) if label not in df]
    if missing_raw:
        problems.append(f"cannot check composites; missing raw columns: {sorted(set(missing_raw))}")
        return max_error
    for i, row in df.iterrows():
        expected = sc.compute_outcomes(row.to_dict())
        for outcome, value in expected.items():
            if outcome not in df:
                problems.append(f"missing composite/outcome column: {outcome}")
                continue
            err = abs(float(row[outcome]) - float(value))
            max_error = max(max_error, err)
            if err > tolerance:
                problems.append(f"row {i} {outcome} differs from computed value by {err}")
                return max_error
    return max_error


def _check_ate_grid(df: pd.DataFrame, conditions: list[str], problems: list[str]) -> pd.DataFrame:
    outcomes = list(sc.OUTCOME_COMPOSITES.keys())
    if "condition" not in df or any(outcome not in df for outcome in outcomes):
        return pd.DataFrame()
    ates = estimate_raw_ates(df, outcomes)
    expected = (len(conditions) - 1) * len(outcomes)
    if len(ates) != expected:
        problems.append(f"expected {expected} condition-outcome ATEs, got {len(ates)}")
    if not ates[["raw_ate", "control_mean", "treatment_mean"]].notna().all().all():
        problems.append("one or more ATE means is missing")
    return ates


def _check_state_specific_stimuli(df: pd.DataFrame, problems: list[str]) -> None:
    if "state_abbr" not in df or "condition" not in df:
        return
    bad_states = sorted(set(df["state_abbr"].dropna().astype(str)) - set(sc.STATE_NAME_TO_ABBR.values()))
    if bad_states:
        problems.append(f"invalid state_abbr value(s): {bad_states}")
        return
    extreme_states = sorted(set(df.loc[df["condition"] == "Extreme weather predictions", "state_abbr"].dropna().astype(str)))
    for state in extreme_states:
        if not sc.get_condition_stimulus("Extreme weather predictions", state).strip():
            problems.append(f"empty extreme-weather stimulus for state {state}")


def _check_calibration_diagnostics(calibration_diagnostics: pd.DataFrame | None, problems: list[str]) -> float | None:
    if calibration_diagnostics is None:
        return None
    required = {"condition", "outcome", "target_ate", "achieved_ate", "absolute_error"}
    missing = required - set(calibration_diagnostics.columns)
    if missing:
        problems.append(f"calibration diagnostics missing column(s): {sorted(missing)}")
        return None
    if calibration_diagnostics.empty:
        problems.append("calibration diagnostics are empty")
        return None
    if not calibration_diagnostics["absolute_error"].map(pd.notna).all():
        problems.append("calibration diagnostics contain missing absolute_error")
    return float(calibration_diagnostics["absolute_error"].max())
