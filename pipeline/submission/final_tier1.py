"""Authoritative Tier-1 assembly: native full G + common F/C ATE shift."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import survey_content as sc
from ate.estimate_ates import estimate_raw_ates
from ate.normalize_effects import OUTCOME_SCALE_BOUNDS, to_percent_of_range
from ate.target_effects import (
    apply_calibration_to_target_ates,
    estimate_target_ates_from_f,
    write_f_stability_diagnostics,
)
from calibration.project_support import (
    bounded_integer_total,
    project_binary_to_count,
    project_integer_to_total,
    project_matrix_to_composite_total,
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = PIPELINE_ROOT / "config" / "benchmark_schema.yaml"
OUTPUTS_DIR = PIPELINE_ROOT / "outputs"
MODERATORS = ["gender", "age_band", "race", "education", "income", "party"]


def _schema() -> dict[str, Any]:
    return yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))


def _donor_key(df: pd.DataFrame) -> pd.Series:
    for col in ("donor_key", "latent_profile_id"):
        if col in df.columns:
            return df[col].astype(str)
    raise ValueError("native G responses need donor_key or latent_profile_id")


def _recompute_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    raw_needed = {
        label
        for _, spec in sc.OUTCOME_COMPOSITES.values()
        for label in (spec if isinstance(spec, list) else [spec])
    }
    absent = sorted(raw_needed - set(df.columns))
    if absent:
        raise ValueError(f"cannot compute outcomes; missing raw item column(s): {absent}")
    outcomes = pd.DataFrame([sc.compute_outcomes(row.to_dict()) for _, row in df.iterrows()], index=df.index)
    out = df.drop(columns=[c for c in outcomes.columns if c in df.columns]).copy()
    for col in outcomes.columns:
        out[col] = outcomes[col]
    return out


def _official_columns() -> list[str]:
    schema = _schema()
    trust_items = sc.OUTCOME_COMPOSITES["trust_multidimensional"][1]
    return [
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


def _raw_item_overlap_check() -> None:
    seen: dict[str, str] = {}
    for outcome, (kind, spec) in sc.OUTCOME_COMPOSITES.items():
        labels = spec if kind == "mean" else [spec]
        for label in labels:
            if label in seen and seen[label] != outcome:
                raise ValueError(f"raw item {label!r} appears in multiple scored constructs: {seen[label]!r}, {outcome!r}")
            seen[label] = outcome


def _prepare_g_native(g_native: pd.DataFrame) -> pd.DataFrame:
    out = _recompute_outcomes(g_native).copy()
    out["donor_key"] = _donor_key(out).to_numpy()
    if "state" not in out.columns:
        if "state_abbr" not in out.columns:
            raise ValueError("native G donor metadata must include state or state_abbr")
        out["state"] = out["state_abbr"]
    if out["state"].isna().any():
        raise ValueError("state is mandatory for every native G donor row")
    if "profile_id" not in out.columns:
        safe = out["condition"].astype(str).str.replace(r"[^A-Za-z0-9]+", "_", regex=True).str.strip("_")
        out["profile_id"] = out["donor_key"] + "__" + safe
    return out


def _validate_full_g(g: pd.DataFrame, *, expected_n_g: int | None) -> None:
    schema = _schema()
    conditions = schema["conditions"]
    if set(g["condition"].astype(str)) != set(conditions):
        raise ValueError("native G responses must contain control and all 16 treatments")
    donors = sorted(g["donor_key"].unique())
    if expected_n_g is not None and len(donors) != expected_n_g:
        raise ValueError(f"expected N_G={expected_n_g}, got {len(donors)}")
    counts = g.groupby("condition")["donor_key"].nunique()
    for condition in conditions:
        if counts.get(condition, 0) != len(donors):
            raise ValueError(f"condition {condition!r} has {counts.get(condition, 0)} donors, expected {len(donors)}")
    per_cell = g.groupby(["condition", "donor_key"]).size()
    if not (per_cell == 1).all():
        raise ValueError("every donor must appear exactly once in every native G condition")
    attrs = [c for c in [*MODERATORS, "state", "state_abbr"] if c in g.columns]
    if not (g.groupby("donor_key")[attrs].nunique(dropna=False) == 1).all().all():
        raise ValueError("donor demographics/state are not identical across native G conditions")
    if g["profile_id"].duplicated().any():
        raise ValueError("native G profile_id values must be globally unique")


def _compute_raw_g_ates(g: pd.DataFrame, outputs_dir: Path) -> pd.DataFrame:
    control = g[g["condition"] == "control"].set_index("donor_key")
    rows = []
    for condition in _schema()["conditions"]:
        if condition == "control":
            continue
        tx = g[g["condition"] == condition].set_index("donor_key").reindex(control.index)
        if tx.isna().any().any():
            raise ValueError(f"native G treatment {condition!r} is missing donor rows")
        for outcome, (low, high) in OUTCOME_SCALE_BOUNDS.items():
            paired = tx[outcome].astype(float) - control[outcome].astype(float)
            paired_ate = float(paired.mean())
            mean_diff = float(tx[outcome].mean() - control[outcome].mean())
            if not np.isclose(paired_ate, mean_diff, atol=1e-10):
                raise ValueError(f"paired and arm-mean raw G ATE disagree for {condition}/{outcome}")
            rows.append(
                {
                    "condition": condition,
                    "outcome": outcome,
                    "outcome_range": high - low,
                    "raw_g_control_mean": float(control[outcome].mean()),
                    "raw_g_treatment_mean": float(tx[outcome].mean()),
                    "raw_g_ate_native": paired_ate,
                    "raw_g_ate_pp": to_percent_of_range(paired_ate, low, high),
                }
            )
    out = pd.DataFrame(rows)
    if len(out) != 16 * 13:
        raise ValueError(f"expected 208 raw G ATEs, got {len(out)}")
    out.to_csv(outputs_dir / "raw_g_ates.csv", index=False)
    return out


def _calibrated_lookup(calibrated: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    return {(r["condition"], r["outcome"]): r for _, r in calibrated.iterrows()}


def _rank_vector(values: pd.Series) -> np.ndarray:
    return values.astype(float).rank(method="first").to_numpy()


def _assert_common_shift_identities(
    *,
    condition: str,
    outcome: str,
    control: pd.Series,
    native_treatment: pd.Series,
    ideal: np.ndarray,
    raw_g_ate: float,
    target_ate: float,
) -> None:
    pre_ate = float(np.mean(ideal) - control.mean())
    if not np.isclose(pre_ate, target_ate, atol=1e-9):
        raise ValueError(f"pre-projection ATE does not equal calibrated target for {condition}/{outcome}")
    left = (ideal - control.to_numpy(dtype=float)) - target_ate
    right = (native_treatment.to_numpy(dtype=float) - control.to_numpy(dtype=float)) - raw_g_ate
    if not np.allclose(left, right, atol=1e-9):
        raise ValueError(f"centered native G treatment heterogeneity changed before projection for {condition}/{outcome}")
    if not np.isclose(np.var(ideal, ddof=0), np.var(native_treatment.to_numpy(dtype=float), ddof=0), atol=1e-9):
        raise ValueError(f"common shift changed treatment variance before projection for {condition}/{outcome}")
    if not np.array_equal(_rank_vector(pd.Series(ideal)), _rank_vector(native_treatment)):
        raise ValueError(f"common shift changed treatment rank ordering before projection for {condition}/{outcome}")


def _project_outcome(
    *,
    native_tx: pd.DataFrame,
    control: pd.DataFrame,
    final_tx: pd.DataFrame,
    shifted_tx: pd.DataFrame,
    condition: str,
    outcome: str,
    raw_g_ate: float,
    target_ate: float,
) -> dict[str, float]:
    common_shift = target_ate - raw_g_ate
    kind, spec = sc.OUTCOME_COMPOSITES[outcome]
    control_outcome = control[outcome].astype(float)
    native_outcome = native_tx[outcome].astype(float)
    ideal_outcome = native_outcome.to_numpy(dtype=float) + common_shift
    _assert_common_shift_identities(
        condition=condition,
        outcome=outcome,
        control=control_outcome,
        native_treatment=native_outcome,
        ideal=ideal_outcome,
        raw_g_ate=raw_g_ate,
        target_ate=target_ate,
    )
    target_total = len(control) * (float(control_outcome.mean()) + target_ate)

    if kind == "item":
        if outcome == "newsletter_signup":
            projected = project_binary_to_count(ideal_outcome, target_count=target_total)
            nearest_total = bounded_integer_total(len(control), 0, 1, target_total)
        else:
            low, high = OUTCOME_SCALE_BOUNDS[outcome]
            projected = project_integer_to_total(ideal_outcome, low=int(low), high=int(high), target_total=target_total)
            nearest_total = bounded_integer_total(len(control), int(low), int(high), target_total)
        final_tx[spec] = projected
        final_tx[outcome] = projected
        shifted_tx[spec] = ideal_outcome
        shifted_tx[outcome] = ideal_outcome
        final_values = final_tx[outcome].astype(float).to_numpy()
    elif kind == "reverse_100":
        raw_ideal = native_tx[spec].astype(float).to_numpy() - common_shift
        raw_target_total = len(control) * 100 - target_total
        projected_raw = project_integer_to_total(raw_ideal, low=0, high=100, target_total=raw_target_total)
        final_tx[spec] = projected_raw
        final_tx[outcome] = 100 - projected_raw
        shifted_tx[spec] = raw_ideal
        shifted_tx[outcome] = ideal_outcome
        final_values = final_tx[outcome].astype(float).to_numpy()
        nearest_total = int(len(control) * 100 - projected_raw.sum())
    elif kind == "mean":
        matrix_ideal = native_tx[spec].to_numpy(dtype=float) + common_shift
        projected_matrix = project_matrix_to_composite_total(matrix_ideal, low=0, high=100, target_total=target_total * len(spec))
        for j, label in enumerate(spec):
            final_tx[label] = projected_matrix[:, j]
            shifted_tx[label] = matrix_ideal[:, j]
        final_tx[outcome] = projected_matrix.mean(axis=1)
        shifted_tx[outcome] = ideal_outcome
        final_values = final_tx[outcome].astype(float).to_numpy()
        nearest_total = int(projected_matrix.sum() / len(spec))
    else:
        raise ValueError(f"unknown composite kind {kind!r}")

    if not np.isclose(float(np.mean(ideal_outcome) - control_outcome.mean()), target_ate, atol=1e-9):
        raise ValueError(f"common-shift target equality failed for {condition}/{outcome}")
    final_ate = float(final_tx[outcome].mean() - control_outcome.mean())
    low, high = OUTCOME_SCALE_BOUNDS[outcome]
    return {
        "common_shift": float(common_shift),
        "pre_projection_ate": float(np.mean(ideal_outcome) - control_outcome.mean()),
        "final_submission_ate": final_ate,
        "target_minus_final": float(target_ate - final_ate),
        "nearest_attainable_total": float(nearest_total),
        "projection_fraction_modified": float(np.mean(np.abs(final_values - ideal_outcome) > 1e-9)),
        "lower_bound_fraction": float(np.mean(np.isclose(final_values, low))),
        "upper_bound_fraction": float(np.mean(np.isclose(final_values, high))),
    }


def _interaction_table(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    control = df[df["condition"] == "control"].set_index("donor_key")
    rows = []
    for condition in _schema()["conditions"]:
        if condition == "control":
            continue
        tx = df[df["condition"] == condition].set_index("donor_key").reindex(control.index)
        for outcome in OUTCOME_SCALE_BOUNDS:
            delta = tx[outcome].astype(float) - control[outcome].astype(float)
            overall = float(delta.mean())
            for moderator in MODERATORS:
                for level in sorted(control[moderator].dropna().astype(str).unique()):
                    keys = control.index[control[moderator].astype(str) == level]
                    rows.append(
                        {
                            "stage": stage,
                            "condition": condition,
                            "outcome": outcome,
                            "moderator": moderator,
                            "level": level,
                            "n": len(keys),
                            "interaction": float(delta.loc[keys].mean() - overall),
                        }
                    )
    return pd.DataFrame(rows)


def _write_interaction_diagnostics(native_g: pd.DataFrame, shifted: pd.DataFrame, final_with_key: pd.DataFrame, outputs_dir: Path) -> pd.DataFrame:
    native = _interaction_table(native_g, "native_g")
    pre = _interaction_table(shifted, "pre_projection")
    final = _interaction_table(final_with_key, "final_projected")
    merged = native.rename(columns={"interaction": "native_interaction"}).drop(columns=["stage"]).merge(
        pre.rename(columns={"interaction": "pre_projection_interaction"}).drop(columns=["stage"]),
        on=["condition", "outcome", "moderator", "level", "n"],
        how="inner",
    ).merge(
        final.rename(columns={"interaction": "final_interaction"}).drop(columns=["stage"]),
        on=["condition", "outcome", "moderator", "level", "n"],
        how="inner",
    )
    merged["native_minus_pre_abs"] = (merged["native_interaction"] - merged["pre_projection_interaction"]).abs()
    merged["projection_abs_change"] = (merged["final_interaction"] - merged["pre_projection_interaction"]).abs()
    merged["projection_sign_agreement"] = np.sign(merged["final_interaction"]) == np.sign(merged["pre_projection_interaction"])
    merged.to_csv(outputs_dir / "final_hte_interactions.csv", index=False)
    return merged


def _ols_diagnostics(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    rows = []
    for outcome in OUTCOME_SCALE_BOUNDS:
        for moderator in MODERATORS:
            cols = ["condition", moderator]
            design = pd.get_dummies(df[cols].astype(str), columns=cols, drop_first=True, dtype=float)
            X = np.column_stack([np.ones(len(design)), design.to_numpy(dtype=float)])
            y = df[outcome].to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            pred = X @ coef
            denom = float(np.sum((y - y.mean()) ** 2))
            r2 = float(1 - np.sum((y - pred) ** 2) / denom) if denom > 0 else 0.0
            rows.append({"stage": stage, "outcome": outcome, "moderator": moderator, "term": "(model)", "coefficient": np.nan, "r2": r2})
            for term, beta in zip(design.columns, coef[1:]):
                if term.startswith(f"{moderator}_"):
                    rows.append({"stage": stage, "outcome": outcome, "moderator": moderator, "term": term, "coefficient": float(beta), "r2": r2})
    return pd.DataFrame(rows)


def _write_demographic_predictability(native_g: pd.DataFrame, final_with_key: pd.DataFrame, outputs_dir: Path) -> pd.DataFrame:
    native = _ols_diagnostics(native_g, "native_g")
    final = _ols_diagnostics(final_with_key, "final_projected")
    all_rows = pd.concat([native, final], ignore_index=True)
    all_rows.to_csv(outputs_dir / "demographic_predictability_full.csv", index=False)
    comp = native.merge(final, on=["outcome", "moderator", "term"], suffixes=("_native", "_final"), how="inner")
    comp["coefficient_change"] = comp["coefficient_final"] - comp["coefficient_native"]
    comp["r2_change"] = comp["r2_final"] - comp["r2_native"]
    comp.to_csv(outputs_dir / "demographic_predictability_native_vs_final.csv", index=False)
    return comp


def build_final_tier1(
    g_native_responses: pd.DataFrame,
    f_responses: pd.DataFrame,
    calibration_model: dict[str, object],
    *,
    outputs_dir: Path | str = OUTPUTS_DIR,
    expected_n_g: int | None = 1000,
    expected_n_f: int | None = 500,
    f_weight_column: str | None = None,
    require_frozen_f_protocol: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build 17-condition Tier-1 output from native full G responses.

    Treatment rows start from native G treatment responses, receive one
    common condition/outcome shift, and are then projected to native support.
    """
    _raw_item_overlap_check()
    outputs = Path(outputs_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    g = _prepare_g_native(g_native_responses)
    _validate_full_g(g, expected_n_g=expected_n_g)
    try:
        g.to_parquet(outputs / "g_native_responses.parquet", index=False)
    except ImportError:
        g.to_csv(outputs / "g_native_responses.csv", index=False)

    raw_g = _compute_raw_g_ates(g, outputs)
    raw_f, _ = estimate_target_ates_from_f(
        f_responses,
        outputs_dir=outputs,
        weight_column=f_weight_column,
        expected_n_f=expected_n_f,
        require_frozen_protocol=require_frozen_f_protocol,
    )
    if len(raw_f) != 16 * 13:
        raise ValueError(f"expected 208 raw F ATEs, got {len(raw_f)}")
    f_stability, f_stability_summary = write_f_stability_diagnostics(f_responses, outputs_dir=outputs, weight_column=f_weight_column)
    calibrated = apply_calibration_to_target_ates(raw_f, calibration_model, outputs_dir=outputs)
    cal_lookup = _calibrated_lookup(calibrated)
    raw_g_lookup = {(r["condition"], r["outcome"]): r for _, r in raw_g.iterrows()}

    control = g[g["condition"] == "control"].set_index("donor_key").sort_index()
    final_parts = [control.reset_index()]
    shifted_parts = [control.reset_index()]
    audit_rows = []

    for condition in _schema()["conditions"]:
        if condition == "control":
            continue
        native_tx = g[g["condition"] == condition].set_index("donor_key").reindex(control.index)
        final_tx = native_tx.copy()
        shifted_tx = native_tx.copy()
        for outcome in OUTCOME_SCALE_BOUNDS:
            key = (condition, outcome)
            raw_row = raw_g_lookup[key]
            cal_row = cal_lookup[key]
            target_ate = float(cal_row["calibrated_ate_native"])
            proj = _project_outcome(
                native_tx=native_tx,
                control=control,
                final_tx=final_tx,
                shifted_tx=shifted_tx,
                condition=condition,
                outcome=outcome,
                raw_g_ate=float(raw_row["raw_g_ate_native"]),
                target_ate=target_ate,
            )
            if not np.isclose(proj["pre_projection_ate"], target_ate, atol=1e-9):
                raise ValueError(f"pre-projection ATE mismatch for {condition}/{outcome}")
            audit_rows.append(
                {
                    "condition": condition,
                    "outcome": outcome,
                    "outcome_range": float(raw_row["outcome_range"]),
                    "raw_g_ate_native": float(raw_row["raw_g_ate_native"]),
                    "raw_f_ate_native": float(cal_row["raw_f_ate_native"]),
                    "raw_f_ate_pp": float(cal_row["raw_f_ate_pp"]),
                    "calibration_model": str(cal_row["calibration_model"]),
                    "alpha": float(cal_row["calibration_alpha"]),
                    "lambda": float(cal_row["calibration_lambda"]),
                    "calibrated_target_ate_native": target_ate,
                    **proj,
                }
            )
        final_parts.append(final_tx.reset_index())
        shifted_parts.append(shifted_tx.reset_index())

    final_with_key = _recompute_outcomes(pd.concat(final_parts, ignore_index=True))
    shifted = _recompute_outcomes(pd.concat(shifted_parts, ignore_index=True))
    expected_rows = len(control) * len(_schema()["conditions"])
    if len(final_with_key) != expected_rows:
        raise ValueError(f"expected {expected_rows} final rows, got {len(final_with_key)}")
    if final_with_key["profile_id"].duplicated().any():
        raise ValueError("final profile_id values are not unique")
    counts = final_with_key.groupby("condition").size()
    if not (counts == len(control)).all():
        raise ValueError("every final condition must contain exactly N_G rows")
    attrs = [c for c in MODERATORS if c in final_with_key.columns]
    if not (final_with_key.groupby("donor_key")[attrs].nunique(dropna=False) == 1).all().all():
        raise ValueError("final donor demographics are not invariant across conditions")

    native_control = g[g["condition"] == "control"].sort_values("donor_key").reset_index(drop=True)
    final_control = final_with_key[final_with_key["condition"] == "control"].sort_values("donor_key").reset_index(drop=True)
    check_cols = [*MODERATORS, *[it["target_label"] for it in sc.load_items()], *sc.OUTCOME_COMPOSITES.keys()]
    check_cols = [c for c in check_cols if c in native_control.columns and c in final_control.columns]
    if not native_control[check_cols].equals(final_control[check_cols]):
        raise ValueError("final control rows differ from native G control rows")

    audit = pd.DataFrame(audit_rows)
    if len(audit) != 16 * 13:
        raise ValueError(f"expected 208 final ATE audit rows, got {len(audit)}")
    audit.to_csv(outputs / "final_ate_audit.csv", index=False)
    interactions = _write_interaction_diagnostics(g, shifted, final_with_key, outputs)
    demographic_predictability = _write_demographic_predictability(g, final_with_key, outputs)
    from submission.validate_tier1 import validate_tier1

    validation_report = validate_tier1(final_with_key)

    final = final_with_key[_official_columns()].copy()
    if "donor_key" in final.columns or "state" in final.columns or "state_abbr" in final.columns:
        raise ValueError("internal donor metadata leaked into final submission")
    achieved = estimate_raw_ates(final_with_key, list(OUTCOME_SCALE_BOUNDS.keys()))
    if len(achieved) != 16 * 13:
        raise ValueError("final output did not produce all 208 benchmark ATEs")
    return final, {
        "raw_g_ates": raw_g,
        "raw_target_ates": raw_f,
        "calibrated_target_ates": calibrated,
        "f_stability_diagnostics": f_stability,
        "f_stability_summary": f_stability_summary,
        "final_ate_audit": audit,
        "final_hte_interactions": interactions,
        "demographic_predictability": demographic_predictability,
        "validation_report": validation_report,
        "n_rows": len(final),
        "n_g": len(control),
        "n_conditions": len(_schema()["conditions"]),
    }
