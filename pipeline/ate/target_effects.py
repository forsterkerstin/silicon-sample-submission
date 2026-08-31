"""Target-benchmark ATE estimation from the F experimental panel.

The F engine supplies paired control/treatment native responses for a fixed
representative target panel. Calibration is applied only after native ATEs
are estimated and converted to percent-of-range units.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import survey_content as sc
from ate.f_reliability import DEFAULT_N_F, profile_level_summary, require_frozen_f_protocol
from ate.normalize_effects import OUTCOME_SCALE_BOUNDS, to_percent_of_range, from_percent_of_range

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
KEY_CANDIDATES = ("f_profile_id", "donor_key", "latent_profile_id", "source_profile_id")


def ensure_outcomes(df: pd.DataFrame, outcomes: Sequence[str] | None = None) -> pd.DataFrame:
    """Return a copy with the 13 benchmark outcomes present."""
    outcomes = list(outcomes or sc.OUTCOME_COMPOSITES.keys())
    missing = [c for c in outcomes if c not in df.columns]
    if not missing:
        return df.copy()
    raw_needed = {
        label
        for outcome in missing
        for _, spec in [sc.OUTCOME_COMPOSITES[outcome]]
        for label in (spec if isinstance(spec, list) else [spec])
    }
    absent = sorted(raw_needed - set(df.columns))
    if absent:
        raise ValueError(f"cannot compute outcomes; missing raw item column(s): {absent}")
    computed = []
    for _, row in df.iterrows():
        all_outcomes = sc.compute_outcomes(row.to_dict())
        computed.append({outcome: all_outcomes[outcome] for outcome in missing})
    outcome_df = pd.DataFrame(computed)
    out = df.drop(columns=[c for c in outcome_df.columns if c in df.columns]).reset_index(drop=True)
    return pd.concat([out, outcome_df], axis=1)


def detect_profile_key(df: pd.DataFrame, *, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for candidate in KEY_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"F responses need one profile key column from {KEY_CANDIDATES}")


def _weighted_mean(values: pd.Series, weights: pd.Series | None) -> float:
    if weights is None:
        return float(values.mean())
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("profile weights must sum to a positive value")
    return float((values * weights).sum() / total)


def compute_f_contrasts(
    f_responses: pd.DataFrame,
    *,
    condition_column: str = "condition",
    control_label: str = "control",
    profile_key_column: str | None = None,
    weight_column: str | None = None,
    outcomes: Sequence[str] | None = None,
    expected_n_f: int | None = None,
) -> pd.DataFrame:
    """One paired row-level contrast per F profile, treatment, and outcome."""
    outcomes = list(outcomes or sc.OUTCOME_COMPOSITES.keys())
    df = ensure_outcomes(f_responses, outcomes=outcomes)
    key = detect_profile_key(df, preferred=profile_key_column)
    if weight_column is not None and weight_column not in df.columns:
        raise ValueError(f"declared F weight column {weight_column!r} is missing")

    controls = df[df[condition_column] == control_label]
    if controls.empty:
        raise ValueError("F responses contain no control rows")
    if expected_n_f is not None and controls[key].nunique() != expected_n_f:
        raise ValueError(f"expected {expected_n_f} unique F profiles, got {controls[key].nunique()}")
    duplicated = controls[key].duplicated()
    if duplicated.any():
        raise ValueError(f"F control rows are not unique by {key}; duplicated {int(duplicated.sum())} key(s)")

    rows: list[dict[str, object]] = []
    demo_cols = [c for c in ["gender", "age_band", "race", "education", "income", "party"] if c in df.columns]
    control_cols = [key, *demo_cols, *outcomes]
    if weight_column:
        control_cols.append(weight_column)
    control = controls[control_cols].rename(columns={o: f"{o}__control" for o in outcomes})

    for condition in sorted(c for c in df[condition_column].unique() if c != control_label):
        tx = df[df[condition_column] == condition]
        duplicated = tx[key].duplicated()
        if duplicated.any():
            raise ValueError(f"F {condition!r} rows are not unique by {key}; duplicated {int(duplicated.sum())} key(s)")
        merged = tx[[key, *outcomes]].merge(control, on=key, how="inner", validate="one_to_one")
        if len(merged) != len(control):
            raise ValueError(f"F condition {condition!r} is not paired with every control profile")
        if expected_n_f is not None and merged[key].nunique() != expected_n_f:
            raise ValueError(f"F condition {condition!r} does not use the same {expected_n_f} profile IDs")
        for outcome in outcomes:
            low, high = OUTCOME_SCALE_BOUNDS[outcome]
            for _, r in merged.iterrows():
                wt = r[weight_column] if weight_column else 1.0
                item = {
                    "f_profile_id": r[key],
                    "condition": condition,
                    "outcome": outcome,
                    "contrast_native": float(r[outcome] - r[f"{outcome}__control"]),
                    "control_value": float(r[f"{outcome}__control"]),
                    "treatment_value": float(r[outcome]),
                    "outcome_min": low,
                    "outcome_max": high,
                    "outcome_range": high - low,
                    "weight": float(wt),
                }
                for demo in demo_cols:
                    item[demo] = r[demo]
                rows.append(item)
    return pd.DataFrame(rows)


def estimate_target_ates_from_f(
    f_responses: pd.DataFrame,
    *,
    outputs_dir: Path | str | None = OUTPUTS_DIR,
    condition_column: str = "condition",
    control_label: str = "control",
    profile_key_column: str | None = None,
    weight_column: str | None = None,
    outcomes: Sequence[str] | None = None,
    expected_n_f: int | None = DEFAULT_N_F,
    require_frozen_protocol: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if require_frozen_protocol:
        require_frozen_f_protocol()
    contrasts = compute_f_contrasts(
        f_responses,
        condition_column=condition_column,
        control_label=control_label,
        profile_key_column=profile_key_column,
        weight_column=weight_column,
        outcomes=outcomes,
        expected_n_f=expected_n_f,
    )
    rows = []
    for (condition, outcome), g in contrasts.groupby(["condition", "outcome"], sort=True):
        low = float(g["outcome_min"].iloc[0])
        high = float(g["outcome_max"].iloc[0])
        weights = g["weight"] if weight_column else None
        ate_native = _weighted_mean(g["contrast_native"], weights)
        profile_summary = profile_level_summary(g["contrast_native"], high - low) if weights is None else {
            "n_f": int(len(g)),
            "raw_ate_native": ate_native,
            "raw_ate_pp": to_percent_of_range(ate_native, low, high),
            "profile_delta_sd": float(g["contrast_native"].std(ddof=1)),
            "profile_ate_se_native": float("nan"),
            "profile_ate_se_pp": float("nan"),
        }
        rows.append(
            {
                "condition": condition,
                "outcome": outcome,
                **profile_summary,
                "synthetic_ate_native": ate_native,
                "synthetic_effect_pp": to_percent_of_range(ate_native, low, high),
                "outcome_min": low,
                "outcome_max": high,
                "outcome_range": high - low,
                "num_profiles": int(len(g)),
                "weights_used": bool(weight_column),
                "population_matching_method": "fixed_representative_us_f_panel",
            }
        )
    raw_ates = pd.DataFrame(rows)
    if outputs_dir is not None:
        out = Path(outputs_dir)
        out.mkdir(parents=True, exist_ok=True)
        raw_ates.to_csv(out / "raw_target_ates.csv", index=False)
    return raw_ates, contrasts


def apply_calibration_to_target_ates(
    raw_target_ates: pd.DataFrame,
    calibration_model: dict[str, object],
    *,
    outputs_dir: Path | str | None = OUTPUTS_DIR,
) -> pd.DataFrame:
    if calibration_model.get("usable_for_production") is False:
        raise ValueError(
            "calibration model is marked unusable for production: "
            f"{calibration_model.get('calibration_status', calibration_model.get('stale_reason', 'unknown reason'))}"
        )
    alpha = float(calibration_model.get("calibration_alpha", calibration_model.get("alpha", 0.0)))
    lam = float(calibration_model.get("calibration_lambda", calibration_model.get("lambda", 1.0)))
    out = raw_target_ates.copy()
    out["calibration_model"] = str(calibration_model.get("model_name", "unknown"))
    out["calibration_alpha"] = alpha
    out["calibration_lambda"] = lam
    raw_pp = out["raw_f_ate_pp"] if "raw_f_ate_pp" in out.columns else out["synthetic_effect_pp"]
    raw_native = out["raw_f_ate_native"] if "raw_f_ate_native" in out.columns else out["synthetic_ate_native"]
    out["calibrated_effect_pp"] = alpha + lam * raw_pp.astype(float)
    out["calibrated_ate_native"] = [
        from_percent_of_range(pp, low, high)
        for pp, low, high in zip(out["calibrated_effect_pp"], out["outcome_min"], out["outcome_max"])
    ]
    out["calibrated_target_ate_native"] = out["calibrated_ate_native"]
    out["raw_f_ate_native"] = raw_native
    out["raw_f_ate_pp"] = raw_pp
    if outputs_dir is not None:
        path = Path(outputs_dir)
        path.mkdir(parents=True, exist_ok=True)
        out.to_csv(path / "calibrated_target_ates.csv", index=False)
    return out


def write_f_stability_diagnostics(
    f_responses: pd.DataFrame,
    *,
    outputs_dir: Path | str | None = OUTPUTS_DIR,
    repetition_column: str = "repetition",
    condition_column: str = "condition",
    control_label: str = "control",
    profile_key_column: str | None = None,
    weight_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Write Monte Carlo stability diagnostics when repeated F runs exist."""
    out_dir = Path(outputs_dir) if outputs_dir is not None else None
    if repetition_column not in f_responses.columns:
        diagnostics = pd.DataFrame(
            [{"status": "single_repetition_or_unavailable", "metric": "not_applicable", "value": np.nan}]
        )
        summary = {"status": "single_repetition_or_unavailable", "n_repetitions": 1}
    else:
        vectors = []
        for rep, group in f_responses.groupby(repetition_column, sort=True):
            raw, _ = estimate_target_ates_from_f(
                group,
                outputs_dir=None,
                condition_column=condition_column,
                control_label=control_label,
                profile_key_column=profile_key_column,
                weight_column=weight_column,
            )
            vectors.append((rep, raw.sort_values(["condition", "outcome"])["raw_f_ate_pp"].to_numpy(dtype=float)))
        rows = []
        for (rep_a, vec_a), (rep_b, vec_b) in itertools.combinations(vectors, 2):
            diff = vec_a - vec_b
            rows.extend(
                [
                    {"rep_a": rep_a, "rep_b": rep_b, "metric": "pearson", "value": float(np.corrcoef(vec_a, vec_b)[0, 1])},
                    {"rep_a": rep_a, "rep_b": rep_b, "metric": "spearman", "value": float(spearmanr(vec_a, vec_b).correlation)},
                    {"rep_a": rep_a, "rep_b": rep_b, "metric": "rmse", "value": float(np.sqrt(np.mean(diff**2)))},
                    {"rep_a": rep_a, "rep_b": rep_b, "metric": "median_abs_diff", "value": float(np.median(np.abs(diff)))},
                    {"rep_a": rep_a, "rep_b": rep_b, "metric": "max_abs_diff", "value": float(np.max(np.abs(diff)))},
                ]
            )
        diagnostics = pd.DataFrame(rows)
        stacked = np.vstack([vec for _, vec in vectors])
        summary = {
            "status": "ok",
            "n_repetitions": len(vectors),
            "mean_cell_sd_pp": float(np.mean(np.std(stacked, axis=0, ddof=1))) if len(vectors) > 1 else 0.0,
            "mean_cell_mcse_pp": float(np.mean(np.std(stacked, axis=0, ddof=1) / np.sqrt(len(vectors)))) if len(vectors) > 1 else 0.0,
        }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        diagnostics.to_csv(out_dir / "f_stability_diagnostics.csv", index=False)
        (out_dir / "f_stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return diagnostics, summary
