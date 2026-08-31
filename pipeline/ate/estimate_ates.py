"""pipeline/ate/estimate_ates.py

Raw (uncalibrated) treatment effects, computed directly from native-scale
responses -- never from a probability distribution. This is the "raw_ate"
half of the new architecture: the separate calibrate_lambda.py /
calibrate_arm.py modules decide what to do with these numbers; this module
only measures what the raw simulation actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def estimate_raw_ates(
    responses: pd.DataFrame,
    outcome_columns: list[str],
    condition_column: str = "condition",
    control_label: str = "control",
) -> pd.DataFrame:
    """For every (condition, outcome) pair (condition != control_label),
    raw_ate = mean(responses[outcome] | condition) - mean(responses[outcome] | control).
    `responses` is one row per simulated respondent, native-scale columns
    (e.g. a Phase-A output: profile_id, condition, + one column per raw item
    or outcome composite). Returns a long DataFrame:
    condition, outcome, raw_ate, control_mean, treatment_mean, n_control, n_treatment.
    """
    control_rows = responses[responses[condition_column] == control_label]
    if control_rows.empty:
        raise ValueError(f"no rows with {condition_column}={control_label!r} to compute a control mean from")

    rows = []
    for outcome in outcome_columns:
        control_series = control_rows[outcome].dropna()
        control_mean = float(control_series.mean())
        for condition, group in responses[responses[condition_column] != control_label].groupby(condition_column):
            treatment_series = group[outcome].dropna()
            if treatment_series.empty:
                continue
            treatment_mean = float(treatment_series.mean())
            rows.append(
                {
                    "condition": condition,
                    "outcome": outcome,
                    "raw_ate": treatment_mean - control_mean,
                    "control_mean": control_mean,
                    "treatment_mean": treatment_mean,
                    "n_control": len(control_series),
                    "n_treatment": len(treatment_series),
                }
            )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class StudyDefinition:
    """Study-specific inputs for a population-aligned ATE estimate."""

    study_id: str
    treatment_label: str
    control_label: str
    outcome_column: str
    target_population: str
    weight_column: str | None = None


@dataclass(frozen=True)
class ProfilePopulation:
    """Synthetic profile responses representing one target population."""

    responses: pd.DataFrame
    target_population: str
    population_matching_method: str
    weight_column: str | None = None


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    if weights is None:
        return float(values.mean())
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")
    if weights.isna().any():
        raise ValueError("weights contain missing values")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("weights must sum to a positive value")
    return float((values * weights).sum() / total_weight)


def estimate_raw_ate(
    study_definition: StudyDefinition,
    profile_population: ProfilePopulation,
    model_config: dict | None = None,
    repetitions: int | None = None,
    condition_column: str = "condition",
) -> dict[str, object]:
    """Estimate F(stimulus, outcome, population) for one study/effect.

    `model_config` and `repetitions` are accepted as explicit protocol
    inputs so callers can keep them identical across calibration and target
    forecasts; this pure ATE reducer does not mutate or interpret them.
    """
    if study_definition.target_population != profile_population.target_population:
        raise ValueError(
            f"study target population {study_definition.target_population!r} does not match "
            f"profile population {profile_population.target_population!r}"
        )
    if not profile_population.population_matching_method:
        raise ValueError("population matching is undefined")

    weight_column = study_definition.weight_column or profile_population.weight_column
    if study_definition.weight_column and profile_population.weight_column and study_definition.weight_column != profile_population.weight_column:
        raise ValueError("study_definition and profile_population specify different weight columns")
    if study_definition.weight_column and study_definition.weight_column not in profile_population.responses:
        raise ValueError(f"declared weight column {study_definition.weight_column!r} is missing")

    responses = profile_population.responses
    treatment = responses[responses[condition_column] == study_definition.treatment_label]
    control = responses[responses[condition_column] == study_definition.control_label]
    if treatment.empty or control.empty:
        raise ValueError("both treatment and control rows are required")
    if study_definition.outcome_column not in responses:
        raise ValueError(f"outcome column {study_definition.outcome_column!r} is missing")

    wt = treatment[weight_column] if weight_column else None
    wc = control[weight_column] if weight_column else None
    treatment_mean = weighted_mean(treatment[study_definition.outcome_column], wt)
    control_mean = weighted_mean(control[study_definition.outcome_column], wc)
    return {
        "study_id": study_definition.study_id,
        "target_population": study_definition.target_population,
        "population_matching_method": profile_population.population_matching_method,
        "weights_used": weight_column is not None,
        "model_config": model_config,
        "repetitions": repetitions,
        "treatment_mean": treatment_mean,
        "control_mean": control_mean,
        "raw_ate": treatment_mean - control_mean,
    }
