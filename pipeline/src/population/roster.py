"""src/population/roster.py

Expands the 1,000 core profiles into the 17,000-row simulation roster: each
latent profile reused, unchanged, once in control and once in all 16
interventions. No outcome columns are generated here; this module stops at
the roster.
"""

from __future__ import annotations

import re

import pandas as pd

from .io import get_logger

logger = get_logger("roster")


class RosterError(Exception):
    """Raised when the assembled roster fails one of §22's structural
    invariants."""


ROSTER_COLUMNS: list[str] = [
    "profile_id", "latent_profile_id", "condition", "condition_replicate",
    "gender", "age", "year_birth", "age_band", "race", "education", "income",
    "party", "state_fips", "state_abbr",
]

_PROFILE_ATTRIBUTE_COLUMNS: list[str] = [
    "gender", "age", "year_birth", "age_band", "race", "education", "income", "party", "state_fips", "state_abbr",
]


def sanitize_condition_slug(condition: str) -> str:
    """Lowercase, alphanumeric-run-preserving slug used only inside
    profile_id (e.g. "Measurement & modeling (1)" -> "measurement-modeling-1").
    The exact original condition string is always preserved separately in
    the `condition` column.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", condition.lower()).strip("-")
    return slug


def build_simulation_roster(
    core_profiles: pd.DataFrame,
    conditions: list[str],
    intervention_replicates: int = 1,
    control_replicates: int = 1,
) -> pd.DataFrame:
    """Build the 17,000-row roster: `intervention_replicates` row(s) per
    latent profile in each of the 16 non-control conditions, and
    `control_replicates` rows per latent profile in "control". `conditions`
    is the exact 17-label list from config/benchmark_schema.yaml (control
    first, per the schema snapshot).
    """
    if conditions[0] != "control" or len(conditions) != 17:
        raise RosterError(f"expected exactly 17 conditions with 'control' first, got {len(conditions)}: {conditions}")

    rows: list[pd.DataFrame] = []
    for condition in conditions:
        n_replicates = control_replicates if condition == "control" else intervention_replicates
        for replicate in range(1, n_replicates + 1):
            chunk = core_profiles[["latent_profile_id"] + _PROFILE_ATTRIBUTE_COLUMNS].copy()
            chunk["condition"] = condition
            chunk["condition_replicate"] = replicate
            slug = sanitize_condition_slug(condition)
            chunk["profile_id"] = chunk["latent_profile_id"] + f"__{slug}__R{replicate}"
            rows.append(chunk)

    roster = pd.concat(rows, ignore_index=True)[ROSTER_COLUMNS]
    validate_roster(roster, core_profiles, conditions, intervention_replicates, control_replicates)
    return roster


def validate_roster(
    roster: pd.DataFrame,
    core_profiles: pd.DataFrame,
    conditions: list[str],
    intervention_replicates: int,
    control_replicates: int,
) -> None:
    """Every §22 structural invariant, raising RosterError on the first
    violation found."""
    n_profiles = len(core_profiles)
    n_interventions = len(conditions) - 1
    expected_rows = n_interventions * intervention_replicates * n_profiles + control_replicates * n_profiles
    if len(roster) != expected_rows:
        raise RosterError(f"expected {expected_rows} roster rows, got {len(roster)}")

    if roster["profile_id"].duplicated().any():
        raise RosterError("profile_id is not globally unique")

    observed_conditions = set(roster["condition"].unique())
    if observed_conditions != set(conditions):
        raise RosterError(f"roster conditions {sorted(observed_conditions)} != schema conditions {sorted(conditions)}")

    counts = roster.groupby("condition").size()
    for condition in conditions:
        expected = control_replicates * n_profiles if condition == "control" else intervention_replicates * n_profiles
        if counts.get(condition, 0) != expected:
            raise RosterError(f"condition '{condition}' has {counts.get(condition, 0)} rows, expected {expected}")

    for condition in conditions:
        n_replicates = control_replicates if condition == "control" else intervention_replicates
        sub = roster.loc[roster["condition"] == condition]
        per_profile_counts = sub.groupby("latent_profile_id").size()
        if len(per_profile_counts) != n_profiles or (per_profile_counts != n_replicates).any():
            raise RosterError(f"condition '{condition}': every latent profile must appear exactly {n_replicates} time(s)")

    attrs_by_profile = core_profiles.set_index("latent_profile_id")[_PROFILE_ATTRIBUTE_COLUMNS]
    roster_attrs = roster.drop_duplicates("latent_profile_id").set_index("latent_profile_id")[_PROFILE_ATTRIBUTE_COLUMNS]
    if not attrs_by_profile.reindex(roster_attrs.index).equals(roster_attrs):
        raise RosterError("profile attributes are not invariant across conditions")

    outcome_like = [c for c in roster.columns if c not in ROSTER_COLUMNS]
    if outcome_like:
        raise RosterError(f"roster must not contain outcome columns yet, found: {outcome_like}")

    logger.info("roster validated: %d rows, %d conditions, %d profiles", len(roster), len(conditions), n_profiles)
