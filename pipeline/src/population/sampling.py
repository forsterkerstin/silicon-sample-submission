"""src/population/sampling.py

Selects the 1,000 ACS PUMS donor profiles from the 40 raked cells (§18),
assigns each a stable latent_profile_id and a probabilistically-imputed
party (§19), and assembles the core profile table (§21).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .constants import AGE_BAND_ORDER, GENDER_ORDER, RACE_ORDER
from .io import get_logger

logger = get_logger("sampling")


class SamplingError(Exception):
    """Raised when a cell's donor pool is smaller than its integer target
    (would require sampling with replacement, which this pipeline refuses to
    do silently) or when a post-sampling invariant is violated.
    """


def sample_donors(recoded_pums: pd.DataFrame, joint_cells: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """For each of the 40 (gender, age_band, race) cells, draw exactly
    `integer_target_n` donors without replacement, with probability
    proportional to pums_person_weight, using the given RNG stream. Cells are
    visited in a fixed canonical order (gender, then AGE_BAND_ORDER, then
    RACE_ORDER) so the draw sequence -- and therefore the result, for a fixed
    seed -- is deterministic. Raises SamplingError if any cell's donor pool
    is smaller than its target (rather than silently sampling with
    replacement).
    """
    selected_frames: list[pd.DataFrame] = []
    for gender in GENDER_ORDER:
        for age_band in AGE_BAND_ORDER:
            for race in RACE_ORDER:
                target_row = joint_cells.loc[
                    (joint_cells["gender"] == gender) & (joint_cells["age_band"] == age_band) & (joint_cells["race"] == race)
                ]
                if len(target_row) != 1:
                    raise SamplingError(f"expected exactly one joint-cell row for ({gender}, {age_band}, {race}), found {len(target_row)}")
                target_n = int(target_row["integer_target_n"].iloc[0])
                if target_n == 0:
                    continue

                pool = recoded_pums.loc[
                    (recoded_pums["gender"] == gender) & (recoded_pums["age_band"] == age_band) & (recoded_pums["race"] == race)
                ]
                if len(pool) < target_n:
                    raise SamplingError(
                        f"donor pool for ({gender}, {age_band}, {race}) has only {len(pool)} donors, "
                        f"need {target_n} without replacement"
                    )

                weights = pool["pums_person_weight"].to_numpy(dtype=float)
                probabilities = weights / weights.sum()
                chosen_positions = rng.choice(len(pool), size=target_n, replace=False, p=probabilities)
                chosen = pool.iloc[chosen_positions].copy()
                chosen["joint_cell_target_n"] = target_n
                selected_frames.append(chosen)

    selected = pd.concat(selected_frames, ignore_index=True)
    if len(selected) != int(joint_cells["integer_target_n"].sum()):
        raise SamplingError(f"selected {len(selected)} donors, expected {int(joint_cells['integer_target_n'].sum())}")
    if selected["donor_id"].duplicated().any():
        raise SamplingError("a donor_id was selected more than once across cells")
    return selected


def assign_latent_profile_ids(selected_donors: pd.DataFrame) -> pd.DataFrame:
    """Sort deterministically by (gender canonical order, age-band canonical
    order, race canonical order, donor_id) and assign latent_profile_id
    LP0001..LP<n> (§18's final step).
    """
    out = selected_donors.copy()
    out["_gender_rank"] = out["gender"].map({g: i for i, g in enumerate(GENDER_ORDER)})
    out["_age_band_rank"] = out["age_band"].map({a: i for i, a in enumerate(AGE_BAND_ORDER)})
    out["_race_rank"] = out["race"].map({r: i for i, r in enumerate(RACE_ORDER)})
    out = out.sort_values(["_gender_rank", "_age_band_rank", "_race_rank", "donor_id"]).reset_index(drop=True)
    out = out.drop(columns=["_gender_rank", "_age_band_rank", "_race_rank"])
    out["latent_profile_id"] = [f"LP{i:04d}" for i in range(1, len(out) + 1)]
    return out


def assign_party(profiles_with_probs: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Draw one party per latent profile from its own fitted probability
    distribution (party_prob_democrat/republican/independent/other) using the
    dedicated party RNG stream -- a genuine categorical draw, never argmax.
    Drawn once per latent profile here, so it is a stable characteristic of
    that profile (never re-sampled per condition downstream).
    """
    prob_cols = ["party_prob_democrat", "party_prob_republican", "party_prob_independent", "party_prob_other"]
    classes = ["Democrat", "Republican", "Independent", "Other"]
    probs = profiles_with_probs[prob_cols].to_numpy(dtype=float)
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise SamplingError("party probabilities do not sum to 1 for at least one profile")
    draws = [classes[rng.choice(len(classes), p=row)] for row in probs]
    return pd.Series(draws, index=profiles_with_probs.index, name="party")


CORE_PROFILE_COLUMNS: list[str] = [
    "latent_profile_id", "donor_id", "SERIALNO", "SPORDER", "gender", "age", "year_birth",
    "age_band", "race", "education", "income", "income_adjusted_2024", "state_fips", "state_abbr",
    "party", "party_prob_republican", "party_prob_democrat", "party_prob_independent", "party_prob_other",
    "pums_person_weight", "joint_cell_target_n", "population_seed", "party_seed",
]


def build_core_profiles(
    profiles_with_party: pd.DataFrame,
    master_seed: int,
    party_seed_key: list[int],
    n_profiles: int = 1000,
) -> pd.DataFrame:
    """Assemble data/processed/population/profiles_core_<n_profiles>.csv's
    exact column set (§21) and run its structural validations (row count,
    unique IDs, no missing required fields).
    """
    out = profiles_with_party.copy()
    out["population_seed"] = master_seed
    out["party_seed"] = "-".join(str(k) for k in party_seed_key)
    out = out[CORE_PROFILE_COLUMNS]

    if len(out) != n_profiles:
        raise SamplingError(f"core profiles must have exactly {n_profiles} rows, got {len(out)}")
    if out["latent_profile_id"].duplicated().any():
        raise SamplingError("latent_profile_id is not unique")
    if out["donor_id"].duplicated().any():
        raise SamplingError("donor_id is not unique")
    if out.isna().any().any():
        missing_cols = out.columns[out.isna().any()].tolist()
        raise SamplingError(f"core profiles have missing values in required column(s): {missing_cols}")
    return out


def quota_audit(core_profiles: pd.DataFrame, quota_age: pd.DataFrame, quota_race: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compare achieved gender x age_band and gender x race margins against
    the operative quota tables; returns the two audit DataFrames (also
    written to reports/population/quota_audit_gender_age.csv /
    quota_audit_gender_race.csv). Raises SamplingError on any exact
    mismatch -- the whole point of controlled integerization is that these
    match exactly.
    """
    achieved_age = core_profiles.groupby(["gender", "age_band"]).size().reset_index(name="achieved_n")
    audit_age = quota_age.merge(achieved_age, on=["gender", "age_band"], how="left").fillna({"achieved_n": 0})
    audit_age["achieved_n"] = audit_age["achieved_n"].astype(int)
    audit_age["exact_match"] = audit_age["achieved_n"] == audit_age["target_n"]

    achieved_race = core_profiles.groupby(["gender", "race"]).size().reset_index(name="achieved_n")
    audit_race = quota_race.merge(achieved_race, on=["gender", "race"], how="left").fillna({"achieved_n": 0})
    audit_race["achieved_n"] = audit_race["achieved_n"].astype(int)
    audit_race["exact_match"] = audit_race["achieved_n"] == audit_race["target_n"]

    if not audit_age["exact_match"].all():
        raise SamplingError(f"gender x age_band quota mismatch:\n{audit_age[~audit_age['exact_match']]}")
    if not audit_race["exact_match"].all():
        raise SamplingError(f"gender x race quota mismatch:\n{audit_race[~audit_race['exact_match']]}")

    return {"gender_age": audit_age, "gender_race": audit_race}
