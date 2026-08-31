"""Shared fixtures for the population-construction test suite. Every test
here uses synthetic or small local fixtures -- none require the real
multi-hundred-MB data/csv_pus.zip or data/CCES24_Common_OUTPUT*.csv, so the
suite runs fast and independent of whether real-data ingestion can complete
(see test_reproducibility.py::test_real_archive_missing_hincp_fails_explicitly
for the one test that does touch the real archive, by design).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from population import ces, raking, roster, sampling  # noqa: E402
from population.constants import AGE_BAND_ORDER, RACE_ORDER, load_benchmark_schema, spawn_rngs  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def schema() -> dict:
    return load_benchmark_schema(REPO_ROOT / "config" / "benchmark_schema.yaml")


@pytest.fixture(scope="session")
def quota_age() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "config" / "quota_gender_age_1000.csv")


@pytest.fixture(scope="session")
def quota_race() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "config" / "quota_gender_race_1000.csv")


def make_synthetic_recoded_pums(seed: int = 0, min_donors: int = 400, max_donors: int = 800) -> pd.DataFrame:
    """A synthetic, fully-recoded PUMS-shaped universe with positive donor
    weight in every one of the 40 (gender, age_band, race) cells -- enough to
    exercise raking/sampling/roster end-to-end without the real archive.
    """
    rng = np.random.default_rng(seed)
    rows = []
    serial = 0
    for gender in ("Male", "Female"):
        for age_band in AGE_BAND_ORDER:
            for race in RACE_ORDER:
                n_donors = rng.integers(min_donors, max_donors)
                for _ in range(n_donors):
                    serial += 1
                    age = {
                        "18-29": rng.integers(18, 30), "30-44": rng.integers(30, 45),
                        "45-59": rng.integers(45, 60), "60+": rng.integers(60, 90),
                    }[age_band]
                    rows.append(
                        {
                            "gender": gender, "age_band": age_band, "race": race,
                            "age": int(age), "year_birth": 2026 - int(age),
                            "education": "Bachelor's degree", "income": "$56,000 to $99,999",
                            "income_adjusted_2024": float(rng.uniform(20_000, 150_000)),
                            "state_fips": "06", "state_abbr": "CA",
                            "pums_person_weight": float(rng.uniform(5, 40)),
                            "SERIALNO": f"2024GQ{serial:07d}", "SPORDER": "01",
                            "SCHL": "21",
                            "donor_id": f"2024GQ{serial:07d}-01",
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_recoded_pums() -> pd.DataFrame:
    return make_synthetic_recoded_pums(seed=0)


@pytest.fixture
def synthetic_pums_factory():
    """Returns make_synthetic_recoded_pums itself, so tests can build fresh
    synthetic universes with a chosen seed without a fragile cross-module
    import of this conftest file.
    """
    return make_synthetic_recoded_pums


def build_population_for_seed(master_seed: int, universe_seed: int, quota_age: pd.DataFrame, quota_race: pd.DataFrame, schema: dict):
    """End-to-end population + roster build on a synthetic PUMS universe,
    used by test_population_outputs.py and test_reproducibility.py. Party
    probabilities are synthetic (Dirichlet draws keyed off master_seed) --
    the party *model* itself is unit-tested separately in
    test_ces_recodes.py; here we only need valid probability vectors to
    exercise sampling.assign_party/build_core_profiles/roster end-to-end.
    """
    universe = make_synthetic_recoded_pums(seed=universe_seed)
    joint_cells = raking.build_joint_cells_table(universe, quota_age, quota_race)

    generators, spawn_keys = spawn_rngs(master_seed)
    selected = sampling.sample_donors(universe, joint_cells, generators["pums_selection"])
    profiles = sampling.assign_latent_profile_ids(selected)

    prob_rng = np.random.default_rng(master_seed + 1)
    probs = prob_rng.dirichlet([1, 1, 1, 1], size=len(profiles))
    profiles["party_prob_democrat"] = probs[:, 0]
    profiles["party_prob_republican"] = probs[:, 1]
    profiles["party_prob_independent"] = probs[:, 2]
    profiles["party_prob_other"] = probs[:, 3]
    profiles["party"] = sampling.assign_party(profiles, generators["party_sampling"])

    core = sampling.build_core_profiles(profiles, master_seed, spawn_keys["party_sampling"])
    roster_df = roster.build_simulation_roster(core, schema["conditions"])
    return core, roster_df


@pytest.fixture
def built_population(quota_age, quota_race, schema):
    core, roster_df = build_population_for_seed(20260831, universe_seed=42, quota_age=quota_age, quota_race=quota_race, schema=schema)
    return core, roster_df


@pytest.fixture
def population_builder():
    """Returns build_population_for_seed itself (see built_population above)
    for tests that need to build more than once with different seeds."""
    return build_population_for_seed


@pytest.fixture
def synthetic_ces_raw() -> pd.DataFrame:
    """A tiny synthetic CES-shaped raw frame (string-coded, matching the real
    CSV's dtype=str convention) covering every substantive code path,
    including missing/refused values, for ces.py's recode unit tests.
    """
    rng = np.random.default_rng(1)
    n = 400
    rows = []
    for i in range(n):
        rows.append(
            {
                "caseid": str(i),
                "commonweight": str(rng.uniform(0.2, 3.0)),
                "inputstate": str(rng.choice(["6", "48", "36", "12"])),
                "birthyr": str(rng.integers(1945, 2005)),
                "gender4": str(rng.choice(["1", "2", "3", "4"], p=[0.45, 0.45, 0.05, 0.05])),
                "educ": str(rng.integers(1, 7)),
                "race": str(rng.choice(["1", "2", "3", "4", "5"], p=[0.6, 0.15, 0.1, 0.1, 0.05])),
                "hispanic": str(rng.choice(["1", "2"], p=[0.1, 0.9])),
                "faminc_new": str(rng.choice(["1", "5", "10", "16", "97"])),
                "pid3": str(rng.choice(["1", "2", "3", "4", "5"], p=[0.35, 0.3, 0.25, 0.05, 0.05])),
            }
        )
    return pd.DataFrame(rows)
