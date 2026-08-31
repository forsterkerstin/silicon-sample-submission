"""Regression tests for validate_personas.py's Other-aware generalization:

- rescaled_quota_reserving_other(): n_other=0 is an EXACT no-op, reproducing
  the published quota_gender_age_1000.csv/quota_gender_race_1000.csv (and
  the 500-row F counterparts) values unchanged -- this is what makes the
  CES generalization a strict superset of the historical PUMS path, not a
  new rule applied to it.
- build_f_panel_with_other(): n_other_f=0 is identical to calling
  build_f_panel() directly; n_other_f>0 draws only from the G master's own
  Other-gender pool via the same deterministic_hash convention used for
  every Male/Female cell.
- build_g_master()'s source_dataset parameter defaults to the historical
  PUMS string (byte-identical for every existing caller).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT / "src", REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from validate_personas import (  # noqa: E402
    QUOTA_GENDER_AGE_1000,
    QUOTA_GENDER_AGE_500,
    QUOTA_GENDER_RACE_1000,
    QUOTA_GENDER_RACE_500,
    build_f_panel,
    build_f_panel_with_other,
    build_g_master,
    load_schema,
    rescaled_quota_reserving_other,
)

SCHEMA_PATH = REPO_ROOT / "config" / "benchmark_schema.yaml"


def test_rescaled_quota_reserving_other_is_exact_noop_for_g_when_n_other_zero():
    age, race = rescaled_quota_reserving_other(QUOTA_GENDER_AGE_1000, QUOTA_GENDER_RACE_1000, n_other=0, tie_prefix="test")
    published_age = pd.read_csv(QUOTA_GENDER_AGE_1000)
    published_race = pd.read_csv(QUOTA_GENDER_RACE_1000)
    pd.testing.assert_frame_equal(
        age.sort_values(["gender", "age_band"]).reset_index(drop=True),
        published_age.sort_values(["gender", "age_band"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        race.sort_values(["gender", "race"]).reset_index(drop=True),
        published_race.sort_values(["gender", "race"]).reset_index(drop=True),
    )


def test_rescaled_quota_reserving_other_is_exact_noop_for_f_when_n_other_zero():
    age, race = rescaled_quota_reserving_other(QUOTA_GENDER_AGE_500, QUOTA_GENDER_RACE_500, n_other=0, tie_prefix="test")
    published_age = pd.read_csv(QUOTA_GENDER_AGE_500)
    published_race = pd.read_csv(QUOTA_GENDER_RACE_500)
    pd.testing.assert_frame_equal(
        age.sort_values(["gender", "age_band"]).reset_index(drop=True),
        published_age.sort_values(["gender", "age_band"]).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        race.sort_values(["gender", "race"]).reset_index(drop=True),
        published_race.sort_values(["gender", "race"]).reset_index(drop=True),
    )


def test_rescaled_quota_reserving_other_reproduces_frozen_ces_992_targets():
    """Golden values from the already-frozen CES donor-source amendment
    (config/population.yaml): 490/510 -> 486/506 when 8 slots are reserved."""
    age, race = rescaled_quota_reserving_other(QUOTA_GENDER_AGE_1000, QUOTA_GENDER_RACE_1000, n_other=8, tie_prefix="test")
    assert int(age.loc[age["gender"] == "Male", "target_n"].sum()) == 486
    assert int(age.loc[age["gender"] == "Female", "target_n"].sum()) == 506
    assert int(age["target_n"].sum()) == 992
    assert int(race["target_n"].sum()) == 992


def test_rescaled_quota_reserving_other_reproduces_frozen_ces_496_targets():
    age, race = rescaled_quota_reserving_other(QUOTA_GENDER_AGE_500, QUOTA_GENDER_RACE_500, n_other=4, tie_prefix="test")
    assert int(age["target_n"].sum()) == 496
    assert int(race["target_n"].sum()) == 496


def test_build_g_master_source_dataset_defaults_to_historical_pums_string():
    core = pd.DataFrame(
        [
            {
                "latent_profile_id": "LP0001",
                "age": 30,
                "age_band": "30-44",
                "gender": "Male",
                "race": "White / Caucasian",
                "education": "Bachelor's degree",
                "income": "$56,000 to $99,999",
                "party": "Democrat",
                "state_abbr": "CA",
            }
        ]
    )
    master = build_g_master(core)
    assert master["source_dataset"].iloc[0] == "ACS PUMS 2024 + CES-derived party assignment"
    master_ces = build_g_master(core, source_dataset="CES 2024 Common Content")
    assert master_ces["source_dataset"].iloc[0] == "CES 2024 Common Content"


def _synthetic_g_master(n_male=20, n_female=20, n_other=4):
    import survey_content as sc

    rows = []
    for i, gender in enumerate(["Male"] * n_male + ["Female"] * n_female + ["Other"] * n_other):
        rows.append(
            {
                "donor_key": f"LP{i:04d}",
                "age": 30,
                "age_band": "30-44",
                "gender": gender,
                "race": "White / Caucasian",
                "education": "Bachelor's degree",
                "income": "$56,000 to $99,999",
                "party": "Democrat",
                "state": "California",
                "state_abbr": "CA",
                "source_row_id": f"D{i}",
                "source_weight": 1.0 + i,
            }
        )
    return pd.DataFrame(rows)


def test_build_f_panel_with_other_zero_matches_build_f_panel_directly():
    schema = load_schema(SCHEMA_PATH)
    g_master = _synthetic_g_master(n_other=0)
    quota_age = pd.DataFrame([{"gender": "Male", "age_band": "30-44", "target_n": 10}, {"gender": "Female", "age_band": "30-44", "target_n": 10}])
    quota_race = pd.DataFrame([{"gender": "Male", "race": "White / Caucasian", "target_n": 10}, {"gender": "Female", "race": "White / Caucasian", "target_n": 10}])
    direct = build_f_panel(g_master, schema, quota_age=quota_age, quota_race=quota_race)
    via_other = build_f_panel_with_other(g_master, schema, quota_age, quota_race, n_other_f=0)
    pd.testing.assert_frame_equal(direct.reset_index(drop=True), via_other.reset_index(drop=True))


def test_build_f_panel_with_other_draws_only_from_g_other_pool():
    schema = load_schema(SCHEMA_PATH)
    g_master = _synthetic_g_master(n_male=20, n_female=20, n_other=4)
    quota_age = pd.DataFrame([{"gender": "Male", "age_band": "30-44", "target_n": 10}, {"gender": "Female", "age_band": "30-44", "target_n": 10}])
    quota_race = pd.DataFrame([{"gender": "Male", "race": "White / Caucasian", "target_n": 10}, {"gender": "Female", "race": "White / Caucasian", "target_n": 10}])
    panel = build_f_panel_with_other(g_master, schema, quota_age, quota_race, n_other_f=2)
    assert len(panel) == 22
    other_in_panel = set(panel.loc[panel["gender"] == "Other", "donor_key"])
    assert len(other_in_panel) == 2
    assert other_in_panel.issubset(set(g_master.loc[g_master["gender"] == "Other", "donor_key"]))
    assert panel["donor_key"].is_unique


def test_build_f_panel_with_other_deterministic():
    schema = load_schema(SCHEMA_PATH)
    g_master = _synthetic_g_master(n_male=20, n_female=20, n_other=4)
    quota_age = pd.DataFrame([{"gender": "Male", "age_band": "30-44", "target_n": 10}, {"gender": "Female", "age_band": "30-44", "target_n": 10}])
    quota_race = pd.DataFrame([{"gender": "Male", "race": "White / Caucasian", "target_n": 10}, {"gender": "Female", "race": "White / Caucasian", "target_n": 10}])
    a = build_f_panel_with_other(g_master, schema, quota_age, quota_race, n_other_f=2)
    b = build_f_panel_with_other(g_master, schema, quota_age, quota_race, n_other_f=2)
    pd.testing.assert_frame_equal(a, b)
