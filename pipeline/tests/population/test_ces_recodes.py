"""CES recodes and the §17-19 party model, verified against
data/CES_2024_GUIDE_vv.pdf and data/CCES24_Common_pre.docx (see
reports/population/ces_variable_audit.md). Also covers §25 tests 21-26
("Party model").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from population import ces, sampling


# --- gender ------------------------------------------------------------------

@pytest.mark.parametrize(
    "code,expected",
    [("1", "Male"), ("2", "Female"), ("3", "Other"), ("4", "Other")],
)
def test_gender_mapping(code, expected):
    assert ces.recode_gender_ces(code) == expected


@pytest.mark.parametrize("code", ["8", "9", None])
def test_gender_missing_codes(code):
    assert ces.recode_gender_ces(code) is None


# --- race / Hispanic ----------------------------------------------------------

def test_race_hispanic_from_race_item_takes_priority():
    # race == 3 ("Hispanic or Latino") is Hispanic regardless of the follow-up.
    assert ces.recode_race_ces("3", "2") == "Hispanic / Latino"


def test_race_hispanic_from_followup_overrides_nonhispanic_race():
    # race == 1 (White) but hispanic == 1 (Yes) -> still Hispanic.
    assert ces.recode_race_ces("1", "1") == "Hispanic / Latino"


@pytest.mark.parametrize("race_code,expected", [("1", "White / Caucasian"), ("2", "Black / African American"), ("4", "Asian / Asian American")])
def test_race_nonhispanic_mappings(race_code, expected):
    assert ces.recode_race_ces(race_code, "2") == expected


@pytest.mark.parametrize("race_code", ["5", "6", "7", "8"])
def test_race_other_bucket(race_code):
    assert ces.recode_race_ces(race_code, "2") == "Other"


def test_race_missing_followup_for_nonhispanic_race_is_none():
    # race==1 (not Hispanic-coded) but the Hispanic follow-up is missing -> unresolved.
    assert ces.recode_race_ces("1", "9") is None
    assert ces.recode_race_ces("1", None) is None


# --- party ---------------------------------------------------------------------

@pytest.mark.parametrize("code,expected", [("1", "Democrat"), ("2", "Republican"), ("3", "Independent"), ("4", "Other"), ("5", "Other")])
def test_party_mapping(code, expected):
    assert ces.recode_party_ces(code) == expected


@pytest.mark.parametrize("code", ["8", "9", None])
def test_party_missing_never_becomes_other(code):
    # §16: do not automatically treat refused/skipped/missing as Other.
    assert ces.recode_party_ces(code) is None


# --- harmonized education / income --------------------------------------------

def test_harmonized_education_ces_and_acs_share_the_same_levels():
    ces_levels = set(ces.HARMONIZED_EDU_FROM_CES.values())
    acs_levels = {ces.harmonized_education_from_schl(c) for c in ["1", "16", "18", "20", "21", "22"]}
    assert acs_levels <= ces_levels


def test_harmonized_education_postgrad_bucket_is_shared_by_masters_professional_doctorate():
    # CES's single Post-grad bucket must be what both Master's/Professional (SCHL 22-23) and Doctorate (24) map to.
    assert ces.harmonized_education_from_schl("22") == "Post-grad"
    assert ces.harmonized_education_from_schl("23") == "Post-grad"
    assert ces.harmonized_education_from_schl("24") == "Post-grad"
    assert ces.harmonized_education_from_ces("6") == "Post-grad"


@pytest.mark.parametrize(
    "amount,expected_code",
    [(-100, "1"), (0, "1"), (9_999, "1"), (10_000, "2"), (99_999, "9"), (100_000, "10"), (499_999, "15"), (500_000, "16"), (10_000_000, "16")],
)
def test_harmonized_income_binning_boundaries(amount, expected_code):
    assert ces.harmonized_income_bracket_from_amount(amount) == expected_code


def test_harmonized_income_ces_missing_codes():
    for code in ("97", "998", "999"):
        assert ces.harmonized_income_from_ces(code) is None


# --- 22/23: adult-population weight used, voter-only weight rejected ---------

def test_adult_population_weight_selected_not_voter_only():
    assert "commonweight" in ces.REQUIRED_COLUMNS
    assert "vvweight" not in ces.REQUIRED_COLUMNS
    assert "vvweight_post" not in ces.REQUIRED_COLUMNS
    assert "commonpostweight" not in ces.REQUIRED_COLUMNS


# --- 21: verified mapping file exists -----------------------------------------

def test_ces_variable_mapping_yaml_exists_and_is_valid(repo_root):
    path = repo_root / "data" / "derived" / "population" / "ces_variable_mapping.yaml"
    assert path.exists()
    with open(path) as f:
        mapping = yaml.safe_load(f)
    assert mapping["weight"]["column"] == "commonweight"
    assert "party" in mapping and "education" in mapping and "household_income" in mapping


# --- training frame construction ----------------------------------------------

def test_build_ces_training_frame_drops_incomplete_rows(synthetic_ces_raw):
    training = ces.build_ces_training_frame(synthetic_ces_raw)
    assert len(training) <= len(synthetic_ces_raw)
    assert training[["gender", "age_band", "race", "harmonized_education", "harmonized_income_ces", "state_abbr", "party"]].notna().all().all()
    assert (training["weight"] > 0).all()


def test_build_ces_training_frame_preserves_original_row_index_for_alignment(synthetic_ces_raw):
    """Regression test for a real alignment bug: the returned frame's index
    must still identify each surviving row's position in ces_raw (so a
    caller can do ces_raw.loc[training.index, "caseid"] to recover the
    correct respondent), not a freshly reset 0..N-1 RangeIndex -- when some
    rows are dropped, a reset index silently misaligns any column pulled
    back from ces_raw by position instead of by the dropped rows' true
    identity."""
    training = ces.build_ces_training_frame(synthetic_ces_raw)
    assert len(training) < len(synthetic_ces_raw)  # fixture includes gender4 codes 3/4, which build_ces_training_frame keeps but this assert just needs SOME rows dropped
    assert set(training.index).issubset(set(synthetic_ces_raw.index))
    # every surviving row's caseid, pulled back via the preserved index, must equal that row's own gender4-derived gender
    recovered_caseid = synthetic_ces_raw.loc[training.index, "caseid"]
    recovered_gender = synthetic_ces_raw.loc[training.index, "gender4"].apply(ces.recode_gender_ces)
    assert (recovered_gender.to_numpy() == training["gender"].to_numpy()).all()
    assert recovered_caseid.is_unique


# --- 24/25: probabilities finite, in [0,1], sum to 1; exact benchmark party levels ---

def test_party_model_probabilities_valid_and_labels_match_benchmark(synthetic_ces_raw, schema):
    training = ces.build_ces_training_frame(synthetic_ces_raw)
    model = ces.fit_final_model(training, max_iterations=200)
    fake_profiles = pd.DataFrame(
        [{"gender": "Male", "age_band": "30-44", "race": "White / Caucasian", "harmonized_education": "4-year", "harmonized_income_ces": "9", "state_abbr": "CA"}]
    )
    probs = ces.predict_party_probabilities(model, fake_profiles)
    assert np.isfinite(probs.to_numpy()).all()
    assert ((probs.to_numpy() >= 0) & (probs.to_numpy() <= 1)).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-10)
    assert set(ces.PARTY_CLASSES) == set(schema["moderators"]["party"])


# --- 26: same latent profile always receives the same party ------------------

def test_same_profile_same_seed_yields_same_party_draw():
    probs_df = pd.DataFrame(
        [{"party_prob_democrat": 0.4, "party_prob_republican": 0.3, "party_prob_independent": 0.2, "party_prob_other": 0.1}] * 5
    )
    draw_a = sampling.assign_party(probs_df, np.random.default_rng(42))
    draw_b = sampling.assign_party(probs_df, np.random.default_rng(42))
    assert draw_a.tolist() == draw_b.tolist()
