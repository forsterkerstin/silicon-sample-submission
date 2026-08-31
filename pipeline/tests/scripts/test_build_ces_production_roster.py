"""Tests for the all-CES N_G=1000 target-donor-source switch
(scripts/build_ces_production_roster.py):

- unit tests for the two new deterministic mechanisms (margin rescaling via
  the existing largest_remainder_allocations, and the RNG-stream-extension
  safety invariant that lets a new 'ces_other_selection' stream be spawned
  without perturbing the existing named streams);
- structural verification of the already-built roster artifact on disk
  (skipped if not built in this environment). No submission, no new
  construction, no target/model information used anywhere in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for p in (SRC_DIR, REPO_ROOT, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from population.constants import RNG_STREAM_NAMES, spawn_rngs  # noqa: E402
from build_ces_production_roster import N_G, N_OTHER, OTHER_STREAM_NAME, rescaled_gender_totals, rescaled_margins, sample_other_donors  # noqa: E402

ROSTER_PATH = REPO_ROOT / "data" / "derived" / "population" / "ces_production_roster_n1000.csv"


# ---- RNG stream extension safety ----


def test_extending_rng_stream_names_does_not_perturb_existing_streams():
    """SeedSequence.spawn(k) must be a strict prefix of spawn(k+1) -- adding
    a new named stream must produce byte-identical Generators for every
    pre-existing name, or the CES donor draw would silently diverge from
    the production PUMS draw's RNG usage."""
    master_seed = 12345
    baseline, baseline_keys = spawn_rngs(master_seed, names=RNG_STREAM_NAMES)
    extended, extended_keys = spawn_rngs(master_seed, names=RNG_STREAM_NAMES + (OTHER_STREAM_NAME,))
    for name in RNG_STREAM_NAMES:
        assert extended_keys[name] == baseline_keys[name]
        assert baseline[name].integers(0, 1_000_000, size=10).tolist() == extended[name].integers(0, 1_000_000, size=10).tolist()
    assert OTHER_STREAM_NAME in extended
    assert OTHER_STREAM_NAME not in baseline


# ---- margin rescaling ----


def test_rescaled_gender_totals_sums_to_reserved_total():
    totals = rescaled_gender_totals(n_reserved_other=8)
    assert int(totals.sum()) == 992
    assert set(totals.index) == {"Male", "Female"}
    # relative order preserved (Female was already the larger published quota)
    assert totals["Female"] > totals["Male"]


def test_rescaled_margins_each_gender_age_and_race_sum_to_gender_total():
    totals = rescaled_gender_totals(n_reserved_other=8)
    age_df, race_df = rescaled_margins(totals)
    for gender in ("Male", "Female"):
        assert int(age_df.loc[age_df["gender"] == gender, "target_n"].sum()) == int(totals[gender])
        assert int(race_df.loc[race_df["gender"] == gender, "target_n"].sum()) == int(totals[gender])
    assert int(age_df["target_n"].sum()) == 992
    assert int(race_df["target_n"].sum()) == 992


def test_rescaling_is_deterministic():
    totals_a = rescaled_gender_totals(n_reserved_other=8)
    totals_b = rescaled_gender_totals(n_reserved_other=8)
    pd.testing.assert_series_equal(totals_a.sort_index(), totals_b.sort_index())
    age_a, race_a = rescaled_margins(totals_a)
    age_b, race_b = rescaled_margins(totals_b)
    pd.testing.assert_frame_equal(age_a, age_b)
    pd.testing.assert_frame_equal(race_a, race_b)


# ---- Other-gender weighted draw ----


def test_sample_other_donors_draws_exactly_n_without_replacement():
    recoded = pd.DataFrame(
        {
            "gender": ["Other"] * 10 + ["Male", "Female"],
            "donor_id": [f"d{i}" for i in range(12)],
            "pums_person_weight": [1.0 + i for i in range(12)],
        }
    )
    rng = np.random.default_rng(0)
    chosen = sample_other_donors(recoded, 4, rng)
    assert len(chosen) == 4
    assert chosen["donor_id"].is_unique
    assert (chosen["gender"] == "Other").all()


def test_sample_other_donors_raises_if_pool_too_small():
    recoded = pd.DataFrame({"gender": ["Other"] * 3, "donor_id": ["a", "b", "c"], "pums_person_weight": [1.0, 1.0, 1.0]})
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError):
        sample_other_donors(recoded, 8, rng)


def test_sample_other_donors_deterministic_given_same_seed():
    recoded = pd.DataFrame(
        {
            "gender": ["Other"] * 20,
            "donor_id": [f"d{i}" for i in range(20)],
            "pums_person_weight": np.linspace(1, 5, 20),
        }
    )
    a = sample_other_donors(recoded, 8, np.random.default_rng(7))
    b = sample_other_donors(recoded, 8, np.random.default_rng(7))
    assert sorted(a["donor_id"]) == sorted(b["donor_id"])


# ---- structural verification of the built roster ----

pytestmark_roster = pytest.mark.skipif(not ROSTER_PATH.exists(), reason="CES production roster not built in this environment")


@pytestmark_roster
def test_roster_has_exactly_1000_unique_rows():
    df = pd.read_csv(ROSTER_PATH)
    assert len(df) == N_G
    assert df["latent_profile_id"].is_unique
    assert df["donor_id"].is_unique


@pytestmark_roster
def test_roster_other_n_is_exactly_frozen_value():
    df = pd.read_csv(ROSTER_PATH)
    assert int((df["gender"] == "Other").sum()) == N_OTHER
    assert int((df["gender"] == "Male").sum()) + int((df["gender"] == "Female").sum()) == N_G - N_OTHER


@pytestmark_roster
def test_roster_required_fields_never_missing():
    df = pd.read_csv(ROSTER_PATH)
    required = ["latent_profile_id", "donor_id", "gender", "age", "age_band", "race", "education", "income", "party", "state_abbr"]
    for col in required:
        assert df[col].notna().all(), f"{col} has missing values"


@pytestmark_roster
def test_roster_no_outcome_or_forbidden_columns():
    df = pd.read_csv(ROSTER_PATH)
    forbidden_substrings = ["outcome", "climate", "trust", "vote_choice", "response", "y_"]
    for col in df.columns:
        low = col.lower()
        assert not any(term in low for term in forbidden_substrings), f"unexpected outcome-adjacent column: {col}"


@pytestmark_roster
def test_roster_other_donors_genuinely_ces_not_relabeled():
    """The Other-gender donors must be their own genuine CES respondents,
    not a relabeled Man/Woman row -- verified by cross-checking each
    Other-labeled donor_id's raw gender4 code in the source CES file."""
    df = pd.read_csv(ROSTER_PATH, dtype={"donor_id": str})
    other_ids = set(df.loc[df["gender"] == "Other", "donor_id"])
    raw = pd.read_csv(REPO_ROOT / "data" / "CCES24_Common_OUTPUT_vv_topost_final.csv", usecols=["caseid", "gender4"], dtype=str)
    raw_gender4 = raw.set_index("caseid")["gender4"]
    for donor_id in other_ids:
        assert raw_gender4.loc[donor_id] in {"3", "4"}, f"donor {donor_id} labeled Other but raw gender4={raw_gender4.loc[donor_id]!r}"
