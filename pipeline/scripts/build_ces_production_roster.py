#!/usr/bin/env python3
"""scripts/build_ces_production_roster.py

Builds and freezes the final all-CES N_G=1000 donor-core roster --
data/derived/population/ces_production_roster_n1000.csv -- for the
population-source-switch amendment (ACS PUMS -> CES 2024 Common Content).

Scope of THIS script, deliberately bounded: freeze the 1,000-row core
roster only. It does NOT touch data/generated/g_personas_master.csv (the
live production file inference/together_batch.py's G-request builder
reads), data/processed/population/simulation_roster_17000.csv, or the
500-row F target panel -- those are all derived FROM a core roster by
scripts/validate_personas.py in one coupled build pass, and are keyed by
donor_key values that would silently point at different real people if the
core roster changed underneath them without also rebuilding those three
together. That live cutover is an explicit, separate, later step.

Two parts beyond the existing scratch feasibility script
(build_scratch_ces_roster.py, reused unmodified for CES recoding and the
verified religion/ideology mappings):

1. Reserve exactly N_OTHER=8 slots for genuine CES gender4 Non-binary/Other
   respondents, using the donor frame's own survey-weighted prevalence
   (0.7634%, independently re-verified against data/CCES24_Common_OUTPUT_vv_topost_final.csv
   itself: commonweight-weighted share of gender4 in {3,4} across all
   60,000 valid-weight rows = 0.76343...%; 1000 * 0.007634 = 7.634 ->
   nearest-integer 8). This is NOT a new benchmark quota -- no age/race
   quota is defined or enforced for these 8 rows; they are drawn by the
   SAME weighted-without-replacement mechanism sample_donors() already uses
   for every other cell, applied to one more pool (gender4 in {Non-binary,
   Other}), via a new named RNG stream spawned alongside (not replacing)
   the existing ones in population.constants.spawn_rngs -- SeedSequence.spawn(k)
   is a strict prefix of spawn(k+1), so this cannot perturb the existing
   Male/Female draw's determinism (verified directly before writing this).

2. Rescale the existing published gender x age and gender x race margins
   (config/quota_gender_age_1000.csv / quota_gender_race_1000.csv, summing
   to 1000) down to 992 (1000 - 8 reserved Other slots), preserving their
   RELATIVE proportions as closely as mathematically possible, via the
   repository's own already-established largest-remainder allocator
   (calibration.study_population.largest_remainder_allocations, reused
   UNMODIFIED -- not reinvented) applied in two stages: gender totals
   (490/510 -> 486/506), then each gender's 4 age-band and 5 race weights
   rescaled to that gender's new total. The existing raking machinery
   (src/population/raking.py: ipf_2d + controlled_integerize, UNMODIFIED)
   is then run against these new margins exactly as it already runs
   against the 1000-total margins for the production PUMS roster.

No paid inference. No LLM calls. No target requests. No production file
(g_personas_master.csv, simulation_roster_17000.csv, f_target_panel.csv) is
modified.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from calibration.study_population import largest_remainder_allocations  # noqa: E402
from population import raking, sampling  # noqa: E402
from population.constants import AGE_BAND_ORDER, RACE_ORDER, RNG_STREAM_NAMES, spawn_rngs  # noqa: E402
from build_scratch_ces_roster import CES_CSV_PATH, build_recoded_ces  # noqa: E402

QUOTA_AGE_PATH = REPO_ROOT / "config" / "quota_gender_age_1000.csv"
QUOTA_RACE_PATH = REPO_ROOT / "config" / "quota_gender_race_1000.csv"
OUT_PATH = REPO_ROOT / "data" / "derived" / "population" / "ces_production_roster_n1000.csv"
POP_CONFIG_PATH = REPO_ROOT / "config" / "population.yaml"

N_G = 1000
N_OTHER = 8  # frozen: round(1000 * 0.007634), the CES commonweight-weighted survey prevalence of gender4 in {Non-binary, Other}
OTHER_STREAM_NAME = "ces_other_selection"


def rescaled_gender_totals(n_reserved_other: int) -> pd.Series:
    quota_age = pd.read_csv(QUOTA_AGE_PATH)
    gender_totals = quota_age.groupby("gender")["target_n"].sum()
    return largest_remainder_allocations(gender_totals, n_f=N_G - n_reserved_other, tie_key="ces_donor_switch_gender_totals_v1")


def rescaled_margins(new_gender_totals: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    quota_age = pd.read_csv(QUOTA_AGE_PATH)
    quota_race = pd.read_csv(QUOTA_RACE_PATH)
    age_rows, race_rows = [], []
    for gender in ("Male", "Female"):
        g_total = int(new_gender_totals[gender])
        age_w = quota_age.loc[quota_age["gender"] == gender].set_index("age_band")["target_n"]
        race_w = quota_race.loc[quota_race["gender"] == gender].set_index("race")["target_n"]
        new_age = largest_remainder_allocations(age_w, n_f=g_total, tie_key=f"ces_donor_switch_age_margin_v1_{gender}")
        new_race = largest_remainder_allocations(race_w, n_f=g_total, tie_key=f"ces_donor_switch_race_margin_v1_{gender}")
        for age_band, n in new_age.items():
            age_rows.append({"gender": gender, "age_band": age_band, "target_n": int(n)})
        for race, n in new_race.items():
            race_rows.append({"gender": gender, "race": race, "target_n": int(n)})
    return pd.DataFrame(age_rows), pd.DataFrame(race_rows)


def sample_other_donors(recoded: pd.DataFrame, n_other: int, rng: np.random.Generator) -> pd.DataFrame:
    """Weighted-without-replacement draw of n_other genuine CES respondents
    whose native gender4 recodes to 'Other', probability proportional to
    their own commonweight -- the identical formula sample_donors() already
    uses per cell, applied here to one pool instead of 40. No target/model
    information is used; weighting by the respondent's own survey weight
    (not any curated selection) is how 'preserve demographic diversity
    under a deterministic rule' is satisfied without inventing an age/race
    quota for a level the benchmark never quota-constrains."""
    pool = recoded.loc[recoded["gender"] == "Other"].reset_index(drop=True)
    if len(pool) < n_other:
        raise RuntimeError(f"Other-gender CES donor pool has only {len(pool)} eligible respondents, need {n_other}")
    weights = pool["pums_person_weight"].to_numpy(dtype=float)
    probabilities = weights / weights.sum()
    chosen_positions = rng.choice(len(pool), size=n_other, replace=False, p=probabilities)
    return pool.iloc[chosen_positions].copy()


def main() -> int:
    pop_cfg = yaml.safe_load(POP_CONFIG_PATH.read_text(encoding="utf-8"))["population"]
    master_seed = int(pop_cfg["master_seed"])
    names = RNG_STREAM_NAMES + (OTHER_STREAM_NAME,)
    generators, spawn_keys = spawn_rngs(master_seed, names=names)
    assert spawn_keys["pums_selection"] == list(np.random.SeedSequence(master_seed).spawn(len(RNG_STREAM_NAMES))[0].spawn_key), "existing RNG streams must be byte-identical to the production build"

    recoded, dropped_incomplete = build_recoded_ces()
    print(f"CES complete-case pool: {len(recoded)} respondents ({dropped_incomplete} dropped for missing required fields)")
    print(f"Other-gender eligible pool: {int((recoded['gender'] == 'Other').sum())}")

    new_gender_totals = rescaled_gender_totals(N_OTHER)
    print(f"rescaled gender totals (992 total): {dict(new_gender_totals)}")
    new_quota_age, new_quota_race = rescaled_margins(new_gender_totals)

    joint_cells = raking.build_joint_cells_table(recoded, new_quota_age, new_quota_race)
    if int(joint_cells["integer_target_n"].sum()) != N_G - N_OTHER:
        raise RuntimeError("rescaled joint-cell table does not sum to 992")

    male_female = sampling.sample_donors(recoded, joint_cells, generators["pums_selection"])
    other = sample_other_donors(recoded, N_OTHER, generators[OTHER_STREAM_NAME])

    combined = pd.concat([male_female, other], ignore_index=True, sort=False)
    if combined["donor_id"].duplicated().any():
        raise RuntimeError("duplicate donor_id across Male/Female + Other draws")
    if len(combined) != N_G:
        raise RuntimeError(f"expected {N_G} total donors, got {len(combined)}")

    profiles = sampling.assign_latent_profile_ids(combined)

    age_audit = new_quota_age.merge(
        profiles.groupby(["gender", "age_band"]).size().rename("achieved_n").reset_index(), on=["gender", "age_band"], how="left"
    ).fillna({"achieved_n": 0})
    race_audit = new_quota_race.merge(
        profiles.groupby(["gender", "race"]).size().rename("achieved_n").reset_index(), on=["gender", "race"], how="left"
    ).fillna({"achieved_n": 0})
    male_female_exact = bool((age_audit["target_n"] == age_audit["achieved_n"]).all() and (race_audit["target_n"] == race_audit["achieved_n"]).all())
    if not male_female_exact:
        raise RuntimeError(f"rescaled Male/Female quota not achieved exactly:\n{age_audit}\n{race_audit}")

    out_cols = [
        "latent_profile_id", "donor_id", "gender", "age", "age_band", "race", "education", "education_approximated",
        "income", "income_approximated", "party", "state_abbr", "religion", "political_ideology", "pums_person_weight",
    ]
    final = profiles[out_cols].rename(columns={"pums_person_weight": "ces_commonweight"})
    final["population_seed"] = master_seed
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PATH, index=False)

    def sha256_file(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    print(f"\nFINAL_CES_ROSTER_BUILT = YES")
    print(f"FINAL_CES_N = {len(final)}")
    print(f"FINAL_CES_UNIQUE_ROWS = {final['donor_id'].nunique()}")
    print(f"FINAL_CES_OTHER_N = {int((final['gender'] == 'Other').sum())}")
    print(f"FINAL_CES_ROSTER_SHA256 = {sha256_file(OUT_PATH)}")
    print(f"education_approximated: {int(final['education_approximated'].sum())} / {len(final)} ({100 * final['education_approximated'].mean():.2f}%)")
    print(f"income_approximated: {int(final['income_approximated'].sum())} / {len(final)} ({100 * final['income_approximated'].mean():.2f}%)")
    print(f"religion missing: {int(final['religion'].isna().sum())} / {len(final)}")
    print(f"political_ideology missing: {int(final['political_ideology'].isna().sum())} / {len(final)}")
    print(f"\nMale/Female quota audit (age):\n{age_audit.to_string(index=False)}")
    print(f"\nMale/Female quota audit (race):\n{race_audit.to_string(index=False)}")
    print(f"\ngender counts:\n{final['gender'].value_counts().to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
