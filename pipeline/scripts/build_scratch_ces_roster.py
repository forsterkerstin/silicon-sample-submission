#!/usr/bin/env python3
"""scripts/build_scratch_ces_roster.py

SCRATCH feasibility artifact only -- proves whether an all-CES N_G=1000
donor roster can match the SAME already-frozen 40-cell (gender, age_band,
race) quota objective as the production PUMS roster, using genuine CES
respondent rows. Writes to data/derived/population/scratch_ces_roster_n1000.csv
-- NEVER touches data/generated/g_personas_master.csv (the real production
roster) or any other frozen production artifact.

Reuses, UNMODIFIED: src/population/sampling.py's sample_donors,
assign_latent_profile_ids, quota_audit; src/population/ces.py's
build_ces_training_frame (gender/age_band/race/state_abbr/party recoding,
byte-identical to the CES columns already used for the production
party-imputation model); the same cached data/derived/population/
joint_cells_40.csv 40-cell target table and config/quota_gender_age_1000.csv
/ quota_gender_race_1000.csv quota tables the production PUMS roster
matches exactly; the same master_seed (20260831) from config/population.yaml.

Two fields (education, income) require an additional, EXPLICITLY DISCLOSED
approximation beyond what build_ces_training_frame already does, because
CES's native brackets do not align 1:1 with the benchmark's category
boundaries -- see EDUCATION_POSTGRAD_DEFAULT / INCOME_MIDPOINT_RULE below.
Two more fields (religion, ideology) are new for this scratch roster (never
needed for CES's existing party-model-training role). Both mappings have
been independently verified against the primary CES documentation (a later
session confirmed data/CES_2024_GUIDE_vv.pdf's printed (label, N) pairs
match this CSV's own value_counts() exactly for both religpew and ideo5 --
see RELIGION_FROM_RELIGPEW_VERIFIED / IDEOLOGY_FROM_IDEO5 below).

No paid inference. No LLM calls. No target requests. No production file is
modified.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from population import ces, sampling  # noqa: E402
from population.constants import spawn_rngs  # noqa: E402

CES_CSV_PATH = REPO_ROOT / "data" / "CCES24_Common_OUTPUT_vv_topost_final.csv"
JOINT_CELLS_PATH = REPO_ROOT / "data" / "derived" / "population" / "joint_cells_40.csv"
QUOTA_AGE_PATH = REPO_ROOT / "config" / "quota_gender_age_1000.csv"
QUOTA_RACE_PATH = REPO_ROOT / "config" / "quota_gender_race_1000.csv"
OUT_PATH = REPO_ROOT / "data" / "derived" / "population" / "scratch_ces_roster_n1000.csv"
POP_CONFIG_PATH = REPO_ROOT / "config" / "population.yaml"

# CES educ (native 6-level) -> benchmark's 6-level education categories.
# 5 of 6 map cleanly; "Post-grad" cannot be split into the benchmark's
# separate Master's/Professional vs. Doctorate levels (CES has one combined
# bucket -- documented in src/population/ces.py's HARMONIZED_EDU_FROM_CES
# docstring). Defaulting every Post-grad respondent to "Master's degree /
# Professional degree" (the larger of the two nationally) is a disclosed,
# deterministic approximation for this feasibility test, NOT a verified
# recovery of each respondent's true terminal degree -- every affected row
# is flagged in the output's education_approximated column.
EDUCATION_FROM_CES_EDUC = {
    "1": "Less than high school",
    "2": "High school diploma / GED",
    "3": "Some college or Associate's degree",
    "4": "Some college or Associate's degree",
    "5": "Bachelor's degree",
    "6": "Master's degree / Professional degree",  # approximated -- see above
}
EDUCATION_POSTGRAD_CODE = "6"

# CES faminc_new brackets vs. the benchmark's 4 income thresholds
# ($30k/$56k/$100k/$168k): $30k and $100k land exactly on a CES bracket
# edge; $56k falls inside CES bracket 6 ($50k-$60k) and $168k falls inside
# CES bracket 12 ($150k-$200k). For those two straddling brackets only, the
# bracket's MIDPOINT determines the benchmark category (midpoint $55k <
# $56k; midpoint $175k > $168k) -- a disclosed, deterministic
# boundary-resolution rule, not an invented objective. Every affected row is
# flagged in income_approximated.
INCOME_STRADDLING_CODES = {"6", "12"}


def _education_from_educ(code: str | None) -> tuple[str | None, bool]:
    if code is None or code not in EDUCATION_FROM_CES_EDUC:
        return None, False
    return EDUCATION_FROM_CES_EDUC[code], code == EDUCATION_POSTGRAD_CODE


def _income_from_faminc(code: str | None) -> tuple[str | None, bool]:
    low, high = ces.CES_INCOME_INTERVALS.get(code, (None, None)) if code else (None, None)
    if code is None or code not in ces.CES_INCOME_INTERVALS:
        return None, False
    thresholds = [(30_000, "Less than $30,000"), (56_000, "$30,000 to $55,999"), (100_000, "$56,000 to $99,999"), (168_000, "$100,000 to $167,999")]
    lo = low if low is not None else -1
    hi = high if high is not None else float("inf")
    midpoint = (lo + hi) / 2 if high is not None else lo + 50_000  # open-ended top bracket: any point past $168k
    if midpoint < 30_000:
        label = "Less than $30,000"
    elif midpoint < 56_000:
        label = "$30,000 to $55,999"
    elif midpoint < 100_000:
        label = "$56,000 to $99,999"
    elif midpoint < 168_000:
        label = "$100,000 to $167,999"
    else:
        label = "$168,000 or more"
    return label, code in INCOME_STRADDLING_CODES


# CES religpew (1-12) -> religion label. VERIFIED (a later session) against
# the authoritative data/CES_2024_GUIDE_vv.pdf, p.60 ("Religion / What is
# your present religion, if any? / religpew"): every one of the 12
# codebook-printed (label, N) pairs matches this CSV's religpew value_counts()
# exactly (19038/10861/623/315/1483/413/521/204/4584/4414/12647/4837,
# missing=60) -- a pure one-to-one categorical recode, no inference. The
# prior session's stated blocker (PDF text extraction unavailable) did not
# hold in this environment (pdfplumber/PyMuPDF both work).
RELIGION_FROM_RELIGPEW_VERIFIED = {
    "1": "Protestant", "2": "Roman Catholic", "3": "Mormon", "4": "Eastern or Greek Orthodox",
    "5": "Jewish", "6": "Muslim", "7": "Buddhist", "8": "Hindu",
    "9": "Atheist", "10": "Agnostic", "11": "Nothing in particular", "12": "Something else",
}

# CES ideo5 (1-6): 1..5 = Very liberal..Very conservative, 6 = Not sure.
# Code 6 and missing are excluded (not fabricated), mirroring this
# pipeline's existing convention for ATP's "Not sure"/"Refused" exclusion.
IDEOLOGY_FROM_IDEO5 = {
    "1": "Very liberal", "2": "Liberal", "3": "Moderate",
    "4": "Conservative", "5": "Very conservative",
}


def load_ces_raw_extended() -> pd.DataFrame:
    cols = list(ces.REQUIRED_COLUMNS) + ["religpew", "ideo5"]
    df = pd.read_csv(CES_CSV_PATH, usecols=cols, dtype=str)
    return df


def build_recoded_ces() -> pd.DataFrame:
    raw = load_ces_raw_extended()
    training = ces.build_ces_training_frame(raw)  # UNMODIFIED reuse: gender/age_band/race/harmonized_*/state_abbr/party/weight, complete-case filtered

    extra = raw.loc[training.index, ["educ", "faminc_new", "religpew", "ideo5"]].copy()
    edu = extra["educ"].apply(_education_from_educ)
    training["education"] = [e[0] for e in edu]
    training["education_approximated"] = [e[1] for e in edu]
    inc = extra["faminc_new"].apply(_income_from_faminc)
    training["income"] = [i[0] for i in inc]
    training["income_approximated"] = [i[1] for i in inc]
    training["religion"] = extra["religpew"].map(RELIGION_FROM_RELIGPEW_VERIFIED)
    training["political_ideology"] = extra["ideo5"].map(IDEOLOGY_FROM_IDEO5)

    training["donor_id"] = raw.loc[training.index, "caseid"].astype(str)
    training["pums_person_weight"] = training["weight"]  # interface alias only -- sample_donors() expects this column name; not a new weight

    complete = training["education"].notna() & training["income"].notna()
    dropped = int((~complete).sum())
    training = training.loc[complete].copy()

    if training["donor_id"].duplicated().any():
        raise RuntimeError("duplicate CES caseid in the complete-case pool")

    return training, dropped


def main() -> int:
    pop_cfg = yaml.safe_load(POP_CONFIG_PATH.read_text(encoding="utf-8"))["population"]
    master_seed = int(pop_cfg["master_seed"])
    generators, _spawn_keys = spawn_rngs(master_seed)

    recoded, dropped_incomplete = build_recoded_ces()
    print(f"CES complete-case pool: {len(recoded)} respondents ({dropped_incomplete} dropped for missing education/income/other required fields)")

    joint_cells = pd.read_csv(JOINT_CELLS_PATH)
    quota_age = pd.read_csv(QUOTA_AGE_PATH)
    quota_race = pd.read_csv(QUOTA_RACE_PATH)

    selected = sampling.sample_donors(recoded, joint_cells, generators["pums_selection"])
    profiles = sampling.assign_latent_profile_ids(selected)

    audit = sampling.quota_audit(profiles, quota_age, quota_race)  # raises SamplingError on any mismatch

    out_cols = [
        "latent_profile_id", "donor_id", "gender", "age_band", "race", "education", "education_approximated",
        "income", "income_approximated", "party", "state_abbr", "religion", "political_ideology", "pums_person_weight",
    ]
    final = profiles[out_cols].rename(columns={"pums_person_weight": "ces_commonweight"})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_PATH, index=False)

    def sha256_file(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    prod_path = REPO_ROOT / "data" / "generated" / "g_personas_master.csv"
    print(f"\nSCRATCH_CES_ROSTER_BUILT = YES")
    print(f"SCRATCH_CES_N = {len(final)}")
    print(f"SCRATCH_CES_OTHER_N = 0 (this roster matches the EXISTING Male/Female-only 40-cell quota table verbatim -- no Other-gender quota exists to draw against; see the separate provenance-decision task)")
    print(f"SCRATCH_CES_ROSTER_SHA256 = {sha256_file(OUT_PATH)}")
    print(f"PUMS_ROSTER_SHA256 = {sha256_file(prod_path) if prod_path.exists() else 'MISSING'}")
    print(f"education_approximated (Post-grad, N/n=6): {int(final['education_approximated'].sum())} / {len(final)} ({100*final['education_approximated'].mean():.1f}%)")
    print(f"income_approximated (straddling brackets): {int(final['income_approximated'].sum())} / {len(final)} ({100*final['income_approximated'].mean():.1f}%)")
    print(f"religion missing (religpew NaN/unmapped): {int(final['religion'].isna().sum())} / {len(final)}")
    print(f"political_ideology missing (ideo5 Not sure/skip): {int(final['political_ideology'].isna().sum())} / {len(final)}")
    print("\nquota audit (gender x age_band):")
    print(audit["gender_age"].to_string(index=False))
    print("\nquota audit (gender x race):")
    print(audit["gender_race"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
