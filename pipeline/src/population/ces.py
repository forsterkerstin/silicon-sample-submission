"""src/population/ces.py

2024 CES Common Content ingestion, recoding, and the weighted multinomial
party-identification model used to impute partisan identity onto the sampled
ACS PUMS donor profiles.

Every variable, code, and weight-selection decision here is verified against
data/CES_2024_GUIDE_vv.pdf and data/CCES24_Common_pre.docx -- see
reports/population/ces_variable_audit.md and
data/derived/population/ces_variable_mapping.yaml for the evidence trail.
Political-attitude and vote-choice variables (pid7, presvote*, ideo5,
registration/validated-vote status) are never used as predictors or targets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .io import get_logger
from .pums import recode_age_band

logger = get_logger("ces")

REQUIRED_COLUMNS: list[str] = [
    "caseid", "commonweight", "inputstate", "birthyr", "gender4",
    "educ", "race", "hispanic", "faminc_new", "pid3",
]

CES_SURVEY_YEAR = 2024  # per the questionnaire's birthyr screenout ("2024 fielding")

GENDER_MAP: dict[str, str] = {"1": "Male", "2": "Female", "3": "Other", "4": "Other"}
GENDER_MISSING = {"8", "9"}

RACE_MAP_NONHISPANIC: dict[str, str] = {
    "1": "White / Caucasian",
    "2": "Black / African American",
    "4": "Asian / Asian American",
    "5": "Other",
    "6": "Other",
    "7": "Other",
    "8": "Other",
}
RACE_HISPANIC_CODE = "3"
HISPANIC_YES_CODE = "1"
HISPANIC_MISSING = {"8", "9"}

PARTY_MAP: dict[str, str] = {"1": "Democrat", "2": "Republican", "3": "Independent", "4": "Other", "5": "Other"}
PARTY_MISSING = {"8", "9"}

#: CES educ (1-6) -> CES's own native education levels, used as the
#: harmonized_education predictor. See ces_variable_audit.md: CES's single
#: "Post-grad" bucket can't be split into the benchmark's separate
#: Master's/Professional vs. Doctorate levels, so the party model harmonizes
#: on CES's coarser 6-level scheme instead of the benchmark's 6 levels.
HARMONIZED_EDU_FROM_CES: dict[str, str] = {
    "1": "No HS", "2": "HS grad", "3": "Some college",
    "4": "2-year", "5": "4-year", "6": "Post-grad",
}

#: CES faminc_new (1-16) -> (low, high) USD interval, low/high None meaning
#: open-ended. Used both to build the CES-side harmonized_income_ces
#: predictor (bracket midpoint is irrelevant, the code itself is used) and to
#: bin the ACS side's income_adjusted_2024 onto the same boundaries.
CES_INCOME_INTERVALS: dict[str, tuple[float | None, float | None]] = {
    "1": (None, 10_000), "2": (10_000, 20_000), "3": (20_000, 30_000),
    "4": (30_000, 40_000), "5": (40_000, 50_000), "6": (50_000, 60_000),
    "7": (60_000, 70_000), "8": (70_000, 80_000), "9": (80_000, 100_000),
    "10": (100_000, 120_000), "11": (120_000, 150_000), "12": (150_000, 200_000),
    "13": (200_000, 250_000), "14": (250_000, 350_000), "15": (350_000, 500_000),
    "16": (500_000, None),
}
CES_INCOME_MISSING = {"97", "998", "999"}

#: predictors shared by CES (to fit) and ACS (to predict on) -- instructions §17.
PARTY_MODEL_PREDICTORS: list[str] = ["gender", "age_band", "race", "harmonized_education", "harmonized_income_ces", "state_abbr"]
PARTY_CLASSES: list[str] = ["Democrat", "Republican", "Independent", "Other"]


def load_ces(csv_path: Path | str) -> pd.DataFrame:
    """Read only the required CES columns (see REQUIRED_COLUMNS), as strings
    so leading-zero-free but still categorical numeric codes are not
    corrupted by type inference. Raises an explicit KeyError-style error via
    pandas if a required column is absent.
    """
    df = pd.read_csv(csv_path, usecols=REQUIRED_COLUMNS, dtype=str)
    logger.info("loaded CES: %d rows, %d columns", len(df), len(df.columns))
    return df


def recode_gender_ces(code: Any) -> str | None:
    """gender4 -> Male/Female/Other, or None if Skipped/Not Asked/missing."""
    s = str(code).strip() if pd.notna(code) else None
    if s in GENDER_MAP:
        return GENDER_MAP[s]
    return None


def recode_race_ces(race_code: Any, hispanic_code: Any) -> str | None:
    """(race, hispanic) -> the benchmark's race/ethnicity levels, or None if
    unresolvable (missing race, or non-Hispanic race with a missing
    Hispanic-origin follow-up). Hispanic-origin takes priority over race,
    mirroring pums.recode_race's HISP-over-RAC1P priority.
    """
    r = str(race_code).strip() if pd.notna(race_code) else None
    if r is None:
        return None
    if r == RACE_HISPANIC_CODE:
        return "Hispanic / Latino"
    h = str(hispanic_code).strip() if pd.notna(hispanic_code) else None
    if h == HISPANIC_YES_CODE:
        return "Hispanic / Latino"
    if h in HISPANIC_MISSING or h is None:
        return None
    if r in RACE_MAP_NONHISPANIC:
        return RACE_MAP_NONHISPANIC[r]
    return None


def recode_party_ces(pid3_code: Any) -> str | None:
    """pid3 -> Democrat/Republican/Independent/Other, or None if
    Skipped/Not Asked/missing (excluded from party-model training, never
    auto-mapped to Other)."""
    s = str(pid3_code).strip() if pd.notna(pid3_code) else None
    if s in PARTY_MAP:
        return PARTY_MAP[s]
    return None


def harmonized_education_from_ces(educ_code: Any) -> str | None:
    """CES educ -> the harmonized_education predictor's native CES levels."""
    s = str(educ_code).strip() if pd.notna(educ_code) else None
    return HARMONIZED_EDU_FROM_CES.get(s)


def harmonized_education_from_schl(schl_code: Any) -> str | None:
    """ACS SCHL -> the same harmonized_education levels as
    harmonized_education_from_ces, so the party model sees one consistent
    education scale on both the CES training data and the ACS profiles it
    predicts onto.
    """
    s = str(schl_code).strip()
    if not s.isdigit():
        return None
    code = int(s)
    if 1 <= code <= 15:
        return "No HS"
    if 16 <= code <= 17:
        return "HS grad"
    if 18 <= code <= 19:
        return "Some college"
    if code == 20:
        return "2-year"
    if code == 21:
        return "4-year"
    if 22 <= code <= 24:
        return "Post-grad"
    return None


def harmonized_income_bracket_from_amount(amount: float) -> str:
    """Bin a numeric dollar amount onto CES's 16 faminc_new brackets (as
    their string codes "1".."16"), open-ended at both ends. Used to bin the
    ACS side's income_adjusted_2024 onto the harmonized_income_ces scale.
    """
    for code, (low, high) in CES_INCOME_INTERVALS.items():
        if low is not None and amount < low:
            continue
        if high is not None and amount >= high:
            continue
        return code
    # amount >= the last interval's low bound with no upper bound (bracket 16)
    return "16"


def harmonized_income_from_ces(faminc_code: Any) -> str | None:
    """CES faminc_new -> the harmonized_income_ces predictor (its own bracket
    code), or None if it's one of the documented non-interval/missing codes
    (97 Prefer not to say, 998 Skipped, 999 Not Asked).
    """
    s = str(faminc_code).strip() if pd.notna(faminc_code) else None
    if s in CES_INCOME_MISSING or s is None:
        return None
    if s in CES_INCOME_INTERVALS:
        return s
    return None


def build_ces_training_frame(ces_raw: pd.DataFrame) -> pd.DataFrame:
    """Recode raw CES rows into the party-model's training frame: predictors
    (gender, age_band, race, harmonized_education, harmonized_income_ces,
    state_abbr), target (party), and weight (commonweight). Rows with any
    missing predictor/target/nonpositive weight are dropped -- this is the
    CES-side §15 sample restriction (valid party identity, valid weight,
    valid age, valid geography, sufficient demographic predictors), not a
    restriction to registered/validated/self-reported voters.

    The returned frame keeps ces_raw's original row index (not reset) so a
    caller can align other ces_raw columns back onto surviving rows via
    ces_raw.loc[training.index, ...] -- e.g. donor_id/religion/ideology for
    a CES-sourced roster. Every existing caller (run_diagnostics,
    fit_final_model) is index-value-agnostic (verified: identical fitted
    coefficients/diagnostics whether the index is reset or preserved), so
    this is purely additive.
    """
    from .pums import STATE_FIPS_TO_ABBR  # local import: avoid a module-level pums<->ces coupling surprise

    age = CES_SURVEY_YEAR - pd.to_numeric(ces_raw["birthyr"], errors="coerce")
    state_fips = ces_raw["inputstate"].str.strip().str.zfill(2)

    out = pd.DataFrame(
        {
            "weight": pd.to_numeric(ces_raw["commonweight"], errors="coerce"),
            "gender": ces_raw["gender4"].apply(recode_gender_ces),
            "age": age,
            "race": [recode_race_ces(r, h) for r, h in zip(ces_raw["race"], ces_raw["hispanic"])],
            "harmonized_education": ces_raw["educ"].apply(harmonized_education_from_ces),
            "harmonized_income_ces": ces_raw["faminc_new"].apply(harmonized_income_from_ces),
            "state_abbr": state_fips.map(STATE_FIPS_TO_ABBR),
            "party": ces_raw["pid3"].apply(recode_party_ces),
        }
    )
    valid_age = out["age"].between(18, 100)
    out.loc[valid_age, "age_band"] = out.loc[valid_age, "age"].astype(int).apply(recode_age_band)

    required = ["weight", "gender", "age_band", "race", "harmonized_education", "harmonized_income_ces", "state_abbr", "party"]
    complete = out[required].notna().all(axis=1) & (out["weight"] > 0)
    n_before = len(out)
    out = out.loc[complete]
    logger.info("CES training frame: %d/%d rows retained after requiring complete predictors/target/weight", len(out), n_before)
    return out


def _stratified_split_indices(
    strata: pd.Series, test_fraction: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One deterministic stratified split using the given RNG stream (not
    sklearn's train_test_split, so the caller's own numpy Generator -- not a
    bare int random_state -- is the literal source of randomness).
    """
    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, group in strata.groupby(strata):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_fraction))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def _build_pipeline(regularization_c: float, max_iterations: int) -> Pipeline:
    """The prespecified party model: OneHotEncode the categorical predictors,
    multinomial L2-regularized logistic regression. Settings are fixed (not
    tuned against any diagnostic), per instructions §17.
    """
    encoder = ColumnTransformer(
        transformers=[("categorical", OneHotEncoder(handle_unknown="ignore"), PARTY_MODEL_PREDICTORS)],
    )
    classifier = LogisticRegression(
        penalty="l2",
        C=regularization_c,
        max_iter=max_iterations,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=0,
    )
    return Pipeline(steps=[("encode", encoder), ("classify", classifier)])


def run_diagnostics(
    training: pd.DataFrame,
    rng: np.random.Generator,
    test_fraction: float = 0.20,
    regularization_c: float = 1.0,
    max_iterations: int = 5000,
) -> dict[str, Any]:
    """Fit the prespecified model on one deterministic stratified train split
    (stratified on `party`) and report diagnostics on the held-out test
    split. Does not select among model specifications -- this is a fixed,
    single fit used only to document performance, per §17.
    """
    train_idx, test_idx = _stratified_split_indices(training["party"], test_fraction, rng)
    train_df, test_df = training.loc[train_idx], training.loc[test_idx]

    pipeline = _build_pipeline(regularization_c, max_iterations)
    pipeline.fit(train_df[PARTY_MODEL_PREDICTORS], train_df["party"], classify__sample_weight=train_df["weight"])
    classes = list(pipeline.named_steps["classify"].classes_)

    proba = pipeline.predict_proba(test_df[PARTY_MODEL_PREDICTORS])
    pred = pipeline.predict(test_df[PARTY_MODEL_PREDICTORS])
    w_test = test_df["weight"].to_numpy()
    y_test = test_df["party"].to_numpy()

    brier_by_class = {}
    for i, c in enumerate(classes):
        y_c = (y_test == c).astype(float)
        brier_by_class[c] = float(np.average((proba[:, i] - y_c) ** 2, weights=w_test))

    observed_shares = _weighted_shares(train_df["party"], train_df["weight"], classes)
    observed_shares_all = _weighted_shares(training["party"], training["weight"], classes)
    predicted_shares = dict(zip(classes, np.average(proba, axis=0, weights=w_test)))

    ess = float((w_test.sum() ** 2) / (w_test**2).sum())

    return {
        "classes": classes,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "weighted_log_loss": float(log_loss(y_test, proba, sample_weight=w_test, labels=classes)),
        "weighted_accuracy": float(accuracy_score(y_test, pred, sample_weight=w_test)),
        "weighted_brier_by_class": brier_by_class,
        "weighted_confusion_matrix": confusion_matrix(y_test, pred, sample_weight=w_test, labels=classes).tolist(),
        "confusion_matrix_labels": classes,
        "observed_weighted_party_shares_train": observed_shares,
        "observed_weighted_party_shares_all_ces": observed_shares_all,
        "predicted_weighted_party_shares_test": predicted_shares,
        "n_test_effective_sample_size": ess,
        "converged": bool(pipeline.named_steps["classify"].n_iter_[0] < max_iterations),
        "n_iter": int(pipeline.named_steps["classify"].n_iter_[0]),
    }


def _weighted_shares(labels: pd.Series, weights: pd.Series, classes: list[str]) -> dict[str, float]:
    total = weights.sum()
    return {c: float(weights[labels == c].sum() / total) for c in classes}


def fit_final_model(training: pd.DataFrame, regularization_c: float = 1.0, max_iterations: int = 5000) -> Pipeline:
    """Refit the same prespecified model on all valid CES rows (no split),
    per §17's "After producing diagnostics, refit the same prespecified model
    on all valid CES rows."
    """
    pipeline = _build_pipeline(regularization_c, max_iterations)
    pipeline.fit(training[PARTY_MODEL_PREDICTORS], training["party"], classify__sample_weight=training["weight"])
    return pipeline


def predict_party_probabilities(pipeline: Pipeline, profiles: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted model to ACS donor profiles (must already carry
    gender, age_band, race, harmonized_education, harmonized_income_ces,
    state_abbr) and return one probability column per party class, validated
    to be finite, in [0, 1], and sum to 1.
    """
    classes = list(pipeline.named_steps["classify"].classes_)
    proba = pipeline.predict_proba(profiles[PARTY_MODEL_PREDICTORS])
    if not np.all(np.isfinite(proba)):
        raise ValueError("party model produced non-finite probabilities")
    if not np.all((proba >= 0) & (proba <= 1)):
        raise ValueError("party model produced probabilities outside [0, 1]")
    row_sums = proba.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-10):
        raise ValueError("party model probabilities do not sum to 1 within 1e-10")
    if set(classes) != set(PARTY_CLASSES):
        raise ValueError(f"fitted model classes {classes} != expected {PARTY_CLASSES}")
    out = pd.DataFrame(proba, columns=[f"party_prob_{c.lower()}" for c in classes], index=profiles.index)
    return out
