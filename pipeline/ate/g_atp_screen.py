"""ATP1/ATP2 mini G-model screen: profile mapping, item construction, and the
frozen primary loss (equal-item-weight normalized Wasserstein-1 on [0,1]).

Source: https://github.com/skrsteski/survey-simulations @
faeb4e1a73567a8c98c69798774b63fdb27c79e1 (MIT license). Local copies at
pipeline/data/atp_survey_simulations/atp{1,2}_human_test.csv (gitignored,
regenerable by re-cloning that commit).

ATP1 = LIVSTAN_W149 (5-point ordinal: standard of living vs. parents).
ATP2 = SCOTUS_JOB_W149 (4-point ordinal: Supreme Court impartiality).
ATP2's 642 respondents are a strict subset of ATP1's 690 -- this is two
items on essentially one panel, not two independent studies, and is scored
with equal ITEM weight (not equal-study weight) for exactly that reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

# --- ATP -> G profile mapping (one representation per concept, chosen
# prospectively; documented, not duplicated across multiple ATP columns) ---
#
# concept              -> ATP column   -> rationale (one ATP column per G concept)
# age                  -> age_cat      -> only age variable available (banded); shown verbatim, never inflated to a fake precise integer
# gender               -> gender       -> only gender variable; shown verbatim ("A man"/"A woman"/"In some other way"), not forced into a binary
# race/ethnicity       -> race_ethn    -> combines race+Hispanic origin into ONE categorical, matching G's single "Race/ethnicity" field; `race` alone is not used (would duplicate/conflict)
# education            -> edu_cat2     -> finer 6-level education, closer to target G's own donor persona granularity than the coarser 3-level `edu_cat` (not used, to avoid duplication)
# income               -> family_income-> dollar-band format matches target G's own donor income convention; `income_tier3` (Lower/Middle/Upper) not used, to avoid duplicating the same concept
# party identification -> party       -> direct categorical match (Democrat/Republican/Independent/Something else); `party_summary`/`party_ideo` not used (party_ideo conflates party with ideology)
# political ideology   -> ideology    -> standalone Very liberal..Very conservative scale; not duplicated via party_ideo
# religion              -> religion_4cat -> coarser 4-category summary preferred over the 13-level `religion` (parsimony, lower re-identification risk from rare categories); not both
# state                -> ABSENT       -> never collected in ATP; omitted, not fabricated
#
# Explicitly excluded (attitudes/behavior/outcome-adjacent, or not part of
# target G's 8-concept profile schema): metro, region, division, years_in_us,
# attend_relig, registered, internet_use, volunteer, birthplace (mislabeled --
# actually records born-again/evangelical status, not birthplace),
# party_lean, hispanic, persona_description (LLM-written attitude narrative;
# leaks other/target attitudes -- never used).
ATP_PROFILE_FIELD_MAP: dict[str, str] = {
    "age_cat": "age",
    "gender": "gender",
    "race_ethn": "race",
    "edu_cat2": "education",
    "family_income": "income",
    "party": "party",
    "ideology": "political_ideology",
    "religion_4cat": "religion",
}

# ATP encodes a non-response as the literal string "Refused" in several of
# these columns (and "Refused" is itself a valid demographic-field value in
# the raw data) -- treated as NOT PRESENT for profile rendering, exactly like
# a missing/NaN value, since "Party identification: Refused" does not
# describe an attribute of the respondent. Never fabricated, never imputed.
_REFUSED_TOKENS = {"refused"}


def atp_row_to_g_profile(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    for atp_col, g_key in ATP_PROFILE_FIELD_MAP.items():
        value = row.get(atp_col) if isinstance(row, dict) else row.get(atp_col)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).strip()
        if not text or text.lower() in _REFUSED_TOKENS:
            continue
        profile[g_key] = text
    return profile


# --- item construction (feeds build_g_external_validation_prompt_render /
# item_json_schema / _questions_block -- the SAME generic item-dict shape F's
# external items already use) ---

ATP1_SUBSTANTIVE_CHOICES = (
    "Much better than your parents",
    "Somewhat better than your parents",
    "About the same as your parents",
    "Somewhat worse than your parents",
    "Much worse than your parents",
)
ATP2_SUBSTANTIVE_CHOICES = ("Excellent", "Good", "Only fair", "Poor")

# Non-substantive response labels excluded from BOTH human eligibility and
# model response support, reproducing datasets/atp{1,2}/evaluate.py's own
# RESPONSE_MAP convention exactly (there, "Not sure" and "Refused/Web blank"
# both map to NaN and are dropped) -- not invented here.
_NON_SUBSTANTIVE_LABELS = {"not sure", "refused/web blank"}


def _item_for(source_id: str, question_text: str, choices: tuple[str, ...]) -> dict[str, Any]:
    numbered = "; ".join(f"{i + 1}={c}" for i, c in enumerate(choices))
    return {
        "qualtrics_label": source_id.lower(),
        "target_label": source_id.lower(),
        "response_key": "response",
        "question_text": f"{question_text}\n\nChoices: {numbered}",
        "response_options": f"Respond with the integer position (1-{len(choices)}) of your chosen option.",
        "scale": "external_native_integer",
        "scale_min": 1,
        "scale_max": len(choices),
    }


def atp1_item() -> dict[str, Any]:
    return _item_for(
        "ATP1",
        "Compared to your parents when they were the age you are now, do you think your own standard of living now is...",
        ATP1_SUBSTANTIVE_CHOICES,
    )


def atp2_item() -> dict[str, Any]:
    return _item_for(
        "ATP2",
        "How would you rate the job Supreme Court justices are doing in keeping their own political views out of how they decide major cases?",
        ATP2_SUBSTANTIVE_CHOICES,
    )


# --- human eligibility (reproduces evaluate.py's dropna-after-RESPONSE_MAP
# convention exactly: "Not sure" and "Refused/Web blank" are excluded) ---


def usable_atp1_respondents(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["answer_label"].str.strip().str.lower().isin(_NON_SUBSTANTIVE_LABELS)].copy()


def usable_atp2_respondents(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["correct_answer"].str.strip().str.lower().isin(_NON_SUBSTANTIVE_LABELS)].copy()


def _reference_positions(labels, choices: tuple[str, ...]) -> list[float]:
    positions = []
    for lbl in labels:
        pos = position_on_unit_interval(lbl, choices)
        if pos is None:
            raise ValueError(f"non-substantive or unrecognized label {lbl!r} survived usable-respondent filtering")
        positions.append(pos)
    return positions


def atp1_human_reference_positions(df: pd.DataFrame) -> list[float]:
    """The frozen human reference distribution for ATP1: every usable
    respondent's answer_label mapped to its [0,1] position. Candidate-
    independent -- computed once from the human data alone."""
    return _reference_positions(usable_atp1_respondents(df)["answer_label"], ATP1_SUBSTANTIVE_CHOICES)


def atp2_human_reference_positions(df: pd.DataFrame) -> list[float]:
    """The frozen human reference distribution for ATP2. ATP2's substantive
    answer is stored in the source's own 'correct_answer' column (not
    'answer_label' -- that quirk is the source data's, not invented here)."""
    return _reference_positions(usable_atp2_respondents(df)["correct_answer"], ATP2_SUBSTANTIVE_CHOICES)


# --- equally-spaced [0,1] position mapping (fresh construction per the
# frozen rule -- NOT the source's own inconsistent 0-4/1-4 RESPONSE_MAP
# integers, which differ in base between ATP1 and ATP2) ---


def position_on_unit_interval(label: str, choices: tuple[str, ...]) -> float | None:
    label_norm = str(label).strip()
    if label_norm.lower() in _NON_SUBSTANTIVE_LABELS:
        return None
    if label_norm not in choices:
        return None
    idx = choices.index(label_norm)
    n = len(choices)
    return idx / (n - 1) if n > 1 else 0.0


def model_response_to_unit_interval(response_position_1_indexed: int, choices: tuple[str, ...]) -> float:
    """response_position_1_indexed: the model's integer answer (1..len(choices)),
    matching the numbered choices shown in the item text."""
    n = len(choices)
    idx = response_position_1_indexed - 1
    if not (0 <= idx < n):
        raise ValueError(f"response position {response_position_1_indexed} outside 1..{n}")
    return idx / (n - 1) if n > 1 else 0.0


# --- primary loss: equal-item-weight mean of normalized W1 (pp) ---


def item_w1_pp(human_positions: list[float], model_positions: list[float]) -> float:
    if not human_positions or not model_positions:
        raise ValueError("item_w1_pp requires non-empty human and model position lists")
    return 100 * float(wasserstein_distance(human_positions, model_positions))


def g_atp_loss(w1_pp_atp1: float, w1_pp_atp2: float) -> float:
    """Equal ITEM weighting (not equal-study weighting), because ATP1/ATP2
    are two items on essentially one panel, not two independent studies."""
    return (w1_pp_atp1 + w1_pp_atp2) / 2


def select_g_star(
    loss_by_model: dict[str, float],
    *,
    invalid_response_rate: dict[str, float],
    realized_cost_usd: dict[str, float],
) -> dict[str, Any]:
    """Lowest g_atp_loss wins. Tie-break identical in shape to ate.f_screen
    and submission.g_screen: lower invalid-response rate, then lower
    realized cost, then lexical model id."""
    models = sorted(loss_by_model.keys())
    if set(models) != set(invalid_response_rate) or set(models) != set(realized_cost_usd):
        raise ValueError("loss_by_model, invalid_response_rate, and realized_cost_usd must cover the same model set")

    def sort_key(model: str) -> tuple:
        return (round(loss_by_model[model], 10), round(invalid_response_rate[model], 10), round(realized_cost_usd[model], 10), model)

    winner = min(models, key=sort_key)
    return {
        "g_star": winner,
        "ranked": sorted(models, key=sort_key),
        "primary_metric_values": loss_by_model,
        "tie_break_rule": "lower invalid-response rate, then lower realized inference cost, then deterministic lexical model id",
    }
