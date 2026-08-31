"""Orchinik et al. (2024) as domain-specific confirmation of the frozen
respondent-simulator choice G* = google/gemma-4-31B-it (vs.
deepseek-ai/DeepSeek-V4-Pro-0813), using the main Bovitz nationally
representative sample.

Superseded role note: an earlier turn this session explored using Orchinik
to estimate a gamma_G shape-validation slope (see outputs/domain_validation/
frozen_domain_validation_protocol.json, commit 7ea86ef). That role is
SUPERSEDED_FOR_CURRENT_METHOD_DECISION -- this module implements the NEW
role only (domain-specific G-vs-DeepSeek confirmation). The prior 50-cell
human ATE surface and its artifacts are untouched and still committed for
provenance; nothing here reads or depends on them, and nothing here may be
used to estimate MU_EXTERNAL or gamma_G, or to select between S1/S2.

Every persona field, condition-material passage, and question text below is
copied verbatim from the real, already-downloaded, already-hashed released
materials (data/domain_validation/orchinik/Bovitz_qualtrics.docx and .qsf,
final_clean.csv) -- nothing is inferred or fabricated. See
scripts/build_orchinik_g_domain_confirmation_manifest.py's module docstring
for the exact extraction provenance.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import survey_content as sc

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = PIPELINE_ROOT / "data" / "domain_validation" / "orchinik" / "final_clean.csv"

EXPECTED_ELIGIBLE_N = 2545

# --- randomized conditions: raw `condition` column value -> paper's display
# label -> exact participant-visible passage (verbatim from Bovitz_qualtrics.docx;
# control sees only a neutral transition screen, no persuasive passage) ---
CONDITION_DISPLAY_LABEL = {"control": "control", "skill": "History", "trust": "Institutions"}

CONDITION_MATERIAL = {
    "control": "",
    "skill": (
        "Climate change might seem like a sudden and modern problem, but the study of climate change is actually "
        "a long-established science. The basic dynamics of carbon dioxide (C02) and atmospheric warming were "
        "first understood in 1860 by physicist John Tyndall, based on earlier insights by Joseph Fourier in 1824. "
        "To track climate, scientists have been recording temperature measurements for hundreds of years, and "
        "CO2 has been measured directly at Mauna Loa Observatory since 1958. Famously, NASA scientist James "
        "Hansen testified to Congress in 1988 stating the greenhouse effect had been detected."
    ),
    "trust": (
        "Scientists and universities take many steps to reduce systemic bias and make their science objective "
        "and open. Scientific journals require conflict of interest statements and other academics, journalists, "
        "and public interest organizations investigate the funding sources of scientists. Those who receive "
        "funding from vested interests are often sanctioned by their peers and their research is taken less "
        "seriously. Scientists want their papers to be available to the public and generate conversation about "
        "important topics, with a survey showing that over 95% of scientists think public access to their "
        "research is important."
    ),
}

CONSENSUS_LEVELS = [50, 75, 90, 97, 99]  # verbatim from the .qsf EmbeddedData block (num1..num5), fixed, not randomized

# --- focal outcomes: target_label prefix (matches final_clean.csv's
# P_<prefix>_given_cons{50,75,90,97,99} columns) -> (display name, exact
# preamble text verbatim from the instrument) ---
FOCAL_OUTCOME_PREAMBLE = {
    "cc": (
        "Suppose that 100 randomly-selected climate scientists were asked whether or not they agree that "
        "human-caused climate change is happening. For each of the levels of agreement below, what do you think "
        "is the likelihood that human-caused climate change is occurring?"
    ),
    "pro_bias": (
        "Suppose that 100 randomly-selected climate scientists were asked whether or not they agree that "
        "human-caused climate change is happening. For each of the levels of agreement below, please consider "
        "the following 2 questions. Let's say that a scientist is \"extremely biased\" if they always express "
        "the same opinion about whether human-caused climate change is occurring, regardless of what the "
        "evidence suggests. That means they would always agree or disagree, no matter what. How likely do you "
        "think it is that a random climate scientist who expresses that human-caused climate change is occurring "
        "is extremely biased?"
    ),
    "anti_bias": (
        "Suppose that 100 randomly-selected climate scientists were asked whether or not they agree that "
        "human-caused climate change is happening. Let's say that a scientist is \"extremely biased\" if they "
        "always express the same opinion about whether human-caused climate change is occurring, regardless of "
        "what the evidence suggests. That means they would always agree or disagree, no matter what. How likely "
        "do you think it is that a random climate scientist who expresses that human-caused climate change is "
        "NOT occurring is extremely biased?"
    ),
    "pro_skill": (
        "Suppose that 100 randomly-selected climate scientists were asked whether or not they agree that "
        "human-caused climate change is happening. For each of the levels of agreement below, please consider "
        "the following 2 questions. Let's say that a scientist is \"capable\" and unbiased if they have the "
        "skills to correctly identify whether climate change is occurring from available evidence. In other "
        "words, they have arrived at their conclusion because of skill. How likely do you think it is that a "
        "random and unbiased climate scientist who expresses that human-caused climate change is occurring is "
        "capable, meaning they arrived at this conclusion due to skill?"
    ),
    "anti_skill": (
        "Suppose that 100 randomly-selected climate scientists were asked whether or not they agree that "
        "human-caused climate change is happening. Let's say that a scientist is \"capable\" and unbiased if "
        "they have the skills to correctly identify whether climate change is occurring from available evidence. "
        "In other words, they have arrived at their conclusion because of skill. How likely do you think it is "
        "that a random and unbiased climate scientist who expresses that human-caused climate change is NOT "
        "occurring is capable, meaning they arrived at this conclusion due to skill?"
    ),
}
FOCAL_OUTCOME_COLUMN_PREFIX = {"cc": "P_cc_given_cons", "pro_bias": "P_pro_bias_given_cons", "anti_bias": "P_anti_bias_given_cons", "pro_skill": "P_pro_skill_given_cons", "anti_skill": "P_anti_skill_given_cons"}
ROW_ITEM_TEXT_TEMPLATE = "Suppose that {level} out of 100 climate scientists expressed agreement."

# --- pretreatment demographics: raw code -> verbatim response-option label
# (from the released Qualtrics instrument; "nick" is an attention-check
# item, not a demographic, and is never used for persona construction) ---
GENDER_LABELS = {"1": "Male", "2": "Female", "5": "Non-Binary", "6": "Not listed", "7": "Prefer not to answer"}
RACE_LABELS = {
    "1": "American Indian or Alaska Native",
    "3": "Black or African American",
    "4": "White/Caucasian",
    "10": "Native Hawaiian or other Pacific Islander",
    "11": "Hispanic/Latino",
    "12": "Indian",
    "13": "Middle Eastern",
    "14": "Chinese",
    "15": "Other",
}
EDU_LABELS = {"1": "Less than a high school degree", "2": "High School Diploma", "3": "Vocational Training", "4": "Attended College", "5": "Bachelor's Degree", "6": "Graduate Degree", "7": "Unknown"}
INCOME_LABELS = {"1": "Less than $20,000", "2": "$20,000 to $39,999", "3": "$40,000 to $59,999", "4": "$60,000 to $79,999", "5": "$80,000 to $99,999", "6": "$100,000 to $149,999", "7": "$150,000 or more"}
PARTY_LABELS = {"1": "Democrat", "2": "Republican", "3": "Independent", "4": "Other"}
POLITICS_LABELS = {"1": "Strongly Democratic", "2": "Democratic", "3": "Lean Democratic", "4": "True Independent", "5": "Lean Republican", "6": "Republican", "7": "Strongly Republican"}
IDEOLOGY_LABELS = {"1": "Strongly Liberal", "2": "Somewhat Liberal", "3": "Moderate", "4": "Somewhat Conservative", "5": "Strongly Conservative"}
GOD_BELIEF_OUT_OF_7 = {str(i): str(i - 1) for i in range(1, 9)}  # code i (1..8) -> belief level (i-1) out of 7, per the instrument's own 0..7 anchored scale

FORBIDDEN_PERSONA_SOURCE_COLUMNS = {
    "prior_cc_occur",
    "prior_cc_conf",
    "prior_consensus_num",
    "prior_consensus_num_conf",
    "prior_sci_biased",
    "prior_sci_biased_yes",
    "P_E_yes_given_cc_unbiased",
    "P_E_no_given_no_cc_unbiased",
    "gov.trust",
    "pol.party.trust",
    "uni.science.trust",
    "priv.science.trust",
    "affpol_thermom_1",
    "affpol_thermom_2",
    "belief_shift_climate",
    "belief_shift_skill",
    "belief_shift_unbiased",
}


def load_eligible_respondents() -> list[dict[str, str]]:
    """Real, already-downloaded, already-hashed Bovitz clean data, filtered
    by the authors' own drop==FALSE eligibility rule (verified elsewhere
    this session to reproduce the paper's own reported N=2,545 exactly)."""
    with open(CLEAN_DATA_PATH, newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("drop") == "FALSE"]
    if len(rows) != EXPECTED_ELIGIBLE_N:
        raise ValueError(f"eligible N mismatch: expected {EXPECTED_ELIGIBLE_N}, got {len(rows)} -- STOP rather than silently using a different N")
    return rows


def _label(mapping: dict[str, str], code: str) -> str | None:
    code = (code or "").strip()
    if not code:
        return None
    return mapping.get(code)


def respondent_to_g_profile(row: dict[str, str]) -> dict[str, Any]:
    """Real pretreatment demographics ONLY -- age, gender, race/ethnicity,
    education, household income, party identification, political ideology
    (social and economic, since Bovitz records these as two distinct
    dimensions rather than one), and belief in God/Gods (the closest
    available pretreatment religiosity measure; Bovitz records no
    denomination). No prior climate belief, consensus estimate, trust/bias/
    skill judgment, institutional-trust rating, affect-thermometer, or any
    other outcome/mediator/posttreatment field ever enters this function.
    Missing/blank raw values are simply omitted, never inferred."""
    # NOTE: keys here are chosen to match inference.prompts.PROFILE_FIELD_ORDER
    # exactly (age/gender/race/education/income/party/political_ideology/
    # religion) -- that frozen renderer only emits keys it recognizes, so
    # any other key name would be silently dropped from the rendered prompt.
    # No fabrication: every value below is still copied verbatim from the
    # real released response options; only the DICT KEY is chosen to match
    # the existing rendering contract. Bovitz has no state/region field, so
    # that slot is legitimately omitted (never fabricated).
    profile: dict[str, Any] = {}

    age = (row.get("age") or "").strip()
    if age.isdigit():
        profile["age"] = int(age)

    gender = _label(GENDER_LABELS, row.get("gender", ""))
    if gender:
        profile["gender"] = gender

    race_raw = (row.get("race") or "").strip()
    if race_raw:
        labels = [RACE_LABELS[c.strip()] for c in race_raw.split(",") if c.strip() in RACE_LABELS]
        if labels:
            profile["race"] = " and ".join(labels)

    edu = _label(EDU_LABELS, row.get("edu", ""))
    if edu:
        profile["education"] = edu

    income = _label(INCOME_LABELS, row.get("income", ""))
    if income:
        profile["income"] = income

    # party: Bovitz's coarse Dem/Rep/Ind/Other item, with the finer 7-point
    # party-lean item appended in parentheses when present -- both are real,
    # both are pretreatment, neither is invented; this preserves both
    # dimensions under the one recognized "party" slot rather than dropping one.
    party = _label(PARTY_LABELS, row.get("party", ""))
    politics = _label(POLITICS_LABELS, row.get("politics", ""))
    if party and politics:
        profile["party"] = f"{party} (lean: {politics})"
    elif party:
        profile["party"] = party
    elif politics:
        profile["party"] = politics

    # political_ideology: Bovitz records social and economic ideology as two
    # distinct 5-point items rather than one combined item -- both are
    # preserved verbatim under the one recognized "political_ideology" slot.
    social = _label(IDEOLOGY_LABELS, row.get("politics_social", ""))
    econ = _label(IDEOLOGY_LABELS, row.get("politics_econ", ""))
    if social and econ:
        profile["political_ideology"] = f"{social} on social issues, {econ} on economic issues"
    elif social:
        profile["political_ideology"] = f"{social} on social issues"
    elif econ:
        profile["political_ideology"] = f"{econ} on economic issues"

    # religion: Bovitz records no denomination, only a 0-7 belief-in-God/Gods
    # intensity item -- reported explicitly as that measure, not relabeled
    # as denominational religion.
    god = _label(GOD_BELIEF_OUT_OF_7, row.get("god", ""))
    if god is not None:
        profile["religion"] = f"belief in God/Gods: {god} out of 7"

    return profile


def build_25_items() -> list[dict[str, Any]]:
    """25 self-contained item dicts (5 focal outcomes x 5 consensus levels),
    in the same {qualtrics_label, target_label, question_text,
    response_options, scale} shape survey_content.load_items() uses, so the
    frozen item_json_schema/_questions_block/build_g_external_validation_
    prompt_render machinery can render them unmodified."""
    items = []
    for outcome, preamble in FOCAL_OUTCOME_PREAMBLE.items():
        for level in CONSENSUS_LEVELS:
            target_label = f"{outcome}_cons{level}"
            items.append(
                {
                    "qualtrics_label": target_label,
                    "target_label": target_label,
                    "question_text": f"{preamble}\n{ROW_ITEM_TEXT_TEMPLATE.format(level=level)}",
                    "response_options": "An integer from 0 to 100.",
                    "scale": sc.SCALE_SLIDER_0_100,
                }
            )
    if len(items) != 25:
        raise ValueError(f"expected exactly 25 items, got {len(items)}")
    return items


def outcome_column(outcome: str, level: int) -> str:
    return f"{FOCAL_OUTCOME_COLUMN_PREFIX[outcome]}{level}"


def all_75_cells() -> list[tuple[str, str, int]]:
    """(condition, outcome, consensus_level) -- 3 arms x 5 outcomes x 5 levels."""
    cells = [(c, o, lvl) for c in ("control", "skill", "trust") for o in FOCAL_OUTCOME_PREAMBLE for lvl in CONSENSUS_LEVELS]
    if len(cells) != 75:
        raise ValueError(f"expected exactly 75 cells, got {len(cells)}")
    return cells
