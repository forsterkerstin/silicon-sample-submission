"""pipeline/survey_content.py

Reads the benchmark's organizer-provided survey materials -- strictly
READ-ONLY, never modified -- and exposes what the elicitation pipeline needs:

  1. get_condition_stimulus(condition, state_abbr) -- the intervention/control
     stimulus text a respondent would have read, parsed from
     survey/survey.json (the Qualtrics Survey Definitions API export),
     matched to a condition title via survey/condition_codenames.csv.
  2. ITEMS / OUTCOME_COMPOSITES -- every raw survey item that needs eliciting
     and how the 13 preregistered outcomes are computed from them, parsed
     from codebook.csv.

Organizer files read (never written): survey/survey.json,
survey/condition_codenames.csv, codebook.csv, all resolved relative to the
repository root (two levels up from this file: pipeline/ -> repo root).
"""

from __future__ import annotations

import csv
import html
import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SURVEY_JSON_PATH = REPO_ROOT / "survey" / "survey.json"
CONDITION_CODENAMES_PATH = REPO_ROOT / "survey" / "condition_codenames.csv"
CODEBOOK_PATH = REPO_ROOT / "codebook.csv"

#: Vintage-2024-style full state name -> USPS abbreviation, for matching the
#: survey's state-branch condition descriptions (which name states in full,
#: e.g. "Alabama") to our profiles' state_abbr field. "Washington, D.C." is
#: the survey's own spelling for DC.
STATE_NAME_TO_ABBR: dict[str, str] = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Washington, D.C.": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

#: "Extreme weather predictions" (code name "practical planarian") branches
#: on the respondent's self-reported state into one of 4 blocks. Extracted
#: directly from survey.json's SurveyFlow branch logic (BranchLogic
#: "LeftOpDesc" state names), not guessed: verified to cover exactly the 50
#: states + DC (27 + 13 + 11 = 51), with "Prefer not to say" the only other
#: branch (falling to the US-general block, never reached by our profiles
#: since every profile has a real state_abbr).
_FLOOD_STATES = {
    "AL", "AR", "DE", "FL", "GA", "IL", "IN", "IA", "KS", "KY", "LA", "MD", "MS", "MO", "NE",
    "NC", "ND", "OH", "OK", "PA", "SC", "SD", "TN", "TX", "VA", "WV", "DC",
}
_WILDFIRE_STATES = {"AK", "AZ", "CA", "CO", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY", "HI"}
_ICE_STATES = {"CT", "ME", "MA", "MI", "MN", "NH", "NJ", "NY", "RI", "VT", "WI"}

_HTML_BREAK_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
#: the "which state do you live in" question's piped-text placeholder,
#: Qualtrics-rendered live from the respondent's own answer in the real
#: survey; substituted with the profile's actual state name here.
_STATE_PIPE_PATTERN = re.compile(r"\$\{q://QID1721185837/ChoiceGroup/SelectedChoices\}")
_ABBR_TO_STATE_NAME = {v: k for k, v in STATE_NAME_TO_ABBR.items()}


def _strip_html(text: str) -> str:
    """Convert Qualtrics rich-text QuestionText into plain text: <br> tags
    become newlines, all other tags are dropped, and HTML entities
    (&ldquo;, &nbsp;, ...) are unescaped.
    """
    text = _HTML_BREAK_PATTERN.sub("\n", text)
    text = _HTML_TAG_PATTERN.sub("", text)
    return html.unescape(text).strip()


def _load_survey_json() -> dict[str, Any]:
    with open(SURVEY_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)["result"]


def _block_db_text(block: dict[str, Any], questions: dict[str, Any]) -> str:
    """Concatenate every DB (descriptive/text block, no response captured)
    question's text within one survey block, in element order.
    """
    parts = []
    for element in block.get("BlockElements", []):
        if element.get("Type") != "Question":
            continue
        question = questions.get(element["QuestionID"], {})
        if question.get("QuestionType") == "DB":
            parts.append(_strip_html(question.get("QuestionText") or ""))
    return "\n\n".join(p for p in parts if p)


def get_common_climate_scientist_context() -> str:
    """Participant-facing transition/definition shown before treatment.

    Source: survey/survey.json question QID1721185798, DataExportTag
    "Transition", in the "Transition to Study" block.
    """
    survey = _load_survey_json()
    question = survey["Questions"]["QID1721185798"]
    if question.get("DataExportTag") != "Transition":
        raise RuntimeError("common climate-scientist context source question changed")
    return _strip_html(question["QuestionText"])


CONSENSUS_INTERACTION_RANDOMIZER_FLOW_ID = "FL_137"
CONSENSUS_INTERACTION_BLOCKS = (
    {
        "block_key": "human_primary_cause",
        "block_id": "BL_6W0XpFkTpycQ3J4",
        "estimate_qid": "QID1721185886",
        "feedback_qid": "QID1721185887",
        "target_label": "consensus_estimate_human_primary_cause",
    },
    {
        "block_key": "co2_warms_planet",
        "block_id": "BL_0ju57PhCZss5AI6",
        "estimate_qid": "QID1721185889",
        "feedback_qid": "QID1721185890",
        "target_label": "consensus_estimate_co2_warms_planet",
    },
    {
        "block_key": "net_zero_before_2085",
        "block_id": "BL_6u8mo7QchfcEHgG",
        "estimate_qid": "QID1721185892",
        "feedback_qid": "QID1721185893",
        "target_label": "consensus_estimate_net_zero_before_2085",
    },
)
CONSENSUS_INTRO_QIDS = ("QID1721185883", "QID1721185884")
CONSENSUS_CLOSING_QID = "QID1721185895"


def get_consensus_interaction_source_audit() -> dict[str, Any]:
    """Qualtrics source map for the interactive Consensus condition.

    The three estimate blocks are shown inside SurveyFlow randomizer FL_137
    with SubSet=3, then each block's feedback page follows its slider page.
    """
    survey = _load_survey_json()
    blocks = survey["Blocks"]
    questions = survey["Questions"]
    branch = survey["SurveyFlow"]["Flow"][37]
    randomizer = branch["Flow"][1]
    entries = []
    for spec in CONSENSUS_INTERACTION_BLOCKS:
        block = blocks[spec["block_id"]]
        qids = [element["QuestionID"] for element in block.get("BlockElements", []) if element.get("Type") == "Question"]
        estimate = questions[spec["estimate_qid"]]
        feedback = questions[spec["feedback_qid"]]
        entries.append(
            {
                **spec,
                "block_description": block.get("Description"),
                "question_ids": qids,
                "slider_qids": [qid for qid in qids if questions[qid].get("QuestionType") == "Slider"],
                "feedback_qids": [qid for qid in qids if questions[qid].get("QuestionType") == "DB"],
                "estimate_question_type": estimate.get("QuestionType"),
                "estimate_selector": estimate.get("Selector"),
                "estimate_choice_display": estimate.get("Choices", {}).get("1", {}).get("Display"),
                "estimate_slider_min": estimate.get("Configuration", {}).get("CSSliderMin"),
                "estimate_slider_max": estimate.get("Configuration", {}).get("CSSliderMax"),
                "estimate_num_decimals": estimate.get("Configuration", {}).get("NumDecimals"),
                "feedback_question_type": feedback.get("QuestionType"),
            }
        )
    return {
        "survey_source": str(SURVEY_JSON_PATH.relative_to(REPO_ROOT)),
        "qsf_source": "survey/survey.qsf",
        "flow_path": "SurveyFlow.Flow[37]",
        "branch_flow_id": branch.get("FlowID"),
        "interactive_randomizer_flow_path": "SurveyFlow.Flow[37].Flow[1]",
        "interactive_randomizer_flow_id": randomizer.get("FlowID"),
        "interactive_randomizer_subset": randomizer.get("SubSet"),
        "participant_entered_estimates_before_feedback": True,
        "blocks": entries,
    }


def get_consensus_stage_a_intro_text() -> str:
    """Participant-facing Consensus introduction/instructions before sliders."""
    survey = _load_survey_json()
    return "\n\n".join(_strip_html(survey["Questions"][qid]["QuestionText"]) for qid in CONSENSUS_INTRO_QIDS)


def get_consensus_estimate_items() -> list[dict[str, Any]]:
    """The three participant-entered Consensus slider questions."""
    survey = _load_survey_json()
    items = []
    for spec in CONSENSUS_INTERACTION_BLOCKS:
        question = survey["Questions"][spec["estimate_qid"]]
        config = question.get("Configuration", {})
        items.append(
            {
                "block_key": spec["block_key"],
                "block_id": spec["block_id"],
                "qualtrics_label": question.get("DataExportTag") or spec["target_label"],
                "target_label": spec["target_label"],
                "question_id": spec["estimate_qid"],
                "question_text": _strip_html(question["QuestionText"]),
                "response_options": question.get("Choices", {}).get("1", {}).get("Display") or "0 to 100 percent",
                "scale": SCALE_SLIDER_0_100,
                "scale_min": int(config.get("CSSliderMin", 0)),
                "scale_max": int(config.get("CSSliderMax", 100)),
            }
        )
    return items


def get_consensus_single_item_feedback_text(block_key: str) -> str:
    """One Consensus item's own feedback text ONLY -- no closing page
    appended (unlike get_consensus_feedback_text, which always appends the
    closing page and is used by the prior all-estimates-then-all-feedback
    Stage A/B implementation). Needed for the benchmark-exact interleaved
    sequence, where each item's feedback is delivered immediately after its
    own estimate and the closing page must appear only once, after the
    third item's feedback."""
    survey = _load_survey_json()
    by_key = {spec["block_key"]: spec for spec in CONSENSUS_INTERACTION_BLOCKS}
    if block_key not in by_key:
        raise KeyError(f"unknown Consensus feedback block key {block_key!r}")
    return _strip_html(survey["Questions"][by_key[block_key]["feedback_qid"]]["QuestionText"])


def get_consensus_closing_text() -> str:
    """The Consensus closing page text ONLY (no item feedback)."""
    survey = _load_survey_json()
    return _strip_html(survey["Questions"][CONSENSUS_CLOSING_QID]["QuestionText"])


def get_consensus_feedback_text(block_keys: list[str] | tuple[str, ...] | None = None) -> str:
    """Participant-facing Consensus feedback pages plus closing page."""
    survey = _load_survey_json()
    by_key = {spec["block_key"]: spec for spec in CONSENSUS_INTERACTION_BLOCKS}
    ordered_keys = list(block_keys) if block_keys is not None else [spec["block_key"] for spec in CONSENSUS_INTERACTION_BLOCKS]
    parts = []
    for key in ordered_keys:
        if key not in by_key:
            raise KeyError(f"unknown Consensus feedback block key {key!r}")
        parts.append(_strip_html(survey["Questions"][by_key[key]["feedback_qid"]]["QuestionText"]))
    parts.append(_strip_html(survey["Questions"][CONSENSUS_CLOSING_QID]["QuestionText"]))
    return "\n\n".join(part for part in parts if part)


def _build_condition_stimuli() -> dict[str, str | tuple[str, ...] | dict[str, str]]:
    """One-time parse of survey.json + condition_codenames.csv into
    {condition_title: stimulus_text} for the 14 ordinary interventions,
    plus three special entries: "control" (the three neutral filler texts),
    "Consensus" (concatenated from its 5-block sequence) and "Extreme
    weather predictions" (a dict keyed by weather category -- resolved
    per-respondent by state in get_condition_stimulus).
    """
    survey = _load_survey_json()
    blocks = survey["Blocks"]
    questions = survey["Questions"]
    block_by_desc = {(b.get("Description") or "").strip(): b for b in blocks.values()}

    code_to_title: dict[str, str] = {}
    with open(CONDITION_CODENAMES_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["title"].strip().lower() != "control":
                code_to_title[row["code_name"]] = row["title"]

    stimuli: dict[str, str | tuple[str, ...] | dict[str, str]] = {}

    stimuli["control"] = tuple(
        _block_db_text(block_by_desc[name], questions)
        for name in ["History of Neckties", "Rules of Baseball", "Different Types of Dances"]
    )

    # Consensus ("jealous jaguar"): intro + {human, CO2, year} (survey-randomized
    # order, all three always shown -- SubSet == full candidate count) + end.
    jaguar_parts = [
        _block_db_text(block_by_desc[name], questions)
        for name in ["jealous jaguar - intro", "jealous jaguar - human", "jealous jaguar - CO2", "jealous jaguar - year", "jealous jaguar - end"]
    ]
    stimuli["Consensus"] = "\n\n".join(p for p in jaguar_parts if p)

    # Extreme weather predictions ("practical planarian"): state-branched.
    stimuli["Extreme weather predictions"] = {
        "floods": _block_db_text(block_by_desc["practical_planarian_floods"], questions),
        "wildfire": _block_db_text(block_by_desc["practical_planarian_wildfire"], questions),
        "ice": _block_db_text(block_by_desc["practical_planarian_ice"], questions),
        "us_general": _block_db_text(block_by_desc["practical_planarian_US_general"], questions),
    }

    # Every remaining code name maps 1:1 to a single block whose Description
    # equals the code name verbatim (including the "; "-joined multi-author names).
    handled_titles = {"Consensus", "Extreme weather predictions"}
    for code_name, title in code_to_title.items():
        if title in handled_titles:
            continue
        block = block_by_desc.get(code_name)
        if block is None:
            raise KeyError(f"condition_codenames.csv code_name {code_name!r} has no matching block in survey.json")
        stimuli[title] = _block_db_text(block, questions)

    return stimuli


_CONDITION_STIMULI: dict[str, str | tuple[str, ...] | dict[str, str]] | None = None


def get_control_stimuli() -> tuple[str, str, str]:
    """The three neutral control filler texts from survey.json, in the same
    order as condition_codenames.csv: neckties, baseball, dances."""
    global _CONDITION_STIMULI
    if _CONDITION_STIMULI is None:
        _CONDITION_STIMULI = _build_condition_stimuli()
    entry = _CONDITION_STIMULI["control"]
    if not isinstance(entry, tuple) or len(entry) != 3:
        raise RuntimeError("control stimuli were not parsed as three neutral filler texts")
    return entry


def get_condition_stimulus(condition: str, state_abbr: str | None = None, control_variant: int | None = None) -> str:
    """The plain-text intervention stimulus a respondent in `condition`
    would have read. For "control", returns one of the three neutral filler
    blocks ("History of Neckties", "Rules of Baseball", "Different Types
    of Dances") instead of an empty/no-treatment prompt. `control_variant`
    is 1-indexed and wraps, matching the roster's condition_replicate field
    when available.

    "Extreme weather predictions" requires `state_abbr` (falls back to the
    survey's own "US general" default block if state_abbr is missing/unmapped).
    """
    global _CONDITION_STIMULI
    if _CONDITION_STIMULI is None:
        _CONDITION_STIMULI = _build_condition_stimuli()

    if condition == "control":
        variants = get_control_stimuli()
        idx = 0 if control_variant is None else (int(control_variant) - 1) % len(variants)
        return variants[idx]

    entry = _CONDITION_STIMULI.get(condition)
    if entry is None:
        raise KeyError(f"no stimulus found for condition {condition!r}")

    if isinstance(entry, dict):  # Extreme weather predictions
        if state_abbr in _FLOOD_STATES:
            category = "floods"
        elif state_abbr in _WILDFIRE_STATES:
            category = "wildfire"
        elif state_abbr in _ICE_STATES:
            category = "ice"
        else:
            category = "us_general"
        text = entry[category]
        state_name = _ABBR_TO_STATE_NAME.get(state_abbr, "your state")
        return _STATE_PIPE_PATTERN.sub(state_name, text)

    return entry


# --- codebook.csv: raw items to elicit + how the 13 outcomes are computed ---

#: scale identifiers used by generate_responses.py to pick a label set.
SCALE_SLIDER_0_100 = "slider_0_100"
SCALE_DONATION_0_10 = "donation_0_10"
SCALE_BINARY_0_1 = "binary_0_1"

_DEMOGRAPHIC_LABELS = {"gender", "year_birth", "race", "education", "income", "party"}


def load_items() -> list[dict[str, str]]:
    """Every raw ("A. Measured items") codebook row that is an actual survey
    item to elicit -- i.e. excluding the 6 demographic/profile fields, which
    come from the roster, not from a survey response. Each entry:
    {qualtrics_label, target_label, question_text, scale}.
    """
    items = []
    with open(CODEBOOK_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["section"] != "A. Measured items":
                continue
            label = row["target_label"]
            if label in _DEMOGRAPHIC_LABELS:
                continue
            if label == "donation_ams":
                scale = SCALE_DONATION_0_10
            elif label == "newsletter_signup":
                scale = SCALE_BINARY_0_1
            else:
                scale = SCALE_SLIDER_0_100
            # codebook.csv's row for qualtrics_label "funding_5" lists
            # target_label "funding_perceptions" directly, but that name is
            # reserved for the *reverse-coded* value (100 - funding_5, per
            # this codebook's own "B. Constructed during cleaning" section
            # and the submission README: "funding_perceptions = 100 -
            # funding_5"). The raw elicited item is renamed here to avoid
            # colliding with the final composite name computed in
            # OUTCOME_COMPOSITES below.
            if label == "funding_perceptions":
                label = "funding_5_raw"
            items.append(
                {
                    "qualtrics_label": row["qualtrics_label"],
                    "target_label": label,
                    "question_text": row["question_text"],
                    "response_options": row["response_options"],
                    "scale": scale,
                }
            )
    return items


#: The 13 preregistered outcomes, each defined as a function of raw item
#: target_labels, per codebook.csv's "B. Constructed during cleaning" rows.
#: A plain string means "use this raw item's value directly"; a tuple means
#: ("mean", [...items...]) or ("reverse_100", item) for funding_perceptions'
#: 100-minus recode.
OUTCOME_COMPOSITES: dict[str, tuple[str, Any]] = {
    "trust_multidimensional": ("mean", ["trust_competence_1", "trust_competence_2", "trust_competence_3",
                                          "trust_integrity_1", "trust_integrity_2", "trust_integrity_3",
                                          "trust_benevolence_1", "trust_benevolence_2", "trust_benevolence_3",
                                          "trust_openness_1", "trust_openness_2", "trust_openness_3"]),
    "trust_post": ("item", "trust_post"),
    "distrust_post": ("item", "distrust_post"),
    "funding_perceptions": ("reverse_100", "funding_5_raw"),
    "policy_role_mean": ("mean", ["policy_role_1", "policy_role_2", "policy_role_3", "policy_role_4"]),
    "inst_trust_mean": ("mean", ["inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa", "inst_trust_universities", "inst_trust_federal_gov"]),
    "belief_post": ("item", "belief_post"),
    "concern_mean": ("mean", ["concern_1", "concern_2", "concern_3"]),
    "policy_general": ("item", "policy_general"),
    "policy_specific_mean": ("mean", [f"policy_specific_{i}" for i in range(1, 8)]),
    "behavior_mean": ("mean", ["behavior_meat", "behavior_transport", "behavior_solar", "behavior_fly", "behavior_talk", "behavior_donate"]),
    "donation_ams": ("item", "donation_ams"),
    "newsletter_signup": ("item", "newsletter_signup"),
}


def compute_outcomes(raw_row: dict[str, Any]) -> dict[str, float | int]:
    """Apply OUTCOME_COMPOSITES to one respondent's raw item values --
    pure arithmetic on already-known values, no elicitation/inference
    dependency of any kind. Shared by every stage that needs the 13
    outcomes from raw items (currently pipeline/submission/build_tier1.py).
    """
    outcomes = {}
    for outcome, (kind, spec) in OUTCOME_COMPOSITES.items():
        if kind == "item":
            outcomes[outcome] = raw_row[spec]
        elif kind == "mean":
            outcomes[outcome] = sum(float(raw_row[label]) for label in spec) / len(spec)
        elif kind == "reverse_100":
            outcomes[outcome] = 100 - raw_row[spec]
    return outcomes
