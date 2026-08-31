"""Versioned G and F prompt compilers.

G and F intentionally use distinct active prompt protocols:

* G: adapted from the demographic-conditioned ATP survey-response template of
  Krsteski et al. (ACL 2026), using demographics only.
* F: adapted from Ashokkumar et al. (Nature 2026) experimental forecasting
  prompt architecture and prompt-ensemble strategy.

The old shared respondent prompt is retained below as inactive legacy text for
reproducibility checks only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import survey_content as sc

G_PROMPT_PROTOCOL = "krsteski_demo_survey_adapted_v1"
F_PROMPT_PROTOCOL = "ashokkumar_experiment_forecast_adapted_v1"
PROMPT_COMPILER_VERSION = "prompt_compiler_v2_split_g_f"
F_VARIANT_ASSIGNMENT_VERSION = "stable_sha256_v1_condition_excluded"
F_RESPONSE_FORMAT_INSTRUCTION_V2_ID = "f_response_format_instruction_v2_json_only_schema_disclosed"
G_QUESTIONNAIRE_VERSION = "post_treatment_blocks_questionnaire_txt_v1"
CONSENSUS_INTERACTION_PROTOCOL_ID = "consensus_two_stage_interactive_adaptation_v1"
CONSENSUS_STAGE_A_OUTCOME_ID = "consensus_stage_a_estimates"

#: The presentation-wrapper component of the G system prompt: P1 is the
#: unmodified original role-adoption sentence (byte-identical to the prompt
#: already in production/target use); P2/P3 are alternative semantically
#: equivalent wrappers (an abandoned, never-executed prompt-ensemble
#: ATE-shape robustness check retired during method development; its frozen
#: proposal manifest is kept at outputs/approach3_prompt_ensemble/ for
#: governance provenance, but the code itself was removed). Everything after the
#: wrapper (G_SHARED_TAIL_INSTRUCTIONS below) is identical across all three
#: variants -- "do not explain or justify your answers" belongs to that
#: shared tail, not the wrapper, so it is deliberately trimmed from the P2/P3
#: wrapper text below to avoid duplicating that instruction.
G_ROLE_WRAPPER_P1 = """You are a survey respondent.

Adopt the respondent profile provided below and answer the survey as this
respondent."""

G_ROLE_WRAPPER_P2 = """You are participating in a survey interview as the person described below. Answer the survey questions from that person's perspective."""

G_ROLE_WRAPPER_P3 = """Complete the following survey as the person described below. Respond from that person's point of view rather than commenting on what they might think."""

G_SHARED_TAIL_INSTRUCTIONS = """Use the response scales exactly as provided. Do not explain or justify your
answers. Do not discuss the study design or what most people would answer.

Return only the requested structured responses."""

G_SYSTEM_PROMPT = f"{G_ROLE_WRAPPER_P1}\n\n{G_SHARED_TAIL_INSTRUCTIONS}"
G_SYSTEM_PROMPT_P2 = f"{G_ROLE_WRAPPER_P2}\n\n{G_SHARED_TAIL_INSTRUCTIONS}"
G_SYSTEM_PROMPT_P3 = f"{G_ROLE_WRAPPER_P3}\n\n{G_SHARED_TAIL_INSTRUCTIONS}"

G_SYSTEM_PROMPT_BY_VARIANT = {"P1": G_SYSTEM_PROMPT, "P2": G_SYSTEM_PROMPT_P2, "P3": G_SYSTEM_PROMPT_P3}

#: Format-only closing-instruction addition, per the G-v2 PROVIDER_SERVING_FORMAT_FAILURE
#: amendment (outputs/target_production/g_wave1_v1_format_failure_amendment.json):
#: the v1 closing sentence alone did not prevent the served model from wrapping its
#: (schema-constrained) JSON in a markdown code fence at scale. Appended verbatim after
#: the existing v1 closing sentence -- never replaces it, so the content-specific clause
#: ("raw item responses only" / "only the three estimate responses") is preserved exactly.
G_FORMAT_INSTRUCTION_V2 = (
    "Return ONLY the raw JSON object matching the supplied schema. Do not use Markdown. "
    "Do not use ```json or any other code fences. Do not place text before or after the JSON. "
    "The response must begin with { and end with }."
)

F_SYSTEM_PROMPT = """You are completing a simulated survey response task for an experimental forecasting study.

Use only the respondent profile, study setting, survey material, and response
scale shown in the request. Return only the requested structured response."""

LEGACY_SHARED_RESPONDENT_SYSTEM_PROMPT_INACTIVE = """You are taking part in a survey.

For this conversation, the respondent profile provided by the researcher
describes you. Treat those profile facts as fixed.

Read the survey material exactly as a participant would and answer the survey
questions from this respondent's perspective at that moment.

Do not act as a researcher, forecaster, expert, or AI assistant.
Do not predict what an average person would answer.
Do not explain or justify your answers.
Do not invent additional demographic or biographical facts that are not given.
Use each response scale exactly as stated.

Return only the requested structured response."""

ROSTER_PROFILE_COLUMNS = [
    "age",
    "age_band",
    "gender",
    "race",
    "education",
    "income",
    "party",
    "political_ideology",
    "state",
    "state_abbr",
    "religion",
]

PROFILE_FIELD_ORDER = [
    ("age", "Age"),
    ("gender", "Gender"),
    ("race", "Race/ethnicity"),
    ("education", "Education"),
    ("income", "Household income"),
    ("party", "Party identification"),
    ("political_ideology", "Political ideology"),
    ("state", "State of residence"),
    ("state_abbr", "State of residence"),
    ("religion", "Religion"),
]

FORBIDDEN_COMPILER_LEAKAGE_TERMS = [
    "expected_effect",
    "hypothesized_effect",
    "target_ate",
    "human_ate",
    "treatment_effect",
    "calibrated_ate",
    "raw_g_ate",
    "raw_f_ate",
    "lambda",
    "alpha",
]

FORBIDDEN_G_PROFILE_TERMS = [
    "climate concern",
    "climate belief",
    "trust in scientists",
    "institutional trust",
    "policy preferences",
    "donation tendency",
    "newsletter tendency",
    "personality",
    "hobbies",
    "occupation",
    "previous survey",
    "prior synthetic",
]

FORBIDDEN_G_QUESTIONNAIRE_METADATA_TERMS = [
    "reverse-coded",
    "recoded",
    "cleaning",
    "raw qualtrics",
    "qualtrics_label",
    "target_label",
    "funding_perceptions",
    "newsletter_signup",
]

G_POST_CONDITION_TRANSITION_TEXT = (
    "You are now moving on to the final section of the study.\n\n"
    "Please answer the following questions to the best of your ability.\n\n"
    "Thank you."
)

G_NEWSLETTER_OFFER_TEXT = """Learn more about climate science

If you’d like to learn more about climate science and solutions, you can subscribe to the newsletter by climate scientist Katharine Hayhoe.

Her newsletter "Talking Climate" provides short, accessible updates on climate science and climate solutions for a general audience.

Signing up takes less than a minute. Please select the free subscription option — there is no need to choose a paid version.

The link below will open the newsletter in a new tab. You can switch back to the current tab and continue the survey right away.

[ Open Talking Climate newsletter (opens in a new tab) ]

Note: Subscribing to this newsletter is optional."""

G_DONATION_INTRO_TEXT = """On the following page, you will have the opportunity to allocate real money between yourself and a non-profit organization.

After data collection is complete, we will randomly select 100 participants from this study to receive a $10 bonus payment.

If you are selected, the amount you allocate to yourself will be paid to you as a bonus, and the amount you allocate to the organization will be donated on your behalf."""

G_OUTCOME_BLOCK_SPECS: list[dict[str, Any]] = [
    {
        "block_id": "trust_multidimensional",
        "intro": (
            "Please answer the following questions on how you perceive climate scientists.\n\n"
            "Answer options below range from 0 to 100. Click on any space within this range and a bar will appear. "
            "Feel free to move that bar around to the number that best represents your answer."
        ),
        "item_labels": sc.OUTCOME_COMPOSITES["trust_multidimensional"][1],
    },
    {
        "block_id": "funding",
        "intro": "",
        "item_labels": ["funding_5_raw"],
    },
    {
        "block_id": "institutional_trust",
        "intro": "How much do you trust the following institutions?",
        "item_labels": ["inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa", "inst_trust_universities", "inst_trust_federal_gov"],
    },
    {
        "block_id": "policy_role",
        "intro": "To what extent do you agree or disagree with the following statements?",
        "item_labels": sc.OUTCOME_COMPOSITES["policy_role_mean"][1],
    },
    {
        "block_id": "trust_post",
        "intro": "",
        "item_labels": ["trust_post"],
    },
    {
        "block_id": "distrust_post",
        "intro": "",
        "item_labels": ["distrust_post"],
    },
    {
        "block_id": "donation",
        "intro": G_DONATION_INTRO_TEXT,
        "item_labels": ["donation_ams"],
    },
    {
        "block_id": "newsletter",
        "intro": "",
        "offer_text": G_NEWSLETTER_OFFER_TEXT,
        "item_labels": ["newsletter_signup"],
    },
    {
        "block_id": "belief_post",
        "intro": "",
        "item_labels": ["belief_post"],
    },
    {
        "block_id": "concern",
        "intro": "Please indicate your views on the following questions.",
        "item_labels": sc.OUTCOME_COMPOSITES["concern_mean"][1],
    },
    {
        "block_id": "behavior",
        "intro": "How likely are you to engage in the following activities in the next twelve months?",
        "item_labels": sc.OUTCOME_COMPOSITES["behavior_mean"][1],
    },
    {
        "block_id": "policy_general",
        "intro": "",
        "item_labels": ["policy_general"],
    },
    {
        "block_id": "policy_specific",
        "intro": "How much do you support or oppose the following policies?",
        "item_labels": sc.OUTCOME_COMPOSITES["policy_specific_mean"][1],
    },
]

G_ITEM_TEXT_OVERRIDES = {
    "funding_5_raw": "Do you think the federal government is spending too much, too little or about the right amount of money on climate change research?",
    "policy_general": 'How much do you oppose or support the following statement? "The U.S. government should do more to reduce global warming"',
    "newsletter_signup": 'Did you subscribe to the "Talking Climate" newsletter on the previous page?',
}

G_RESPONSE_OPTIONS_OVERRIDES = {
    "funding_5_raw": "0 = far too little, 50 = about the right amount, 100 = far too much",
    "donation_ams": "$0-$10 in whole-dollar choices ($1 increments; integers only).",
    "newsletter_signup": "Yes or No",
}

F_ITEM_TEXT_OVERRIDES = {
    "newsletter_signup": G_ITEM_TEXT_OVERRIDES["newsletter_signup"],
}

F_RESPONSE_OPTIONS_OVERRIDES = {
    "donation_ams": "$0-$10 in $1 increments.",
    "newsletter_signup": G_RESPONSE_OPTIONS_OVERRIDES["newsletter_signup"],
}

F_INTRO_VARIANTS = [
    ("intro_a", "Social scientists often conduct research studies using online surveys."),
    ("intro_b", "Researchers sometimes study how survey materials relate to participants' answers."),
    ("intro_c", "The following is a simulated response to one online survey condition."),
]

F_PROFILE_LABEL_VARIANTS = [
    ("profile_label_a", "Respondent profile"),
    ("profile_label_b", "Participant profile"),
    ("profile_label_c", "Survey participant"),
]

F_PROFILE_FORMAT_VARIANTS = [
    ("profile_format_bullets", "bullets"),
    ("profile_format_sentence", "sentence"),
]

F_SURVEY_FORMAT_VARIANTS = [
    ("survey_format_pages", "pages"),
    ("survey_format_sections", "sections"),
]


@dataclass(frozen=True)
class PromptRender:
    role: str
    protocol_id: str
    request_key: str
    system_prompt: str
    user_prompt: str
    response_schema: dict[str, Any]
    stimulus_text: str
    nonstimulus_text: str
    prompt_variant_id: str = ""
    variant_assignment: dict[str, str] | None = None
    questionnaire_order: dict[str, Any] | None = None
    response_key_map: dict[str, str] | None = None
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    provenance: dict[str, Any] | None = None

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history,
            {"role": "user", "content": self.user_prompt},
        ]


def stable_hash(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def schema_hash(schema: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def _is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and str(value) == "nan":
        return False
    return str(value).strip() != ""


def profile_description(profile: Mapping[str, object], *, style: str = "bullets") -> str:
    """Compile demographic-only profile text in the frozen field order."""
    lines: list[tuple[str, object]] = []
    seen_labels = set()
    for key, label in PROFILE_FIELD_ORDER:
        if label in seen_labels:
            continue
        value = profile.get(key)
        if not _is_present(value):
            continue
        lines.append((label, value))
        seen_labels.add(label)
    text = "\n".join(f"- {label}: {value}" for label, value in lines)
    if style == "sentence":
        return "; ".join(f"{label}: {value}" for label, value in lines) + "."
    return text


def persona_description(profile: Mapping[str, object]) -> str:
    """Backward-compatible name for demographic-only profile rendering."""
    return "RESPONDENT PROFILE\n" + profile_description(profile)


def _bounds_for_item(item: Mapping[str, Any]) -> tuple[int, int]:
    if "scale_min" in item and "scale_max" in item:
        return int(item["scale_min"]), int(item["scale_max"])
    scale = item["scale"]
    if scale == sc.SCALE_SLIDER_0_100:
        return 0, 100
    if scale == sc.SCALE_DONATION_0_10:
        return 0, 10
    if scale == sc.SCALE_BINARY_0_1:
        return 0, 1
    raise ValueError(f"unknown scale {scale!r}")


def item_json_schema(items: list[dict[str, Any]]) -> dict:
    properties: dict[str, dict] = {}
    required: list[str] = []
    for item in items:
        low, high = _bounds_for_item(item)
        label = item.get("response_key", item["qualtrics_label"])
        if item.get("scale") == sc.SCALE_BINARY_0_1 and low == 0 and high == 1:
            properties[label] = {"type": "integer", "enum": [0, 1]}
        else:
            properties[label] = {"type": "integer", "minimum": low, "maximum": high}
        required.append(label)
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def items_for_scored_outcome(outcome: str, all_items: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    all_items = all_items or sc.load_items()
    by_label = {item["target_label"]: item for item in all_items}
    kind, spec = sc.OUTCOME_COMPOSITES[outcome]
    labels = spec if kind == "mean" else [spec]
    return [by_label[label] for label in labels]


def _questions_block(items: list[dict[str, Any]]) -> str:
    rendered = []
    for item in items:
        low, high = _bounds_for_item(item)
        lines = [f"{item.get('response_key', item['qualtrics_label'])}: {_f_item_text(item)}"]
        options = _f_response_options(item)
        if options:
            if options[-1] not in ".!?":
                options = f"{options}."
            lines.append(f"Response options: {options}")
        if item.get("scale") == sc.SCALE_BINARY_0_1 and low == 0 and high == 1:
            lines.append("Answer 1 for Yes and 0 for No.")
        else:
            lines.append(f"Answer with an integer from {low} to {high}.")
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


def _assert_one_resolved_stimulus(condition_stimulus: str, *, allow_empty: bool = False) -> None:
    if not isinstance(condition_stimulus, str):
        raise ValueError("every prompt request must have exactly one resolved non-empty stimulus string")
    if not condition_stimulus.strip() and not allow_empty:
        raise ValueError("every prompt request must have exactly one resolved non-empty stimulus string")


def _g_item_text(item: Mapping[str, Any]) -> str:
    override = G_ITEM_TEXT_OVERRIDES.get(str(item["target_label"]))
    if override is not None:
        return override
    text = str(item["question_text"])
    if "—" in text:
        return text.split("—", 1)[1].strip()
    return text


def _g_response_options(item: Mapping[str, Any]) -> str:
    return G_RESPONSE_OPTIONS_OVERRIDES.get(str(item["target_label"]), str(item["response_options"]))


def _f_item_text(item: Mapping[str, Any]) -> str:
    return F_ITEM_TEXT_OVERRIDES.get(str(item.get("target_label", "")), str(item["question_text"]))


def _f_response_options(item: Mapping[str, Any]) -> str:
    return F_RESPONSE_OPTIONS_OVERRIDES.get(str(item.get("target_label", "")), str(item["response_options"]))


def f_target_outcome_context(outcome: str) -> str:
    """Participant-facing context required by isolated target F outcomes."""
    if outcome == "newsletter_signup":
        return G_NEWSLETTER_OFFER_TEXT
    if outcome == "donation_ams":
        return G_DONATION_INTRO_TEXT
    return ""


def f_target_condition_material(condition_stimulus: str) -> str:
    """Target F condition material includes the common pre-condition context."""
    common = sc.get_common_climate_scientist_context().strip()
    stimulus = str(condition_stimulus).strip()
    if stimulus.startswith(common):
        return stimulus
    return f"{common}\n\n{stimulus}"


def target_f_control_variant(f_profile_id: str, replicate_id: int = 1) -> int:
    """Deterministically balance the three target control fillers by profile."""
    match = re.search(r"(\d+)$", str(f_profile_id))
    if match:
        profile_index = int(match.group(1))
    else:
        profile_index = int(stable_hash("target_f_control_filler", f_profile_id)[:12], 16)
    return ((profile_index - 1) % 3) + 1


def consensus_interaction_order(subject_id: str, replicate_id: int = 1, *, order_replicate_id: int | None = None) -> list[dict[str, Any]]:
    """Respondent-specific FL_137 order for Consensus estimate/feedback blocks.

    The key deliberately excludes condition_id. The same ordered block keys are
    used for Stage A estimate questions and the subsequent Stage B feedback.

    order_replicate_id, when given, is used for the item-order hash INSTEAD
    of replicate_id -- lets a caller vary replicate_id (to get a fresh
    request_key/seed, e.g. a bounded production retry attempt) while
    preserving the SAME administered item order as an earlier attempt.
    Defaults to None, which falls back to replicate_id -- every existing
    call site is byte-identical to before this parameter existed."""
    effective_replicate_id = replicate_id if order_replicate_id is None else order_replicate_id
    items = sc.get_consensus_estimate_items()
    return sorted(
        items,
        key=lambda item: stable_hash(
            subject_id,
            effective_replicate_id,
            CONSENSUS_INTERACTION_PROTOCOL_ID,
            sc.CONSENSUS_INTERACTION_RANDOMIZER_FLOW_ID,
            item["block_key"],
        ),
    )


def _consensus_stage_a_items(subject_id: str, replicate_id: int = 1, *, order_replicate_id: int | None = None) -> list[dict[str, Any]]:
    rendered = []
    for index, item in enumerate(consensus_interaction_order(subject_id, replicate_id, order_replicate_id=order_replicate_id), start=1):
        stage_item = dict(item)
        stage_item["response_key"] = f"Q{index:03d}"
        rendered.append(stage_item)
    return rendered


def _consensus_stage_a_prompt_text(profile: Mapping[str, object], *, profile_label: str, profile_style: str, study_setting: str = "") -> str:
    profile_text = profile_description(profile, style=profile_style)
    setting = f"\n\nSTUDY SETTING\n{study_setting}" if study_setting else ""
    return (
        f"{profile_label}\n{profile_text}{setting}\n\n"
        f"SURVEY MATERIAL\n{sc.get_common_climate_scientist_context()}\n\n"
        f"{sc.get_consensus_stage_a_intro_text()}\n\n"
        "SCIENTIFIC AGREEMENT ESTIMATE QUESTIONS\n"
    )


def _json_response(response: Mapping[str, Any]) -> str:
    return json.dumps(dict(response), sort_keys=True, separators=(",", ":"))


def consensus_stage_a_record(
    render: PromptRender,
    response: Mapping[str, Any],
    *,
    role: str,
    subject_id: str,
    replicate_id: int = 1,
) -> dict[str, Any]:
    """Create the provenance object required before rendering Consensus Stage B."""
    if render.protocol_id != CONSENSUS_INTERACTION_PROTOCOL_ID:
        raise ValueError("Consensus Stage A record requires a Stage A render")
    expected_required = set(render.response_schema.get("required", []))
    observed = set(response)
    if expected_required != observed:
        raise ValueError(f"Stage A response keys do not match schema: expected {sorted(expected_required)}, observed {sorted(observed)}")
    response_text = _json_response(response)
    order_keys = [item["block_key"] for item in render.provenance.get("consensus_stage_a_items", [])] if render.provenance else []
    return {
        "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID,
        "role": role,
        "subject_id": str(subject_id),
        "replicate_id": int(replicate_id),
        "stage_a_request_key": render.request_key,
        "stage_a_prompt_hash": text_hash(render.system_prompt + "\n" + render.user_prompt),
        "stage_a_response_schema_hash": schema_hash(render.response_schema),
        "stage_a_response": dict(response),
        "stage_a_response_json": response_text,
        "stage_a_messages": [
            {"role": "user", "content": render.user_prompt},
            {"role": "assistant", "content": response_text},
        ],
        "consensus_order_keys": order_keys,
    }


def _validate_consensus_stage_a_record(record: Mapping[str, Any], *, role: str, subject_id: str, replicate_id: int) -> None:
    checks = {
        "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID,
        "role": role,
        "subject_id": str(subject_id),
        "replicate_id": int(replicate_id),
    }
    for key, expected in checks.items():
        if record.get(key) != expected:
            raise ValueError(f"Consensus Stage A record mismatch for {key}: expected {expected!r}, observed {record.get(key)!r}")
    messages = record.get("stage_a_messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Consensus Stage A record is missing conversation history messages")
    if "stage_a_response_json" not in record:
        raise ValueError("Consensus Stage A record is missing exact response JSON")


def _consensus_feedback_for_record(record: Mapping[str, Any]) -> str:
    order_keys = record.get("consensus_order_keys") or [item["block_key"] for item in sc.get_consensus_estimate_items()]
    return sc.get_consensus_feedback_text(list(order_keys))


def build_g_consensus_stage_a_prompt_render(
    profile: Mapping[str, object],
    *,
    donor_key: str,
    replicate_id: int = 1,
    order_replicate_id: int | None = None,
    system_prompt: str | None = None,
    prompt_variant: str = "P1",
    response_format_instruction_version: str = "v1",
) -> PromptRender:
    """G Consensus Stage A: estimate questions only, before any feedback.

    system_prompt/prompt_variant default to the original P1 path -- see
    build_g_prompt_render's docstring for the same backward-compatibility
    guarantee. response_format_instruction_version="v1" (default) leaves
    the closing instruction exactly as before; "v2" appends
    G_FORMAT_INSTRUCTION_V2 (format-only, see its docstring).
    order_replicate_id -- see consensus_interaction_order's docstring;
    defaults to None (falls back to replicate_id), so every existing call
    site is byte-identical to before this parameter existed."""
    if response_format_instruction_version not in ("v1", "v2"):
        raise ValueError(f"unknown response_format_instruction_version: {response_format_instruction_version!r}")
    items = _consensus_stage_a_items(donor_key, replicate_id, order_replicate_id=order_replicate_id)
    resolved_system_prompt = G_SYSTEM_PROMPT if system_prompt is None else system_prompt
    closing = "Return a single JSON object with one integer value per question key. Return only the three estimate responses."
    if response_format_instruction_version == "v2":
        closing = f"{closing} {G_FORMAT_INSTRUCTION_V2}"
    user = _consensus_stage_a_prompt_text(profile, profile_label="RESPONDENT PROFILE", profile_style="bullets") + _questions_block(items) + f"\n\n{closing}"
    request_key = f"G|{donor_key}|Consensus|stage_a|replicate_{replicate_id}"
    if prompt_variant != "P1":
        request_key = f"{request_key}|variant_{prompt_variant}"
    if response_format_instruction_version != "v1":
        request_key = f"{request_key}|fmt_{response_format_instruction_version}"
    return PromptRender(
        role="G",
        protocol_id=CONSENSUS_INTERACTION_PROTOCOL_ID,
        request_key=request_key,
        system_prompt=resolved_system_prompt,
        user_prompt=user,
        response_schema=item_json_schema(items),
        stimulus_text=sc.get_consensus_stage_a_intro_text(),
        nonstimulus_text=user.replace(sc.get_consensus_stage_a_intro_text(), "<<STIMULUS>>", 1),
        questionnaire_order={
            "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID,
            "qualtrics_randomizer_flow_id": sc.CONSENSUS_INTERACTION_RANDOMIZER_FLOW_ID,
            "ordered_block_keys": [item["block_key"] for item in items],
            "condition_id_excluded": True,
        },
        response_key_map={item["response_key"]: item["target_label"] for item in items},
        provenance={"consensus_stage_a_items": items},
    )


def build_g_consensus_stage_b_prompt_render(
    profile: Mapping[str, object],
    items: list[dict[str, Any]],
    stage_a_record: Mapping[str, Any],
    *,
    donor_key: str,
    replicate_id: int = 1,
    system_prompt: str | None = None,
    prompt_variant: str = "P1",
    response_format_instruction_version: str = "v1",
) -> PromptRender:
    """G Consensus Stage B: prior Stage A history, feedback, then full G questionnaire.

    system_prompt/prompt_variant default to the original P1 path -- see
    build_g_prompt_render's docstring for the same backward-compatibility
    guarantee. response_format_instruction_version -- see
    build_g_consensus_stage_a_prompt_render's docstring."""
    if response_format_instruction_version not in ("v1", "v2"):
        raise ValueError(f"unknown response_format_instruction_version: {response_format_instruction_version!r}")
    _validate_consensus_stage_a_record(stage_a_record, role="G", subject_id=donor_key, replicate_id=replicate_id)
    resolved_system_prompt = G_SYSTEM_PROMPT if system_prompt is None else system_prompt
    feedback = _consensus_feedback_for_record(stage_a_record)
    questionnaire_text, rendered_items, questionnaire_order = _build_g_questionnaire(items, donor_key)
    closing = "Return a single JSON object with one integer value per question key. Return raw item responses only."
    if response_format_instruction_version == "v2":
        closing = f"{closing} {G_FORMAT_INSTRUCTION_V2}"
    user = f"SURVEY MATERIAL\n{feedback}\n\nSURVEY QUESTIONS\n{questionnaire_text}\n\n{closing}"
    request_key = f"G|{donor_key}|Consensus|stage_b|replicate_{replicate_id}"
    if prompt_variant != "P1":
        request_key = f"{request_key}|variant_{prompt_variant}"
    if response_format_instruction_version != "v1":
        request_key = f"{request_key}|fmt_{response_format_instruction_version}"
    return PromptRender(
        role="G",
        protocol_id=G_PROMPT_PROTOCOL,
        request_key=request_key,
        system_prompt=resolved_system_prompt,
        user_prompt=user,
        response_schema=item_json_schema(rendered_items),
        stimulus_text=feedback,
        nonstimulus_text=user.replace(feedback, "<<STIMULUS>>", 1),
        questionnaire_order={**questionnaire_order, "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID},
        response_key_map={item["response_key"]: item["target_label"] for item in rendered_items},
        conversation_history=list(stage_a_record["stage_a_messages"]),
        provenance={
            "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID,
            "stage_a_request_key": stage_a_record["stage_a_request_key"],
            "stage_a_prompt_hash": stage_a_record["stage_a_prompt_hash"],
            "stage_a_response_schema_hash": stage_a_record["stage_a_response_schema_hash"],
            "feedback_prompt_material_hash": text_hash(feedback),
        },
    )


def build_f_consensus_stage_a_prompt_render(
    profile: Mapping[str, object],
    *,
    f_profile_id: str,
    replicate_id: int = 1,
    study_setting: str = "This is an online survey shown to a broad sample of adult respondents.",
) -> PromptRender:
    """F Consensus Stage A: one estimate set per f_profile_id x replicate_id."""
    items = _consensus_stage_a_items(f_profile_id, replicate_id)
    variant = f_variant_assignment("target", f_profile_id, CONSENSUS_STAGE_A_OUTCOME_ID, replicate_id)
    intro = _variant_text(F_INTRO_VARIANTS, variant["intro_variant_id"])
    profile_label = _variant_text(F_PROFILE_LABEL_VARIANTS, variant["profile_label_variant_id"]).upper()
    profile_style = _variant_text(F_PROFILE_FORMAT_VARIANTS, variant["profile_format_variant_id"])
    user = (
        f"{intro}\n\n"
        + _consensus_stage_a_prompt_text(profile, profile_label=profile_label, profile_style=profile_style, study_setting=study_setting)
        + _questions_block(items)
        + "\n\nReturn a single JSON object with one integer value per question key. Return only the three estimate responses."
    )
    request_key = f"F|target|{f_profile_id}|Consensus|stage_a|replicate_{replicate_id}"
    return PromptRender(
        role="F",
        protocol_id=CONSENSUS_INTERACTION_PROTOCOL_ID,
        request_key=request_key,
        system_prompt=F_SYSTEM_PROMPT,
        user_prompt=user,
        response_schema=item_json_schema(items),
        stimulus_text=sc.get_consensus_stage_a_intro_text(),
        nonstimulus_text=user.replace(sc.get_consensus_stage_a_intro_text(), "<<STIMULUS>>", 1),
        prompt_variant_id=variant["prompt_variant_id"],
        variant_assignment=variant,
        response_key_map={item["response_key"]: item["target_label"] for item in items},
        provenance={
            "consensus_stage_a_items": items,
            "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID,
        },
    )


def build_f_consensus_stage_b_prompt_render(
    profile: Mapping[str, object],
    outcome: str,
    stage_a_record: Mapping[str, Any],
    *,
    f_profile_id: str,
    replicate_id: int = 1,
    all_items: list[dict[str, Any]] | None = None,
    study_setting: str = "This is an online survey shown to a broad sample of adult respondents.",
    response_format_instruction_version: str = "v1",
) -> PromptRender:
    """F Consensus Stage B: Stage A history plus feedback, one scored outcome.

    response_format_instruction_version: forwarded unchanged to
    build_f_prompt_render_from_items (see build_f_prompt_render)."""
    _validate_consensus_stage_a_record(stage_a_record, role="F", subject_id=f_profile_id, replicate_id=replicate_id)
    feedback = _consensus_feedback_for_record(stage_a_record)
    render = build_f_prompt_render_from_items(
        profile,
        feedback,
        items_for_scored_outcome(outcome, all_items),
        study_id="target",
        f_profile_id=f_profile_id,
        outcome_id=outcome,
        replicate_id=replicate_id,
        condition_id="Consensus",
        study_setting=study_setting,
        response_format_instruction_version=response_format_instruction_version,
        outcome_context=f_target_outcome_context(outcome),
        conversation_history=list(stage_a_record["stage_a_messages"]),
        provenance={
            "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID,
            "stage_a_request_key": stage_a_record["stage_a_request_key"],
            "stage_a_prompt_hash": stage_a_record["stage_a_prompt_hash"],
            "stage_a_response_schema_hash": stage_a_record["stage_a_response_schema_hash"],
            "feedback_prompt_material_hash": text_hash(feedback),
        },
    )
    return render


G_SECONDARY_RANDOMIZER_BLOCK_IDS = (
    "trust_post",
    "donation",
    "distrust_post",
    "policy_role",
    "funding",
    "institutional_trust",
    "newsletter",
)
G_TERTIARY_RANDOMIZER_BLOCK_IDS = (
    "belief_post",
    "concern",
    "behavior",
    "policy_general",
    "policy_specific",
)


def _stable_block_permutation(donor_key: str, randomizer_id: str, block_ids: Sequence[str]) -> list[str]:
    return sorted(block_ids, key=lambda block_id: stable_hash(donor_key, G_PROMPT_PROTOCOL, G_QUESTIONNAIRE_VERSION, randomizer_id, block_id))


def g_outcome_block_order(donor_key: str, item_labels: set[str] | None = None) -> dict[str, Any]:
    """Stable post-treatment questionnaire block order for G.

    The multidimensional trust block is always first. Remaining present
    blocks are ordered by a stable donor/protocol/questionnaire hash that
    deliberately excludes condition_id, so a donor sees the same non-stimulus
    questionnaire structure in control and treatment requests.
    """
    item_labels = item_labels or {label for spec in G_OUTCOME_BLOCK_SPECS for label in spec["item_labels"]}
    present_ids = {spec["block_id"] for spec in G_OUTCOME_BLOCK_SPECS if set(spec["item_labels"]) & item_labels}
    if not present_ids:
        return {"questionnaire_version": G_QUESTIONNAIRE_VERSION, "assignment_key_hash": stable_hash(donor_key, G_PROMPT_PROTOCOL, G_QUESTIONNAIRE_VERSION), "ordered_block_ids": []}
    primary = ["trust_multidimensional"] if "trust_multidimensional" in present_ids else []
    secondary = _stable_block_permutation(donor_key, "secondary_outcomes_FL_55", [bid for bid in G_SECONDARY_RANDOMIZER_BLOCK_IDS if bid in present_ids])
    tertiary = _stable_block_permutation(donor_key, "tertiary_outcomes_FL_49", [bid for bid in G_TERTIARY_RANDOMIZER_BLOCK_IDS if bid in present_ids])
    ordered = [*primary, *secondary, *tertiary]
    return {
        "questionnaire_version": G_QUESTIONNAIRE_VERSION,
        "assignment_key_hash": stable_hash(donor_key, G_PROMPT_PROTOCOL, G_QUESTIONNAIRE_VERSION),
        "assignment_algorithm": "qualtrics_primary_fixed_then_secondary_FL_55_then_tertiary_FL_49_stable_sha256_donor_protocol_questionnaire_version_randomizer_block",
        "condition_id_excluded": True,
        "qualtrics_randomizer_structure": {
            "primary_fixed": primary,
            "secondary_outcomes_FL_55": secondary,
            "tertiary_outcomes_FL_49": tertiary,
        },
        "ordered_block_ids": ordered,
    }


def _build_g_questionnaire(items: list[dict[str, Any]], donor_key: str) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    by_target = {item["target_label"]: item for item in items}
    order = g_outcome_block_order(donor_key, set(by_target))
    spec_by_id = {spec["block_id"]: spec for spec in G_OUTCOME_BLOCK_SPECS}
    rendered_items: list[dict[str, Any]] = []
    parts = [
        "POST-CONDITION TRANSITION",
        G_POST_CONDITION_TRANSITION_TEXT,
    ]
    q_index = 1
    for page_index, block_id in enumerate(order["ordered_block_ids"], start=1):
        spec = spec_by_id[block_id]
        block_lines = [f"QUESTION PAGE {page_index}"]
        if spec.get("offer_text"):
            block_lines.extend(["NEWSLETTER OFFER PAGE", str(spec["offer_text"])])
        if spec.get("intro"):
            block_lines.append(str(spec["intro"]))
        for label in spec["item_labels"]:
            if label not in by_target:
                continue
            source_item = by_target[label]
            response_key = f"Q{q_index:03d}"
            q_index += 1
            item = dict(source_item)
            item["response_key"] = response_key
            rendered_items.append(item)
            low, high = _bounds_for_item(item)
            text = _g_item_text(item)
            options = _g_response_options(item)
            answer_instruction = f"Answer with an integer from {low} to {high}."
            if item["scale"] == sc.SCALE_BINARY_0_1:
                answer_instruction = "Answer 1 for Yes and 0 for No."
            block_lines.append(f"{response_key}. {text}\nResponse options: {options}\n{answer_instruction}")
        parts.append("\n\n".join(block_lines))
    return "\n\n".join(parts), rendered_items, order


def build_g_prompt_render(
    profile: Mapping[str, object],
    condition_stimulus: str,
    items: list[dict[str, Any]],
    *,
    donor_key: str = "",
    condition_id: str = "",
    replicate_id: int = 1,
    system_prompt: str | None = None,
    prompt_variant: str = "P1",
    response_format_instruction_version: str = "v1",
) -> PromptRender:
    """G prompt: one demographic-only respondent and one full questionnaire.

    system_prompt/prompt_variant default to the original, unmodified P1
    production path (system_prompt=None resolves to G_SYSTEM_PROMPT,
    prompt_variant="P1" leaves request_key exactly as before) -- every
    existing call site is byte-identical to before this addition. Passing a
    non-default system_prompt/prompt_variant (see G_SYSTEM_PROMPT_BY_VARIANT)
    is used only by the Approach-3 prompt-ensemble ATE-shape check.
    response_format_instruction_version="v1" (default) leaves the closing
    instruction exactly as before; "v2" appends G_FORMAT_INSTRUCTION_V2 --
    see the G-v2 PROVIDER_SERVING_FORMAT_FAILURE amendment.
    """
    if condition_id == "Consensus":
        raise ValueError("target G Consensus requires build_g_consensus_stage_a_prompt_render/build_g_consensus_stage_b_prompt_render")
    if response_format_instruction_version not in ("v1", "v2"):
        raise ValueError(f"unknown response_format_instruction_version: {response_format_instruction_version!r}")
    _assert_one_resolved_stimulus(condition_stimulus)
    resolved_system_prompt = G_SYSTEM_PROMPT if system_prompt is None else system_prompt
    profile_text = profile_description(profile)
    questionnaire_text, rendered_items, questionnaire_order = _build_g_questionnaire(items, donor_key)
    closing = "Return a single JSON object with one integer value per question key. Return raw item responses only."
    if response_format_instruction_version == "v2":
        closing = f"{closing} {G_FORMAT_INSTRUCTION_V2}"
    user = f"RESPONDENT PROFILE\n{profile_text}\n\nSURVEY MATERIAL\n{sc.get_common_climate_scientist_context()}\n\n{condition_stimulus}\n\nSURVEY QUESTIONS\n{questionnaire_text}\n\n{closing}"
    request_key = f"G|{donor_key}|{condition_id}|replicate_{replicate_id}"
    if prompt_variant != "P1":
        request_key = f"{request_key}|variant_{prompt_variant}"
    if response_format_instruction_version != "v1":
        request_key = f"{request_key}|fmt_{response_format_instruction_version}"
    return PromptRender(
        role="G",
        protocol_id=G_PROMPT_PROTOCOL,
        request_key=request_key,
        system_prompt=resolved_system_prompt,
        user_prompt=user,
        response_schema=item_json_schema(rendered_items),
        stimulus_text=condition_stimulus,
        nonstimulus_text=user.replace(condition_stimulus, "<<STIMULUS>>", 1),
        questionnaire_order=questionnaire_order,
        response_key_map={item["response_key"]: item["target_label"] for item in rendered_items},
    )


G_EXTERNAL_VALIDATION_PROTOCOL_ID = "g_external_validation_v1"


def build_g_external_validation_prompt_render(
    profile: Mapping[str, object],
    items: list[dict[str, Any]],
    *,
    external_material: str = "",
    source_id: str,
    respondent_id: str,
    replicate_id: int = 1,
    response_format_instruction_version: str = "v1",
) -> PromptRender:
    """External human-validation G prompt: a demographic-only persona
    answering an EXTERNAL survey instrument, not the target krsteski
    questionnaire. Reuses the frozen G respondent system instruction
    (G_SYSTEM_PROMPT) and the same generic profile/question-block/schema
    machinery already used for F's external items (profile_description,
    _questions_block, item_json_schema) -- but injects NO target-specific
    content: no climate-scientist context, no target control material, no
    target questions, no target intervention labels, no fabricated filler.

    If the external instrument's administration had no preceding
    participant-visible material for this item (external_material=""), no
    "SURVEY MATERIAL" section is rendered at all -- nothing is invented to
    fill it.

    This function never calls or mutates build_g_prompt_render(); it uses a
    distinct protocol_id (G_EXTERNAL_VALIDATION_PROTOCOL_ID) and request_key
    prefix so it can never collide with target-G request_keys.

    response_format_instruction_version="v1" (default) leaves the closing
    instruction exactly as before -- every existing call site (Howe, the
    ATP G screen) is byte-identical to before this parameter was added.
    "v2" appends G_FORMAT_INSTRUCTION_V2 (format-only, see its docstring),
    same as the other three G builders' G-v2 amendment.
    """
    if response_format_instruction_version not in ("v1", "v2"):
        raise ValueError(f"unknown response_format_instruction_version: {response_format_instruction_version!r}")
    profile_text = profile_description(profile)
    material = external_material.strip()
    material_block = f"SURVEY MATERIAL\n{material}\n\n" if material else ""
    questions_text = _questions_block(items)
    closing = "Return a single JSON object with one integer value per question key. Return raw item responses only."
    if response_format_instruction_version == "v2":
        closing = f"{closing} {G_FORMAT_INSTRUCTION_V2}"
    user = f"RESPONDENT PROFILE\n{profile_text}\n\n{material_block}SURVEY QUESTIONS\n{questions_text}\n\n{closing}"
    request_key = f"G_EXTERNAL|{source_id}|{respondent_id}|replicate_{replicate_id}"
    if response_format_instruction_version != "v1":
        request_key = f"{request_key}|fmt_{response_format_instruction_version}"
    nonstimulus_text = user.replace(material, "<<STIMULUS>>", 1) if material else user
    return PromptRender(
        role="G",
        protocol_id=G_EXTERNAL_VALIDATION_PROTOCOL_ID,
        request_key=request_key,
        system_prompt=G_SYSTEM_PROMPT,
        user_prompt=user,
        response_schema=item_json_schema(items),
        stimulus_text=material,
        nonstimulus_text=nonstimulus_text,
        response_key_map={item.get("response_key", item["qualtrics_label"]): item["target_label"] for item in items},
        provenance={"external_source_id": source_id, "external_respondent_id": respondent_id},
    )


def build_g_block_messages(
    profile: Mapping[str, object],
    condition_stimulus: str,
    items: list[dict[str, Any]],
    *,
    donor_key: str = "",
    condition_id: str = "",
    replicate_id: int = 1,
) -> list[dict[str, str]]:
    return build_g_prompt_render(
        profile,
        condition_stimulus,
        items,
        donor_key=donor_key,
        condition_id=condition_id,
        replicate_id=replicate_id,
    ).messages


def f_variant_assignment(study_id: str, f_profile_id: str, outcome_id: str, replicate_id: int = 1) -> dict[str, str]:
    """Stable F prompt-variant assignment. Condition id is intentionally absent."""
    key = stable_hash(study_id, f_profile_id, outcome_id, replicate_id, F_PROMPT_PROTOCOL)
    value = int(key[:16], 16)
    intro = F_INTRO_VARIANTS[value % len(F_INTRO_VARIANTS)][0]
    profile_label = F_PROFILE_LABEL_VARIANTS[(value // len(F_INTRO_VARIANTS)) % len(F_PROFILE_LABEL_VARIANTS)][0]
    profile_format = F_PROFILE_FORMAT_VARIANTS[(value // 17) % len(F_PROFILE_FORMAT_VARIANTS)][0]
    survey_format = F_SURVEY_FORMAT_VARIANTS[(value // 31) % len(F_SURVEY_FORMAT_VARIANTS)][0]
    variant_id = f"{intro}+{profile_label}+{profile_format}+{survey_format}"
    return {
        "assignment_key_hash": key,
        "prompt_variant_id": variant_id,
        "intro_variant_id": intro,
        "profile_label_variant_id": profile_label,
        "profile_format_variant_id": profile_format,
        "survey_format_variant_id": survey_format,
        "assignment_algorithm": F_VARIANT_ASSIGNMENT_VERSION,
    }


def _variant_text(variants: Sequence[tuple[str, str]], variant_id: str) -> str:
    return dict(variants)[variant_id]


def build_f_prompt_render(
    profile: Mapping[str, object],
    condition_stimulus: str,
    outcome: str,
    *,
    study_id: str,
    f_profile_id: str,
    replicate_id: int = 1,
    condition_id: str = "",
    study_setting: str = "This is an online survey shown to a broad sample of adult respondents.",
    all_items: list[dict[str, Any]] | None = None,
    response_format_instruction_version: str = "v1",
) -> PromptRender:
    """F prompt: one profile, one condition, one scored outcome block.

    response_format_instruction_version: forwarded unchanged to
    build_f_prompt_render_from_items -- "v1" (default) reproduces prior
    behavior byte-for-byte for every existing caller; "v2" opts into the R1
    root-cause format-only remediation.
    """
    if study_id == "target" and condition_id == "Consensus":
        raise ValueError("target F Consensus requires build_f_consensus_stage_a_prompt_render/build_f_consensus_stage_b_prompt_render")
    items = items_for_scored_outcome(outcome, all_items)
    resolved_condition_stimulus = f_target_condition_material(condition_stimulus) if study_id == "target" else condition_stimulus
    return build_f_prompt_render_from_items(
        profile,
        resolved_condition_stimulus,
        items,
        study_id=study_id,
        f_profile_id=f_profile_id,
        outcome_id=outcome,
        replicate_id=replicate_id,
        condition_id=condition_id,
        study_setting=study_setting,
        response_format_instruction_version=response_format_instruction_version,
        outcome_context=f_target_outcome_context(outcome),
    )


def _f_response_format_instruction_v2(items: list[dict[str, Any]]) -> str:
    """Format-only replacement for the F closing instruction sentence, added
    per the R1 root-cause amendment (provider-side structured-output
    enforcement was unreliable at scale; the frozen closing sentence's "one
    integer value per item label" wording is ambiguous against archived
    outcome text that itself contains a letter label like "A." for a
    single-extracted sub-item, and an unenforced model would sometimes key
    its JSON on that letter instead of the schema's real key).

    Mechanically derived from item_json_schema(items) -- the SAME schema
    already sent via response_format -- so the plain-text copy shown to the
    model and the structural schema constraint can never drift apart. Adds
    no new response variable, no semantic instruction, and does not touch
    item wording/labels/response anchors, which remain exactly as archived.
    """
    schema = item_json_schema(items)
    keys = schema["required"]
    keys_text = ", ".join(f'"{k}"' for k in keys)
    schema_text = json.dumps(schema, indent=2, sort_keys=True)
    return (
        "Respond ONLY with one JSON object. Do not use Markdown formatting or code fences -- "
        "output the raw JSON object and nothing else.\n"
        f"The object must contain exactly {'this key' if len(keys) == 1 else 'these keys'}, each an integer: {keys_text}.\n"
        "Do NOT use a survey item's own label (for example \"A\") as a JSON key -- use exactly "
        f"the key name{'s' if len(keys) != 1 else ''} listed above.\n"
        "Your response must satisfy exactly this JSON schema:\n"
        f"{schema_text}"
    )


def build_f_prompt_render_from_items(
    profile: Mapping[str, object],
    condition_stimulus: str,
    items: list[dict[str, Any]],
    *,
    study_id: str,
    f_profile_id: str,
    outcome_id: str,
    replicate_id: int = 1,
    condition_id: str = "",
    study_setting: str = "This is an online survey shown to a broad sample of adult respondents.",
    outcome_context: str = "",
    conversation_history: list[dict[str, str]] | None = None,
    provenance: dict[str, Any] | None = None,
    intentional_no_material_control: bool = False,
    response_format_instruction_version: str = "v1",
) -> PromptRender:
    """F prompt from resolved source items.

    This is the same active F protocol as build_f_prompt_render(), but it lets
    external calibration studies supply their own participant-facing outcome
    item after archive parsing rather than forcing benchmark composite names.

    intentional_no_material_control must be set True ONLY by a caller that has
    already established, from frozen source provenance (the t_hypothesis==0
    designated control arm whose own archived transcript shows nothing between
    demographics and the outcome question), that this specific request is a
    genuine no-material control condition -- never as a general workaround for
    missing/unresolved stimulus text.

    response_format_instruction_version: "v1" (default) reproduces the
    original closing instruction byte-for-byte -- every existing caller is
    unaffected unless it explicitly opts in. "v2" substitutes the R1
    root-cause amendment's format-only instruction (see
    _f_response_format_instruction_v2); everything else about the render is
    identical between the two versions for the same inputs.
    """
    if response_format_instruction_version not in ("v1", "v2"):
        raise ValueError(f"unknown response_format_instruction_version: {response_format_instruction_version!r}")
    _assert_one_resolved_stimulus(condition_stimulus, allow_empty=intentional_no_material_control)
    if not items:
        raise ValueError("F prompt requires at least one resolved outcome item")
    variant = f_variant_assignment(study_id, f_profile_id, outcome_id, replicate_id)
    intro = _variant_text(F_INTRO_VARIANTS, variant["intro_variant_id"])
    profile_label = _variant_text(F_PROFILE_LABEL_VARIANTS, variant["profile_label_variant_id"])
    profile_style = _variant_text(F_PROFILE_FORMAT_VARIANTS, variant["profile_format_variant_id"])
    survey_style = _variant_text(F_SURVEY_FORMAT_VARIANTS, variant["survey_format_variant_id"])
    profile_text = profile_description(profile, style=profile_style)

    if survey_style == "pages":
        material_label = "The survey page says:"
        question_label = "The outcome question page says:"
    else:
        material_label = "Survey material:"
        question_label = "Outcome questions:"
    outcome_context_block = f"{outcome_context.strip()}\n\n" if outcome_context.strip() else ""
    closing_instruction = (
        "Return a single JSON object with one integer value per item label on the native response scale."
        if response_format_instruction_version == "v1"
        else _f_response_format_instruction_v2(items)
    )

    user = (
        f"{intro}\n\n"
        f"{profile_label.upper()}\n{profile_text}\n\n"
        f"STUDY SETTING\n{study_setting}\n\n"
        f"SURVEY MATERIAL\n{material_label}\n{condition_stimulus}\n\n"
        f"OUTCOME QUESTIONS\n{question_label}\n{outcome_context_block}{_questions_block(items)}\n\n"
        f"{closing_instruction}"
    )
    request_key = f"F|{study_id}|{f_profile_id}|{condition_id}|{outcome_id}|replicate_{replicate_id}"
    if condition_stimulus:
        nonstimulus_text = user.replace(condition_stimulus, "<<STIMULUS>>", 1)
    else:
        # str.replace("", ...) would otherwise insert the marker at position 0,
        # not at the (empty) SURVEY MATERIAL slot -- only reachable when
        # intentional_no_material_control authorized an empty stimulus above.
        marker = f"SURVEY MATERIAL\n{material_label}\n\n\n"
        nonstimulus_text = user.replace(marker, f"SURVEY MATERIAL\n{material_label}\n<<STIMULUS>>\n\n", 1)
    return PromptRender(
        role="F",
        protocol_id=F_PROMPT_PROTOCOL,
        request_key=request_key,
        system_prompt=F_SYSTEM_PROMPT,
        user_prompt=user,
        response_schema=item_json_schema(items),
        stimulus_text=condition_stimulus,
        nonstimulus_text=nonstimulus_text,
        prompt_variant_id=variant["prompt_variant_id"],
        variant_assignment=variant,
        response_key_map={item["qualtrics_label"]: item["target_label"] for item in items},
        conversation_history=conversation_history or [],
        provenance=provenance,
    )


def build_f_outcome_block_messages(
    profile: Mapping[str, object],
    condition_stimulus: str,
    outcome: str,
    *,
    study_id: str = "target",
    f_profile_id: str = "",
    replicate_id: int = 1,
    condition_id: str = "",
    all_items: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], dict]:
    render = build_f_prompt_render(
        profile,
        condition_stimulus,
        outcome,
        study_id=study_id,
        f_profile_id=f_profile_id,
        replicate_id=replicate_id,
        condition_id=condition_id,
        all_items=all_items,
    )
    return render.messages, render.response_schema


def build_block_messages(profile: Mapping[str, object], condition_stimulus: str, items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Backward-compatible wrapper for active G prompts."""
    return build_g_block_messages(profile, condition_stimulus, items)


def build_outcome_block_messages(
    profile: Mapping[str, object],
    condition_stimulus: str,
    outcome: str,
    all_items: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, str]], dict]:
    """Backward-compatible wrapper for active F prompts."""
    return build_f_outcome_block_messages(profile, condition_stimulus, outcome, all_items=all_items)


def normalize_prompt_without_stimulus(render: PromptRender, sentinel: str = "<<STIMULUS>>") -> str:
    if render.nonstimulus_text:
        return render.nonstimulus_text
    return render.user_prompt.replace(render.stimulus_text, sentinel, 1)


def validate_compiler_no_leakage(render: PromptRender, *, condition_id: str = "") -> list[str]:
    """Check compiler-added text only; participant-facing stimulus is ignored."""
    problems = []
    haystack = f"{render.system_prompt}\n{render.nonstimulus_text}".lower()
    for term in FORBIDDEN_COMPILER_LEAKAGE_TERMS:
        if term.lower() in haystack:
            problems.append(f"compiler-added forbidden leakage term {term!r}")
    if render.role == "G":
        for term in FORBIDDEN_G_PROFILE_TERMS:
            if term.lower() in haystack:
                problems.append(f"G compiler-added forbidden persona term {term!r}")
        for term in FORBIDDEN_G_QUESTIONNAIRE_METADATA_TERMS:
            if term.lower() in haystack:
                problems.append(f"G compiler-added forbidden questionnaire metadata term {term!r}")
        if condition_id and condition_id.lower() in haystack:
            problems.append(f"G compiler-added internal condition label {condition_id!r}")
    return problems
