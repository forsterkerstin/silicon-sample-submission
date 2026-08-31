"""Benchmark-exact Consensus interaction: prospective correction of the
FAIL_MATERIAL_SEQUENCE_MISMATCH finding against the public instrument
(survey/questionnaire.txt lines 507-543, corroborated by
survey/survey.qsf's FL_137 block structure).

Two things the prior consensus_two_stage_interactive_adaptation_v1
implementation (build_g_consensus_stage_a_prompt_render /
build_g_consensus_stage_b_prompt_render in inference/prompts.py, left
UNMODIFIED and UNUSED by any new manifest going forward) got wrong,
corrected here:

1. Item order: the public spec requires "[Randomize with #3 always in the
   middle]" (net_zero_before_2085 is always item #3). The only two legal
   full orders are:
       1-3-2: human_primary_cause, net_zero_before_2085, co2_warms_planet
       2-3-1: co2_warms_planet, net_zero_before_2085, human_primary_cause
   assign_consensus_exact_order() picks one of these two, deterministically
   from donor_key alone (no replicate_id/attempt_id dependency at all --
   the assignment can never shift across a technical retry).

2. Feedback timing: the public spec requires "Feedback: Given directly
   after each item" -- each item's correction must be shown immediately
   after that item's own estimate and before the next item's estimate.
   This module implements four genuinely sequential, chained requests per
   donor (STEP_1, STEP_2, STEP_3, OUTCOMES), each one carrying forward the
   PRIOR steps' real conversation (question + the model's own real answer)
   as conversation_history, with the completed step's feedback text
   prepended to the NEXT step's question. No step's render function ever
   has access to a later step's feedback -- there is structurally no way
   for a not-yet-asked item's correct answer to leak in, since each
   builder only ever receives the PRIOR steps' already-materialized
   records as input.

Every builder function's donor/persona/scientific content is completely
independent of attempt_id -- attempt_id changes ONLY the request_key
(hence seed/custom_id), never anything else. A stage's builder for attempt
N>1 takes the EXACT SAME prior-step record(s) as attempt 1 -- retrying a
downstream stage can never resample or alter an already-locked upstream
stage's response.
"""

from __future__ import annotations

from typing import Any, Mapping

import survey_content as sc
from inference.prompts import (
    G_FORMAT_INSTRUCTION_V2,
    G_SYSTEM_PROMPT,
    PromptRender,
    _build_g_questionnaire,
    _questions_block,
    item_json_schema,
    profile_description,
    stable_hash,
)

CONSENSUS_EXACT_PROTOCOL_ID = "consensus_benchmark_exact_v1"
MIDDLE_BLOCK_KEY = "net_zero_before_2085"
OUTER_BLOCK_KEYS = ("human_primary_cause", "co2_warms_planet")
LEGAL_ORDERS = (
    (OUTER_BLOCK_KEYS[0], MIDDLE_BLOCK_KEY, OUTER_BLOCK_KEYS[1]),  # 1-3-2
    (OUTER_BLOCK_KEYS[1], MIDDLE_BLOCK_KEY, OUTER_BLOCK_KEYS[0]),  # 2-3-1
)
STEP_NAMES = ("step1", "step2", "step3", "outcomes")

_CLOSING_INSTRUCTION = "Return a single JSON object with one integer value per question key. Return raw item responses only."


def assign_consensus_exact_order(donor_key: str) -> tuple[str, str, str]:
    """Deterministic, donor-only (never replicate/attempt-dependent) choice
    between the two legal orders. Computed once, before any response is
    observed; never re-evaluated after the fact."""
    h = stable_hash(donor_key, CONSENSUS_EXACT_PROTOCOL_ID, sc.CONSENSUS_INTERACTION_RANDOMIZER_FLOW_ID)
    return LEGAL_ORDERS[int(h, 16) % 2]


def _items_by_key() -> dict[str, dict[str, Any]]:
    return {item["block_key"]: item for item in sc.get_consensus_estimate_items()}


def _single_item_render(item: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    rendered = dict(item)
    rendered["response_key"] = "Q001"
    schema = item_json_schema([rendered])
    return rendered, schema


def _request_key(donor_key: str, step_name: str, attempt_id: int) -> str:
    return f"G|{donor_key}|ConsensusExact|{step_name}|attempt_{attempt_id}"


def build_step1_prompt_render(profile: Mapping[str, object], *, donor_key: str, attempt_id: int = 1) -> PromptRender:
    order = assign_consensus_exact_order(donor_key)
    item, schema = _single_item_render(_items_by_key()[order[0]])
    intro_text = sc.get_consensus_stage_a_intro_text()
    user = f"RESPONDENT PROFILE\n{profile_description(profile)}\n\nSURVEY MATERIAL\n{sc.get_common_climate_scientist_context()}\n\n{intro_text}\n\nSURVEY QUESTIONS\n{_questions_block([item])}\n\n{_CLOSING_INSTRUCTION} {G_FORMAT_INSTRUCTION_V2}"
    return PromptRender(
        role="G",
        protocol_id=CONSENSUS_EXACT_PROTOCOL_ID,
        request_key=_request_key(donor_key, "step1", attempt_id),
        system_prompt=G_SYSTEM_PROMPT,
        user_prompt=user,
        response_schema=schema,
        stimulus_text=intro_text,
        nonstimulus_text=user.replace(intro_text, "<<STIMULUS>>", 1),
        response_key_map={"Q001": item["target_label"]},
        provenance={"order": order, "block_key": order[0], "step_name": "step1"},
    )


def step_record(render: PromptRender, response: Mapping[str, Any], *, donor_key: str, attempt_id: int) -> dict[str, Any]:
    """Analogous to inference.prompts.consensus_stage_a_record: captures
    exactly what the NEXT step needs, and nothing scientific beyond the
    respondent's own already-given answer (which the next step's prompt
    must legitimately echo back as conversation history -- this is prompt
    construction, not a retry/reconciliation membership decision)."""
    expected_required = set(render.response_schema.get("required", []))
    observed = set(response)
    if expected_required != observed:
        raise ValueError(f"step response keys do not match schema: expected {sorted(expected_required)}, observed {sorted(observed)}")
    import json

    response_json_text = json.dumps(dict(response), sort_keys=True, separators=(",", ":"))
    return {
        "protocol_id": CONSENSUS_EXACT_PROTOCOL_ID,
        "donor_key": str(donor_key),
        "attempt_id": int(attempt_id),
        "step_name": render.provenance["step_name"],
        "block_key": render.provenance.get("block_key"),
        "order": render.provenance["order"],
        "request_key": render.request_key,
        # deliberately EXCLUDES the system prompt: this chains forward as
        # the NEXT step's conversation_history, and that step's own render
        # supplies system_prompt=G_SYSTEM_PROMPT itself (PromptRender.messages
        # = [system, *conversation_history, user]) -- including it here would
        # duplicate the system turn.
        "messages": [*render.conversation_history, {"role": "user", "content": render.user_prompt}, {"role": "assistant", "content": response_json_text}],
        "response": dict(response),
        "response_json": response_json_text,
    }


def _validate_chain(step1_record: Mapping[str, Any], donor_key: str) -> tuple[str, str, str]:
    if step1_record["donor_key"] != str(donor_key):
        raise ValueError(f"step1_record donor_key {step1_record['donor_key']!r} does not match {donor_key!r}")
    order = tuple(step1_record["order"])
    if order != assign_consensus_exact_order(donor_key):
        raise ValueError(f"step1_record order {order} does not match the frozen deterministic assignment for donor {donor_key!r}")
    if order not in LEGAL_ORDERS:
        raise ValueError(f"order {order} is not one of the two legal benchmark-exact orders")
    return order


def build_step2_prompt_render(profile: Mapping[str, object], *, donor_key: str, step1_record: Mapping[str, Any], attempt_id: int = 1) -> PromptRender:
    order = _validate_chain(step1_record, donor_key)
    if step1_record["step_name"] != "step1":
        raise ValueError("build_step2_prompt_render requires a step1 record")
    feedback = sc.get_consensus_single_item_feedback_text(order[0])
    item, schema = _single_item_render(_items_by_key()[order[1]])  # always MIDDLE_BLOCK_KEY
    user = f"SURVEY MATERIAL\n{feedback}\n\nSURVEY QUESTIONS\n{_questions_block([item])}\n\n{_CLOSING_INSTRUCTION} {G_FORMAT_INSTRUCTION_V2}"
    return PromptRender(
        role="G",
        protocol_id=CONSENSUS_EXACT_PROTOCOL_ID,
        request_key=_request_key(donor_key, "step2", attempt_id),
        system_prompt=G_SYSTEM_PROMPT,
        user_prompt=user,
        response_schema=schema,
        stimulus_text=feedback,
        nonstimulus_text=user.replace(feedback, "<<STIMULUS>>", 1),
        response_key_map={"Q001": item["target_label"]},
        conversation_history=list(step1_record["messages"]),
        provenance={"order": order, "block_key": order[1], "step_name": "step2"},
    )


def build_step3_prompt_render(profile: Mapping[str, object], *, donor_key: str, step1_record: Mapping[str, Any], step2_record: Mapping[str, Any], attempt_id: int = 1) -> PromptRender:
    order = _validate_chain(step1_record, donor_key)
    if step2_record["step_name"] != "step2" or tuple(step2_record["order"]) != order:
        raise ValueError("build_step3_prompt_render requires a step2 record chained from the same step1_record's order")
    feedback = sc.get_consensus_single_item_feedback_text(order[1])
    item, schema = _single_item_render(_items_by_key()[order[2]])
    user = f"SURVEY MATERIAL\n{feedback}\n\nSURVEY QUESTIONS\n{_questions_block([item])}\n\n{_CLOSING_INSTRUCTION} {G_FORMAT_INSTRUCTION_V2}"
    return PromptRender(
        role="G",
        protocol_id=CONSENSUS_EXACT_PROTOCOL_ID,
        request_key=_request_key(donor_key, "step3", attempt_id),
        system_prompt=G_SYSTEM_PROMPT,
        user_prompt=user,
        response_schema=schema,
        stimulus_text=feedback,
        nonstimulus_text=user.replace(feedback, "<<STIMULUS>>", 1),
        response_key_map={"Q001": item["target_label"]},
        conversation_history=list(step2_record["messages"]),
        provenance={"order": order, "block_key": order[2], "step_name": "step3"},
    )


def build_outcomes_prompt_render(
    profile: Mapping[str, object],
    items: list[dict[str, Any]],
    *,
    donor_key: str,
    step1_record: Mapping[str, Any],
    step2_record: Mapping[str, Any],
    step3_record: Mapping[str, Any],
    attempt_id: int = 1,
) -> PromptRender:
    order = _validate_chain(step1_record, donor_key)
    if step2_record["step_name"] != "step2" or tuple(step2_record["order"]) != order:
        raise ValueError("build_outcomes_prompt_render requires a step2 record chained from the same step1_record's order")
    if step3_record["step_name"] != "step3" or tuple(step3_record["order"]) != order:
        raise ValueError("build_outcomes_prompt_render requires a step3 record chained from the same step1_record's order")
    final_feedback = sc.get_consensus_single_item_feedback_text(order[2])
    closing = sc.get_consensus_closing_text()
    questionnaire_text, rendered_items, questionnaire_order = _build_g_questionnaire(items, donor_key)
    user = f"SURVEY MATERIAL\n{final_feedback}\n\n{closing}\n\nSURVEY QUESTIONS\n{questionnaire_text}\n\n{_CLOSING_INSTRUCTION} {G_FORMAT_INSTRUCTION_V2}"
    return PromptRender(
        role="G",
        protocol_id=CONSENSUS_EXACT_PROTOCOL_ID,
        request_key=_request_key(donor_key, "outcomes", attempt_id),
        system_prompt=G_SYSTEM_PROMPT,
        user_prompt=user,
        response_schema=item_json_schema(rendered_items),
        stimulus_text=final_feedback,
        nonstimulus_text=user.replace(final_feedback, "<<STIMULUS>>", 1),
        questionnaire_order=questionnaire_order,
        response_key_map={item["response_key"]: item["target_label"] for item in rendered_items},
        conversation_history=list(step3_record["messages"]),
        provenance={"order": order, "step_name": "outcomes"},
    )
