"""Regression tests for the P1/P2/P3 prompt-wrapper addition (Approach 3).

Proves: P1 is exactly the pre-existing frozen production G system prompt
(golden hash unchanged); P2/P3 differ from P1 only in the presentation
wrapper, with everything else (shared tail instructions, rendered user
prompt, response schema, request-key base) byte-identical; and every
existing build_g_prompt_render/build_g_consensus_stage_*_prompt_render call
site (system_prompt=None, prompt_variant="P1") is unchanged from before this
addition.
"""

from __future__ import annotations

import survey_content as sc
from inference.prompts import (
    G_ROLE_WRAPPER_P1,
    G_ROLE_WRAPPER_P2,
    G_ROLE_WRAPPER_P3,
    G_SHARED_TAIL_INSTRUCTIONS,
    G_SYSTEM_PROMPT,
    G_SYSTEM_PROMPT_BY_VARIANT,
    G_SYSTEM_PROMPT_P2,
    G_SYSTEM_PROMPT_P3,
    build_g_prompt_render,
    text_hash,
)

# Golden hash of the pre-existing, unmodified G_SYSTEM_PROMPT (also asserted
# in tests/inference/test_g_external_validation_prompt.py's
# _GOLDEN_SYSTEM_PROMPT_HASH -- repeated here as the load-bearing P1 proof
# for this specific addition).
P1_GOLDEN_SHA256 = "bd227a7dd8fbc554711e48172ab92d9b051720d02750d357d9b47893222d1a8f"


def test_p1_is_byte_identical_to_pre_existing_production_prompt():
    assert text_hash(G_SYSTEM_PROMPT) == P1_GOLDEN_SHA256
    assert G_SYSTEM_PROMPT == f"{G_ROLE_WRAPPER_P1}\n\n{G_SHARED_TAIL_INSTRUCTIONS}"


def test_p2_p3_share_the_exact_same_tail_instructions_as_p1():
    assert G_SYSTEM_PROMPT_P2.endswith(G_SHARED_TAIL_INSTRUCTIONS)
    assert G_SYSTEM_PROMPT_P3.endswith(G_SHARED_TAIL_INSTRUCTIONS)
    assert G_SYSTEM_PROMPT.endswith(G_SHARED_TAIL_INSTRUCTIONS)
    # no duplicated "do not explain or justify" instruction anywhere
    for prompt in (G_SYSTEM_PROMPT, G_SYSTEM_PROMPT_P2, G_SYSTEM_PROMPT_P3):
        assert prompt.count("Do not explain or justify") == 1


def test_p2_p3_hashes_are_stable_and_distinct_from_p1():
    p1_hash = text_hash(G_SYSTEM_PROMPT)
    p2_hash = text_hash(G_SYSTEM_PROMPT_P2)
    p3_hash = text_hash(G_SYSTEM_PROMPT_P3)
    assert len({p1_hash, p2_hash, p3_hash}) == 3
    assert G_SYSTEM_PROMPT_BY_VARIANT == {"P1": G_SYSTEM_PROMPT, "P2": G_SYSTEM_PROMPT_P2, "P3": G_SYSTEM_PROMPT_P3}


def _target_profile():
    return {
        "age": 40,
        "gender": "Female",
        "race_ethnicity": "White / Caucasian",
        "education": "Bachelor's degree",
        "household_income": "$56,000 to $99,999",
        "party_id": "Independent",
        "political_ideology": "Moderate",
        "state_abbr": "CA",
        "religion": "Protestant",
    }


def test_only_system_prompt_differs_between_p1_p2_p3_renders():
    items = sc.load_items()
    renders = {
        variant: build_g_prompt_render(
            _target_profile(), "Stimulus text.", items, donor_key="D1", condition_id="control", system_prompt=(None if variant == "P1" else G_SYSTEM_PROMPT_BY_VARIANT[variant]), prompt_variant=variant
        )
        for variant in ("P1", "P2", "P3")
    }
    user_prompts = {v: r.user_prompt for v, r in renders.items()}
    assert user_prompts["P1"] == user_prompts["P2"] == user_prompts["P3"]
    schemas = {v: r.response_schema for v, r in renders.items()}
    assert schemas["P1"] == schemas["P2"] == schemas["P3"]
    system_prompts = {v: r.system_prompt for v, r in renders.items()}
    assert len(set(system_prompts.values())) == 3
    # request_key base is identical; only the variant suffix differs
    assert renders["P1"].request_key == "G|D1|control|replicate_1"
    assert renders["P2"].request_key == "G|D1|control|replicate_1|variant_P2"
    assert renders["P3"].request_key == "G|D1|control|replicate_1|variant_P3"


def test_existing_call_sites_unchanged_when_omitting_new_params():
    items = sc.load_items()
    explicit_p1 = build_g_prompt_render(_target_profile(), "Stimulus text.", items, donor_key="D1", condition_id="control", system_prompt=None, prompt_variant="P1")
    default_call = build_g_prompt_render(_target_profile(), "Stimulus text.", items, donor_key="D1", condition_id="control")
    assert explicit_p1.request_key == default_call.request_key
    assert explicit_p1.system_prompt == default_call.system_prompt == G_SYSTEM_PROMPT
    assert explicit_p1.user_prompt == default_call.user_prompt
