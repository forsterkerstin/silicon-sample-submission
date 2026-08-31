from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
import re
import html

import pandas as pd
import pytest
import yaml

import survey_content as sc
from inference.model_config import load_model_config, model_candidates, provider_parameters, selected_model
from inference.prompts import (
    F_PROMPT_PROTOCOL,
    F_SYSTEM_PROMPT,
    G_PROMPT_PROTOCOL,
    G_QUESTIONNAIRE_VERSION,
    G_SECONDARY_RANDOMIZER_BLOCK_IDS,
    G_SYSTEM_PROMPT,
    G_TERTIARY_RANDOMIZER_BLOCK_IDS,
    LEGACY_SHARED_RESPONDENT_SYSTEM_PROMPT_INACTIVE,
    build_f_consensus_stage_a_prompt_render,
    build_f_consensus_stage_b_prompt_render,
    build_f_prompt_render,
    build_f_prompt_render_from_items,
    build_g_consensus_stage_a_prompt_render,
    build_g_consensus_stage_b_prompt_render,
    build_g_prompt_render,
    build_outcome_block_messages,
    consensus_stage_a_record,
    f_variant_assignment,
    f_target_outcome_context,
    item_json_schema,
    normalize_prompt_without_stimulus,
    target_f_control_variant,
    validate_compiler_no_leakage,
)
from render_prompt_validation import f_leakage_terms, missing_placeholder_hits
from inference.request_logging import request_key_f, request_key_g, seed_from_request_key

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PIPELINE_ROOT.parent


@pytest.fixture(scope="session")
def prompt_artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("prompt_validation")
    env = os.environ.copy()
    subprocess.run(
        [sys.executable, str(PIPELINE_ROOT / "scripts" / "render_prompt_validation.py"), "--output-dir", str(out)],
        cwd=PIPELINE_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "manifest": json.loads((out / "prompt_protocol_manifest.json").read_text(encoding="utf-8")),
        "audit": pd.read_csv(out / "prompt_audit.csv"),
        "pairing": pd.read_csv(out / "f_pairing_audit.csv"),
        "leakage": pd.read_csv(out / "f_effect_leakage_audit.csv"),
        "source_fidelity": pd.read_csv(out / "f_source_fidelity_audit.csv"),
        "target_context": pd.read_csv(out / "f_target_outcome_context_audit.csv"),
        "control_filler": pd.read_csv(out / "f_target_control_filler_audit.csv"),
        "consensus_flow": json.loads((out / "f_consensus_flow_audit.json").read_text(encoding="utf-8")),
        "summary": json.loads((out / "summary.json").read_text(encoding="utf-8")),
        "out": out,
    }


def target_profile(state_abbr: str = "CA") -> dict[str, object]:
    abbr_to_state = {abbr: name for name, abbr in sc.STATE_NAME_TO_ABBR.items()}
    return {
        "age": 40,
        "gender": "Female",
        "race": "Asian / Asian American",
        "education": "Bachelor's degree",
        "income": "$56,000 to $99,999",
        "party": "Independent",
        "state_abbr": state_abbr,
        "state": abbr_to_state[state_abbr],
    }


def deterministic_stage_a_response(render) -> dict[str, int]:
    return {key: 50 + index for index, key in enumerate(render.response_schema["required"], start=1)}


def f_stage_a_record(profile: dict[str, object], f_profile_id: str = "F1"):
    stage_a = build_f_consensus_stage_a_prompt_render(profile, f_profile_id=f_profile_id, replicate_id=1)
    return consensus_stage_a_record(stage_a, deterministic_stage_a_response(stage_a), role="F", subject_id=f_profile_id, replicate_id=1)


def g_stage_a_record(profile: dict[str, object], donor_key: str = "D1"):
    stage_a = build_g_consensus_stage_a_prompt_render(profile, donor_key=donor_key, replicate_id=1)
    return consensus_stage_a_record(stage_a, deterministic_stage_a_response(stage_a), role="G", subject_id=donor_key, replicate_id=1)


def test_model_config_declares_candidates_and_frozen_selection():
    """F*/G* were independently selected as google/gemma-4-31B-it (F-screen
    commit 6a25d3b7..., G-ATP-screen commit d09ece8...) and materialized into
    config/model_config.yaml's selected_f_model/selected_g_model by the
    replacement-R1-passed / calibration-production-preparation amendments --
    selected_model(require_frozen=True) must succeed and match, not raise."""
    cfg = load_model_config()

    assert model_candidates("g") == ["deepseek-ai/DeepSeek-V4-Pro-0813", "google/gemma-4-31B-it"]
    assert model_candidates("f") == ["deepseek-ai/DeepSeek-V4-Pro-0813", "google/gemma-4-31B-it"]
    assert cfg["f_protocol"]["n_f"] == 500
    assert cfg["f_protocol"]["f_num_draws"] == 1
    assert cfg["prompting"]["g_prompt_protocol"] == G_PROMPT_PROTOCOL
    assert cfg["prompting"]["f_prompt_protocol"] == F_PROMPT_PROTOCOL
    assert selected_model("g", require_frozen=True) == "google/gemma-4-31B-it"
    assert selected_model("f", require_frozen=True) == "google/gemma-4-31B-it"


def test_selected_model_require_frozen_false_returns_none_when_unset(tmp_path):
    """Synthetic (not the real, now-frozen repo config): require_frozen=False
    must return None rather than raise when a role's selection is genuinely
    unset -- require_frozen=True on the same unset config must still raise."""
    cfg_path = tmp_path / "model_config.yaml"
    cfg_path.write_text(
        "model_selection:\n"
        "  candidate_bakeoff_required: true\n"
        "  g_model_candidates: [deepseek-ai/DeepSeek-V4-Pro-0813, google/gemma-4-31B-it]\n"
        "  f_model_candidates: [deepseek-ai/DeepSeek-V4-Pro-0813, google/gemma-4-31B-it]\n"
        "  selected_g_model:\n"
        "  selected_f_model:\n",
        encoding="utf-8",
    )
    assert selected_model("g", require_frozen=False, path=cfg_path) is None
    assert selected_model("f", require_frozen=False, path=cfg_path) is None
    with pytest.raises(RuntimeError, match="not frozen"):
        selected_model("g", require_frozen=True, path=cfg_path)


def test_provider_parameters_omit_unsupported_reasoning_effort():
    params = provider_parameters(supports_reasoning_effort=False)

    assert "reasoning_effort" not in params
    assert params["reasoning_effort_omitted_reason"]


def test_active_g_protocol_is_krsteski_demo_survey_adapted_v1():
    assert G_PROMPT_PROTOCOL == "krsteski_demo_survey_adapted_v1"
    assert G_SYSTEM_PROMPT.startswith("You are a survey respondent.")


def test_active_f_protocol_is_ashokkumar_experiment_forecast_adapted_v1():
    assert F_PROMPT_PROTOCOL == "ashokkumar_experiment_forecast_adapted_v1"
    assert F_SYSTEM_PROMPT.startswith("You are completing a simulated survey response task")


def test_no_active_shared_g_f_system_prompt_remains():
    assert G_SYSTEM_PROMPT != F_SYSTEM_PROMPT
    assert LEGACY_SHARED_RESPONDENT_SYSTEM_PROMPT_INACTIVE not in {G_SYSTEM_PROMPT, F_SYSTEM_PROMPT}


def test_g_persona_contains_no_previous_synthetic_survey_responses():
    render = build_g_prompt_render(target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")
    assert "previous survey" not in render.nonstimulus_text.lower()
    assert "prior synthetic" not in render.nonstimulus_text.lower()
    assert "you choose" not in render.nonstimulus_text.lower()


def test_g_persona_contains_no_fabricated_climate_attitude_or_trust_variables():
    render = build_g_prompt_render(target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")
    forbidden = ["climate concern", "climate belief", "trust in scientists", "policy preferences", "donation tendency", "newsletter tendency", "personality", "hobbies", "occupation"]
    for term in forbidden:
        assert term not in render.nonstimulus_text.lower()


def test_g_exposes_exactly_one_condition_material():
    render = build_g_prompt_render(target_profile(), "One and only one stimulus text.", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")
    assert render.user_prompt.count("SURVEY MATERIAL") == 1
    assert render.user_prompt.count("One and only one stimulus text.") == 1


def test_g_prompt_renders_questionnaire_blocks_without_scoring_metadata():
    render = build_g_prompt_render(target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")
    prompt_lower = render.nonstimulus_text.lower()

    assert "post-condition transition" in prompt_lower
    assert "you are now moving on to the final section of the study" in prompt_lower
    assert "learn more about climate science" in prompt_lower
    assert 'did you subscribe to the "talking climate" newsletter on the previous page?' in prompt_lower
    for forbidden in ["reverse-coded", "recoded", "cleaning", "raw qualtrics", "funding_perceptions", "newsletter_signup"]:
        assert forbidden not in prompt_lower


def test_g_prompt_includes_participant_facing_common_climate_scientist_context():
    render = build_g_prompt_render(target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="control")
    context = sc.get_common_climate_scientist_context()

    assert context in render.user_prompt
    assert render.user_prompt.index(context) < render.user_prompt.index("Stimulus")
    assert "Climate scientists study changes in the Earth's climate over time" in context
    assert "prospect of working as a climate scientist" not in render.user_prompt


def test_g_prompt_omits_model_visible_slider_integer_sentence():
    render = build_g_prompt_render(target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="control")

    assert "All 0–100 slider items are also integers." not in render.user_prompt
    assert "All 0-100 slider items are also integers." not in render.user_prompt


def test_g_questionnaire_order_is_stable_by_donor_not_condition():
    profile = target_profile()
    control = build_g_prompt_render(profile, "Control stimulus", sc.load_items(), donor_key="D1", condition_id="control")
    treatment = build_g_prompt_render(profile, "Treatment stimulus", sc.load_items(), donor_key="D1", condition_id="Corporate reliance")

    assert control.questionnaire_order["questionnaire_version"] == G_QUESTIONNAIRE_VERSION
    assert control.questionnaire_order["condition_id_excluded"] is True
    assert control.questionnaire_order["ordered_block_ids"][0] == "trust_multidimensional"
    assert control.questionnaire_order["qualtrics_randomizer_structure"]["secondary_outcomes_FL_55"]
    assert control.questionnaire_order["qualtrics_randomizer_structure"]["tertiary_outcomes_FL_49"]
    assert control.questionnaire_order == treatment.questionnaire_order
    assert normalize_prompt_without_stimulus(control) == normalize_prompt_without_stimulus(treatment)


def _strip_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def test_g_randomizer_structure_matches_raw_qualtrics_secondary_and_tertiary_blocks():
    survey = json.loads((REPO_ROOT / "survey" / "survey.json").read_text(encoding="utf-8"))["result"]
    flow = survey["SurveyFlow"]["Flow"]
    secondary = flow[41]["Flow"][0]
    tertiary = flow[42]["Flow"][0]
    blocks = {block["ID"]: block["Description"] for block in survey["Blocks"].values()}
    secondary_desc = [blocks[child["ID"]] for child in secondary["Flow"]]
    tertiary_desc = [blocks[child["ID"]] for child in tertiary["Flow"]]

    assert secondary["Type"] == "BlockRandomizer"
    assert secondary["FlowID"] == "FL_55"
    assert secondary["SubSet"] == 7
    assert tertiary["Type"] == "BlockRandomizer"
    assert tertiary["FlowID"] == "FL_49"
    assert tertiary["SubSet"] == 5
    assert set(G_SECONDARY_RANDOMIZER_BLOCK_IDS) == {"trust_post", "donation", "distrust_post", "policy_role", "funding", "institutional_trust", "newsletter"}
    assert set(G_TERTIARY_RANDOMIZER_BLOCK_IDS) == {"belief_post", "concern", "behavior", "policy_general", "policy_specific"}
    assert secondary_desc == [
        "trust single post",
        "donation",
        "distrust single post",
        "scientists' role in policy ",
        "funding",
        "institutional trust",
        "subscription newsletter",
    ]
    assert tertiary_desc == [
        "belief post",
        "climate change concern",
        "individual level behavior",
        "support general climate policies",
        "support specific climate policies",
    ]


def test_donation_bonus_intro_is_verbatim_participant_facing_qualtrics_material():
    survey = json.loads((REPO_ROOT / "survey" / "survey.json").read_text(encoding="utf-8"))["result"]
    question = survey["Questions"]["QID1721185865"]
    source_text = _strip_html(question["QuestionText"])
    render = build_g_prompt_render(target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="control")

    assert question["DataExportTag"] == "Q16"
    assert "After data collection is complete, we will randomly select 100 participants from this study" in source_text
    assert "After data collection is complete, we will randomly select 100 participants from this study" in render.user_prompt


def test_g_prompt_uses_neutral_question_keys_and_maps_to_target_labels():
    render = build_g_prompt_render(target_profile(), "Stimulus", sc.load_items(), donor_key="D1", condition_id="control")

    assert render.response_schema["required"][0] == "Q001"
    assert all(label.startswith("Q") for label in render.response_schema["required"])
    assert "trust_competent_1:" not in render.user_prompt
    assert "funding_5:" not in render.user_prompt
    assert render.response_key_map["Q001"] == "trust_competence_1"
    assert "newsletter_signup" in set(render.response_key_map.values())


def test_g_response_key_mapping_is_request_specific_and_round_trips_for_different_donor_orders():
    items = sc.load_items()
    donor_a = "D1"
    donor_b = next(
        f"D{i}"
        for i in range(2, 200)
        if build_g_prompt_render(target_profile(), "Stimulus", items, donor_key=f"D{i}", condition_id="control").response_key_map
        != build_g_prompt_render(target_profile(), "Stimulus", items, donor_key=donor_a, condition_id="control").response_key_map
    )
    render_a = build_g_prompt_render(target_profile(), "Stimulus", items, donor_key=donor_a, condition_id="control")
    render_b = build_g_prompt_render(target_profile(), "Stimulus", items, donor_key=donor_b, condition_id="control")
    raw_a = {key: i for i, key in enumerate(render_a.response_schema["required"], start=1)}
    raw_b = {key: i for i, key in enumerate(render_b.response_schema["required"], start=1)}
    parsed_a = {render_a.response_key_map[key]: raw_a[key] for key in render_a.response_schema["required"]}
    parsed_b = {render_b.response_key_map[key]: raw_b[key] for key in render_b.response_schema["required"]}

    assert render_a.questionnaire_order["ordered_block_ids"] != render_b.questionnaire_order["ordered_block_ids"]
    assert render_a.response_key_map != render_b.response_key_map
    assert set(parsed_a) == {item["target_label"] for item in items}
    assert set(parsed_b) == {item["target_label"] for item in items}
    assert parsed_a["donation_ams"] == raw_a[next(key for key, label in render_a.response_key_map.items() if label == "donation_ams")]
    assert parsed_b["donation_ams"] == raw_b[next(key for key, label in render_b.response_key_map.items() if label == "donation_ams")]


def test_same_g_donor_has_identical_response_key_mapping_across_all_conditions():
    condition_names = yaml.safe_load((PIPELINE_ROOT / "config" / "benchmark_schema.yaml").read_text(encoding="utf-8"))["conditions"]
    items = sc.load_items()
    profile = target_profile("TX")
    reference = None
    for condition in condition_names:
        stimulus = sc.get_condition_stimulus(condition, profile["state_abbr"], control_variant=1)
        if condition == "Consensus":
            record = g_stage_a_record(profile, "D-stable")
            render = build_g_consensus_stage_b_prompt_render(profile, items, record, donor_key="D-stable", replicate_id=1)
        else:
            render = build_g_prompt_render(profile, stimulus, items, donor_key="D-stable", condition_id=condition)
        if reference is None:
            reference = render.response_key_map
        assert render.response_key_map == reference
        assert render.questionnaire_order["condition_id_excluded"] is True


def test_g_has_no_participant_facing_internal_treatment_label():
    condition = "Corporate reliance"
    render = build_g_prompt_render(target_profile(), sc.get_condition_stimulus(condition), sc.load_items(), donor_key="D1", condition_id=condition)
    assert condition.lower() not in render.nonstimulus_text.lower()
    assert validate_compiler_no_leakage(render, condition_id=condition) == []


def test_all_g_raw_response_schemas_enforce_native_support():
    schema = item_json_schema(sc.load_items())
    item_by_label = {item["qualtrics_label"]: item for item in sc.load_items()}
    for label, spec in schema["properties"].items():
        item = item_by_label[label]
        if item["target_label"] == "newsletter_signup":
            assert spec == {"type": "integer", "enum": [0, 1]}
        elif item["target_label"] == "donation_ams":
            assert spec["minimum"] == 0 and spec["maximum"] == 10
        else:
            assert spec["minimum"] == 0 and spec["maximum"] == 100


def test_g_composites_are_computed_mechanically():
    raw = {item["target_label"]: 50 for item in sc.load_items()}
    raw["funding_5_raw"] = 20
    outcomes = sc.compute_outcomes(raw)
    assert outcomes["trust_multidimensional"] == 50
    assert outcomes["funding_perceptions"] == 80


def test_state_exists_for_all_g_donors():
    g = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv")
    assert g["state_abbr"].notna().all()
    assert set(g["state_abbr"]) <= set(sc.STATE_NAME_TO_ABBR.values())


def test_every_extreme_weather_prompt_resolves_by_state():
    for state in sc.STATE_NAME_TO_ABBR.values():
        assert sc.get_condition_stimulus("Extreme weather predictions", state).strip()


def test_g_production_request_count_is_17000_when_n_g_is_1000():
    design = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "tier1_design_skeleton.csv")
    assert design["donor_key"].nunique() == 1000
    assert design["condition"].nunique() == 17
    assert len(design) == 17_000


def test_f_request_contains_one_scored_outcome_battery_only():
    messages, schema = build_outcome_block_messages(target_profile(), "Stimulus", "trust_multidimensional")
    assert len(schema["required"]) == 12
    assert all(label.startswith("trust_") for label in schema["required"])
    assert "donation" not in messages[1]["content"].lower()
    assert "newsletter" not in messages[1]["content"].lower()


def test_f_newsletter_context_is_participant_facing_and_before_signup_response():
    control = build_f_prompt_render(target_profile(), "Control stimulus", "newsletter_signup", study_id="target", f_profile_id="F1", condition_id="control")
    treatment = build_f_consensus_stage_b_prompt_render(
        target_profile(),
        "newsletter_signup",
        f_stage_a_record(target_profile(), "F1"),
        f_profile_id="F1",
    )
    context = f_target_outcome_context("newsletter_signup")

    for render, stimulus in [(control, "Control stimulus"), (treatment, "Treatment stimulus")]:
        assert context in render.user_prompt
        if stimulus in render.user_prompt:
            assert render.user_prompt.index(stimulus) < render.user_prompt.index(context)
        assert render.user_prompt.index(context) < render.user_prompt.index('newsletter: Did you subscribe to the "Talking Climate" newsletter on the previous page?')
        assert "Response options: Yes or No." in render.user_prompt
        assert "Answer 1 for Yes and 0 for No." in render.user_prompt
        assert "Answer with an integer from 0 to 1." not in render.user_prompt
        assert "recoded" not in render.user_prompt.lower()
        assert "cleaning" not in render.user_prompt.lower()
    assert normalize_prompt_without_stimulus(control) == normalize_prompt_without_stimulus(treatment)


def test_f_donation_context_is_participant_facing_and_before_donation_response():
    render = build_f_prompt_render(target_profile(), "Stimulus", "donation_ams", study_id="target", f_profile_id="F1", condition_id="control")
    context = f_target_outcome_context("donation_ams")

    assert context in render.user_prompt
    assert render.user_prompt.index("Stimulus") < render.user_prompt.index(context)
    assert render.user_prompt.index(context) < render.user_prompt.index("donation: Of the $10 bonus")
    assert "Response options: $0-$10 in $1 increments." in render.user_prompt
    assert "All 0–100 slider items are also integers." not in render.user_prompt
    assert "All 0-100 slider items are also integers." not in render.user_prompt


def test_target_f_includes_common_climate_scientist_context_before_condition_material():
    stimulus = sc.get_condition_stimulus("Funding")
    render = build_f_prompt_render(target_profile(), stimulus, "trust_post", study_id="target", f_profile_id="F1", condition_id="Funding")
    context = sc.get_common_climate_scientist_context()

    assert context in render.user_prompt
    assert render.user_prompt.index(context) < render.user_prompt.index(stimulus)
    assert "prospect of working as a climate scientist" not in render.user_prompt


def test_f_never_receives_human_or_calibrated_ate():
    render = build_f_prompt_render(target_profile(), "Stimulus", "trust_post", study_id="s", f_profile_id="F1", condition_id="t")
    forbidden = ["human_ate", "calibrated_ate", "target_ate", "raw_f_ate", "raw_g_ate", "treatment_effect"]
    for term in forbidden:
        assert term not in render.user_prompt.lower()
    assert f_leakage_terms(render) == []
    external = build_f_prompt_render_from_items(
        target_profile(),
        "Stimulus",
        [{"qualtrics_label": "response", "target_label": "response", "question_text": "Question", "response_options": "", "scale": "external_native_integer", "scale_min": 1, "scale_max": 5}],
        study_id="s",
        f_profile_id="F1",
        outcome_id="s:y:hyp1",
        study_setting="This is an online survey shown to adult respondents.",
    )
    assert "external primary calibration archive" not in external.user_prompt.lower()


def test_external_f_item_render_uses_same_protocol_and_native_source_scale():
    item = {
        "qualtrics_label": "response",
        "target_label": "response",
        "question_text": "Would you support this policy?",
        "response_options": "",
        "scale": "external_native_integer",
        "scale_min": 1,
        "scale_max": 2,
    }
    external = build_f_prompt_render_from_items(
        {"gender": "Female", "race": "White"},
        "Archived source material",
        [item],
        study_id="ArchiveStudy",
        f_profile_id="ArchiveStudy__F001",
        outcome_id="ArchiveStudy:response:hyp1",
        condition_id="control",
    )
    target = build_f_prompt_render(target_profile(), "Stimulus", "trust_post", study_id="target", f_profile_id="F1", condition_id="control")

    assert external.protocol_id == target.protocol_id == F_PROMPT_PROTOCOL
    assert external.response_schema["properties"]["response"]["minimum"] == 1
    assert external.response_schema["properties"]["response"]["maximum"] == 2
    assert "percent" not in external.user_prompt.lower()
    assert "Original response scale" not in external.user_prompt


def test_f_prompt_variant_key_excludes_condition_id():
    a = f_variant_assignment("study", "F1", "trust_post", 1)
    b = f_variant_assignment("study", "F1", "trust_post", 1)
    assert a == b


def test_paired_control_treatment_f_requests_use_same_prompt_variant():
    profile = target_profile()
    control = build_f_prompt_render(profile, "Control stimulus", "trust_post", study_id="s", f_profile_id="F1", condition_id="control")
    treatment = build_f_prompt_render(profile, "Treatment stimulus", "trust_post", study_id="s", f_profile_id="F1", condition_id="treatment")
    assert control.prompt_variant_id == treatment.prompt_variant_id
    assert control.variant_assignment == treatment.variant_assignment


def test_after_stimulus_normalization_paired_f_prompt_text_is_identical():
    profile = target_profile()
    control = build_f_prompt_render(profile, "Control stimulus", "trust_post", study_id="s", f_profile_id="F1", condition_id="control")
    treatment = build_f_prompt_render(profile, "Treatment stimulus", "trust_post", study_id="s", f_profile_id="F1", condition_id="treatment")
    assert normalize_prompt_without_stimulus(control) == normalize_prompt_without_stimulus(treatment)


def test_f_retains_n_f_500():
    assert load_model_config()["f_protocol"]["n_f"] == 500
    assert len(pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv")) == 500


def test_target_f_control_fillers_are_balanced_and_stable_by_profile_across_outcomes():
    f_panel = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv")
    counts = f_panel["f_profile_id"].map(lambda value: target_f_control_variant(str(value), 1)).value_counts().sort_index().to_dict()

    assert counts == {1: 167, 2: 167, 3: 166}
    profile_id = str(f_panel.iloc[1]["f_profile_id"])
    assert target_f_control_variant(profile_id, 1) == target_f_control_variant(profile_id, 2)
    assert target_f_control_variant(profile_id, 1) == target_f_control_variant(profile_id, 99)
    expected = sc.get_condition_stimulus("control", control_variant=target_f_control_variant(profile_id, 1))
    for outcome in sc.OUTCOME_COMPOSITES:
        render = build_f_prompt_render(
            target_profile(),
            sc.get_condition_stimulus("control", control_variant=target_f_control_variant(profile_id, 1)),
            outcome,
            study_id="target",
            f_profile_id=profile_id,
            condition_id="control",
        )
        assert expected in render.user_prompt


def test_consensus_stage_a_contains_estimates_only_no_feedback_or_true_answers():
    render = build_g_consensus_stage_a_prompt_render(target_profile(), donor_key="D1", replicate_id=1)
    text = render.user_prompt

    assert "make estimations about scientific agreement" in text
    assert "99% of scientists" not in text
    assert "100% of scientists" not in text
    assert "66% of scientists" not in text
    assert "Surveys of scientists show" not in text
    assert set(render.response_schema["required"]) == {"Q001", "Q002", "Q003"}


def test_consensus_stage_b_contains_exact_stage_a_response_as_history():
    profile = target_profile()
    record = g_stage_a_record(profile, "D1")
    render = build_g_consensus_stage_b_prompt_render(profile, sc.load_items(), record, donor_key="D1", replicate_id=1)

    assert render.conversation_history == record["stage_a_messages"]
    assert render.messages[1]["role"] == "user"
    assert render.messages[2]["role"] == "assistant"
    assert render.messages[2]["content"] == record["stage_a_response_json"]
    assert "99% of scientists" in render.user_prompt
    assert "100% of scientists" in render.user_prompt
    assert "66% of scientists" in render.user_prompt


def test_f_consensus_stage_a_reused_across_outcomes_and_not_condition_control():
    profile = target_profile()
    f_profile_id = "F42"
    stage_a = build_f_consensus_stage_a_prompt_render(profile, f_profile_id=f_profile_id, replicate_id=1)
    record = consensus_stage_a_record(stage_a, deterministic_stage_a_response(stage_a), role="F", subject_id=f_profile_id, replicate_id=1)
    newsletter = build_f_consensus_stage_b_prompt_render(profile, "newsletter_signup", record, f_profile_id=f_profile_id, replicate_id=1)
    donation = build_f_consensus_stage_b_prompt_render(profile, "donation_ams", record, f_profile_id=f_profile_id, replicate_id=1)
    control = build_f_prompt_render(profile, "Control stimulus", "newsletter_signup", study_id="target", f_profile_id=f_profile_id, condition_id="control")

    assert stage_a.request_key == f"F|target|{f_profile_id}|Consensus|stage_a|replicate_1"
    assert newsletter.provenance["stage_a_request_key"] == stage_a.request_key
    assert donation.provenance["stage_a_request_key"] == stage_a.request_key
    assert newsletter.provenance["stage_a_prompt_hash"] == donation.provenance["stage_a_prompt_hash"]
    assert "SCIENTIFIC AGREEMENT ESTIMATE QUESTIONS" not in control.user_prompt
    assert not control.conversation_history


def test_consensus_stage_a_record_from_another_profile_is_rejected():
    profile = target_profile()
    record = f_stage_a_record(profile, "F1")

    with pytest.raises(ValueError, match="subject_id"):
        build_f_consensus_stage_b_prompt_render(profile, "trust_post", record, f_profile_id="F2", replicate_id=1)


def test_target_consensus_one_shot_builders_are_blocked():
    with pytest.raises(ValueError, match="Consensus requires"):
        build_g_prompt_render(target_profile(), sc.get_condition_stimulus("Consensus"), sc.load_items(), donor_key="D1", condition_id="Consensus")
    with pytest.raises(ValueError, match="Consensus requires"):
        build_f_prompt_render(target_profile(), sc.get_condition_stimulus("Consensus"), "trust_post", study_id="target", f_profile_id="F1", condition_id="Consensus")


def test_f_respondent_level_outputs_never_enter_final_tier1_rows():
    submission = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "tier1_submission_skeleton.csv", nrows=1)
    assert "f_profile_id" not in submission.columns
    assert "raw_f_ate" not in " ".join(submission.columns)


def test_prompt_protocol_manifest_is_written(prompt_artifacts):
    manifest = prompt_artifacts["manifest"]
    assert manifest["g_protocol_id"] == G_PROMPT_PROTOCOL
    assert manifest["f_protocol_id"] == F_PROMPT_PROTOCOL
    assert manifest["g_not_used"] == "Krsteski persona-guided prior-response synthesis"


def test_prompt_examples_can_be_rendered_without_api_calls(prompt_artifacts):
    assert prompt_artifacts["summary"]["g_examples"] >= 6
    assert prompt_artifacts["summary"]["f_examples"] >= 5
    assert prompt_artifacts["summary"]["f_pair_examples"] >= 10
    assert list((prompt_artifacts["out"] / "g_prompt_examples").glob("*.txt"))
    assert list((prompt_artifacts["out"] / "f_prompt_examples").glob("*.txt"))
    assert list((prompt_artifacts["out"] / "f_pair_examples").glob("*.txt"))
    assert (prompt_artifacts["out"] / "f_outcome_context_manual_read.md").exists()
    assert (prompt_artifacts["out"] / "f_consensus_flow_audit.json").exists()


def test_metadata_leakage_audit_passes_before_production(prompt_artifacts):
    audit = prompt_artifacts["audit"]
    assert not audit.empty
    assert set(audit["validation_status"]) == {"PASS"}
    leakage = prompt_artifacts["leakage"]
    assert not leakage.empty
    assert set(leakage["status"]) == {"PASS"}
    assert leakage["leakage_terms"].fillna("").eq("").all()
    rendered = "\n".join(path.read_text(encoding="utf-8") for path in (prompt_artifacts["out"] / "f_pair_examples").glob("*.txt"))
    assert "external primary calibration archive" not in rendered.lower()
    assert "Original response scale" not in rendered


def test_f_pairing_audit_passes(prompt_artifacts):
    pairing = prompt_artifacts["pairing"]
    assert not pairing.empty
    assert pairing["nonstimulus_equal"].all()
    assert set(pairing["status"]) == {"PASS"}
    explicit_pairs = pairing[pairing["n_conditions"].eq(2)]
    assert not explicit_pairs.empty
    assert explicit_pairs["schema_equal"].all()
    assert explicit_pairs["condition_id_excluded_from_variant"].all()


def test_f_source_fidelity_audit_covers_external_primary_examples(prompt_artifacts):
    source = prompt_artifacts["source_fidelity"]

    assert source["study_id"].nunique() >= 3
    assert source["effect_id"].nunique() >= 3
    assert set(source["arm"]) == {"control", "treatment"}
    assert "pipeline/data/archive_70studies/llm_responses.RDS" in set(source["source_path"])
    assert "exact participant-facing archive pages" in " ".join(source["stimulus_source_classification"])
    assert source["outcome_max"].ge(source["outcome_min"]).all()


def test_f_target_control_filler_audit_counts_all_three_fillers(prompt_artifacts):
    control = prompt_artifacts["control_filler"]

    assert len(control) == 500
    assert control["control_filler"].value_counts().sort_index().to_dict() == {"baseball": 167, "dances": 166, "neckties": 167}
    assert control.groupby("f_profile_id")["control_filler"].nunique().eq(1).all()


def test_f_consensus_flow_audit_records_interactive_estimate_then_feedback(prompt_artifacts):
    flow = prompt_artifacts["consensus_flow"]

    assert flow["participant_entered_estimates_before_feedback"] is True
    assert flow["interactive_randomizer_flow_id"] == "FL_137"
    assert flow["interactive_randomizer_subset"] == 3
    slider_qids = [qid for block in flow["blocks"] for qid in block["slider_qids"]]
    assert slider_qids == ["QID1721185886", "QID1721185889", "QID1721185892"]


def test_raw_f_prompt_urls_are_not_markdown_escaped():
    stimulus = sc.get_condition_stimulus("Funding")
    render = build_f_prompt_render(target_profile(), stimulus, "trust_post", study_id="target", f_profile_id="F1", condition_id="Funding")

    assert "https://" in render.user_prompt
    assert "https\\://" not in render.user_prompt


def test_f_target_outcome_context_audit_classifies_all_13_outcomes(prompt_artifacts):
    context = prompt_artifacts["target_context"]

    assert set(context["outcome"]) == set(sc.OUTCOME_COMPOSITES)
    required = context[~context["self_contained_question"]]
    assert set(required["outcome"]) == {"donation_ams", "newsletter_signup"}
    assert required["context_included_in_F"].all()
    assert context[context["self_contained_question"]]["required_participant_context"].eq(
        "none beyond scored item/battery wording and native response scale"
    ).all()


def test_f_missing_demographic_profile_omits_absent_fields_without_placeholders(prompt_artifacts):
    path = prompt_artifacts["out"] / "f_pair_examples" / "external_braman751_scholar_credibility_hyp1_control.txt"
    text = path.read_text(encoding="utf-8")
    profile_section = text.split("STUDY SETTING", 1)[0]

    assert "- Race/ethnicity:" not in profile_section
    assert "- Gender:" in profile_section
    assert missing_placeholder_hits(type("RenderLike", (), {"user_prompt": text})()) == []


def test_legacy_bisbee_common_shared_prompt_path_is_not_active():
    active = G_SYSTEM_PROMPT + "\n" + F_SYSTEM_PROMPT
    assert "bisbee" not in active.lower()
    assert "shared respondent" not in active.lower()


def test_legacy_krsteski_persona_guided_prior_response_method_is_not_active(prompt_artifacts):
    manifest = prompt_artifacts["manifest"]
    assert manifest["g_not_used"] == "Krsteski persona-guided prior-response synthesis"
    assert "previous response" not in G_SYSTEM_PROMPT.lower()


def test_binary_schema_uses_enum():
    item = next(i for i in sc.load_items() if i["target_label"] == "newsletter_signup")
    schema = item_json_schema([item])

    spec = schema["properties"][item["qualtrics_label"]]
    assert spec == {"type": "integer", "enum": [0, 1]}


def test_request_keys_and_seeds_are_deterministic_and_role_specific():
    g_key = request_key_g(donor_key="G1", condition="control", replicate=1)
    f_key_1 = request_key_f(study_id="s", f_profile_id="F1", condition="t", outcome="y", replicate=1)
    f_key_2 = request_key_f(study_id="s", f_profile_id="F1", condition="t", outcome="y", replicate=2)

    assert g_key != f_key_1
    assert seed_from_request_key(f_key_1) == seed_from_request_key(f_key_1)
    assert seed_from_request_key(f_key_1) != seed_from_request_key(f_key_2)
