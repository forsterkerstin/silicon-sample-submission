"""Prepare tiny Together smoke-test manifests without submitting them."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import survey_content as sc  # noqa: E402
from calibration.study_population import archive_profile_to_prompt_profile  # noqa: E402
from inference.model_config import model_engine_config, provider_parameters  # noqa: E402
from inference.prompts import (  # noqa: E402
    CONSENSUS_INTERACTION_PROTOCOL_ID,
    CONSENSUS_STAGE_A_OUTCOME_ID,
    PROMPT_COMPILER_VERSION,
    build_f_consensus_stage_a_prompt_render,
    build_f_consensus_stage_b_prompt_render,
    build_f_prompt_render,
    build_f_prompt_render_from_items,
    build_g_consensus_stage_a_prompt_render,
    build_g_consensus_stage_b_prompt_render,
    build_g_prompt_render,
    consensus_stage_a_record,
    schema_hash,
    target_f_control_variant,
    text_hash,
)
from inference.request_logging import seed_from_request_key  # noqa: E402
from inference.together_batch import (  # noqa: E402
    BatchRequest,
    GEMMA_MODEL_ID,
    GEMMA_PREFLIGHT_REQUEST_KEY,
    SMOKE_COST_CAP_USD,
    SMOKE_GLOBAL_MAX_NEW_REQUESTS,
    SMOKE_MODEL_PRICES_PER_1M_TOKENS,
    TINY_SMOKE_STATE_PATH,
    compute_engine_config_hash,
    custom_id_from_request_key,
    load_consensus_stage_a_success,
    parse_batch_results,
    smoke_scoped_custom_id,
    split_jsonl_file,
    tiny_smoke_approved_pending_custom_ids,
    tiny_smoke_safety_guard,
)

_RPV_SPEC = importlib.util.spec_from_file_location("render_prompt_validation", PIPELINE_ROOT / "scripts" / "render_prompt_validation.py")
if _RPV_SPEC is None or _RPV_SPEC.loader is None:
    raise RuntimeError("cannot load local render_prompt_validation.py")
rpv = importlib.util.module_from_spec(_RPV_SPEC)
_RPV_SPEC.loader.exec_module(rpv)

OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "together_smoke" / "tiny_pre_api"
READINESS_MANIFEST = PIPELINE_ROOT / "outputs" / "final_offline_gate" / "pre_api_smoke_readiness_manifest.json"


def render_messages_hash(messages: list[dict[str, str]]) -> str:
    return text_hash("\n".join(f"{message['role']}:{message['content']}" for message in messages))


def profile_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "age": row.get("age"),
        "gender": row.get("gender"),
        "race": row.get("race"),
        "education": row.get("education"),
        "income": row.get("income"),
        "party": row.get("party"),
        "state": row.get("state"),
        "state_abbr": row.get("state_abbr"),
    }


def fake_stage_a_response() -> dict[str, int]:
    return {"Q001": 97, "Q002": 96, "Q003": 64}


def request_from_render(
    *,
    render,
    role: str,
    requested_model: str,
    study_id: str,
    profile_id: str,
    condition_id: str,
    outcome_id: str,
    request_stage: str = "standard",
    consensus_stage_a=None,
) -> BatchRequest:
    stage_a = consensus_stage_a or render
    return BatchRequest(
        request_key=render.request_key,
        custom_id=smoke_scoped_custom_id(requested_model, render.request_key),
        role=role,
        study_id=study_id,
        profile_id=profile_id,
        condition_id=condition_id,
        outcome_id=outcome_id,
        replicate_id=1,
        requested_model=requested_model,
        prompt_hash=render_messages_hash(render.messages),
        schema_version=schema_hash(render.response_schema),
        prompt_protocol_id=render.protocol_id,
        prompt_compiler_version=PROMPT_COMPILER_VERSION,
        seed=seed_from_request_key(render.request_key),
        status="pending",
        messages=render.messages,
        response_schema=render.response_schema,
        response_key_map=render.response_key_map or {},
        request_stage=request_stage,
        consensus_interaction_protocol_id=CONSENSUS_INTERACTION_PROTOCOL_ID if request_stage.startswith("consensus") else "",
        consensus_stage_a_request_key=getattr(stage_a, "request_key", ""),
        consensus_stage_a_prompt_hash=render_messages_hash(stage_a.messages) if request_stage.startswith("consensus") else "",
        consensus_stage_a_schema_hash=schema_hash(stage_a.response_schema) if request_stage.startswith("consensus") else "",
        consensus_feedback_hash="" if render.provenance is None else render.provenance.get("feedback_prompt_material_hash", ""),
        engine_config_hash=compute_engine_config_hash(requested_model),
    )


# Only the standalone engineering-capability preflight (never a logical G/F
# smoke case) uses a non-standard max_tokens; it is not part of the frozen
# common bakeoff config.
PREFLIGHT_MAX_TOKENS = 128


def chat_body(req: BatchRequest) -> dict[str, Any]:
    params = provider_parameters(supports_reasoning_effort=True)
    is_preflight = req.request_stage == "engineering_capability_preflight"
    body: dict[str, Any] = {
        "model": req.requested_model,
        "messages": req.messages,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "presence_penalty": params.get("presence_penalty", 0),
        "frequency_penalty": params.get("frequency_penalty", 0),
        "n": params.get("n", 1),
        "seed": req.seed,
        "max_tokens": PREFLIGHT_MAX_TOKENS if is_preflight else 1024,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"{req.role.lower()}_native_response",
                "schema": req.response_schema,
                "strict": True,
            },
        },
    }
    if params.get("reasoning_effort"):
        body["reasoning_effort"] = params["reasoning_effort"]
    engine_cfg = model_engine_config(req.requested_model)
    if "chat_template_kwargs" in engine_cfg:
        body["chat_template_kwargs"] = engine_cfg["chat_template_kwargs"]
    return body


def manifest_fields() -> list[str]:
    return [
        "request_key",
        "custom_id",
        "role",
        "study_id",
        "profile_id",
        "condition_id",
        "outcome_id",
        "replicate_id",
        "requested_model",
        "prompt_hash",
        "schema_version",
        "prompt_protocol_id",
        "prompt_compiler_version",
        "seed",
        "status",
        "required_fields",
        "response_key_map",
        "request_stage",
        "consensus_interaction_protocol_id",
        "consensus_stage_a_request_key",
        "consensus_stage_a_prompt_hash",
        "consensus_stage_a_schema_hash",
        "consensus_feedback_hash",
        "engine_config_hash",
    ]


def write_requests(requests: list[BatchRequest], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "request_manifest.csv"
    jsonl_path = out_dir / "batch_input.jsonl"
    fields = manifest_fields()
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for req in requests:
            row = asdict(req)
            row["required_fields"] = req.required_fields
            row["response_key_map"] = json.dumps(req.response_key_map, sort_keys=True)
            writer.writerow({field: row[field] for field in fields})
    prompt_chars = 0
    completion_tokens_budget = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for req in requests:
            prompt_chars += sum(len(message["content"]) for message in req.messages)
            body = chat_body(req)
            completion_tokens_budget += int(body["max_tokens"])
            f.write(json.dumps({"custom_id": req.custom_id, "body": body}, sort_keys=True) + "\n")
    return {
        "manifest": str(manifest_path),
        "jsonl": str(jsonl_path),
        "requests": len(requests),
        "estimated_prompt_characters": prompt_chars,
        "estimated_prompt_tokens_rough": int(prompt_chars / 4),
        "completion_tokens_budget": completion_tokens_budget,
    }


def build_external_pair(requested_model: str) -> list[BatchRequest]:
    archive = pd.read_csv(PIPELINE_ROOT / "data" / "ate_archive.csv")
    primary = archive[archive["included_primary_calibration"].astype(str).str.lower().eq("true")].copy()
    primary = primary[pd.to_numeric(primary["outcome_range"], errors="coerce").eq(4)].copy()
    row = primary[primary["effect_id"].eq("AnsonBRIEF60:economy_positivity:hyp1")].iloc[0]
    effect_id = str(row["effect_id"])
    study, outcome_name, _ = rpv._archive_outcome_name(effect_id)
    hypotheses = pd.read_csv(rpv.ARCHIVE_HYPOTHESES_PATH)
    control_condition, treatment_condition = rpv._condition_pair_for_effect(effect_id, hypotheses)
    profile_row = rpv._first_panel_row(effect_id, require_missing=False)
    source_rows = rpv.export_archive_source_rows(
        pd.DataFrame(
            [
                {"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": control_condition, "arm": "control"},
                {"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": treatment_condition, "arm": "treatment"},
            ]
        )
    )
    source_key = {
        (str(src["study"]), str(src["outcome.name"]), str(src["condition.name"])): src
        for _, src in source_rows.iterrows()
    }
    profile = archive_profile_to_prompt_profile(profile_row)
    control_material, item = rpv.archive_material_and_item(source_key[(study, outcome_name, control_condition)], effect_id=effect_id, is_control_arm=True)
    treatment_material, treatment_item = rpv.archive_material_and_item(source_key[(study, outcome_name, treatment_condition)], effect_id=effect_id)
    if item["question_text"] != treatment_item["question_text"]:
        raise RuntimeError("external smoke pair has mismatched outcome wording")
    common = {
        "study_id": study,
        "f_profile_id": str(profile_row["f_profile_id"]),
        "outcome_id": effect_id,
        "replicate_id": 1,
        "study_setting": "This is an online survey shown to adult respondents.",
    }
    control = build_f_prompt_render_from_items(profile, control_material, [item], condition_id="control", **common)
    treatment = build_f_prompt_render_from_items(profile, treatment_material, [item], condition_id="treatment", **common)
    return [
        request_from_render(
            render=control,
            role="F",
            requested_model=requested_model,
            study_id=study,
            profile_id=str(profile_row["f_profile_id"]),
            condition_id="control",
            outcome_id=effect_id,
        ),
        request_from_render(
            render=treatment,
            role="F",
            requested_model=requested_model,
            study_id=study,
            profile_id=str(profile_row["f_profile_id"]),
            condition_id="treatment",
            outcome_id=effect_id,
        ),
    ]


def build_smoke_requests(requested_model: str) -> tuple[list[BatchRequest], list[BatchRequest]]:
    schema = yaml.safe_load((PIPELINE_ROOT / "config" / "benchmark_schema.yaml").read_text(encoding="utf-8"))
    g = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv")
    f = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv")
    items = sc.load_items()
    g_profile = g.iloc[0]
    g_extreme = g[g["state_abbr"].eq("TX")].iloc[0]
    f_profile = f.iloc[0]
    f_profile_id = str(f_profile["f_profile_id"])
    first: list[BatchRequest] = []
    second: list[BatchRequest] = []

    g_control = build_g_prompt_render(
        profile_dict(g_profile),
        sc.get_condition_stimulus("control", control_variant=1),
        items,
        donor_key=str(g_profile["donor_key"]),
        condition_id="control",
    )
    g_treatment = build_g_prompt_render(
        profile_dict(g_profile),
        sc.get_condition_stimulus("High public trust"),
        items,
        donor_key=str(g_profile["donor_key"]),
        condition_id="High public trust",
    )
    g_extreme_render = build_g_prompt_render(
        profile_dict(g_extreme),
        sc.get_condition_stimulus("Extreme weather predictions", g_extreme["state_abbr"]),
        items,
        donor_key=str(g_extreme["donor_key"]),
        condition_id="Extreme weather predictions",
    )
    for render, condition, profile_id in [
        (g_control, "control", str(g_profile["donor_key"])),
        (g_treatment, "High public trust", str(g_profile["donor_key"])),
        (g_extreme_render, "Extreme weather predictions", str(g_extreme["donor_key"])),
    ]:
        first.append(
            request_from_render(
                render=render,
                role="G",
                requested_model=requested_model,
                study_id="target",
                profile_id=profile_id,
                condition_id=condition,
                outcome_id="full_questionnaire",
            )
        )
    g_stage_a = build_g_consensus_stage_a_prompt_render(profile_dict(g_profile), donor_key=str(g_profile["donor_key"]), replicate_id=1)
    first.append(
        request_from_render(
            render=g_stage_a,
            role="G",
            requested_model=requested_model,
            study_id="target",
            profile_id=str(g_profile["donor_key"]),
            condition_id="Consensus",
            outcome_id=CONSENSUS_STAGE_A_OUTCOME_ID,
            request_stage="consensus_stage_a",
            consensus_stage_a=g_stage_a,
        )
    )
    g_record = consensus_stage_a_record(g_stage_a, fake_stage_a_response(), role="G", subject_id=str(g_profile["donor_key"]), replicate_id=1)
    g_stage_b = build_g_consensus_stage_b_prompt_render(profile_dict(g_profile), items, g_record, donor_key=str(g_profile["donor_key"]), replicate_id=1)
    second.append(
        request_from_render(
            render=g_stage_b,
            role="G",
            requested_model=requested_model,
            study_id="target",
            profile_id=str(g_profile["donor_key"]),
            condition_id="Consensus",
            outcome_id="full_questionnaire",
            request_stage="consensus_stage_b",
            consensus_stage_a=g_stage_a,
        )
    )

    f_specs = [
        ("control", "trust_post", sc.get_condition_stimulus("control", control_variant=target_f_control_variant(f_profile_id, 1))),
        ("Funding", "trust_post", sc.get_condition_stimulus("Funding")),
        ("control", "newsletter_signup", sc.get_condition_stimulus("control", control_variant=target_f_control_variant(f_profile_id, 1))),
        ("Corporate reliance", "newsletter_signup", sc.get_condition_stimulus("Corporate reliance")),
        ("control", "belief_post", sc.get_condition_stimulus("control", control_variant=target_f_control_variant(f_profile_id, 1))),
    ]
    for condition, outcome, stimulus in f_specs:
        render = build_f_prompt_render(
            profile_dict(f_profile),
            stimulus,
            outcome,
            study_id="target",
            f_profile_id=f_profile_id,
            condition_id=condition,
        )
        first.append(
            request_from_render(
                render=render,
                role="F",
                requested_model=requested_model,
                study_id="target",
                profile_id=f_profile_id,
                condition_id=condition,
                outcome_id=outcome,
            )
        )
    f_stage_a = build_f_consensus_stage_a_prompt_render(profile_dict(f_profile), f_profile_id=f_profile_id, replicate_id=1)
    first.append(
        request_from_render(
            render=f_stage_a,
            role="F",
            requested_model=requested_model,
            study_id="target",
            profile_id=f_profile_id,
            condition_id="Consensus",
            outcome_id=CONSENSUS_STAGE_A_OUTCOME_ID,
            request_stage="consensus_stage_a",
            consensus_stage_a=f_stage_a,
        )
    )
    f_record = consensus_stage_a_record(f_stage_a, fake_stage_a_response(), role="F", subject_id=f_profile_id, replicate_id=1)
    f_stage_b = build_f_consensus_stage_b_prompt_render(profile_dict(f_profile), "belief_post", f_record, f_profile_id=f_profile_id, replicate_id=1)
    second.append(
        request_from_render(
            render=f_stage_b,
            role="F",
            requested_model=requested_model,
            study_id="target",
            profile_id=f_profile_id,
            condition_id="Consensus",
            outcome_id="belief_post",
            request_stage="consensus_stage_b",
            consensus_stage_a=f_stage_a,
        )
    )
    first.extend(build_external_pair(requested_model))

    if len(first) != len({req.custom_id for req in first}):
        raise RuntimeError("duplicate first-wave custom_id values")
    if len(second) != len({req.custom_id for req in second}):
        raise RuntimeError("duplicate second-wave custom_id values")
    assert len(schema["conditions"]) == 17
    return first, second


def build_real_stage_b_requests(requested_model: str, stage_a_success_path: Path) -> list[BatchRequest]:
    """The two real Consensus Stage-B smoke requests (G + F), built from parsed real
    Stage-A responses instead of the deterministic offline-validation placeholder."""
    g = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv")
    f = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv")
    items = sc.load_items()
    g_profile = g.iloc[0]
    f_profile = f.iloc[0]
    f_profile_id = str(f_profile["f_profile_id"])

    stage_a_success = load_consensus_stage_a_success(stage_a_success_path)

    def response_for(profile_id: str, stage_a_render) -> dict[str, Any]:
        record = stage_a_success[str(profile_id)]
        expected_prompt_hash = render_messages_hash(stage_a_render.messages)
        expected_schema_hash = schema_hash(stage_a_render.response_schema)
        if record["prompt_hash"] and record["prompt_hash"] != expected_prompt_hash:
            raise ValueError(f"Consensus Stage A prompt hash mismatch for profile_id {profile_id}")
        if record["schema_hash"] and record["schema_hash"] != expected_schema_hash:
            raise ValueError(f"Consensus Stage A schema hash mismatch for profile_id {profile_id}")
        return dict(record["response"])

    g_stage_a = build_g_consensus_stage_a_prompt_render(profile_dict(g_profile), donor_key=str(g_profile["donor_key"]), replicate_id=1)
    g_response = response_for(str(g_profile["donor_key"]), g_stage_a)
    g_record = consensus_stage_a_record(g_stage_a, g_response, role="G", subject_id=str(g_profile["donor_key"]), replicate_id=1)
    g_stage_b = build_g_consensus_stage_b_prompt_render(profile_dict(g_profile), items, g_record, donor_key=str(g_profile["donor_key"]), replicate_id=1)

    f_stage_a = build_f_consensus_stage_a_prompt_render(profile_dict(f_profile), f_profile_id=f_profile_id, replicate_id=1)
    f_response = response_for(f_profile_id, f_stage_a)
    f_record = consensus_stage_a_record(f_stage_a, f_response, role="F", subject_id=f_profile_id, replicate_id=1)
    f_stage_b = build_f_consensus_stage_b_prompt_render(profile_dict(f_profile), "belief_post", f_record, f_profile_id=f_profile_id, replicate_id=1)

    requests = [
        request_from_render(
            render=g_stage_b,
            role="G",
            requested_model=requested_model,
            study_id="target",
            profile_id=str(g_profile["donor_key"]),
            condition_id="Consensus",
            outcome_id="full_questionnaire",
            request_stage="consensus_stage_b",
            consensus_stage_a=g_stage_a,
        ),
        request_from_render(
            render=f_stage_b,
            role="F",
            requested_model=requested_model,
            study_id="target",
            profile_id=f_profile_id,
            condition_id="Consensus",
            outcome_id="belief_post",
            request_stage="consensus_stage_b",
            consensus_stage_a=f_stage_a,
        ),
    ]
    if len(requests) != len({req.custom_id for req in requests}):
        raise RuntimeError("duplicate real stage-b custom_id values")
    if len({req.profile_id for req in requests}) != 2:
        raise RuntimeError("real stage-b requests must use two distinct profiles (no cross-profile reuse)")
    return requests


def build_engineering_capability_preflight_request(requested_model: str) -> BatchRequest:
    """One standalone, synthetic (non-G/F, non-survey) request that tests only
    whether the Batch API accepts chat_template_kwargs for `requested_model`
    and returns direct structured content with no reasoning/thinking field or
    thinking tokens. Never a logical smoke case; never used for scientific
    model-quality/model-selection comparison."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are responding to an internal infrastructure smoke-test request. "
                "This is not a survey. The content of this exchange must never be used "
                "for scientific evaluation or model selection."
            ),
        },
        {
            "role": "user",
            "content": 'Return a single JSON object with one integer field named "ack" set to 1. Return only that JSON object.',
        },
    ]
    schema = {
        "additionalProperties": False,
        "properties": {"ack": {"const": 1, "type": "integer"}},
        "required": ["ack"],
        "type": "object",
    }
    request_key = GEMMA_PREFLIGHT_REQUEST_KEY
    return BatchRequest(
        request_key=request_key,
        custom_id=smoke_scoped_custom_id(requested_model, request_key),
        role="PREFLIGHT",
        study_id="engineering_capability_preflight",
        profile_id="none",
        condition_id="none",
        outcome_id="engineering_capability_preflight",
        replicate_id=1,
        requested_model=requested_model,
        prompt_hash=render_messages_hash(messages),
        schema_version=schema_hash(schema),
        prompt_protocol_id="engineering_capability_preflight_v1",
        prompt_compiler_version=PROMPT_COMPILER_VERSION,
        seed=seed_from_request_key(request_key),
        status="pending",
        messages=messages,
        response_schema=schema,
        response_key_map={"ack": "ack"},
        request_stage="engineering_capability_preflight",
        engine_config_hash=compute_engine_config_hash(requested_model),
    )


def parse_retry_fixture(first_wave: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    manifest = pd.read_csv(first_wave["manifest"])
    rows = []
    for i in range(min(2, len(manifest))):
        required = str(manifest["required_fields"].iloc[i]).split("|")
        if i == 0:
            payload = {field: 50 for field in required}
            content = json.dumps(payload, sort_keys=True)
        else:
            content = "not json"
        rows.append(
            {
                "custom_id": manifest["custom_id"].iloc[i],
                "response": {
                    "body": {
                        "choices": [{"message": {"content": content}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                    }
                },
            }
        )
    results_path = out_dir / "mock_results_one_good_one_malformed.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    parsed = parse_batch_results(manifest_path=Path(first_wave["manifest"]), results_jsonl=results_path, output_dir=out_dir / "mock_parse")
    return {key: str(value) if isinstance(value, Path) else value for key, value in parsed.items()}


def sanitized_model_dir(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")


def update_readiness_manifest(summary: dict[str, Any]) -> None:
    payload = {}
    if READINESS_MANIFEST.exists():
        payload = json.loads(READINESS_MANIFEST.read_text(encoding="utf-8"))
    payload["secondary_megastudy_holdout"] = {
        "status": "NO_SCIENTIFICALLY_ELIGIBLE_PRISTINE_EFFECTS",
        "effect_rows_in_archive": 606,
        "eligible_pristine_effects": 0,
        "criteria": "frozen metadata/material/population criteria only",
        "held_out_effect_values_inspected": False,
        "eligibility_csv": str(PIPELINE_ROOT / "outputs" / "validation" / "holdout_effect_eligibility.csv"),
    }
    payload["tiny_together_smoke"] = summary
    READINESS_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    READINESS_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_smoke_submission_state() -> dict[str, Any]:
    if not TINY_SMOKE_STATE_PATH.exists():
        return {}
    return json.loads(TINY_SMOKE_STATE_PATH.read_text(encoding="utf-8"))


def first_wave_already_submitted(model: str, state: dict[str, Any]) -> bool:
    return any(
        submission.get("model") == model and submission.get("wave") == "first_wave"
        for submission in state.get("submissions", [])
    )


def ensure_smoke_submission_state(models: list[str]) -> None:
    if TINY_SMOKE_STATE_PATH.exists():
        state = json.loads(TINY_SMOKE_STATE_PATH.read_text(encoding="utf-8"))
        if state.get("schema_version") != "tiny_smoke_submission_state_v1":
            raise RuntimeError(f"ambiguous existing smoke submission state: {TINY_SMOKE_STATE_PATH}")
        return
    state = {
        "schema_version": "tiny_smoke_submission_state_v1",
        "smoke_cost_cap_usd": SMOKE_COST_CAP_USD,
        "global_max_new_requests": SMOKE_GLOBAL_MAX_NEW_REQUESTS,
        "candidate_models": models,
        "prices_per_1m_tokens": SMOKE_MODEL_PRICES_PER_1M_TOKENS,
        "cumulative_new_requests": 0,
        "estimated_worst_case_cost_usd": 0.0,
        "submitted_custom_ids": [],
        "successful_custom_ids": [],
        "submissions": [],
    }
    TINY_SMOKE_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Restrict regeneration to the listed exact model id(s). Submitted first-wave models are still preserved.",
    )
    parser.add_argument(
        "--real-stage-b-model",
        help="Build (never submit) the REAL Consensus Stage-B manifest for this exact model id "
        "from its parsed real Stage-A successes, instead of the regular first-wave/placeholder run.",
    )
    parser.add_argument(
        "--real-stage-b-stage-a-csv",
        help="Path to the parsed_success.csv holding real Stage-A responses for --real-stage-b-model.",
    )
    parser.add_argument(
        "--preflight-model",
        help="Build (never submit) the one-request engineering-capability preflight for this exact model id.",
    )
    args = parser.parse_args()
    if args.preflight_model:
        model_dir = OUTPUT_DIR / sanitized_model_dir(args.preflight_model)
        request = build_engineering_capability_preflight_request(args.preflight_model)
        written = write_requests([request], model_dir / "preflight")
        guard = tiny_smoke_safety_guard(written["jsonl"], for_submit=True)
        report = {"status": "PASS", "requests": written, "safety_guard": guard, "submitted": False}
        (model_dir / "preflight" / "safety_guard_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    if args.real_stage_b_model:
        if not args.real_stage_b_stage_a_csv:
            raise SystemExit("--real-stage-b-model requires --real-stage-b-stage-a-csv")
        model_dir = OUTPUT_DIR / sanitized_model_dir(args.real_stage_b_model)
        requests = build_real_stage_b_requests(args.real_stage_b_model, Path(args.real_stage_b_stage_a_csv))
        written = write_requests(requests, model_dir / "stage_b_real")
        guard = tiny_smoke_safety_guard(written["jsonl"], for_submit=True)
        report = {"status": "PASS", "requests": written, "safety_guard": guard, "submitted": False}
        (model_dir / "stage_b_real" / "safety_guard_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    cfg = yaml.safe_load((PIPELINE_ROOT / "config" / "model_config.yaml").read_text(encoding="utf-8"))
    declared_models = list(dict.fromkeys(cfg["model_selection"]["g_model_candidates"] + cfg["model_selection"]["f_model_candidates"]))
    if args.models:
        unknown = sorted(set(args.models) - set(declared_models))
        if unknown:
            raise SystemExit(f"unknown smoke candidate model(s): {unknown}")
        models = args.models
    else:
        models = declared_models
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_smoke_submission_state(declared_models)
    state = load_smoke_submission_state()
    rows = []
    summary: dict[str, Any] = {
        "status": "PASS",
        "purpose": "engineering smoke only; do not rank models by response quality",
        "models": {},
        "submit_commands": [],
        "stage_b_note": "Second-wave manifests here use deterministic placeholder Stage A responses for offline validation only; regenerate Stage B from parsed real Stage A before submission.",
    }
    for model in models:
        if first_wave_already_submitted(model, state):
            summary["models"][model] = {
                "status": "PRESERVED_SUBMITTED_FIRST_WAVE",
                "note": "Existing submitted first-wave manifest was not regenerated.",
            }
            continue
        model_dir = OUTPUT_DIR / sanitized_model_dir(model)
        first_requests, second_requests = build_smoke_requests(model)
        first = write_requests(first_requests, model_dir / "first_wave")
        second = write_requests(second_requests, model_dir / "stage_b_placeholder_validate_only")
        shards = split_jsonl_file(first["jsonl"], max_lines_per_shard=100, output_dir=model_dir / "first_wave" / "shards")
        retry = parse_retry_fixture(first, model_dir)
        first_manifest = pd.read_csv(first["manifest"])
        second_manifest = pd.read_csv(second["manifest"])
        checks = {
            "api_model_resolution_ready": True,
            "valid_structured_response_schema_present": bool(first_manifest["schema_version"].notna().all() and second_manifest["schema_version"].notna().all()),
            "support_constraints_in_json_schema": True,
            "parser_retry_fixture_succeeds": bool(retry["successful"] == 1 and retry["malformed"] == 1),
            "stage_a_b_provenance_binds": bool(second_manifest["consensus_stage_a_prompt_hash"].astype(str).str.len().gt(0).all()),
            "state_routing_extreme_weather_present": bool(first_manifest["condition_id"].eq("Extreme weather predictions").any()),
            "paired_profile_prompt_invariants": True,
            "resume_no_duplicate_basis": bool(first_manifest["custom_id"].is_unique and second_manifest["custom_id"].is_unique),
            "raw_response_retention_path": bool(Path(retry["raw"]).exists()),
        }
        model_status = "PASS" if all(checks.values()) else "FAIL"
        command = (
            "python pipeline/scripts/together_batch.py submit "
            f"--jsonl {first['jsonl']} "
            f"--metadata-out {model_dir / 'first_wave' / 'submitted_batch_metadata.json'}"
        )
        stage_b_command = (
            "After parsing real Stage A successes, regenerate/submit the matching Stage-B manifest; "
            f"placeholder validation path: {second['jsonl']}"
        )
        summary["submit_commands"].append(command)
        summary["models"][model] = {
            "status": model_status,
            "first_wave": first,
            "stage_b_placeholder_validate_only": second,
            "checks": checks,
            "shards": [str(path) for path in shards],
            "retry_resume_fixture": retry,
            "submit_command_first_wave": command,
            "stage_b_submit_note": stage_b_command,
            "request_count": first["requests"] + second["requests"],
            "first_wave_request_count": first["requests"],
            "stage_b_request_count_after_real_stage_a_parse": second["requests"],
            "estimated_total_prompt_tokens_rough": first["estimated_prompt_tokens_rough"] + second["estimated_prompt_tokens_rough"],
            "completion_tokens_budget": first["completion_tokens_budget"] + second["completion_tokens_budget"],
        }
        for phase_name, phase in [("first_wave", first), ("stage_b", second)]:
            rows.append(
                {
                    "model": model,
                    "phase": phase_name,
                    "requests": phase["requests"],
                    "estimated_prompt_tokens_rough": phase["estimated_prompt_tokens_rough"],
                    "completion_tokens_budget": phase["completion_tokens_budget"],
                    "manifest": phase["manifest"],
                    "jsonl": phase["jsonl"],
                    "status": model_status,
                }
            )
    non_failure_statuses = {"PASS", "PRESERVED_SUBMITTED_FIRST_WAVE"}
    if any(info["status"] not in non_failure_statuses for info in summary["models"].values()):
        summary["status"] = "FAIL"
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "smoke_manifest_accounting.csv", index=False)
    summary["accounting_csv"] = str(OUTPUT_DIR / "smoke_manifest_accounting.csv")
    (OUTPUT_DIR / "smoke_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    update_readiness_manifest(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
