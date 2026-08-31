"""TogetherAI batch preparation, dry-run, resume, and result parsing.

This module is intentionally offline-first. Preparing a manifest/JSONL never
contacts Together; paid actions live behind explicit submit/status/retrieve
functions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import yaml

import survey_content as sc
from inference.model_config import inference_parameters, model_engine_config, provider_parameters
from inference.prompts import (
    CONSENSUS_INTERACTION_PROTOCOL_ID,
    CONSENSUS_STAGE_A_OUTCOME_ID,
    G_SYSTEM_PROMPT_BY_VARIANT,
    PROMPT_COMPILER_VERSION,
    build_f_consensus_stage_a_prompt_render,
    build_f_consensus_stage_b_prompt_render,
    build_f_prompt_render,
    build_g_consensus_stage_a_prompt_render,
    build_g_consensus_stage_b_prompt_render,
    build_g_prompt_render,
    consensus_stage_a_record,
    schema_hash,
    target_f_control_variant,
    text_hash,
)
from inference.request_logging import request_key_f, request_key_g, seed_from_request_key

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
SCHEMA_PATH = PIPELINE_ROOT / "config" / "benchmark_schema.yaml"
G_MASTER_PATH = PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv"
F_PANEL_PATH = PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv"
DEFAULT_OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "together_batch"
# Together Batch API documented limits (docs.together.ai/docs/inference/batch/overview,
# recorded 2026-08-25): up to 50,000 requests per batch, up to 100 MB input file.
# Together recommends ~1,000-10,000 requests/batch as a practical best-practice
# range (not a hard limit) for latency/manageability, distinct from the hard cap.
TOGETHER_BATCH_MAX_REQUESTS = 50_000
TOGETHER_BATCH_MAX_INPUT_FILE_BYTES = 100 * 1024 * 1024
TOGETHER_BATCH_RECOMMENDED_REQUESTS_RANGE = (1_000, 10_000)
TOGETHER_BATCH_LIMIT_SOURCE = "https://docs.together.ai/docs/inference/batch/overview"
TINY_SMOKE_ROOT = PIPELINE_ROOT / "outputs" / "together_smoke" / "tiny_pre_api"
TINY_SMOKE_STATE_PATH = TINY_SMOKE_ROOT / "smoke_submission_state.json"
SMOKE_COST_CAP_USD = 0.30
# 24 already-submitted historical requests (DeepSeek + Qwen first-wave, both
# retained -- Qwen is engineering-screened-out, not deleted) + 12 Gemma
# first-wave + 2 DeepSeek real Stage-B + 2 Gemma real Stage-B + 1 Gemma
# engineering-capability preflight = 41. Raising this cap does NOT by itself
# admit arbitrary new requests -- see tiny_smoke_approved_pending_custom_ids().
SMOKE_GLOBAL_MAX_NEW_REQUESTS = 41
SMOKE_MODEL_PRICES_PER_1M_TOKENS = {
    # Together Batch API prices checked 2026-08-25 from together.ai/pricing.
    "deepseek-ai/DeepSeek-V4-Pro-0813": {"input": 1.32, "output": 3.96},
    "Qwen/Qwen3.8-2.4T-A95B": {"input": 2.50, "output": 6.25},
    "google/gemma-4-31B-it": {"input": 0.20, "output": 0.50},
}


@dataclass(frozen=True)
class BatchRequest:
    request_key: str
    custom_id: str
    role: str
    study_id: str
    profile_id: str
    condition_id: str
    outcome_id: str
    replicate_id: int
    requested_model: str
    prompt_hash: str
    schema_version: str
    prompt_protocol_id: str
    prompt_compiler_version: str
    seed: int
    status: str
    messages: list[dict[str, str]]
    response_schema: dict[str, Any]
    response_key_map: dict[str, str]
    request_stage: str = "standard"
    consensus_interaction_protocol_id: str = ""
    consensus_stage_a_request_key: str = ""
    consensus_stage_a_prompt_hash: str = ""
    consensus_stage_a_schema_hash: str = ""
    consensus_feedback_hash: str = ""
    engine_config_hash: str = ""

    @property
    def required_fields(self) -> str:
        return "|".join(self.response_schema.get("required", []))


def _profile_dict(row: pd.Series) -> dict[str, Any]:
    out = {
        "age": row.get("age"),
        "gender": row.get("gender"),
        "race": row.get("race"),
        "education": row.get("education"),
        "income": row.get("income"),
        "party": row.get("party"),
        "state": row.get("state"),
        "state_abbr": row.get("state_abbr"),
    }
    for optional in ("political_ideology", "religion"):
        if optional in row and pd.notna(row[optional]) and str(row[optional]).strip():
            out[optional] = row[optional]
    return out


def custom_id_from_request_key(request_key: str) -> str:
    digest = hashlib.sha256(request_key.encode("utf-8")).hexdigest()[:40]
    prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", request_key.split("|", 1)[0]).strip("-") or "req"
    return f"{prefix}-{digest}"[:64]


def smoke_scoped_custom_id(model: str, request_key: str) -> str:
    """Model-scoped smoke custom_id: hashes `model|request_key`, so the same
    logical request_key never collides across candidate models."""
    return custom_id_from_request_key(f"{model}|{request_key}")


DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-V4-Pro-0813"
QWEN_MODEL_ID = "Qwen/Qwen3.8-2.4T-A95B"
GEMMA_MODEL_ID = "google/gemma-4-31B-it"

# The 12 logical first-wave smoke request_keys, identical across every
# candidate model (only the model-scoped custom_id and requested_model differ).
TINY_SMOKE_FIRST_WAVE_REQUEST_KEYS = (
    "G|LP0001|control|replicate_1",
    "G|LP0001|High public trust|replicate_1",
    "G|LP0009|Extreme weather predictions|replicate_1",
    "G|LP0001|Consensus|stage_a|replicate_1",
    "F|target|F0001|control|trust_post|replicate_1",
    "F|target|F0001|Funding|trust_post|replicate_1",
    "F|target|F0001|control|newsletter_signup|replicate_1",
    "F|target|F0001|Corporate reliance|newsletter_signup|replicate_1",
    "F|target|F0001|control|belief_post|replicate_1",
    "F|target|F0001|Consensus|stage_a|replicate_1",
    "F|AnsonBRIEF60|AnsonBRIEF60__9db762e4cca15f5f__F004|control|AnsonBRIEF60:economy_positivity:hyp1|replicate_1",
    "F|AnsonBRIEF60|AnsonBRIEF60__9db762e4cca15f5f__F004|treatment|AnsonBRIEF60:economy_positivity:hyp1|replicate_1",
)
# The 2 logical Consensus Stage-B request_keys (G donor + F profile).
TINY_SMOKE_STAGE_B_REQUEST_KEYS = (
    "G|LP0001|Consensus|stage_b|replicate_1",
    "F|target|F0001|Consensus|belief_post|replicate_1",
)
GEMMA_PREFLIGHT_REQUEST_KEY = "G|PREFLIGHT|engineering_capability_check|replicate_1"


def tiny_smoke_approved_pending_custom_ids() -> frozenset[str]:
    """The closed, exact set of custom_ids allowed to be submitted next -- on
    top of whatever tiny_smoke_state.json already records as
    submitted/successful. This is an identity allowlist, not a request-count
    ceiling: raising SMOKE_GLOBAL_MAX_NEW_REQUESTS does not by itself admit
    any request not named here.

    Qwen is deliberately absent from every entry: it is
    engineering-screened-out, and its Consensus Stage-B must remain forbidden
    even though DeepSeek's Stage-B, Gemma's first wave, and Gemma's Stage-B
    are approved."""
    approved = {smoke_scoped_custom_id(GEMMA_MODEL_ID, key) for key in TINY_SMOKE_FIRST_WAVE_REQUEST_KEYS}
    for key in TINY_SMOKE_STAGE_B_REQUEST_KEYS:
        approved.add(smoke_scoped_custom_id(DEEPSEEK_MODEL_ID, key))
        approved.add(smoke_scoped_custom_id(GEMMA_MODEL_ID, key))
    approved.add(smoke_scoped_custom_id(GEMMA_MODEL_ID, GEMMA_PREFLIGHT_REQUEST_KEY))
    return frozenset(approved)


def _render_prompt_hash(messages: list[dict[str, str]]) -> str:
    return text_hash("\n".join(f"{message['role']}:{message['content']}" for message in messages))


def _conditions() -> list[str]:
    return list(yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))["conditions"])


def load_consensus_stage_a_success(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load parsed Stage A successes for second-wave Consensus Stage B batches."""
    df = pd.read_csv(path)
    required = {"profile_id", "outcome_id", "parsed_output", "response_key_map"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Consensus Stage A success file is missing columns: {sorted(missing)}")
    df = df[df["outcome_id"].eq(CONSENSUS_STAGE_A_OUTCOME_ID)].copy()
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        profile_id = str(row["profile_id"])
        if profile_id in out:
            raise ValueError(f"duplicate Consensus Stage A success for profile_id {profile_id}")
        parsed = json.loads(row["parsed_output"])
        key_map = json.loads(row["response_key_map"] or "{}")
        inverse_map = {target: key for key, target in key_map.items()}
        response = {inverse_map.get(key, key): value for key, value in parsed.items()}
        out[profile_id] = {
            "response": response,
            "prompt_hash": str(row.get("consensus_stage_a_prompt_hash") or row.get("prompt_hash") or ""),
            "schema_hash": str(row.get("consensus_stage_a_schema_hash") or row.get("schema_version") or ""),
        }
    return out


def _chat_body(req: BatchRequest) -> dict[str, Any]:
    params = provider_parameters(supports_reasoning_effort=True)
    body: dict[str, Any] = {
        "model": req.requested_model,
        "messages": req.messages,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "presence_penalty": params.get("presence_penalty", 0),
        "frequency_penalty": params.get("frequency_penalty", 0),
        "n": params.get("n", 1),
        "seed": req.seed,
        "max_tokens": 1024,
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


def compute_engine_config_hash(model: str) -> str:
    """Hash of the declared model-engine config (e.g. chat_template_kwargs) for
    `model`, empty string if the model has no declared engine config. Distinct
    from prompt_hash/schema_version -- this is serving-config provenance, not
    prompt/schema provenance."""
    engine_cfg = model_engine_config(model)
    if not engine_cfg:
        return ""
    return schema_hash({"model": model, "chat_template_kwargs": engine_cfg.get("chat_template_kwargs", {})})


def _stage_a_response_for(consensus_stage_a_success: dict[str, dict[str, Any]], profile_id: str, render) -> dict[str, Any]:
    try:
        record = consensus_stage_a_success[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"missing Consensus Stage A success for profile_id {profile_id}") from exc
    expected_prompt_hash = _render_prompt_hash(render.messages)
    expected_schema_hash = schema_hash(render.response_schema)
    if record.get("prompt_hash") and record["prompt_hash"] != expected_prompt_hash:
        raise ValueError(f"Consensus Stage A prompt hash mismatch for profile_id {profile_id}")
    if record.get("schema_hash") and record["schema_hash"] != expected_schema_hash:
        raise ValueError(f"Consensus Stage A schema hash mismatch for profile_id {profile_id}")
    return dict(record["response"])


def iter_g_requests(
    *,
    requested_model: str,
    max_requests: int | None = None,
    successful_custom_ids: set[str] | None = None,
    consensus_stage_a_success: dict[str, dict[str, Any]] | None = None,
    prompt_variant: str = "P1",
    response_format_instruction_version: str = "v1",
) -> Iterable[BatchRequest]:
    """prompt_variant/response_format_instruction_version default to "P1"/"v1"
    -- every existing call site is byte-identical to before these parameters
    were added (see inference.prompts.G_SYSTEM_PROMPT_BY_VARIANT /
    build_g_prompt_render's docstring for the same backward-compatibility
    guarantee). A non-default prompt_variant is used only by the Approach-3
    prompt-ensemble ATE-shape check; response_format_instruction_version="v2"
    is used only by the G-v2 PROVIDER_SERVING_FORMAT_FAILURE replacement."""
    successful_custom_ids = successful_custom_ids or set()
    system_prompt = None if prompt_variant == "P1" else G_SYSTEM_PROMPT_BY_VARIANT[prompt_variant]
    profiles = pd.read_csv(G_MASTER_PATH)
    items = sc.load_items()
    emitted = 0
    for _, row in profiles.iterrows():
        donor_key = str(row["donor_key"])
        for condition in _conditions():
            if condition == "Consensus":
                stage_a = build_g_consensus_stage_a_prompt_render(
                    _profile_dict(row),
                    donor_key=donor_key,
                    replicate_id=1,
                    system_prompt=system_prompt,
                    prompt_variant=prompt_variant,
                    response_format_instruction_version=response_format_instruction_version,
                )
                if consensus_stage_a_success is None:
                    render = stage_a
                    request_stage = "consensus_stage_a"
                    outcome_id = CONSENSUS_STAGE_A_OUTCOME_ID
                else:
                    response = _stage_a_response_for(consensus_stage_a_success, donor_key, stage_a)
                    record = consensus_stage_a_record(stage_a, response, role="G", subject_id=donor_key, replicate_id=1)
                    render = build_g_consensus_stage_b_prompt_render(
                        _profile_dict(row),
                        items,
                        record,
                        donor_key=donor_key,
                        replicate_id=1,
                        system_prompt=system_prompt,
                        prompt_variant=prompt_variant,
                        response_format_instruction_version=response_format_instruction_version,
                    )
                    request_stage = "consensus_stage_b"
                    outcome_id = "full_questionnaire"
                key = render.request_key
                custom_id = custom_id_from_request_key(key)
                status = "already_successful" if custom_id in successful_custom_ids else "pending"
                yield BatchRequest(
                    request_key=key,
                    custom_id=custom_id,
                    role="G",
                    study_id="target",
                    profile_id=donor_key,
                    condition_id=condition,
                    outcome_id=outcome_id,
                    replicate_id=1,
                    requested_model=requested_model,
                    prompt_hash=_render_prompt_hash(render.messages),
                    schema_version=schema_hash(render.response_schema),
                    prompt_protocol_id=render.protocol_id,
                    prompt_compiler_version=PROMPT_COMPILER_VERSION,
                    seed=seed_from_request_key(key),
                    status=status,
                    messages=render.messages,
                    response_schema=render.response_schema,
                    response_key_map=render.response_key_map or {},
                    request_stage=request_stage,
                    consensus_interaction_protocol_id=CONSENSUS_INTERACTION_PROTOCOL_ID,
                    consensus_stage_a_request_key=stage_a.request_key,
                    consensus_stage_a_prompt_hash=_render_prompt_hash(stage_a.messages),
                    consensus_stage_a_schema_hash=schema_hash(stage_a.response_schema),
                    consensus_feedback_hash="" if render.provenance is None else render.provenance.get("feedback_prompt_material_hash", ""),
                )
                emitted += 1
                if max_requests is not None and emitted >= max_requests:
                    return
                continue
            stimulus = sc.get_condition_stimulus(
                condition,
                state_abbr=row.get("state_abbr"),
                control_variant=1,
            )
            render = build_g_prompt_render(
                _profile_dict(row),
                stimulus,
                items,
                donor_key=donor_key,
                condition_id=condition,
                system_prompt=system_prompt,
                prompt_variant=prompt_variant,
                response_format_instruction_version=response_format_instruction_version,
            )
            key = render.request_key
            assert key == request_key_g(donor_key=donor_key, condition=condition, replicate=1) or prompt_variant != "P1" or response_format_instruction_version != "v1"
            custom_id = custom_id_from_request_key(key)
            status = "already_successful" if custom_id in successful_custom_ids else "pending"
            yield BatchRequest(
                request_key=key,
                custom_id=custom_id,
                role="G",
                study_id="target",
                profile_id=donor_key,
                condition_id=condition,
                outcome_id="full_questionnaire",
                replicate_id=1,
                requested_model=requested_model,
                prompt_hash=_render_prompt_hash(render.messages),
                schema_version=schema_hash(render.response_schema),
                prompt_protocol_id=render.protocol_id,
                prompt_compiler_version=PROMPT_COMPILER_VERSION,
                seed=seed_from_request_key(key),
                status=status,
                messages=render.messages,
                response_schema=render.response_schema,
                response_key_map=render.response_key_map or {},
            )
            emitted += 1
            if max_requests is not None and emitted >= max_requests:
                return


def iter_f_requests(
    *,
    requested_model: str,
    max_requests: int | None = None,
    successful_custom_ids: set[str] | None = None,
    consensus_stage_a_success: dict[str, dict[str, Any]] | None = None,
) -> Iterable[BatchRequest]:
    successful_custom_ids = successful_custom_ids or set()
    profiles = pd.read_csv(F_PANEL_PATH)
    outcomes = list(sc.OUTCOME_COMPOSITES.keys())
    emitted = 0
    for _, row in profiles.iterrows():
        f_profile_id = str(row["f_profile_id"])
        for condition in _conditions():
            if condition == "Consensus":
                stage_a = build_f_consensus_stage_a_prompt_render(_profile_dict(row), f_profile_id=f_profile_id, replicate_id=1)
                if consensus_stage_a_success is None:
                    renders = [(CONSENSUS_STAGE_A_OUTCOME_ID, stage_a, "consensus_stage_a")]
                else:
                    response = _stage_a_response_for(consensus_stage_a_success, f_profile_id, stage_a)
                    record = consensus_stage_a_record(stage_a, response, role="F", subject_id=f_profile_id, replicate_id=1)
                    renders = [
                        (
                            outcome,
                            build_f_consensus_stage_b_prompt_render(
                                _profile_dict(row),
                                outcome,
                                record,
                                f_profile_id=f_profile_id,
                                replicate_id=1,
                                response_format_instruction_version="v2",
                            ),
                            "consensus_stage_b",
                        )
                        for outcome in outcomes
                    ]
                for outcome_id, render, request_stage in renders:
                    key = render.request_key
                    custom_id = custom_id_from_request_key(key)
                    status = "already_successful" if custom_id in successful_custom_ids else "pending"
                    yield BatchRequest(
                        request_key=key,
                        custom_id=custom_id,
                        role="F",
                        study_id="target",
                        profile_id=f_profile_id,
                        condition_id=condition,
                        outcome_id=outcome_id,
                        replicate_id=1,
                        requested_model=requested_model,
                        prompt_hash=_render_prompt_hash(render.messages),
                        schema_version=schema_hash(render.response_schema),
                        prompt_protocol_id=render.protocol_id,
                        prompt_compiler_version=PROMPT_COMPILER_VERSION,
                        seed=seed_from_request_key(key),
                        status=status,
                        messages=render.messages,
                        response_schema=render.response_schema,
                        response_key_map=render.response_key_map or {},
                        request_stage=request_stage,
                        consensus_interaction_protocol_id=CONSENSUS_INTERACTION_PROTOCOL_ID,
                        consensus_stage_a_request_key=stage_a.request_key,
                        consensus_stage_a_prompt_hash=_render_prompt_hash(stage_a.messages),
                        consensus_stage_a_schema_hash=schema_hash(stage_a.response_schema),
                        consensus_feedback_hash="" if render.provenance is None else render.provenance.get("feedback_prompt_material_hash", ""),
                    )
                    emitted += 1
                    if max_requests is not None and emitted >= max_requests:
                        return
                continue
            stimulus = sc.get_condition_stimulus(
                condition,
                state_abbr=row.get("state_abbr"),
                control_variant=target_f_control_variant(f_profile_id, 1) if condition == "control" else None,
            )
            for outcome in outcomes:
                key = request_key_f(study_id="target", f_profile_id=f_profile_id, condition=condition, outcome=outcome, replicate=1)
                custom_id = custom_id_from_request_key(key)
                render = build_f_prompt_render(
                    _profile_dict(row),
                    stimulus,
                    outcome,
                    study_id="target",
                    f_profile_id=f_profile_id,
                    condition_id=condition,
                    response_format_instruction_version="v2",
                )
                status = "already_successful" if custom_id in successful_custom_ids else "pending"
                yield BatchRequest(
                    request_key=key,
                    custom_id=custom_id,
                    role="F",
                    study_id="target",
                    profile_id=f_profile_id,
                    condition_id=condition,
                    outcome_id=outcome,
                    replicate_id=1,
                    requested_model=requested_model,
                    prompt_hash=_render_prompt_hash(render.messages),
                    schema_version=schema_hash(render.response_schema),
                    prompt_protocol_id=render.protocol_id,
                    prompt_compiler_version=PROMPT_COMPILER_VERSION,
                    seed=seed_from_request_key(key),
                    status=status,
                    messages=render.messages,
                    response_schema=render.response_schema,
                    response_key_map=render.response_key_map or {},
                )
                emitted += 1
                if max_requests is not None and emitted >= max_requests:
                    return


def successful_custom_ids(paths: Sequence[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            if "custom_id" in df:
                ids.update(df["custom_id"].dropna().astype(str))
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                custom_id = obj.get("custom_id")
                if custom_id and "error" not in obj:
                    ids.add(str(custom_id))
    return ids


def prepare_batch(
    *,
    role: str,
    requested_model: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    max_requests: int | None = None,
    successful_result_paths: Sequence[Path] = (),
    consensus_stage_a_success_path: Path | str | None = None,
    prompt_variant: str = "P1",
    response_format_instruction_version: str = "v1",
) -> dict[str, Any]:
    """prompt_variant/response_format_instruction_version default to
    "P1"/"v1" -- every existing call site is byte-identical to before these
    parameters were added; non-default values are only meaningful for
    role="G" (prompt_variant: Approach-3 prompt-ensemble ATE-shape check;
    response_format_instruction_version="v2": the G-v2
    PROVIDER_SERVING_FORMAT_FAILURE replacement) and are ignored for role="F"."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    successes = successful_custom_ids(successful_result_paths)
    consensus_stage_a_success = load_consensus_stage_a_success(consensus_stage_a_success_path) if consensus_stage_a_success_path is not None else None
    if role == "G":
        requests = list(
            iter_g_requests(
                requested_model=requested_model,
                max_requests=max_requests,
                successful_custom_ids=successes,
                consensus_stage_a_success=consensus_stage_a_success,
                prompt_variant=prompt_variant,
                response_format_instruction_version=response_format_instruction_version,
            )
        )
    else:
        requests = list(
            iter_f_requests(
                requested_model=requested_model,
                max_requests=max_requests,
                successful_custom_ids=successes,
                consensus_stage_a_success=consensus_stage_a_success,
            )
        )
    custom_ids = [req.custom_id for req in requests]
    if len(custom_ids) != len(set(custom_ids)):
        raise ValueError("duplicate custom_id values in batch manifest")

    manifest_path = out / "request_manifest.csv"
    jsonl_path = out / "batch_input.jsonl"
    accounting_path = out / "batch_accounting.json"
    manifest_fields = [
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
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        for req in requests:
            writer.writerow(
                {
                    field: (
                        req.required_fields
                        if field == "required_fields"
                        else json.dumps(req.response_key_map, sort_keys=True)
                        if field == "response_key_map"
                        else getattr(req, field)
                    )
                    for field in manifest_fields
                }
            )

    submitted = 0
    prompt_chars = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for req in requests:
            if req.status == "already_successful":
                continue
            body = _chat_body(req)
            prompt_chars += sum(len(m["content"]) for m in req.messages)
            f.write(json.dumps({"custom_id": req.custom_id, "body": body}, sort_keys=True) + "\n")
            submitted += 1

    accounting = {
        "role": role,
        "requested_model": requested_model,
        "manifest_requests": len(requests),
        "already_successful_requests": sum(1 for req in requests if req.status == "already_successful"),
        "submitted_requests": submitted,
        "estimated_prompt_characters": prompt_chars,
        "estimated_prompt_tokens_rough": int(prompt_chars / 4),
        "completion_tokens_budget": submitted * 1024,
        "consensus_stage_a_success_path": "" if consensus_stage_a_success_path is None else str(consensus_stage_a_success_path),
    }
    accounting_path.write_text(json.dumps(accounting, indent=2) + "\n", encoding="utf-8")
    return {"manifest": manifest_path, "jsonl": jsonl_path, "accounting": accounting_path, **accounting}


def split_jsonl_file(jsonl_path: Path | str, *, max_lines_per_shard: int, output_dir: Path | str | None = None) -> list[Path]:
    """Deterministically split a JSONL batch input into numbered shards."""
    if max_lines_per_shard <= 0:
        raise ValueError("max_lines_per_shard must be positive")
    source = Path(jsonl_path)
    if not source.exists():
        raise FileNotFoundError(source)
    out_dir = Path(output_dir) if output_dir is not None else source.parent / "shards"
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    handle = None
    try:
        with open(source, encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if line_no % max_lines_per_shard == 0:
                    if handle is not None:
                        handle.close()
                    shard = out_dir / f"{source.stem}_part{len(shards) + 1:04d}{source.suffix}"
                    shards.append(shard)
                    handle = open(shard, "w", encoding="utf-8")
                if handle is not None:
                    handle.write(line)
    finally:
        if handle is not None:
            handle.close()
    return shards


def _under_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def is_tiny_smoke_jsonl(jsonl_path: Path | str) -> bool:
    return _under_path(Path(jsonl_path), TINY_SMOKE_ROOT)


def _load_smoke_state(state_path: Path = TINY_SMOKE_STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        raise RuntimeError(f"smoke accounting state is missing: {state_path}")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"smoke accounting state is ambiguous/unreadable: {exc}") from exc
    required = {"schema_version", "smoke_cost_cap_usd", "global_max_new_requests", "submitted_custom_ids", "successful_custom_ids"}
    missing = required - set(state)
    if missing:
        raise RuntimeError(f"smoke accounting state is ambiguous; missing {sorted(missing)}")
    if float(state["smoke_cost_cap_usd"]) != SMOKE_COST_CAP_USD:
        raise RuntimeError("smoke accounting state cost cap does not match current guard")
    if int(state["global_max_new_requests"]) != SMOKE_GLOBAL_MAX_NEW_REQUESTS:
        raise RuntimeError("smoke accounting state global request cap does not match current guard")
    for key in ("submitted_custom_ids", "successful_custom_ids"):
        if not isinstance(state[key], list):
            raise RuntimeError(f"smoke accounting state {key} must be a list")
        if len(state[key]) != len(set(state[key])):
            raise RuntimeError(f"smoke accounting state {key} contains duplicate custom_id values")
    return state


def _declared_smoke_model(jsonl_path: Path) -> str:
    candidates = yaml.safe_load((PIPELINE_ROOT / "config" / "model_config.yaml").read_text(encoding="utf-8"))["model_selection"]
    candidate_ids = list(dict.fromkeys(candidates["g_model_candidates"] + candidates["f_model_candidates"]))
    model_dir = jsonl_path.resolve().relative_to(TINY_SMOKE_ROOT.resolve()).parts[0]
    matches = [
        model
        for model in candidate_ids
        if re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") == model_dir
    ]
    if len(matches) != 1:
        raise RuntimeError(f"input file model directory {model_dir!r} does not identify one declared smoke candidate")
    if matches[0] not in SMOKE_MODEL_PRICES_PER_1M_TOKENS:
        raise RuntimeError(f"missing smoke price for declared model {matches[0]!r}")
    return matches[0]


def _smoke_wave(jsonl_path: Path) -> tuple[str, int]:
    rel_parts = jsonl_path.resolve().relative_to(TINY_SMOKE_ROOT.resolve()).parts
    if any(part.startswith("preflight") for part in rel_parts):
        return "engineering_capability_preflight", 1
    if "first_wave" in rel_parts:
        return "first_wave", 12
    if any(part.startswith("stage_b") for part in rel_parts):
        return "consensus_stage_b", 2
    raise RuntimeError("smoke input path does not identify first_wave, consensus Stage-B, or preflight wave")


def _read_jsonl_requests(jsonl_path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"smoke JSONL has invalid JSON on line {line_no}: {exc}") from exc
    return rows


def tiny_smoke_safety_guard(jsonl_path: Path | str, *, for_submit: bool = True) -> dict[str, Any]:
    """Fail-closed cost and duplicate guard for tiny pre-API smoke batches."""
    path = Path(jsonl_path)
    if not path.exists():
        raise RuntimeError(f"smoke input file is missing: {path}")
    if not _under_path(path, TINY_SMOKE_ROOT):
        raise RuntimeError(f"input file is outside tiny smoke directory: {path}")
    if for_submit and any("placeholder_validate_only" in part for part in path.parts):
        raise RuntimeError("refusing to submit placeholder Stage-B validation manifest; regenerate Stage B from real Stage A first")
    state = _load_smoke_state(TINY_SMOKE_STATE_PATH)
    declared_model = _declared_smoke_model(path)
    wave, expected_count = _smoke_wave(path)
    requests = _read_jsonl_requests(path)
    request_count = len(requests)
    if request_count != expected_count:
        raise RuntimeError(f"tiny smoke {wave} must contain exactly {expected_count} requests; observed {request_count}")
    custom_ids = [str(row.get("custom_id", "")) for row in requests]
    if not all(custom_ids) or len(custom_ids) != len(set(custom_ids)):
        raise RuntimeError("smoke JSONL custom_id values are missing or duplicated")
    bodies = [row.get("body") for row in requests]
    models = sorted({str(body.get("model", "")) for body in bodies if isinstance(body, dict)})
    if models != [declared_model]:
        raise RuntimeError(f"model ID differs from declared smoke candidate; expected {declared_model!r}, observed {models!r}")
    max_tokens = [int(body.get("max_tokens", 0) or 0) for body in bodies if isinstance(body, dict)]
    if len(max_tokens) != request_count or min(max_tokens) <= 0:
        raise RuntimeError("smoke JSONL has missing/invalid maximum output-token budgets")
    already = set(custom_ids) & (set(state["submitted_custom_ids"]) | set(state["successful_custom_ids"]))
    if already:
        raise RuntimeError(f"JSONL contains custom_id already recorded as submitted/successful: {sorted(already)[:5]}")
    if for_submit:
        not_approved = set(custom_ids) - tiny_smoke_approved_pending_custom_ids()
        if not_approved:
            raise RuntimeError(
                f"JSONL contains custom_id(s) not in the approved pending allowlist "
                f"(raising SMOKE_GLOBAL_MAX_NEW_REQUESTS does not admit these): {sorted(not_approved)[:5]}"
            )

    estimated_input_tokens = int(sum(sum(len(message.get("content", "")) for message in body["messages"]) for body in bodies) / 4)
    maximum_output_tokens = int(sum(max_tokens))
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[declared_model]
    batch_cost = (
        estimated_input_tokens * prices["input"] / 1_000_000
        + maximum_output_tokens * prices["output"] / 1_000_000
    )
    cumulative_new_requests = int(state.get("cumulative_new_requests", 0)) + request_count
    cumulative_cost = float(state.get("estimated_worst_case_cost_usd", 0.0)) + batch_cost
    if cumulative_new_requests > SMOKE_GLOBAL_MAX_NEW_REQUESTS:
        raise RuntimeError(
            f"tiny smoke global request cap exceeded: {cumulative_new_requests} > {SMOKE_GLOBAL_MAX_NEW_REQUESTS}"
        )
    if cumulative_cost > SMOKE_COST_CAP_USD:
        raise RuntimeError(
            f"tiny smoke worst-case cost cap exceeded: ${cumulative_cost:.4f} > ${SMOKE_COST_CAP_USD:.2f}"
        )
    return {
        "model": declared_model,
        "wave": wave,
        "request_count": request_count,
        "cumulative_new_requests": cumulative_new_requests,
        "estimated_input_tokens": estimated_input_tokens,
        "maximum_output_tokens": maximum_output_tokens,
        "estimated_worst_case_cost_usd": round(cumulative_cost, 6),
        "batch_estimated_worst_case_cost_usd": round(batch_cost, 6),
        "smoke_cost_cap_usd": SMOKE_COST_CAP_USD,
        "jsonl": str(path),
        "state_path": str(TINY_SMOKE_STATE_PATH),
        "submission_allowed": True,
    }


def record_tiny_smoke_submission(guard: dict[str, Any], submit_result: dict[str, Any], *, state_path: Path = TINY_SMOKE_STATE_PATH) -> dict[str, Any]:
    state = _load_smoke_state(state_path)
    requests = _read_jsonl_requests(Path(guard["jsonl"]))
    submitted = list(dict.fromkeys([*state["submitted_custom_ids"], *[str(row["custom_id"]) for row in requests]]))
    state["submitted_custom_ids"] = submitted
    state["cumulative_new_requests"] = int(guard["cumulative_new_requests"])
    state["estimated_worst_case_cost_usd"] = float(guard["estimated_worst_case_cost_usd"])
    state.setdefault("submissions", []).append(
        {
            "model": guard["model"],
            "wave": guard["wave"],
            "request_count": guard["request_count"],
            "estimated_worst_case_cost_usd": guard["batch_estimated_worst_case_cost_usd"],
            "submit_result": submit_result,
        }
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return state


def _message_content_from_output(obj: dict[str, Any]) -> str | None:
    body = obj.get("response", {}).get("body", obj.get("body", obj))
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices:
        return None
    first = choices[0]
    if "message" in first and isinstance(first["message"], dict):
        return first["message"].get("content")
    return first.get("text")


def parse_batch_results(*, manifest_path: Path, results_jsonl: Path, output_dir: Path | str) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)
    by_custom_id = manifest.set_index("custom_id").to_dict("index")
    success_rows: list[dict[str, Any]] = []
    malformed_rows: list[dict[str, Any]] = []
    raw_path = out / "raw_responses.jsonl"
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    with open(raw_path, "w", encoding="utf-8") as raw_out, open(results_jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed_rows.append({"line_no": line_no, "custom_id": "", "error": f"invalid_json_line: {exc}", "raw_content": line.strip()})
                continue
            custom_id = str(obj.get("custom_id", ""))
            meta = by_custom_id.get(custom_id)
            content = _message_content_from_output(obj)
            raw_out.write(json.dumps({"custom_id": custom_id, "request_key": "" if meta is None else meta["request_key"], "raw_content": content, "raw_line": obj}, sort_keys=True) + "\n")
            body = obj.get("response", {}).get("body", obj.get("body", obj))
            if isinstance(body, dict) and isinstance(body.get("usage"), dict):
                for key in usage:
                    usage[key] += int(body["usage"].get(key, 0) or 0)
            if meta is None:
                malformed_rows.append({"line_no": line_no, "custom_id": custom_id, "error": "custom_id_not_in_manifest", "raw_content": content})
                continue
            if "error" in obj:
                malformed_rows.append({"line_no": line_no, "custom_id": custom_id, "error": json.dumps(obj["error"], sort_keys=True), "raw_content": content})
                continue
            try:
                parsed = json.loads(content or "")
            except json.JSONDecodeError as exc:
                malformed_rows.append({"line_no": line_no, "custom_id": custom_id, "error": f"invalid_response_json: {exc}", "raw_content": content})
                continue
            required = [field for field in str(meta["required_fields"]).split("|") if field]
            missing = [field for field in required if field not in parsed]
            if missing:
                malformed_rows.append({"line_no": line_no, "custom_id": custom_id, "error": f"missing_required_fields: {missing}", "raw_content": content})
                continue
            key_map = json.loads(meta.get("response_key_map") or "{}")
            parsed_target = {key_map.get(key, key): parsed[key] for key in required}
            success_rows.append({**meta, "custom_id": custom_id, "parsed_output": json.dumps(parsed_target, sort_keys=True), "status": "success"})

    success = pd.DataFrame(success_rows)
    malformed = pd.DataFrame(malformed_rows)
    success_path = out / "parsed_success.csv"
    malformed_path = out / "malformed_outputs.csv"
    retry_path = out / "retry_manifest.csv"
    success.to_csv(success_path, index=False)
    malformed.to_csv(malformed_path, index=False)
    retry_ids = set(malformed["custom_id"].dropna().astype(str)) if not malformed.empty and "custom_id" in malformed else set()
    manifest[manifest["custom_id"].isin(retry_ids)].to_csv(retry_path, index=False)
    accounting = {"successful": int(len(success)), "malformed": int(len(malformed)), **usage}
    (out / "result_accounting.json").write_text(json.dumps(accounting, indent=2) + "\n", encoding="utf-8")
    return {"success": success_path, "malformed": malformed_path, "retry_manifest": retry_path, "raw": raw_path, **accounting}


def submit_batch(jsonl_path: Path, *, endpoint: str = "/v1/chat/completions") -> dict[str, Any]:
    from together import Together

    client = Together()
    file_resp = client.files.upload(file=str(jsonl_path), purpose="batch-api", check=False)
    batch = client.batches.create(input_file_id=file_resp.id, endpoint=endpoint)
    return {"input_file_id": file_resp.id, "batch": _to_plain(getattr(batch, "job", batch))}


def retrieve_batch(batch_id: str) -> dict[str, Any]:
    from together import Together

    client = Together()
    return _to_plain(client.batches.retrieve(batch_id))


def download_file(file_id: str, output_path: Path) -> Path:
    from together import Together

    client = Together()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with client.files.with_streaming_response.content(id=file_id) as response:
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    return output_path


def _to_plain(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        try:
            return _to_plain(value.model_dump(mode="json"))
        except TypeError:
            return _to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return _to_plain(value.dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return json.loads(json.dumps(value, default=str))


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable version of SDK metadata.

    Datetime/date objects are preserved as ISO-8601 strings.
    """
    return _to_plain(value)
