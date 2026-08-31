"""Build (never submit) the Orchinik G-vs-DeepSeek domain-confirmation
manifests: one request per eligible Bovitz respondent per candidate model,
each simulated ONLY under the condition that respondent actually received.

Provenance: participant-visible condition passages, focal-outcome question
text, and consensus-level values are copied verbatim from the real,
already-downloaded, already-hashed Bovitz_qualtrics.docx/.qsf (see
ate/orchinik_g_domain_confirmation.py's module docstring and inline
comments for exactly where each piece came from). Persona fields are real
pretreatment demographics only, mapped onto inference.prompts.
PROFILE_FIELD_ORDER's recognized keys.

Reuses, unmodified: inference.prompts.build_g_external_validation_prompt_render
(the frozen external-human-validation G builder -- no target-specific
content, no new model-specific prompt), inference.together_batch._chat_body
(applies each model's own already-frozen serving config automatically),
and inference.together_batch.smoke_scoped_custom_id (so the same
respondent's Gemma and DeepSeek requests never collide).

No LLM calls. No target requests submitted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ate.orchinik_g_domain_confirmation import (  # noqa: E402
    CONDITION_MATERIAL,
    build_25_items,
    load_eligible_respondents,
    respondent_to_g_profile,
)
from inference.prompts import PROMPT_COMPILER_VERSION, build_g_external_validation_prompt_render, schema_hash, text_hash  # noqa: E402
from inference.request_logging import seed_from_request_key  # noqa: E402
from inference.together_batch import BatchRequest, SMOKE_MODEL_PRICES_PER_1M_TOKENS, _chat_body, compute_engine_config_hash, smoke_scoped_custom_id  # noqa: E402

MODELS = ["google/gemma-4-31B-it", "deepseek-ai/DeepSeek-V4-Pro-0813"]
SOURCE_ID = "orchinik2024_bovitz"
OUT_ROOT = PIPELINE_ROOT / "outputs" / "domain_validation" / "orchinik_g_domain_confirmation"

MANIFEST_FIELDS = [
    "request_key", "custom_id", "role", "study_id", "profile_id", "condition_id",
    "outcome_id", "replicate_id", "requested_model", "prompt_hash", "schema_version",
    "prompt_protocol_id", "prompt_compiler_version", "seed", "status", "required_fields",
    "response_key_map", "request_stage", "engine_config_hash",
]


def build_requests_for_model(requested_model: str, respondents: list[dict[str, str]], items: list[dict[str, Any]], *, response_format_instruction_version: str = "v1") -> list[BatchRequest]:
    requests: list[BatchRequest] = []
    for row in respondents:
        respondent_id = str(row["ID"]).strip()
        if not respondent_id:
            raise ValueError("respondent row has no usable id")
        condition = row["condition"]
        material = CONDITION_MATERIAL[condition]
        profile = respondent_to_g_profile(row)
        render = build_g_external_validation_prompt_render(
            profile, items, external_material=material, source_id=SOURCE_ID, respondent_id=respondent_id, response_format_instruction_version=response_format_instruction_version
        )
        custom_id = smoke_scoped_custom_id(requested_model, render.request_key)
        seed = seed_from_request_key(f"{requested_model}|{render.request_key}")
        requests.append(
            BatchRequest(
                request_key=render.request_key,
                custom_id=custom_id,
                role="G",
                study_id=SOURCE_ID,
                profile_id=respondent_id,
                condition_id=condition,
                outcome_id="orchinik_focal_battery",
                replicate_id=1,
                requested_model=requested_model,
                prompt_hash=text_hash("\n".join(f"{m['role']}:{m['content']}" for m in render.messages)),
                schema_version=schema_hash(render.response_schema),
                prompt_protocol_id=render.protocol_id,
                prompt_compiler_version=PROMPT_COMPILER_VERSION,
                seed=seed,
                status="pending",
                messages=render.messages,
                response_schema=render.response_schema,
                response_key_map=render.response_key_map or {},
                request_stage="domain_confirmation",
                engine_config_hash=compute_engine_config_hash(requested_model),
            )
        )
    if len({r.custom_id for r in requests}) != len(requests):
        raise RuntimeError("duplicate custom_id in Orchinik domain-confirmation manifest")
    return requests


def write_requests(requests: list[BatchRequest], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "request_manifest.csv"
    jsonl_path = out_dir / "batch_input.jsonl"
    prompt_chars = 0
    max_tokens_total = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf, open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for req in requests:
            row = asdict(req)
            row["required_fields"] = req.required_fields
            row["response_key_map"] = json.dumps(req.response_key_map, sort_keys=True)
            writer.writerow({field: row[field] for field in MANIFEST_FIELDS})
            body = _chat_body(req)
            prompt_chars += sum(len(m["content"]) for m in req.messages)
            max_tokens_total += int(body["max_tokens"])
            jf.write(json.dumps({"custom_id": req.custom_id, "body": body}, sort_keys=True) + "\n")
    return {
        "requests": len(requests),
        "manifest_sha256": hashlib.sha256((out_dir / "request_manifest.csv").read_bytes()).hexdigest(),
        "jsonl_sha256": hashlib.sha256((out_dir / "batch_input.jsonl").read_bytes()).hexdigest(),
        "estimated_prompt_characters": prompt_chars,
        "estimated_prompt_tokens_rough": int(prompt_chars / 4),
        "maximum_output_tokens": max_tokens_total,
    }


def worst_case_cost(model: str, estimated_prompt_tokens_rough: int, maximum_output_tokens: int) -> float:
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[model]
    return estimated_prompt_tokens_rough * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000


def main() -> dict:
    respondents = load_eligible_respondents()
    items = build_25_items()

    all_custom_ids: set[str] = set()
    summary: dict[str, Any] = {"eligible_respondents": len(respondents), "items_per_respondent": len(items), "models": {}}
    for model in MODELS:
        requests = build_requests_for_model(model, respondents, items)
        model_dir_name = model.replace("/", "_")
        stats = write_requests(requests, OUT_ROOT / model_dir_name)
        stats["worst_case_cost_usd"] = round(worst_case_cost(model, stats["estimated_prompt_tokens_rough"], stats["maximum_output_tokens"]), 6)
        summary["models"][model] = stats
        overlap = all_custom_ids & {r.custom_id for r in requests}
        if overlap:
            raise RuntimeError(f"custom_id collision across models: {sorted(overlap)[:5]}")
        all_custom_ids |= {r.custom_id for r in requests}

    summary["total_requests"] = sum(m["requests"] for m in summary["models"].values())
    summary["total_worst_case_cost_usd"] = round(sum(m["worst_case_cost_usd"] for m in summary["models"].values()), 6)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
