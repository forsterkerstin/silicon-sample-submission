"""Materialize (never submit) the ATP1/ATP2 G-model screen manifest.

Implements the ATP_G_SCREEN_FROZEN amendment in
outputs/final_offline_gate/model_selection_r_f_rule_manifest.json (v5):
one request per usable ATP1/ATP2 respondent, using ONLY the already-frozen
build_g_external_validation_prompt_render prompt path (which does not alter
the production build_g_prompt_render target-G path) and the already-frozen
ATP profile mapping / eligibility convention in ate/g_atp_screen.py. This
script builds request manifests only; it never calls the Together API.

Source: https://github.com/skrsteski/survey-simulations @
faeb4e1a73567a8c98c69798774b63fdb27c79e1 (local copies gitignored, see
.gitignore).
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ate.g_atp_screen import (  # noqa: E402
    atp1_item,
    atp2_item,
    atp_row_to_g_profile,
    usable_atp1_respondents,
    usable_atp2_respondents,
)
from inference.prompts import (  # noqa: E402
    PROMPT_COMPILER_VERSION,
    build_g_external_validation_prompt_render,
    schema_hash,
    text_hash,
)
from inference.request_logging import seed_from_request_key  # noqa: E402
from inference.together_batch import (  # noqa: E402
    BatchRequest,
    SMOKE_MODEL_PRICES_PER_1M_TOKENS,
    _chat_body,
    compute_engine_config_hash,
    custom_id_from_request_key,
)

OUT_ROOT = PIPELINE_ROOT / "outputs" / "scientific_bakeoff" / "g_model_screen"
DATA_DIR = PIPELINE_ROOT / "data" / "atp_survey_simulations"
REQUEST_STAGE = "g_model_screen"
MODELS = ["deepseek-ai/DeepSeek-V4-Pro-0813", "google/gemma-4-31B-it"]
EXPECTED_ATP1_N = 650
EXPECTED_ATP2_N = 576

MANIFEST_FIELDS = [
    "request_key", "custom_id", "role", "study_id", "profile_id", "condition_id",
    "outcome_id", "replicate_id", "requested_model", "prompt_hash", "schema_version",
    "prompt_protocol_id", "prompt_compiler_version", "seed", "status", "required_fields",
    "response_key_map", "request_stage", "engine_config_hash",
]


def _sources() -> list[tuple[str, dict[str, Any], pd.DataFrame]]:
    a1 = pd.read_csv(DATA_DIR / "atp1_human_test.csv")
    a2 = pd.read_csv(DATA_DIR / "atp2_human_test.csv")
    return [
        ("ATP1", atp1_item(), usable_atp1_respondents(a1)),
        ("ATP2", atp2_item(), usable_atp2_respondents(a2)),
    ]


def build_requests_for_model(requested_model: str, sources: list[tuple[str, dict[str, Any], pd.DataFrame]]) -> list[BatchRequest]:
    requests: list[BatchRequest] = []
    for source_id, item, respondents in sources:
        for _, row in respondents.iterrows():
            respondent_id = str(row["id"])
            profile = atp_row_to_g_profile(row)
            render = build_g_external_validation_prompt_render(
                profile, [item], external_material="", source_id=source_id, respondent_id=respondent_id
            )
            requests.append(
                BatchRequest(
                    request_key=render.request_key,
                    custom_id=custom_id_from_request_key(f"{requested_model}|{render.request_key}"),
                    role="G",
                    study_id=source_id,
                    profile_id=respondent_id,
                    condition_id="",
                    outcome_id=source_id,
                    replicate_id=1,
                    requested_model=requested_model,
                    prompt_hash=text_hash("\n".join(f"{m['role']}:{m['content']}" for m in render.messages)),
                    schema_version=schema_hash(render.response_schema),
                    prompt_protocol_id=render.protocol_id,
                    prompt_compiler_version=PROMPT_COMPILER_VERSION,
                    seed=seed_from_request_key(render.request_key),
                    status="pending",
                    messages=render.messages,
                    response_schema=render.response_schema,
                    response_key_map=render.response_key_map or {},
                    request_stage=REQUEST_STAGE,
                    engine_config_hash=compute_engine_config_hash(requested_model),
                )
            )
    if len({r.custom_id for r in requests}) != len(requests):
        raise RuntimeError("duplicate custom_id in G ATP screen manifest")
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
        "manifest": str(manifest_path),
        "jsonl": str(jsonl_path),
        "requests": len(requests),
        "estimated_prompt_characters": prompt_chars,
        "estimated_prompt_tokens_rough": int(prompt_chars / 4),
        "maximum_output_tokens": max_tokens_total,
    }


def worst_case_cost(model: str, estimated_prompt_tokens_rough: int, maximum_output_tokens: int) -> float:
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[model]
    return estimated_prompt_tokens_rough * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000


def main() -> int:
    sources = _sources()
    counts = {source_id: len(respondents) for source_id, _item, respondents in sources}
    if counts.get("ATP1") != EXPECTED_ATP1_N:
        raise RuntimeError(f"expected {EXPECTED_ATP1_N} usable ATP1 respondents, got {counts.get('ATP1')}")
    if counts.get("ATP2") != EXPECTED_ATP2_N:
        raise RuntimeError(f"expected {EXPECTED_ATP2_N} usable ATP2 respondents, got {counts.get('ATP2')}")
    expected_per_model = counts["ATP1"] + counts["ATP2"]

    summary: dict[str, Any] = {
        "purpose": "ATP1/ATP2 G-model screen manifest only; not submitted",
        "atp1_usable_n": counts["ATP1"],
        "atp2_usable_n": counts["ATP2"],
        "expected_requests_per_model": expected_per_model,
        "expected_requests_total": expected_per_model * len(MODELS),
        "models": {},
    }

    per_model_requests: dict[str, list[BatchRequest]] = {}
    for model in MODELS:
        requests = build_requests_for_model(model, sources)
        if len(requests) != expected_per_model:
            raise RuntimeError(f"{model}: expected {expected_per_model} requests, got {len(requests)}")
        per_model_requests[model] = requests

    # Byte-identical scientific content across candidates: same request_key
    # set, and for each request_key the SAME prompt_hash/schema_version --
    # only requested_model/custom_id/engine_config_hash may differ.
    by_key = {model: {r.request_key: (r.prompt_hash, r.schema_version) for r in reqs} for model, reqs in per_model_requests.items()}
    reference_model = MODELS[0]
    for model in MODELS[1:]:
        if set(by_key[model]) != set(by_key[reference_model]):
            raise RuntimeError(f"{model} and {reference_model} do not share an identical request_key set")
        mismatched = [k for k in by_key[reference_model] if by_key[reference_model][k] != by_key[model][k]]
        if mismatched:
            raise RuntimeError(f"{model} and {reference_model} diverge in prompt/schema content for {mismatched[:5]}")

    for model in MODELS:
        model_dir = OUT_ROOT / re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")
        written = write_requests(per_model_requests[model], model_dir)
        cost = worst_case_cost(model, written["estimated_prompt_tokens_rough"], written["maximum_output_tokens"])
        written["worst_case_cost_usd"] = round(cost, 4)
        summary["models"][model] = written
        print(model, "->", json.dumps(written, indent=2))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
