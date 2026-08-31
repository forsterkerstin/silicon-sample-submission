"""Build (never submit) the benchmark-exact Consensus STEP_1 manifest: the
first of four sequential, chained requests per Consensus donor (see
inference/consensus_benchmark_exact.py's module docstring for the full
STEP_1 -> STEP_2 -> STEP_3 -> OUTCOMES design and why only STEP_1 can be
built as a real manifest right now -- STEP_2/STEP_3/OUTCOMES each require
the REAL, already-retrieved response from the immediately preceding stage,
which does not exist until STEP_1 is actually submitted and retrieved).

All 1,000 Consensus donors (the full G donor population -- every donor
receives a Consensus row, same as every other condition). Item order is the
donor-only-derived assign_consensus_exact_order (never replicate/attempt-
dependent). attempt_id=1 for every row (this is the FIRST production
attempt for every donor's STEP_1; retries, if ever needed, are attempt_id
2/3 built later from the SAME donor/order, via the bounded stage-retry
ledger in inference/consensus_exact_retry_engine.py).

No LLM calls. No target requests submitted. No scientific target-G value
is computed, read, or exposed anywhere in this script.
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

import pandas as pd  # noqa: E402

from inference.consensus_benchmark_exact import CONSENSUS_EXACT_PROTOCOL_ID, build_step1_prompt_render  # noqa: E402
from inference.model_config import selected_model  # noqa: E402
from inference.prompts import PROMPT_COMPILER_VERSION, schema_hash  # noqa: E402
from inference.request_logging import seed_from_request_key  # noqa: E402
from inference.together_batch import (  # noqa: E402
    G_MASTER_PATH,
    SMOKE_MODEL_PRICES_PER_1M_TOKENS,
    BatchRequest,
    _chat_body,
    _profile_dict,
    _render_prompt_hash,
    compute_engine_config_hash,
    custom_id_from_request_key,
)

OUT_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_exact" / "step1"
MANIFEST_FIELDS = [
    "request_key", "custom_id", "role", "study_id", "profile_id", "condition_id",
    "outcome_id", "replicate_id", "requested_model", "prompt_hash", "schema_version",
    "prompt_protocol_id", "prompt_compiler_version", "seed", "status", "required_fields",
    "response_key_map", "request_stage", "engine_config_hash",
]
EXPECTED_DONOR_COUNT = 1000


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_requests(requested_model: str) -> list[BatchRequest]:
    profiles = pd.read_csv(G_MASTER_PATH)
    if len(profiles) != EXPECTED_DONOR_COUNT:
        raise RuntimeError(f"G donor population is {len(profiles)}, expected exactly {EXPECTED_DONOR_COUNT}")
    requests: list[BatchRequest] = []
    for _, row in profiles.iterrows():
        donor_key = str(row["donor_key"])
        profile = _profile_dict(row)
        render = build_step1_prompt_render(profile, donor_key=donor_key, attempt_id=1)
        key = render.request_key
        custom_id = custom_id_from_request_key(key)
        requests.append(
            BatchRequest(
                request_key=key,
                custom_id=custom_id,
                role="G",
                study_id="target",
                profile_id=donor_key,
                condition_id="Consensus",
                outcome_id="consensus_exact_step1_estimate",
                replicate_id=1,
                requested_model=requested_model,
                prompt_hash=_render_prompt_hash(render.messages),
                schema_version=schema_hash(render.response_schema),
                prompt_protocol_id=render.protocol_id,
                prompt_compiler_version=PROMPT_COMPILER_VERSION,
                seed=seed_from_request_key(key),
                status="pending",
                messages=render.messages,
                response_schema=render.response_schema,
                response_key_map=render.response_key_map or {},
                request_stage="consensus_exact_step1",
                engine_config_hash=compute_engine_config_hash(requested_model),
            )
        )
    if len({r.custom_id for r in requests}) != len(requests):
        raise RuntimeError("duplicate custom_id in Consensus-exact STEP_1 manifest")
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
        "manifest_sha256": _sha256_file(manifest_path),
        "jsonl_sha256": _sha256_file(jsonl_path),
        "estimated_prompt_characters": prompt_chars,
        "estimated_prompt_tokens_rough": int(prompt_chars / 4),
        "maximum_output_tokens": max_tokens_total,
    }


def main() -> dict:
    g_star = selected_model("g", require_frozen=True)
    requests = build_requests(g_star)
    stats = write_requests(requests, OUT_ROOT)
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[g_star]
    stats["worst_case_cost_usd"] = round(stats["estimated_prompt_tokens_rough"] * prices["input"] / 1_000_000 + stats["maximum_output_tokens"] * prices["output"] / 1_000_000, 6)
    stats["model"] = g_star
    stats["protocol_id"] = CONSENSUS_EXACT_PROTOCOL_ID
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
