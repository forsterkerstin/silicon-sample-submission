"""Build (never submit) the CURRENT target G Wave-1 completion manifests:
exactly the intended production identities that still need a production
attempt right now -- the 1,473 identities whose attempt-1 (the 16,990-
request G-v2 full production replacement) was schema_invalid or
provider_error, plus the 10 identities the engineering smoke happened to
cover (smoke is not production, so those start at attempt 1).

Uses inference.target_g_retry_engine's bounded 3-attempt machinery: each
identity's OWN next_attempt_number (1 for smoke-only, 2 for previously-
failed) determines its replicate_id, so the SAME donor/condition/persona/
stimulus/questionnaire/model/sampling config is reused exactly and only the
stochastic draw (request_key/seed/custom_id) is fresh. Consensus Stage-A
requests always preserve their attempt-1 item order (order_replicate_id=1).

Excludes every one of the 15,517 already-valid production identities --
FIRST_VALID_RESPONSE_WINS, never regenerated.

No LLM calls. No target requests submitted. No scientific target-G value
(mean/ATE/distribution/ranking) is computed, read, or exposed anywhere in
this script.
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

from inference.model_config import selected_model  # noqa: E402
from inference.target_g_retry_engine import STAGES, build_attempt_ledger, build_completion_requests, identities_pending_next_attempt  # noqa: E402
from inference.together_batch import SMOKE_MODEL_PRICES_PER_1M_TOKENS, _chat_body  # noqa: E402

OUT_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion"
MANIFEST_FIELDS = [
    "request_key", "custom_id", "role", "study_id", "profile_id", "condition_id",
    "outcome_id", "replicate_id", "requested_model", "prompt_hash", "schema_version",
    "prompt_protocol_id", "prompt_compiler_version", "seed", "status", "required_fields",
    "response_key_map", "request_stage", "engine_config_hash",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _worst_case_cost(model: str, estimated_prompt_tokens_rough: int, maximum_output_tokens: int) -> float:
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[model]
    return estimated_prompt_tokens_rough * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000


def write_requests(requests: list, out_dir: Path) -> dict[str, Any]:
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

    ledger = build_attempt_ledger()
    pending = identities_pending_next_attempt(ledger)
    requests_by_stage = build_completion_requests(ledger, pending, requested_model=g_star)

    all_custom_ids: set[str] = set()
    summary: dict[str, Any] = {"model": g_star, "stages": {}}
    for stage in STAGES:
        requests = requests_by_stage[stage]
        overlap = all_custom_ids & {r.custom_id for r in requests}
        if overlap:
            raise RuntimeError(f"custom_id collision across stages: {sorted(overlap)[:5]}")
        all_custom_ids |= {r.custom_id for r in requests}

        stats = write_requests(requests, OUT_ROOT / stage)
        stats["worst_case_cost_usd"] = round(_worst_case_cost(g_star, stats["estimated_prompt_tokens_rough"], stats["maximum_output_tokens"]), 6)
        # true production attempt_number (1 for smoke-only identities, 2 for
        # previously-failed) -- from the ledger, NOT from req.replicate_id
        # (which is the wire-level discriminator and can differ from
        # attempt_number for smoke-only identities; see build_completion_requests).
        attempt_number_counts: dict[str, int] = {}
        for identity in pending[stage]:
            key = str(ledger[identity]["next_attempt_number"])
            attempt_number_counts[key] = attempt_number_counts.get(key, 0) + 1
        stats["attempt_number_counts"] = attempt_number_counts
        summary["stages"][stage] = stats

    summary["total_requests"] = sum(s["requests"] for s in summary["stages"].values())
    summary["total_worst_case_cost_usd"] = round(sum(s["worst_case_cost_usd"] for s in summary["stages"].values()), 6)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
