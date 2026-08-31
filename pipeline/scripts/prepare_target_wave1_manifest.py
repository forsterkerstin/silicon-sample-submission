#!/usr/bin/env python3
"""scripts/prepare_target_wave1_manifest.py

Builds (never submits) the canonical target Wave-1 manifests for G and F,
using ONLY the already-frozen target request builders
(inference.together_batch.prepare_batch -> iter_g_requests/iter_f_requests),
unmodified. Wave 1 = every target request that does not depend on a target
model output, plus Consensus Stage A (prepare_batch's default behavior,
omitting consensus_stage_a_success_path, already produces exactly this
shape -- no new request-selection logic is written here).

Output root: outputs/target_production/wave1/{G,F}/google_gemma-4-31B-it/
-- under target_production_guard.TARGET_PRODUCTION_ROOT, so a future
declare_target_phase/target_production_safety_guard call (still blocked
today: assert_target_production_prerequisites_frozen() raises until the
calibration artifact and final method manifest are frozen) can reach it.

Partitioning: deterministic positional chunking (every request's row order
from iter_g_requests/iter_f_requests already groups all of one donor's/
profile's requests contiguously) into the smallest K (a divisor of the
per-entity block size) whose worst-case shard byte size and request count
both stay under the operational ceilings -- same dynamic-K-search pattern
already used for the calibration-production manifest.

No LLM calls. No target requests submitted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

from inference.model_config import selected_model  # noqa: E402
from inference.target_production_guard import TARGET_PRODUCTION_ROOT  # noqa: E402
from inference.together_batch import SMOKE_MODEL_PRICES_PER_1M_TOKENS, TOGETHER_BATCH_MAX_INPUT_FILE_BYTES, TOGETHER_BATCH_MAX_REQUESTS, prepare_batch  # noqa: E402

WAVE1_ROOT = TARGET_PRODUCTION_ROOT / "wave1"
OPERATIONAL_FILE_SIZE_CEILING_BYTES = int(95 * 1024 * 1024)
# G: 17 rows/donor (16 non-Consensus + 1 Consensus Stage A). F: 209 rows/profile (16*13 + 1).
CANDIDATE_PARTITION_COUNTS = (1, 2, 4, 5, 10, 20, 25)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_and_write_partitions(jsonl_path: Path, manifest_df: pd.DataFrame, rows_per_entity: int, out_root: Path) -> tuple[int, list[dict[str, Any]]]:
    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) != len(manifest_df):
        raise RuntimeError(f"jsonl has {len(lines)} lines, manifest has {len(manifest_df)} rows -- must match 1:1 in order")
    line_bytes = [len(line.encode("utf-8")) for line in lines]
    n_entities = len(lines) // rows_per_entity
    if len(lines) % rows_per_entity != 0:
        raise RuntimeError(f"{len(lines)} rows is not an exact multiple of rows_per_entity={rows_per_entity}")

    chosen_k = None
    for k in CANDIDATE_PARTITION_COUNTS:
        if n_entities % k != 0:
            continue
        entities_per_part = n_entities // k
        rows_per_part = entities_per_part * rows_per_entity
        max_bytes = 0
        max_requests = 0
        for p in range(k):
            part_bytes = sum(line_bytes[p * rows_per_part : (p + 1) * rows_per_part])
            max_bytes = max(max_bytes, part_bytes)
            max_requests = max(max_requests, rows_per_part)
        if max_bytes < OPERATIONAL_FILE_SIZE_CEILING_BYTES and max_requests < TOGETHER_BATCH_MAX_REQUESTS:
            chosen_k = k
            break
    if chosen_k is None:
        raise RuntimeError(f"no candidate K in {CANDIDATE_PARTITION_COUNTS} keeps every partition under the size/request ceilings")

    entities_per_part = n_entities // chosen_k
    rows_per_part = entities_per_part * rows_per_entity
    partitions = []
    for p in range(chosen_k):
        lo, hi = p * rows_per_part, (p + 1) * rows_per_part
        part_dir = out_root / f"part{p + 1}"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_jsonl = part_dir / "batch_input.jsonl"
        part_jsonl.write_text("".join(lines[lo:hi]), encoding="utf-8")
        part_manifest = manifest_df.iloc[lo:hi]
        part_manifest.to_csv(part_dir / "request_manifest.csv", index=False)
        partitions.append(
            {
                "partition": f"part{p + 1}",
                "requests": hi - lo,
                "jsonl_size_mb": round(part_jsonl.stat().st_size / (1024 * 1024), 2),
                "sha256": sha256_file(part_jsonl),
            }
        )
    return chosen_k, partitions


def worst_case_cost(model: str, prompt_tokens_rough: int, completion_tokens_budget: int) -> float:
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[model]
    return prompt_tokens_rough * prices["input"] / 1_000_000 + completion_tokens_budget * prices["output"] / 1_000_000


def build_role(role: str, rows_per_entity: int) -> dict[str, Any]:
    model = selected_model(role.lower(), require_frozen=True)
    if not model:
        raise RuntimeError(f"{role}* is not frozen")
    out_dir = WAVE1_ROOT / role / model.replace("/", "_")
    result = prepare_batch(role=role, requested_model=model, output_dir=out_dir)
    manifest_df = pd.read_csv(result["manifest"])
    manifest_sha = sha256_file(result["manifest"])
    jsonl_sha = sha256_file(result["jsonl"])
    cost = worst_case_cost(model, result["estimated_prompt_tokens_rough"], result["completion_tokens_budget"])

    k, partitions = choose_and_write_partitions(result["jsonl"], manifest_df, rows_per_entity, out_dir)

    return {
        "role": role,
        "model": model,
        "requests": result["manifest_requests"],
        "manifest_sha256": manifest_sha,
        "jsonl_sha256": jsonl_sha,
        "worst_case_cost_usd": round(cost, 4),
        "partition_count": k,
        "partitions": partitions,
    }


def main() -> int:
    WAVE1_ROOT.mkdir(parents=True, exist_ok=True)
    g_result = build_role("G", rows_per_entity=17)
    f_result = build_role("F", rows_per_entity=209)

    summary = {"G": g_result, "F": f_result}
    (WAVE1_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
