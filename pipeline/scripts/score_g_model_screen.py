"""Score the retrieved G-ATP model-screen batches under the frozen v5 rules.

Applies ONLY the already-frozen, already-tested pipeline (ate/g_atp_screen.py,
ate/g_atp_screen_validation.py) to the retrieved raw batch output. Nothing
here computes a new metric, threshold, or tie-break -- this script is glue
that loads real retrieved data and hands it to the frozen scorer
(score_g_atp_candidate_from_raw, select_g_star), mirroring
score_f_model_screen.py's structure for F.

Never called before both candidates' raw output is retrieved and hashed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ate.g_atp_screen import (  # noqa: E402
    ATP1_SUBSTANTIVE_CHOICES,
    ATP2_SUBSTANTIVE_CHOICES,
    atp1_human_reference_positions,
    atp2_human_reference_positions,
    select_g_star,
)
from ate.g_atp_screen_validation import EXPECTED_REQUESTS_PER_MODEL_ATP, score_g_atp_candidate_from_raw  # noqa: E402
from inference.together_batch import SMOKE_MODEL_PRICES_PER_1M_TOKENS  # noqa: E402

MODELS = ["deepseek-ai/DeepSeek-V4-Pro-0813", "google/gemma-4-31B-it"]
MODEL_DIRS = {
    "deepseek-ai/DeepSeek-V4-Pro-0813": "deepseek-ai_DeepSeek-V4-Pro-0813",
    "google/gemma-4-31B-it": "google_gemma-4-31B-it",
}
G_MODEL_SCREEN_ROOT = PIPELINE_ROOT / "outputs" / "scientific_bakeoff" / "g_model_screen"
DATA_DIR = PIPELINE_ROOT / "data" / "atp_survey_simulations"
CHOICES_BY_SOURCE = {"ATP1": ATP1_SUBSTANTIVE_CHOICES, "ATP2": ATP2_SUBSTANTIVE_CHOICES}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_by_custom_id(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[str(record["custom_id"])] = record
    return out


def load_schema_by_custom_id(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    """Reconstructs the exact request-specific JSON Schema from the FROZEN,
    unmodified batch_input.jsonl body.response_format -- never invents or
    infers a schema."""
    out: dict[str, dict[str, Any]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            schema = record["body"]["response_format"]["json_schema"]["schema"]
            out[str(record["custom_id"])] = schema
    return out


def realized_usage_cost(model: str, raw_by_custom_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prompt_tokens = completion_tokens = 0
    for record in raw_by_custom_id.values():
        body = record.get("response", {}).get("body", {})
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens += int(usage.get("completion_tokens", 0) or 0)
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[model]
    cost = prompt_tokens * prices["input"] / 1_000_000 + completion_tokens * prices["output"] / 1_000_000
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "realized_cost_usd": round(cost, 6)}


def main() -> int:
    human_reference_positions = {
        "ATP1": atp1_human_reference_positions(pd.read_csv(DATA_DIR / "atp1_human_test.csv")),
        "ATP2": atp2_human_reference_positions(pd.read_csv(DATA_DIR / "atp2_human_test.csv")),
    }

    provenance: dict[str, Any] = {"models": {}}
    candidate_results: dict[str, Any] = {}

    for model in MODELS:
        model_dir = G_MODEL_SCREEN_ROOT / MODEL_DIRS[model]
        retrieved_dir = model_dir / "retrieved"
        output_path = retrieved_dir / "batch_output.jsonl"
        status_path = retrieved_dir / "batch_status.json"
        error_path = retrieved_dir / "batch_error.jsonl"

        status = json.loads(status_path.read_text(encoding="utf-8"))
        raw_by_custom_id = load_raw_by_custom_id(output_path)
        schema_by_custom_id = load_schema_by_custom_id(model_dir / "batch_input.jsonl")
        manifest = pd.read_csv(model_dir / "request_manifest.csv")
        usage = realized_usage_cost(model, raw_by_custom_id)

        provenance["models"][model] = {
            "batch_id": status.get("id"),
            "status": status.get("status"),
            "created_at": status.get("created_at"),
            "completed_at": status.get("completed_at"),
            "output_file_id": status.get("output_file_id"),
            "error_file_id": status.get("error_file_id"),
            "batch_output_sha256": sha256_file(output_path),
            "batch_output_lines": len(raw_by_custom_id),
            "batch_error_present": error_path.exists(),
            **usage,
        }

        result = score_g_atp_candidate_from_raw(
            model=model,
            manifest=manifest,
            raw_by_custom_id=raw_by_custom_id,
            schema_by_custom_id=schema_by_custom_id,
            human_reference_positions=human_reference_positions,
            choices_by_source=CHOICES_BY_SOURCE,
            expected_total=EXPECTED_REQUESTS_PER_MODEL_ATP,
        )
        candidate_results[model] = {"score": result, "usage": usage}

    all_final = all(candidate_results[m]["score"]["is_final"] for m in MODELS)
    selection = None
    if all_final:
        loss_by_model = {m: candidate_results[m]["score"]["g_atp_loss"] for m in MODELS}
        invalid_response_rate = {m: candidate_results[m]["score"]["invalid_report"]["overall_invalid_rate"] for m in MODELS}
        realized_cost_usd = {m: candidate_results[m]["usage"]["realized_cost_usd"] for m in MODELS}
        selection = select_g_star(loss_by_model, invalid_response_rate=invalid_response_rate, realized_cost_usd=realized_cost_usd)

    output = {
        "provenance": provenance,
        "candidate_results": candidate_results,
        "all_final": all_final,
        "selection": selection,
    }
    out_path = G_MODEL_SCREEN_ROOT / "g_screen_scoring_result.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "all_final": all_final,
                "selection": selection,
                "per_model": {
                    m: {
                        "is_final": candidate_results[m]["score"]["is_final"],
                        "decision_status": candidate_results[m]["score"]["decision_status"],
                        "g_atp_loss": candidate_results[m]["score"]["g_atp_loss"],
                        "w1_pp_atp1": candidate_results[m]["score"]["w1_pp_atp1"],
                        "w1_pp_atp2": candidate_results[m]["score"]["w1_pp_atp2"],
                        "invalid_rate": candidate_results[m]["score"]["invalid_report"]["overall_invalid_rate"],
                        "realized_cost_usd": candidate_results[m]["usage"]["realized_cost_usd"],
                    }
                    for m in MODELS
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
