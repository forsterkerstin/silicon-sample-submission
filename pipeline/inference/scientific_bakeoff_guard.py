"""Fail-closed cost/duplicate/allowlist guard for the scientific bakeoff.

This is a SEPARATE ledger from the engineering-smoke guard in together_batch.py
(TINY_SMOKE_*) -- it never reads or resets the smoke state, and the smoke
$0.30 cap never applies here. Each scientific phase (e.g. "f_reliability_pilot")
declares its own explicit cost cap and its own explicit approved custom_id
allowlist; nothing is admitted by request-count headroom alone.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from inference.together_batch import (
    DEEPSEEK_MODEL_ID,
    GEMMA_MODEL_ID,
    QWEN_MODEL_ID,
    SMOKE_MODEL_PRICES_PER_1M_TOKENS,
    TOGETHER_BATCH_MAX_INPUT_FILE_BYTES,
    TOGETHER_BATCH_MAX_REQUESTS,
    _read_jsonl_requests,
    _under_path,
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
SCIENTIFIC_BAKEOFF_ROOT = PIPELINE_ROOT / "outputs" / "scientific_bakeoff"
SCIENTIFIC_BAKEOFF_STATE_PATH = SCIENTIFIC_BAKEOFF_ROOT / "scientific_bakeoff_submission_state.json"
# Same Together prices already configured for engineering smoke -- pricing is a
# provider fact, not a phase-specific concept, so this is imported, not duplicated.
SCIENTIFIC_MODEL_PRICES_PER_1M_TOKENS = SMOKE_MODEL_PRICES_PER_1M_TOKENS
ALLOWED_SCIENTIFIC_MODELS = frozenset({DEEPSEEK_MODEL_ID, GEMMA_MODEL_ID})
DENIED_SCIENTIFIC_MODELS = frozenset({QWEN_MODEL_ID})
# request_stage values (BatchRequest.request_stage) that are valid DURING a
# scientific-development phase. "target production" requests use request_key
# prefixes like "G|target|..." / "F|target|..." (see inference/prompts.py) and
# must never be admitted here regardless of phase.
TARGET_PRODUCTION_STUDY_ID = "target"


def _load_state(state_path: Path = SCIENTIFIC_BAKEOFF_STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "schema_version": "scientific_bakeoff_submission_state_v1",
            "phases": {},
            "submitted_custom_ids": [],
            "successful_custom_ids": [],
        }
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "scientific_bakeoff_submission_state_v1":
        raise RuntimeError(f"ambiguous existing scientific-bakeoff state: {state_path}")
    return state


def declare_phase(
    phase: str,
    *,
    approved_custom_ids: set[str],
    cost_cap_usd: float,
    state_path: Path = SCIENTIFIC_BAKEOFF_STATE_PATH,
) -> dict[str, Any]:
    """Register (or re-verify) one phase's exact approved allowlist and cost cap.

    Re-declaring an existing phase with a DIFFERENT allowlist/cap is refused --
    a phase's allowlist and cap are frozen the first time they're declared, not
    silently expandable."""
    state = _load_state(state_path)
    phases = state.setdefault("phases", {})
    existing = phases.get(phase)
    if existing is not None:
        if set(existing["approved_custom_ids"]) != set(approved_custom_ids):
            raise RuntimeError(f"phase {phase!r} allowlist already declared and differs from the requested set")
        if float(existing["cost_cap_usd"]) != float(cost_cap_usd):
            raise RuntimeError(f"phase {phase!r} cost cap already declared as ${existing['cost_cap_usd']}, not ${cost_cap_usd}")
        return existing
    phases[phase] = {
        "approved_custom_ids": sorted(approved_custom_ids),
        "cost_cap_usd": float(cost_cap_usd),
        "cumulative_requests": 0,
        "cumulative_worst_case_cost_usd": 0.0,
        "submissions": [],
    }
    SCIENTIFIC_BAKEOFF_ROOT.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return phases[phase]


def scientific_bakeoff_safety_guard(
    jsonl_path: Path | str,
    manifest_path: Path | str,
    *,
    phase: str,
    for_submit: bool = True,
    state_path: Path = SCIENTIFIC_BAKEOFF_STATE_PATH,
) -> dict[str, Any]:
    path = Path(jsonl_path)
    manifest_path = Path(manifest_path)
    if not path.exists():
        raise RuntimeError(f"scientific-bakeoff input file is missing: {path}")
    if not manifest_path.exists():
        raise RuntimeError(f"scientific-bakeoff manifest is missing: {manifest_path}")
    if not _under_path(path, SCIENTIFIC_BAKEOFF_ROOT):
        raise RuntimeError(f"input file is outside the scientific-bakeoff directory: {path}")

    file_size_bytes = path.stat().st_size
    if file_size_bytes > TOGETHER_BATCH_MAX_INPUT_FILE_BYTES:
        raise RuntimeError(
            f"input file is {file_size_bytes / (1024*1024):.2f} MB, exceeds the documented "
            f"Together Batch API limit of {TOGETHER_BATCH_MAX_INPUT_FILE_BYTES / (1024*1024):.0f} MB"
        )

    state = _load_state(state_path)
    phase_state = state.get("phases", {}).get(phase)
    if phase_state is None:
        raise RuntimeError(f"phase {phase!r} has not been declared (no approved allowlist/cost cap on record); call declare_phase() first")
    approved = set(phase_state["approved_custom_ids"])
    cost_cap_usd = float(phase_state["cost_cap_usd"])

    requests = _read_jsonl_requests(path)
    if len(requests) > TOGETHER_BATCH_MAX_REQUESTS:
        raise RuntimeError(f"batch has {len(requests)} requests, exceeds the documented Together Batch API limit of {TOGETHER_BATCH_MAX_REQUESTS}")
    manifest_rows = {row["custom_id"]: row for row in csv.DictReader(open(manifest_path, encoding="utf-8"))}

    custom_ids = [str(row.get("custom_id", "")) for row in requests]
    if not all(custom_ids) or len(custom_ids) != len(set(custom_ids)):
        raise RuntimeError("scientific-bakeoff JSONL custom_id values are missing or duplicated")

    bodies = [row.get("body") for row in requests]
    models = sorted({str(body.get("model", "")) for body in bodies if isinstance(body, dict)})
    denied_models = [m for m in models if m in DENIED_SCIENTIFIC_MODELS]
    if denied_models:
        raise RuntimeError(f"scientific bakeoff refuses these models outright: {denied_models}")
    unrecognized_models = [m for m in models if m not in ALLOWED_SCIENTIFIC_MODELS]
    if unrecognized_models:
        raise RuntimeError(f"scientific bakeoff only allows {sorted(ALLOWED_SCIENTIFIC_MODELS)}; observed {unrecognized_models}")
    if len(models) != 1:
        raise RuntimeError(f"scientific-bakeoff JSONL must target exactly one model per batch; observed {models}")
    declared_model = models[0]

    missing_manifest = [cid for cid in custom_ids if cid not in manifest_rows]
    if missing_manifest:
        raise RuntimeError(f"custom_id(s) in JSONL have no matching manifest row: {missing_manifest[:5]}")
    target_production_ids = [
        cid for cid in custom_ids if manifest_rows[cid].get("study_id") == TARGET_PRODUCTION_STUDY_ID
    ]
    if target_production_ids:
        raise RuntimeError(
            f"scientific bakeoff refuses target-production requests during phase {phase!r}: {target_production_ids[:5]}"
        )
    off_phase_ids = [cid for cid in custom_ids if manifest_rows[cid].get("request_stage") != phase]
    if off_phase_ids:
        raise RuntimeError(f"custom_id(s) do not carry request_stage={phase!r}: {off_phase_ids[:5]}")

    already = set(custom_ids) & (set(state.get("submitted_custom_ids", [])) | set(state.get("successful_custom_ids", [])))
    if already:
        raise RuntimeError(f"JSONL contains custom_id already recorded as submitted/successful: {sorted(already)[:5]}")

    if for_submit:
        not_approved = set(custom_ids) - approved
        if not_approved:
            raise RuntimeError(
                f"JSONL contains custom_id(s) not in phase {phase!r}'s approved allowlist "
                f"(no numeric cap alone admits these): {sorted(not_approved)[:5]}"
            )

    max_tokens = [int(body.get("max_tokens", 0) or 0) for body in bodies if isinstance(body, dict)]
    if len(max_tokens) != len(requests) or min(max_tokens) <= 0:
        raise RuntimeError("scientific-bakeoff JSONL has missing/invalid maximum output-token budgets")

    estimated_input_tokens = int(sum(sum(len(m.get("content", "")) for m in body["messages"]) for body in bodies) / 4)
    maximum_output_tokens = int(sum(max_tokens))
    prices = SCIENTIFIC_MODEL_PRICES_PER_1M_TOKENS[declared_model]
    batch_cost = estimated_input_tokens * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000

    cumulative_requests = int(phase_state["cumulative_requests"]) + len(requests)
    cumulative_cost = float(phase_state["cumulative_worst_case_cost_usd"]) + batch_cost
    if cumulative_cost > cost_cap_usd:
        raise RuntimeError(f"phase {phase!r} worst-case cost cap exceeded: ${cumulative_cost:.4f} > ${cost_cap_usd:.2f}")

    return {
        "phase": phase,
        "model": declared_model,
        "request_count": len(requests),
        "cumulative_phase_requests": cumulative_requests,
        "estimated_input_tokens": estimated_input_tokens,
        "maximum_output_tokens": maximum_output_tokens,
        "batch_worst_case_cost_usd": round(batch_cost, 6),
        "cumulative_phase_worst_case_cost_usd": round(cumulative_cost, 6),
        "phase_cost_cap_usd": cost_cap_usd,
        "jsonl": str(path),
        "state_path": str(state_path),
        "submission_allowed": True,
    }


def record_scientific_bakeoff_submission(
    guard: dict[str, Any],
    submit_result: dict[str, Any],
    *,
    state_path: Path = SCIENTIFIC_BAKEOFF_STATE_PATH,
) -> dict[str, Any]:
    """Automatic retries are never performed by this module -- record is called
    exactly once per real submit_batch() call, by the caller, never in a loop."""
    state = _load_state(state_path)
    requests = _read_jsonl_requests(Path(guard["jsonl"]))
    phase_state = state["phases"][guard["phase"]]
    phase_state["cumulative_requests"] = int(guard["cumulative_phase_requests"])
    phase_state["cumulative_worst_case_cost_usd"] = float(guard["cumulative_phase_worst_case_cost_usd"])
    phase_state.setdefault("submissions", []).append(
        {"model": guard["model"], "request_count": guard["request_count"], "batch_worst_case_cost_usd": guard["batch_worst_case_cost_usd"], "submit_result": submit_result}
    )
    submitted = list(dict.fromkeys([*state.get("submitted_custom_ids", []), *[str(r["custom_id"]) for r in requests]]))
    state["submitted_custom_ids"] = submitted
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return state
