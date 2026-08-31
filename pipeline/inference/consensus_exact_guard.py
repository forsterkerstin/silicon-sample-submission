"""Least-privilege submission guard for the benchmark-exact Consensus
pipeline (outputs/target_production/consensus_exact/). Mirrors
inference/target_g_completion_guard.py's architecture: each phase is bound
to exactly one canonical frozen file by exact path AND SHA256, one exact
request count, one model, one cost cap. No caller-supplied --jsonl is ever
trusted, so no substitution is possible.

All four phases now exist: consensus_exact_step1, consensus_exact_step2,
consensus_exact_step3, and consensus_exact_outcomes (1,000 requests each,
each built only once the immediately preceding stage's real retrieved
responses fully resolved first-valid -- see
inference/consensus_benchmark_exact.py). This module will need extending
(a new PHASES entry with real pinned hashes) for any future stage, rather
than ever accepting a caller-supplied hash.

This guard authorizes ONLY consensus_exact_step1/step2/step3/outcomes. It
does NOT authorize: the old 82-request Consensus-A completion manifest
(explicitly disabled -- see is_disabled_legacy_consensus_jsonl), standard
target-G completion, Orchinik, or any other target/development inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from inference.together_batch import (
    SMOKE_MODEL_PRICES_PER_1M_TOKENS,
    TOGETHER_BATCH_MAX_INPUT_FILE_BYTES,
    TOGETHER_BATCH_MAX_REQUESTS,
    _read_jsonl_requests,
    _under_path,
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
CONSENSUS_EXACT_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_exact"
STEP1_DIR = CONSENSUS_EXACT_ROOT / "step1"
STEP2_DIR = CONSENSUS_EXACT_ROOT / "step2"
STEP3_DIR = CONSENSUS_EXACT_ROOT / "step3"
OUTCOMES_DIR = CONSENSUS_EXACT_ROOT / "outcomes"
STATE_PATH = CONSENSUS_EXACT_ROOT / "consensus_exact_submission_state.json"

# The legacy (FAIL_MATERIAL_SEQUENCE_MISMATCH) 82-request Consensus-A
# completion manifest, frozen and archived but never submittable again.
LEGACY_CONSENSUS_A_COMPLETION_JSONL = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion" / "consensus_stage_a" / "batch_input.jsonl"

TARGET_STUDY_ID = "target"
CONDITION_ID = "Consensus"

PHASES: dict[str, dict[str, Any]] = {
    "consensus_exact_step1": {
        "model": "google/gemma-4-31B-it",
        "request_stage": "consensus_exact_step1",
        "manifest_path": STEP1_DIR / "request_manifest.csv",
        "jsonl_path": STEP1_DIR / "batch_input.jsonl",
        "manifest_sha256": "3f3e49611a8c51489fb6c227508e020e44fad74710470675e4076a0d06d3eb4f",
        "jsonl_sha256": "641c510993b152ef10c28569ec063f74ddf4bc67f583ba60f5e47c88c84b384d",
        "expected_request_count": 1000,
        "cost_cap_usd": 0.622864,
    },
    "consensus_exact_step2": {
        "model": "google/gemma-4-31B-it",
        "request_stage": "consensus_exact_step2",
        "manifest_path": STEP2_DIR / "request_manifest.csv",
        "jsonl_path": STEP2_DIR / "batch_input.jsonl",
        "manifest_sha256": "f57bfaf2bfe6c2613937a38394e58ef0712304c6f6785a60666103ae2250625e",
        "jsonl_sha256": "07e4eb4104c7c99d2a1731bc39e580fbc90f19fbbc8226fee1e52c5e85df86d6",
        "expected_request_count": 1000,
        "cost_cap_usd": 0.680756,
    },
    "consensus_exact_step3": {
        "model": "google/gemma-4-31B-it",
        "request_stage": "consensus_exact_step3",
        "manifest_path": STEP3_DIR / "request_manifest.csv",
        "jsonl_path": STEP3_DIR / "batch_input.jsonl",
        "manifest_sha256": "997096c8dfdcbdc9d609b99db2afd2d53d6be3edbffdca23b586de7c2a7df801",
        "jsonl_sha256": "4aabb951701f38d8567366f80320b4f360325f9d5eaf485653f18c83bbf20a8a",
        "expected_request_count": 1000,
        "cost_cap_usd": 0.73728,
    },
    "consensus_exact_outcomes": {
        "model": "google/gemma-4-31B-it",
        "request_stage": "consensus_exact_outcomes",
        "manifest_path": OUTCOMES_DIR / "request_manifest.csv",
        "jsonl_path": OUTCOMES_DIR / "batch_input.jsonl",
        "manifest_sha256": "9feb7b78d9e2b1aeb80360270b242aeabc2f54549aa516ff84c0c156c631f571",
        "jsonl_sha256": "7b535527ba6647a36f13be547034bc896caf289568fbdd527fcff30c465b70c3",
        "expected_request_count": 1000,
        "cost_cap_usd": 1.277945,
    },
}


class ConsensusExactNotAuthorized(RuntimeError):
    """Fail-closed refusal: the requested phase, file, or content does not
    exactly match this module's hard-coded canonical scope."""


def is_disabled_legacy_consensus_jsonl(jsonl_path: Path | str) -> bool:
    """The old (FAIL_MATERIAL_SEQUENCE_MISMATCH) 82-request Consensus-A
    completion manifest -- explicitly, permanently disabled. Never
    submittable through this or any guard."""
    return Path(jsonl_path).resolve() == LEGACY_CONSENSUS_A_COMPLETION_JSONL.resolve()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_phase(phase: str) -> dict[str, Any]:
    spec = PHASES.get(phase)
    if spec is None:
        raise ConsensusExactNotAuthorized(f"unknown Consensus-exact phase {phase!r}; only {sorted(PHASES)} are recognized")
    return spec


def _verify_canonical_files_unaltered(spec: dict[str, Any]) -> None:
    manifest_path = spec["manifest_path"]
    jsonl_path = spec["jsonl_path"]
    if not manifest_path.exists():
        raise ConsensusExactNotAuthorized(f"canonical manifest is missing: {manifest_path}")
    if not jsonl_path.exists():
        raise ConsensusExactNotAuthorized(f"canonical jsonl is missing: {jsonl_path}")
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != spec["manifest_sha256"]:
        raise ConsensusExactNotAuthorized(f"canonical manifest {manifest_path} SHA256 mismatch: expected {spec['manifest_sha256']}, got {actual_manifest_sha256}")
    actual_jsonl_sha256 = _sha256_file(jsonl_path)
    if actual_jsonl_sha256 != spec["jsonl_sha256"]:
        raise ConsensusExactNotAuthorized(f"canonical jsonl {jsonl_path} SHA256 mismatch: expected {spec['jsonl_sha256']}, got {actual_jsonl_sha256}")


def _load_state(state_path: Path = STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {"schema_version": "consensus_exact_submission_state_v1", "phases": {}, "submitted_custom_ids": [], "successful_custom_ids": []}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "consensus_exact_submission_state_v1":
        raise RuntimeError(f"ambiguous existing Consensus-exact state: {state_path}")
    return state


def declare_consensus_exact_phase(phase: str, *, state_path: Path = STATE_PATH) -> dict[str, Any]:
    spec = _resolved_phase(phase)
    _verify_canonical_files_unaltered(spec)

    manifest_rows = list(csv.DictReader(open(spec["manifest_path"], encoding="utf-8")))
    approved_custom_ids = {row["custom_id"] for row in manifest_rows}
    if len(approved_custom_ids) != spec["expected_request_count"]:
        raise ConsensusExactNotAuthorized(f"canonical manifest for phase {phase!r} has {len(approved_custom_ids)} unique custom_ids, expected exactly {spec['expected_request_count']}")

    state = _load_state(state_path)
    phases = state.setdefault("phases", {})
    existing = phases.get(phase)
    if existing is not None:
        if set(existing["approved_custom_ids"]) != approved_custom_ids:
            raise RuntimeError(f"phase {phase!r} allowlist on disk no longer matches the previously declared allowlist")
        return existing
    phases[phase] = {
        "model": spec["model"],
        "approved_custom_ids": sorted(approved_custom_ids),
        "cost_cap_usd": spec["cost_cap_usd"],
        "cumulative_requests": 0,
        "cumulative_worst_case_cost_usd": 0.0,
        "submissions": [],
    }
    CONSENSUS_EXACT_ROOT.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return phases[phase]


def consensus_exact_safety_guard(jsonl_path: Path | str, *, phase: str, for_submit: bool = True, state_path: Path = STATE_PATH) -> dict[str, Any]:
    if is_disabled_legacy_consensus_jsonl(jsonl_path):
        raise ConsensusExactNotAuthorized(
            "this is the legacy (FAIL_MATERIAL_SEQUENCE_MISMATCH) Consensus-A completion manifest -- permanently disabled, "
            "never submittable; use the benchmark-exact pipeline instead"
        )
    spec = _resolved_phase(phase)
    _verify_canonical_files_unaltered(spec)

    path = Path(jsonl_path).resolve()
    canonical_path = spec["jsonl_path"].resolve()
    if path != canonical_path:
        raise ConsensusExactNotAuthorized(f"phase {phase!r} only accepts its own canonical jsonl ({canonical_path}); refusing substituted path {path}")

    file_size_bytes = path.stat().st_size
    if file_size_bytes > TOGETHER_BATCH_MAX_INPUT_FILE_BYTES:
        raise ConsensusExactNotAuthorized(f"input file is {file_size_bytes / (1024*1024):.2f} MB, exceeds the Together Batch API limit")

    state = _load_state(state_path)
    phase_state = state.get("phases", {}).get(phase)
    if phase_state is None:
        raise RuntimeError(f"phase {phase!r} has not been declared; call declare_consensus_exact_phase() first")
    approved = set(phase_state["approved_custom_ids"])
    cost_cap_usd = float(phase_state["cost_cap_usd"])

    requests = _read_jsonl_requests(path)
    if len(requests) != spec["expected_request_count"]:
        raise ConsensusExactNotAuthorized(f"phase {phase!r} expects exactly {spec['expected_request_count']} requests; jsonl has {len(requests)}")
    if len(requests) > TOGETHER_BATCH_MAX_REQUESTS:
        raise ConsensusExactNotAuthorized(f"batch has {len(requests)} requests, exceeds the Together Batch API limit of {TOGETHER_BATCH_MAX_REQUESTS}")

    manifest_rows = {row["custom_id"]: row for row in csv.DictReader(open(spec["manifest_path"], encoding="utf-8"))}
    custom_ids = [str(row.get("custom_id", "")) for row in requests]
    if not all(custom_ids) or len(custom_ids) != len(set(custom_ids)):
        raise ConsensusExactNotAuthorized("Consensus-exact JSONL custom_id values are missing or duplicated")
    if set(custom_ids) != approved:
        missing = approved - set(custom_ids)
        extra = set(custom_ids) - approved
        raise ConsensusExactNotAuthorized(f"phase {phase!r} requires exactly its full approved allowlist; missing={len(missing)} extra={len(extra)}")

    missing_manifest = [cid for cid in custom_ids if cid not in manifest_rows]
    if missing_manifest:
        raise ConsensusExactNotAuthorized(f"custom_id(s) in JSONL have no matching manifest row: {missing_manifest[:5]}")
    off_namespace_ids = [
        cid
        for cid in custom_ids
        if manifest_rows[cid].get("study_id") != TARGET_STUDY_ID
        or manifest_rows[cid].get("condition_id") != CONDITION_ID
        or manifest_rows[cid].get("request_stage") != spec["request_stage"]
        or manifest_rows[cid].get("requested_model") != spec["model"]
    ]
    if off_namespace_ids:
        raise ConsensusExactNotAuthorized(f"custom_id(s) do not carry the expected Consensus-exact namespace: {off_namespace_ids[:5]}")

    bodies = [row.get("body") for row in requests]
    models = sorted({str(body.get("model", "")) for body in bodies if isinstance(body, dict)})
    if models != [spec["model"]]:
        raise ConsensusExactNotAuthorized(f"phase {phase!r} only allows model {spec['model']!r}; observed {models}")

    already = set(custom_ids) & (set(state.get("submitted_custom_ids", [])) | set(state.get("successful_custom_ids", [])))
    if already:
        raise ConsensusExactNotAuthorized(f"JSONL contains custom_id already recorded as submitted/successful (no retries): {sorted(already)[:5]}")

    if for_submit:
        not_approved = set(custom_ids) - approved
        if not_approved:
            raise ConsensusExactNotAuthorized(f"JSONL contains custom_id(s) not in phase {phase!r}'s approved allowlist: {sorted(not_approved)[:5]}")

    max_tokens = [int(body.get("max_tokens", 0) or 0) for body in bodies if isinstance(body, dict)]
    if len(max_tokens) != len(requests) or min(max_tokens) <= 0:
        raise ConsensusExactNotAuthorized("Consensus-exact JSONL has missing/invalid maximum output-token budgets")

    estimated_input_tokens = int(sum(sum(len(m.get("content", "")) for m in body["messages"]) for body in bodies) / 4)
    maximum_output_tokens = int(sum(max_tokens))
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[spec["model"]]
    batch_cost = estimated_input_tokens * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000

    cumulative_requests = int(phase_state["cumulative_requests"]) + len(requests)
    cumulative_cost = float(phase_state["cumulative_worst_case_cost_usd"]) + batch_cost
    if round(cumulative_cost, 6) > cost_cap_usd:
        raise ConsensusExactNotAuthorized(f"phase {phase!r} worst-case cost cap exceeded: ${cumulative_cost:.6f} > ${cost_cap_usd:.6f}")

    return {
        "phase": phase,
        "model": spec["model"],
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
        "automatic_follow_on_inference_authorized": False,
    }


def record_consensus_exact_submission(guard: dict[str, Any], submit_result: dict[str, Any], *, state_path: Path = STATE_PATH) -> dict[str, Any]:
    state = _load_state(state_path)
    requests = _read_jsonl_requests(Path(guard["jsonl"]))
    phase_state = state["phases"][guard["phase"]]
    phase_state["cumulative_requests"] = int(guard["cumulative_phase_requests"])
    phase_state["cumulative_worst_case_cost_usd"] = float(guard["cumulative_phase_worst_case_cost_usd"])
    phase_state.setdefault("submissions", []).append({"model": guard["model"], "request_count": guard["request_count"], "batch_worst_case_cost_usd": guard["batch_worst_case_cost_usd"], "submit_result": submit_result})
    submitted = list(dict.fromkeys([*state.get("submitted_custom_ids", []), *[str(r["custom_id"]) for r in requests]]))
    state["submitted_custom_ids"] = submitted
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return state


def is_consensus_exact_jsonl(jsonl_path: Path | str) -> bool:
    return _under_path(Path(jsonl_path), CONSENSUS_EXACT_ROOT)
