"""Least-privilege submission guard for the CURRENT target G Wave-1
completion manifests (outputs/target_production/wave1_g_completion/).

Mirrors inference/orchinik_domain_confirmation_guard.py's architecture:
two hard-coded phases (target_g_wave1_completion_standard,
target_g_wave1_completion_consensus_a), each bound to exactly one canonical
frozen file by both exact path and exact SHA256 (manifest CSV, JSONL, and
the target G-v2 format-failure amendment this completion build depends on)
-- no caller-supplied --jsonl is ever trusted, so no substitution is
possible even with scientifically-identical content. Each phase enforces
its own frozen request count and cost cap.

This guard authorizes ONLY the current completion manifests (1,401 standard
+ 82 consensus_stage_a = 1,483 requests, $1.695305 total worst-case). It
does NOT authorize: Attempt 3, Consensus Stage B, F, Orchinik, or any other
target/development inference -- those have no phase entry here at all, so
they fail closed by construction (unknown phase name), not by a runtime
check that could be bypassed.
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
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
COMPLETION_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_completion"
G_V2_AMENDMENT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "g_wave1_v1_format_failure_amendment.json"
EXPECTED_G_V2_AMENDMENT_SHA256 = "1ec34ad251fd86b841b0639213c8f84d65126205b413b744456a554a28b207e7"
STATE_PATH = COMPLETION_ROOT / "target_g_completion_submission_state.json"

TARGET_STUDY_ID = "target"
COMPLETION_REQUEST_KEY_SUFFIX = "|fmt_v2"
COMBINED_TOTAL_REQUEST_COUNT = 1483
COMBINED_TOTAL_COST_CAP_USD = 1.695305

PHASES: dict[str, dict[str, Any]] = {
    "target_g_wave1_completion_standard": {
        "model": "google/gemma-4-31B-it",
        "request_stage": "standard",
        "manifest_path": COMPLETION_ROOT / "standard" / "request_manifest.csv",
        "jsonl_path": COMPLETION_ROOT / "standard" / "batch_input.jsonl",
        "manifest_sha256": "d78eb983bceaa97171973bba77c41318bd4dd0377bcd23e35ea36545ea711c56",
        "jsonl_sha256": "70603a986f0d22dae743a511189af5d9b504fed71211c5d6b769c6646148e174",
        "expected_request_count": 1401,
        "cost_cap_usd": 1.642233,
    },
}

# DISABLED, 2026-08-27: the offline public-instrument audit classified the
# Consensus sequence this 82-request manifest was built under as
# FAIL_MATERIAL_SEQUENCE_MISMATCH (item order was an unconstrained
# permutation instead of the public spec's "#3 always in the middle", and
# feedback was batched at the end of Stage B instead of interleaved
# immediately after each item -- see outputs/target_production/
# consensus_protocol_amendment.json). This manifest is archived for
# provenance but MUST NEVER be submitted -- kept out of PHASES entirely
# (not merely flagged) so it fails closed by unknown-phase-name, the same
# guarantee every other never-authorized scope in this pipeline gets.
# Superseded by inference.consensus_exact_guard's consensus_exact_step1
# phase (a completely different, chained multi-turn pipeline).
DISABLED_LEGACY_CONSENSUS_A_COMPLETION_PHASE = {
    "target_g_wave1_completion_consensus_a": {
        "model": "google/gemma-4-31B-it",
        "request_stage": "consensus_stage_a",
        "manifest_path": COMPLETION_ROOT / "consensus_stage_a" / "request_manifest.csv",
        "jsonl_path": COMPLETION_ROOT / "consensus_stage_a" / "batch_input.jsonl",
        "manifest_sha256": "4e6eaaf9bea584e37d30362164859a1fcd743e96d4d95435884b1c3bd8515f9a",
        "jsonl_sha256": "c57f8714a1697ec64317fc113c3341cf886f3c800dbcd0e0bd8d4e64c66959e9",
        "expected_request_count": 82,
        "cost_cap_usd": 0.053072,
        "disabled": True,
        "disabled_reason": "FAIL_MATERIAL_SEQUENCE_MISMATCH -- see outputs/target_production/consensus_protocol_amendment.json",
    },
}


class TargetGCompletionNotAuthorized(RuntimeError):
    """Fail-closed refusal: the requested phase, file, or content does not
    exactly match this module's hard-coded canonical scope."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_phase(phase: str) -> dict[str, Any]:
    spec = PHASES.get(phase)
    if spec is None:
        raise TargetGCompletionNotAuthorized(f"unknown target G completion phase {phase!r}; only {sorted(PHASES)} are recognized")
    return spec


def _verify_canonical_files_unaltered(spec: dict[str, Any]) -> None:
    manifest_path = spec["manifest_path"]
    jsonl_path = spec["jsonl_path"]
    if not manifest_path.exists():
        raise TargetGCompletionNotAuthorized(f"canonical manifest is missing: {manifest_path}")
    if not jsonl_path.exists():
        raise TargetGCompletionNotAuthorized(f"canonical jsonl is missing: {jsonl_path}")
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != spec["manifest_sha256"]:
        raise TargetGCompletionNotAuthorized(f"canonical manifest {manifest_path} SHA256 mismatch: expected {spec['manifest_sha256']}, got {actual_manifest_sha256}")
    actual_jsonl_sha256 = _sha256_file(jsonl_path)
    if actual_jsonl_sha256 != spec["jsonl_sha256"]:
        raise TargetGCompletionNotAuthorized(f"canonical jsonl {jsonl_path} SHA256 mismatch: expected {spec['jsonl_sha256']}, got {actual_jsonl_sha256}")
    if not G_V2_AMENDMENT_PATH.exists() or _sha256_file(G_V2_AMENDMENT_PATH) != EXPECTED_G_V2_AMENDMENT_SHA256:
        raise TargetGCompletionNotAuthorized(f"target G-v2 format-failure amendment missing or SHA256 mismatch: {G_V2_AMENDMENT_PATH}")


def _load_state(state_path: Path = STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "schema_version": "target_g_completion_submission_state_v1",
            "phases": {},
            "submitted_custom_ids": [],
            "successful_custom_ids": [],
            "combined_cumulative_requests": 0,
            "combined_cumulative_worst_case_cost_usd": 0.0,
        }
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "target_g_completion_submission_state_v1":
        raise RuntimeError(f"ambiguous existing target G completion state: {state_path}")
    return state


def declare_target_g_completion_phase(phase: str, *, state_path: Path = STATE_PATH) -> dict[str, Any]:
    """Register one of the two hard-coded phases. Verifies the canonical
    manifest/jsonl/amendment files on disk still match their frozen SHA256
    before ever registering anything."""
    spec = _resolved_phase(phase)
    _verify_canonical_files_unaltered(spec)

    manifest_rows = list(csv.DictReader(open(spec["manifest_path"], encoding="utf-8")))
    approved_custom_ids = {row["custom_id"] for row in manifest_rows}
    if len(approved_custom_ids) != spec["expected_request_count"]:
        raise TargetGCompletionNotAuthorized(
            f"canonical manifest for phase {phase!r} has {len(approved_custom_ids)} unique custom_ids, expected exactly {spec['expected_request_count']}"
        )

    state = _load_state(state_path)
    phases = state.setdefault("phases", {})
    existing = phases.get(phase)
    if existing is not None:
        if set(existing["approved_custom_ids"]) != approved_custom_ids:
            raise RuntimeError(f"phase {phase!r} allowlist on disk no longer matches the previously declared allowlist -- refusing to silently widen or narrow scope")
        return existing
    phases[phase] = {
        "model": spec["model"],
        "approved_custom_ids": sorted(approved_custom_ids),
        "cost_cap_usd": spec["cost_cap_usd"],
        "cumulative_requests": 0,
        "cumulative_worst_case_cost_usd": 0.0,
        "submissions": [],
    }
    COMPLETION_ROOT.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return phases[phase]


def target_g_completion_safety_guard(
    jsonl_path: Path | str,
    *,
    phase: str,
    for_submit: bool = True,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    """Fail-closed on any mismatch. --jsonl must be byte-identical to the
    ONE canonical file this phase is bound to -- both the exact resolved
    path and its SHA256 must match."""
    spec = _resolved_phase(phase)
    _verify_canonical_files_unaltered(spec)

    path = Path(jsonl_path).resolve()
    canonical_path = spec["jsonl_path"].resolve()
    if path != canonical_path:
        raise TargetGCompletionNotAuthorized(f"phase {phase!r} only accepts its own canonical jsonl ({canonical_path}); refusing substituted path {path}")

    file_size_bytes = path.stat().st_size
    if file_size_bytes > TOGETHER_BATCH_MAX_INPUT_FILE_BYTES:
        raise TargetGCompletionNotAuthorized(f"input file is {file_size_bytes / (1024*1024):.2f} MB, exceeds the Together Batch API limit")

    state = _load_state(state_path)
    phase_state = state.get("phases", {}).get(phase)
    if phase_state is None:
        raise RuntimeError(f"phase {phase!r} has not been declared; call declare_target_g_completion_phase() first")
    approved = set(phase_state["approved_custom_ids"])
    cost_cap_usd = float(phase_state["cost_cap_usd"])

    requests = _read_jsonl_requests(path)
    if len(requests) != spec["expected_request_count"]:
        raise TargetGCompletionNotAuthorized(f"phase {phase!r} expects exactly {spec['expected_request_count']} requests; jsonl has {len(requests)}")
    if len(requests) > TOGETHER_BATCH_MAX_REQUESTS:
        raise TargetGCompletionNotAuthorized(f"batch has {len(requests)} requests, exceeds the Together Batch API limit of {TOGETHER_BATCH_MAX_REQUESTS}")

    manifest_rows = {row["custom_id"]: row for row in csv.DictReader(open(spec["manifest_path"], encoding="utf-8"))}
    custom_ids = [str(row.get("custom_id", "")) for row in requests]
    if not all(custom_ids) or len(custom_ids) != len(set(custom_ids)):
        raise TargetGCompletionNotAuthorized("target G completion JSONL custom_id values are missing or duplicated")
    if set(custom_ids) != approved:
        missing = approved - set(custom_ids)
        extra = set(custom_ids) - approved
        raise TargetGCompletionNotAuthorized(
            f"phase {phase!r} requires exactly its full approved allowlist; missing={len(missing)} extra={len(extra)} (sample missing={sorted(missing)[:3]}, sample extra={sorted(extra)[:3]})"
        )

    missing_manifest = [cid for cid in custom_ids if cid not in manifest_rows]
    if missing_manifest:
        raise TargetGCompletionNotAuthorized(f"custom_id(s) in JSONL have no matching manifest row: {missing_manifest[:5]}")
    off_namespace_ids = [
        cid
        for cid in custom_ids
        if manifest_rows[cid].get("study_id") != TARGET_STUDY_ID
        or manifest_rows[cid].get("request_stage") != spec["request_stage"]
        or manifest_rows[cid].get("requested_model") != spec["model"]
        or not str(manifest_rows[cid].get("request_key", "")).endswith(COMPLETION_REQUEST_KEY_SUFFIX)
    ]
    if off_namespace_ids:
        raise TargetGCompletionNotAuthorized(
            f"custom_id(s) do not carry the expected target G completion namespace (study_id={TARGET_STUDY_ID!r}, "
            f"request_stage={spec['request_stage']!r}, requested_model={spec['model']!r}, "
            f"request_key ending {COMPLETION_REQUEST_KEY_SUFFIX!r}): {off_namespace_ids[:5]}"
        )

    bodies = [row.get("body") for row in requests]
    models = sorted({str(body.get("model", "")) for body in bodies if isinstance(body, dict)})
    if models != [spec["model"]]:
        raise TargetGCompletionNotAuthorized(f"phase {phase!r} only allows model {spec['model']!r}; observed {models}")

    already = set(custom_ids) & (set(state.get("submitted_custom_ids", [])) | set(state.get("successful_custom_ids", [])))
    if already:
        raise TargetGCompletionNotAuthorized(f"JSONL contains custom_id already recorded as submitted/successful (no retries): {sorted(already)[:5]}")

    if for_submit:
        not_approved = set(custom_ids) - approved
        if not_approved:
            raise TargetGCompletionNotAuthorized(f"JSONL contains custom_id(s) not in phase {phase!r}'s approved allowlist: {sorted(not_approved)[:5]}")

    max_tokens = [int(body.get("max_tokens", 0) or 0) for body in bodies if isinstance(body, dict)]
    if len(max_tokens) != len(requests) or min(max_tokens) <= 0:
        raise TargetGCompletionNotAuthorized("target G completion JSONL has missing/invalid maximum output-token budgets")

    estimated_input_tokens = int(sum(sum(len(m.get("content", "")) for m in body["messages"]) for body in bodies) / 4)
    maximum_output_tokens = int(sum(max_tokens))
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[spec["model"]]
    batch_cost = estimated_input_tokens * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000

    cumulative_requests = int(phase_state["cumulative_requests"]) + len(requests)
    cumulative_cost = float(phase_state["cumulative_worst_case_cost_usd"]) + batch_cost
    # cost_cap_usd is itself recorded to 6 decimal places; compare at that
    # same precision to avoid spurious refusal from float summation-order
    # noise below the 6th decimal (see the analogous fix in
    # inference/orchinik_domain_confirmation_guard.py).
    if round(cumulative_cost, 6) > cost_cap_usd:
        raise TargetGCompletionNotAuthorized(f"phase {phase!r} worst-case cost cap exceeded: ${cumulative_cost:.6f} > ${cost_cap_usd:.6f}")

    combined_cumulative_requests = int(state.get("combined_cumulative_requests", 0)) + len(requests)
    combined_cumulative_cost = float(state.get("combined_cumulative_worst_case_cost_usd", 0.0)) + batch_cost
    if combined_cumulative_requests > COMBINED_TOTAL_REQUEST_COUNT:
        raise TargetGCompletionNotAuthorized(f"combined target G completion request cap exceeded: {combined_cumulative_requests} > {COMBINED_TOTAL_REQUEST_COUNT}")
    if round(combined_cumulative_cost, 6) > COMBINED_TOTAL_COST_CAP_USD:
        raise TargetGCompletionNotAuthorized(f"combined target G completion worst-case cost cap exceeded: ${combined_cumulative_cost:.6f} > ${COMBINED_TOTAL_COST_CAP_USD:.6f}")

    return {
        "phase": phase,
        "model": spec["model"],
        "request_count": len(requests),
        "cumulative_phase_requests": cumulative_requests,
        "combined_cumulative_requests": combined_cumulative_requests,
        "estimated_input_tokens": estimated_input_tokens,
        "maximum_output_tokens": maximum_output_tokens,
        "batch_worst_case_cost_usd": round(batch_cost, 6),
        "cumulative_phase_worst_case_cost_usd": round(cumulative_cost, 6),
        "combined_cumulative_worst_case_cost_usd": round(combined_cumulative_cost, 6),
        "phase_cost_cap_usd": cost_cap_usd,
        "combined_total_request_count_cap": COMBINED_TOTAL_REQUEST_COUNT,
        "combined_total_cost_cap_usd": COMBINED_TOTAL_COST_CAP_USD,
        "jsonl": str(path),
        "state_path": str(state_path),
        "submission_allowed": True,
        "attempt_3_authorized": False,
        "consensus_stage_b_authorized": False,
        "automatic_follow_on_inference_authorized": False,
    }


def record_target_g_completion_submission(
    guard: dict[str, Any],
    submit_result: dict[str, Any],
    *,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    """Called exactly once per real submit_batch() call, by the caller,
    never in a loop -- this module performs no automatic retries or
    follow-on submissions itself."""
    state = _load_state(state_path)
    requests = _read_jsonl_requests(Path(guard["jsonl"]))
    phase_state = state["phases"][guard["phase"]]
    phase_state["cumulative_requests"] = int(guard["cumulative_phase_requests"])
    phase_state["cumulative_worst_case_cost_usd"] = float(guard["cumulative_phase_worst_case_cost_usd"])
    phase_state.setdefault("submissions", []).append(
        {"model": guard["model"], "request_count": guard["request_count"], "batch_worst_case_cost_usd": guard["batch_worst_case_cost_usd"], "submit_result": submit_result}
    )
    state["combined_cumulative_requests"] = int(guard["combined_cumulative_requests"])
    state["combined_cumulative_worst_case_cost_usd"] = float(guard["combined_cumulative_worst_case_cost_usd"])
    submitted = list(dict.fromkeys([*state.get("submitted_custom_ids", []), *[str(r["custom_id"]) for r in requests]]))
    state["submitted_custom_ids"] = submitted
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return state
