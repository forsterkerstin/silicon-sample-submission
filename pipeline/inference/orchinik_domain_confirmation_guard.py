"""Least-privilege submission guard for the Orchinik G-vs-DeepSeek domain-
confirmation v2 manifests (outputs/domain_validation/
orchinik_g_domain_confirmation_v2/).

Built per the pre-submission audit's finding that this path currently falls
through to together_batch.py's unguarded generic submission path (it is
under neither TINY_SMOKE_ROOT, SCIENTIFIC_BAKEOFF_ROOT, nor
TARGET_PRODUCTION_ROOT). Unlike scientific_bakeoff_guard/target_production_guard,
which accept a caller-supplied --manifest and an approved_custom_ids allowlist
declared at runtime, this guard trusts NOTHING supplied by the caller except
the phase name: each of the two phases is bound, as a hard-coded constant, to
exactly one canonical frozen file (exact path AND exact SHA256 for both the
manifest CSV and the JSONL), one exact model, one exact request count, and one
exact worst-case cost cap. There is no declare-time parameterization and no
way to widen a phase's scope after the fact -- if the file at the canonical
path does not hash to the frozen constant, the guard fails closed before ever
reading its contents as trusted input.

Scope is exactly google/gemma-4-31B-it (2,545 requests) +
deepseek-ai/DeepSeek-V4-Pro-0813 (2,545 requests) = 5,090 requests, total
worst-case cost $29.835322. This module authorizes nothing else: not
retries, not additional models, not altered manifests, not target G/F, not
calibration, not gamma validation, not the withheld all-three-condition
Orchinik simulation, not any other domain validation. record_*_submission is
called exactly once per real submit_batch() call by the CLI caller -- this
module never resubmits or chains a follow-on call itself.
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
ORCHINIK_V2_ROOT = PIPELINE_ROOT / "outputs" / "domain_validation" / "orchinik_g_domain_confirmation_v2"
SERVING_AMENDMENT_PATH = ORCHINIK_V2_ROOT / "serving_amendment.json"
SERVING_AMENDMENT_SHA256_PATH = ORCHINIK_V2_ROOT / "serving_amendment.sha256.txt"
STATE_PATH = ORCHINIK_V2_ROOT / "orchinik_domain_confirmation_submission_state.json"

EXPECTED_SERVING_AMENDMENT_SHA256 = "835661585f4dae5a5f817746b14c88c125593d7d2d4290b8ebba7e98e463dd4f"
ORCHINIK_STUDY_ID = "orchinik2024_bovitz"
ORCHINIK_REQUEST_STAGE = "domain_confirmation"
ORCHINIK_V2_REQUEST_KEY_SUFFIX = "|fmt_v2"

COMBINED_TOTAL_REQUEST_COUNT = 5090
COMBINED_TOTAL_COST_CAP_USD = 29.835322

# Each phase is bound, as an immutable constant, to exactly one canonical
# frozen file. No declare-time parameter can widen or redirect this.
PHASES: dict[str, dict[str, Any]] = {
    "orchinik_g_domain_confirmation_v2_gemma": {
        "model": "google/gemma-4-31B-it",
        "manifest_path": ORCHINIK_V2_ROOT / "google_gemma-4-31B-it" / "request_manifest.csv",
        "jsonl_path": ORCHINIK_V2_ROOT / "google_gemma-4-31B-it" / "batch_input.jsonl",
        "manifest_sha256": "97f8f351f1229883e57871845cefb7ebc5a9a44a1071e026c4e8a55637a5c60a",
        "jsonl_sha256": "612b2ec9ea4854cf9f455a3e81cc6360293b7d36df419e17aec20f1b7e546f95",
        "expected_request_count": 2545,
        "cost_cap_usd": 3.699383,
    },
    "orchinik_g_domain_confirmation_v2_deepseek": {
        "model": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "manifest_path": ORCHINIK_V2_ROOT / "deepseek-ai_DeepSeek-V4-Pro-0813" / "request_manifest.csv",
        "jsonl_path": ORCHINIK_V2_ROOT / "deepseek-ai_DeepSeek-V4-Pro-0813" / "batch_input.jsonl",
        "manifest_sha256": "4cd0bb37e6de4feeb05b04f8851ecb773b7ca9570c4e194305a3aafeec6a8ba6",
        "jsonl_sha256": "8965f570870f4548bf870ac952b3ace24fef478dec7e6d11e7d96bbbd6358fe0",
        "expected_request_count": 2545,
        "cost_cap_usd": 26.135939,
    },
}


class OrchinikDomainConfirmationNotAuthorized(RuntimeError):
    """Fail-closed refusal: the requested phase, file, or content does not
    exactly match this module's hard-coded canonical scope."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_phase(phase: str) -> dict[str, Any]:
    spec = PHASES.get(phase)
    if spec is None:
        raise OrchinikDomainConfirmationNotAuthorized(f"unknown Orchinik domain-confirmation phase {phase!r}; only {sorted(PHASES)} are recognized")
    return spec


def _verify_canonical_files_unaltered(spec: dict[str, Any]) -> None:
    manifest_path = spec["manifest_path"]
    jsonl_path = spec["jsonl_path"]
    if not manifest_path.exists():
        raise OrchinikDomainConfirmationNotAuthorized(f"canonical manifest is missing: {manifest_path}")
    if not jsonl_path.exists():
        raise OrchinikDomainConfirmationNotAuthorized(f"canonical jsonl is missing: {jsonl_path}")
    actual_manifest_sha256 = _sha256_file(manifest_path)
    if actual_manifest_sha256 != spec["manifest_sha256"]:
        raise OrchinikDomainConfirmationNotAuthorized(f"canonical manifest {manifest_path} SHA256 mismatch: expected {spec['manifest_sha256']}, got {actual_manifest_sha256}")
    actual_jsonl_sha256 = _sha256_file(jsonl_path)
    if actual_jsonl_sha256 != spec["jsonl_sha256"]:
        raise OrchinikDomainConfirmationNotAuthorized(f"canonical jsonl {jsonl_path} SHA256 mismatch: expected {spec['jsonl_sha256']}, got {actual_jsonl_sha256}")
    if not SERVING_AMENDMENT_SHA256_PATH.exists():
        raise OrchinikDomainConfirmationNotAuthorized(f"serving amendment hash file is missing: {SERVING_AMENDMENT_SHA256_PATH}")
    actual_amendment_sha256 = SERVING_AMENDMENT_SHA256_PATH.read_text(encoding="utf-8").strip()
    if actual_amendment_sha256 != EXPECTED_SERVING_AMENDMENT_SHA256:
        raise OrchinikDomainConfirmationNotAuthorized(f"serving amendment SHA256 mismatch: expected {EXPECTED_SERVING_AMENDMENT_SHA256}, got {actual_amendment_sha256}")
    if not SERVING_AMENDMENT_PATH.exists() or _sha256_file(SERVING_AMENDMENT_PATH) != EXPECTED_SERVING_AMENDMENT_SHA256:
        raise OrchinikDomainConfirmationNotAuthorized(f"serving amendment file content does not match its own recorded SHA256: {SERVING_AMENDMENT_PATH}")


def _load_state(state_path: Path = STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "schema_version": "orchinik_domain_confirmation_submission_state_v1",
            "phases": {},
            "submitted_custom_ids": [],
            "successful_custom_ids": [],
            "combined_cumulative_requests": 0,
            "combined_cumulative_worst_case_cost_usd": 0.0,
        }
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "orchinik_domain_confirmation_submission_state_v1":
        raise RuntimeError(f"ambiguous existing Orchinik domain-confirmation state: {state_path}")
    return state


def declare_orchinik_domain_confirmation_phase(phase: str, *, state_path: Path = STATE_PATH) -> dict[str, Any]:
    """Register one of the two hard-coded phases. Verifies the canonical
    manifest/jsonl/serving-amendment files on disk still match their frozen
    SHA256 before ever registering anything -- an on-disk edit to a
    "canonical" file is refused here, not silently trusted."""
    spec = _resolved_phase(phase)
    _verify_canonical_files_unaltered(spec)

    manifest_rows = list(csv.DictReader(open(spec["manifest_path"], encoding="utf-8")))
    approved_custom_ids = {row["custom_id"] for row in manifest_rows}
    if len(approved_custom_ids) != spec["expected_request_count"]:
        raise OrchinikDomainConfirmationNotAuthorized(
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
    ORCHINIK_V2_ROOT.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return phases[phase]


def orchinik_domain_confirmation_safety_guard(
    jsonl_path: Path | str,
    *,
    phase: str,
    for_submit: bool = True,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    """Fail-closed on any mismatch. --jsonl must be byte-identical to the
    ONE canonical file this phase is bound to -- both the exact resolved
    path and its SHA256 must match; there is no way to submit a different
    file, even one with identical scientific content, under this phase."""
    spec = _resolved_phase(phase)
    _verify_canonical_files_unaltered(spec)

    path = Path(jsonl_path).resolve()
    canonical_path = spec["jsonl_path"].resolve()
    if path != canonical_path:
        raise OrchinikDomainConfirmationNotAuthorized(f"phase {phase!r} only accepts its own canonical jsonl ({canonical_path}); refusing substituted path {path}")

    file_size_bytes = path.stat().st_size
    if file_size_bytes > TOGETHER_BATCH_MAX_INPUT_FILE_BYTES:
        raise OrchinikDomainConfirmationNotAuthorized(f"input file is {file_size_bytes / (1024*1024):.2f} MB, exceeds the Together Batch API limit")

    state = _load_state(state_path)
    phase_state = state.get("phases", {}).get(phase)
    if phase_state is None:
        raise RuntimeError(f"phase {phase!r} has not been declared; call declare_orchinik_domain_confirmation_phase() first")
    approved = set(phase_state["approved_custom_ids"])
    cost_cap_usd = float(phase_state["cost_cap_usd"])

    requests = _read_jsonl_requests(path)
    if len(requests) != spec["expected_request_count"]:
        raise OrchinikDomainConfirmationNotAuthorized(f"phase {phase!r} expects exactly {spec['expected_request_count']} requests; jsonl has {len(requests)}")
    if len(requests) > TOGETHER_BATCH_MAX_REQUESTS:
        raise OrchinikDomainConfirmationNotAuthorized(f"batch has {len(requests)} requests, exceeds the Together Batch API limit of {TOGETHER_BATCH_MAX_REQUESTS}")

    manifest_rows = {row["custom_id"]: row for row in csv.DictReader(open(spec["manifest_path"], encoding="utf-8"))}
    custom_ids = [str(row.get("custom_id", "")) for row in requests]
    if not all(custom_ids) or len(custom_ids) != len(set(custom_ids)):
        raise OrchinikDomainConfirmationNotAuthorized("Orchinik domain-confirmation JSONL custom_id values are missing or duplicated")
    if set(custom_ids) != approved:
        missing = approved - set(custom_ids)
        extra = set(custom_ids) - approved
        raise OrchinikDomainConfirmationNotAuthorized(
            f"phase {phase!r} requires exactly its full approved allowlist; missing={len(missing)} extra={len(extra)} (sample missing={sorted(missing)[:3]}, sample extra={sorted(extra)[:3]})"
        )

    missing_manifest = [cid for cid in custom_ids if cid not in manifest_rows]
    if missing_manifest:
        raise OrchinikDomainConfirmationNotAuthorized(f"custom_id(s) in JSONL have no matching manifest row: {missing_manifest[:5]}")
    off_namespace_ids = [
        cid
        for cid in custom_ids
        if manifest_rows[cid].get("study_id") != ORCHINIK_STUDY_ID
        or manifest_rows[cid].get("request_stage") != ORCHINIK_REQUEST_STAGE
        or manifest_rows[cid].get("requested_model") != spec["model"]
        or not str(manifest_rows[cid].get("request_key", "")).endswith(ORCHINIK_V2_REQUEST_KEY_SUFFIX)
    ]
    if off_namespace_ids:
        raise OrchinikDomainConfirmationNotAuthorized(
            f"custom_id(s) do not carry the expected Orchinik v2 namespace (study_id={ORCHINIK_STUDY_ID!r}, "
            f"request_stage={ORCHINIK_REQUEST_STAGE!r}, requested_model={spec['model']!r}, "
            f"request_key ending {ORCHINIK_V2_REQUEST_KEY_SUFFIX!r}): {off_namespace_ids[:5]}"
        )

    bodies = [row.get("body") for row in requests]
    models = sorted({str(body.get("model", "")) for body in bodies if isinstance(body, dict)})
    if models != [spec["model"]]:
        raise OrchinikDomainConfirmationNotAuthorized(f"phase {phase!r} only allows model {spec['model']!r}; observed {models}")

    already = set(custom_ids) & (set(state.get("submitted_custom_ids", [])) | set(state.get("successful_custom_ids", [])))
    if already:
        raise OrchinikDomainConfirmationNotAuthorized(f"JSONL contains custom_id already recorded as submitted/successful (no retries): {sorted(already)[:5]}")

    if for_submit:
        not_approved = set(custom_ids) - approved
        if not_approved:
            raise OrchinikDomainConfirmationNotAuthorized(f"JSONL contains custom_id(s) not in phase {phase!r}'s approved allowlist: {sorted(not_approved)[:5]}")

    max_tokens = [int(body.get("max_tokens", 0) or 0) for body in bodies if isinstance(body, dict)]
    if len(max_tokens) != len(requests) or min(max_tokens) <= 0:
        raise OrchinikDomainConfirmationNotAuthorized("Orchinik domain-confirmation JSONL has missing/invalid maximum output-token budgets")

    estimated_input_tokens = int(sum(sum(len(m.get("content", "")) for m in body["messages"]) for body in bodies) / 4)
    maximum_output_tokens = int(sum(max_tokens))
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[spec["model"]]
    batch_cost = estimated_input_tokens * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000

    cumulative_requests = int(phase_state["cumulative_requests"]) + len(requests)
    cumulative_cost = float(phase_state["cumulative_worst_case_cost_usd"]) + batch_cost
    # cost_cap_usd is itself a value recorded to 6 decimal places (see
    # scripts/build_orchinik_g_domain_confirmation_manifest_v2.py's
    # round(..., 6)); comparing at that same precision avoids a spurious
    # refusal from float summation-order noise below the 6th decimal
    # (observed: a live re-sum landing at 26.135939280000002 against a
    # frozen cap of 26.135939 -- equal to 6dp, not actually over cap).
    if round(cumulative_cost, 6) > cost_cap_usd:
        raise OrchinikDomainConfirmationNotAuthorized(f"phase {phase!r} worst-case cost cap exceeded: ${cumulative_cost:.6f} > ${cost_cap_usd:.6f}")

    combined_cumulative_requests = int(state.get("combined_cumulative_requests", 0)) + len(requests)
    combined_cumulative_cost = float(state.get("combined_cumulative_worst_case_cost_usd", 0.0)) + batch_cost
    if combined_cumulative_requests > COMBINED_TOTAL_REQUEST_COUNT:
        raise OrchinikDomainConfirmationNotAuthorized(f"combined Orchinik domain-confirmation request cap exceeded: {combined_cumulative_requests} > {COMBINED_TOTAL_REQUEST_COUNT}")
    if round(combined_cumulative_cost, 6) > COMBINED_TOTAL_COST_CAP_USD:
        raise OrchinikDomainConfirmationNotAuthorized(f"combined Orchinik domain-confirmation worst-case cost cap exceeded: ${combined_cumulative_cost:.6f} > ${COMBINED_TOTAL_COST_CAP_USD:.6f}")

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
        "automatic_follow_on_inference_authorized": False,
    }


def record_orchinik_domain_confirmation_submission(
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
