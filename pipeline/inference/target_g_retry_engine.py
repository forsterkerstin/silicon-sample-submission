"""Bounded 3-attempt production retry engine for target G Wave-1.

Engineering only: this module NEVER reads or exposes scientific response
VALUES (parsed questionnaire answers) -- only provider-delivery status
(schema_valid / schema_invalid / provider_error / not_attempted) and
identity/provenance bookkeeping. Retry eligibility is determined solely by
that status, never by outcome/condition/demographic/plausibility/mean/ATE.

Policy (frozen, see outputs/target_production/target_g_retry_policy.json):
    MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY = 3
    FIRST_VALID_RESPONSE_WINS = true (an identity with a schema-valid
        production response is locked -- never regenerated, never re-chosen)
    SMOKE_IS_NOT_PRODUCTION = true (engineering-smoke responses can never
        become a selected production response, and count toward zero
        attempts for the 10 identities the smoke happened to also cover)

"Intended production identity" = donor/condition/stage triple, independent
of attempt number and format version: derived by stripping both the
`|fmt_v\\d+` and `|replicate_\\d+` suffixes from a request_key. `replicate_id`
is repurposed as the production-attempt discriminator for target G Wave-1
(this study never used replicate_id for a legitimate multiple-draws-per-cell
design) -- attempt N always renders with replicate_id=N, which changes ONLY
the request_key/seed (hence custom_id and stochastic draw), never donor,
condition, persona, stimulus, model, or sampling parameters. For Consensus
Stage-A specifically, item ORDER is also a function of replicate_id in the
frozen prompt compiler (consensus_interaction_order); this module always
passes order_replicate_id=1 for every attempt, so the questionnaire order
administered to a donor is identical across all of that donor's attempts,
matching what a real "resend the same unanswered survey" retry means.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PIPELINE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))

import pandas as pd  # noqa: E402
import survey_content as sc  # noqa: E402

import score_target_g_wave1_v2_replacement as attempt1_scorer  # noqa: E402
from inference.prompts import (  # noqa: E402
    CONSENSUS_STAGE_A_OUTCOME_ID,
    PROMPT_COMPILER_VERSION,
    build_g_consensus_stage_a_prompt_render,
    build_g_prompt_render,
    schema_hash,
)
from inference.request_logging import seed_from_request_key  # noqa: E402
from inference.together_batch import (  # noqa: E402
    BatchRequest,
    G_MASTER_PATH,
    _profile_dict,
    _render_prompt_hash,
    compute_engine_config_hash,
    custom_id_from_request_key,
)

WAVE1_V2_BY_STAGE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "by_stage"
WAVE1_V2_SUBMISSION_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "submission"

STAGES = ("standard", "consensus_stage_a")
MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY = 3
SCHEMA_VALID = "SCHEMA_VALID"
SCHEMA_INVALID = "SCHEMA_INVALID"
PROVIDER_ERROR = "PROVIDER_ERROR"
NOT_ATTEMPTED = "NOT_ATTEMPTED"
TERMINAL_VALID_STATUS = SCHEMA_VALID
RETRYABLE_STATUSES = {SCHEMA_INVALID, PROVIDER_ERROR, NOT_ATTEMPTED}
EXPECTED_UNIVERSE_SIZE = 17000

_FMT_SUFFIX_RE = re.compile(r"\|fmt_v\d+$")
_REPLICATE_SUFFIX_RE = re.compile(r"\|replicate_\d+")


def intended_identity_from_request_key(request_key: str) -> str:
    """Strips the format-version suffix and the replicate_id (attempt
    number) component, leaving only donor/condition/stage. Stable across
    every attempt and format version of the same intended identity."""
    stripped = _FMT_SUFFIX_RE.sub("", request_key)
    stripped = _REPLICATE_SUFFIX_RE.sub("", stripped)
    return stripped


def load_intended_universe() -> dict[str, dict[str, Any]]:
    """The full 17,000-identity target G Wave-1 universe, keyed by intended
    identity. Reads ONLY the already-frozen v2 by-stage manifests (never
    mutated by this module)."""
    universe: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        manifest_path = WAVE1_V2_BY_STAGE_ROOT / stage / "request_manifest.csv"
        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                identity = intended_identity_from_request_key(row["request_key"])
                if identity in universe:
                    raise RuntimeError(f"duplicate intended identity across the universe: {identity}")
                universe[identity] = {
                    "request_stage": stage,
                    "profile_id": row["profile_id"],
                    "condition_id": row["condition_id"],
                    "v2_request_key": row["request_key"],
                    "v2_custom_id": row["custom_id"],
                }
    if len(universe) != EXPECTED_UNIVERSE_SIZE:
        raise RuntimeError(f"intended production universe is {len(universe)}, expected exactly {EXPECTED_UNIVERSE_SIZE}")
    return universe


def classify_attempt1_responses() -> dict[str, dict[str, Any]]:
    """Per-custom_id classification of the real, already-retrieved attempt-1
    (G-v2 full production replacement) responses -- reuses
    scripts/score_target_g_wave1_v2_replacement.py's own loading primitives
    (no duplicated parsing logic) so this can never silently drift from the
    committed validation report. Recomputes from the raw retrieved files
    every call -- never trusts a cached total."""
    from ate.f_screen_validation import validate_response

    out: dict[str, dict[str, Any]] = {}
    for stage, part in attempt1_scorer.PARTS:
        part_dir = attempt1_scorer.SUBMISSION_ROOT / stage / part
        schema_by_cid = attempt1_scorer._load_schema_by_cid(part_dir / "batch_input.jsonl")
        manifest_by_cid: dict[str, str] = {}
        with open(part_dir / "request_manifest.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manifest_by_cid[row["custom_id"]] = intended_identity_from_request_key(row["request_key"])
        output_by_cid, _, _ = attempt1_scorer._load_jsonl_by_cid(part_dir / "retrieved" / "batch_output.jsonl")
        error_by_cid, _, _ = attempt1_scorer._load_jsonl_by_cid(part_dir / "retrieved" / "batch_error.jsonl")

        for cid, identity in manifest_by_cid.items():
            if cid in out:
                raise RuntimeError(f"custom_id {cid} appears in more than one attempt-1 part")
            if cid in output_by_cid:
                rec = output_by_cid[cid]
                fp = rec.get("response", {}).get("body", {}).get("system_fingerprint")
                v = validate_response(rec, schema_by_cid.get(cid))
                status = SCHEMA_VALID if v["valid"] else SCHEMA_INVALID
                out[cid] = {"identity": identity, "stage": stage, "part": part, "status": status, "system_fingerprint": fp}
            elif cid in error_by_cid:
                out[cid] = {"identity": identity, "stage": stage, "part": part, "status": PROVIDER_ERROR, "system_fingerprint": None}
            else:
                raise RuntimeError(f"custom_id {cid} is missing entirely from attempt-1 retrieved output (expected accounting_closes=true; re-run the scorer)")
    return out


def verify_attempt1_classification_matches_committed_report(classification: dict[str, dict[str, Any]]) -> None:
    """Fail closed if the freshly recomputed classification disagrees with
    the committed validation report's totals -- never trust a cached
    number over the real artifacts."""
    import json

    report = json.loads(attempt1_scorer.OUT_PATH.read_text(encoding="utf-8"))
    counts = {SCHEMA_VALID: 0, SCHEMA_INVALID: 0, PROVIDER_ERROR: 0}
    for c in classification.values():
        counts[c["status"]] += 1
    expected = {SCHEMA_VALID: report["totals"]["schema_valid"], SCHEMA_INVALID: report["totals"]["schema_invalid"], PROVIDER_ERROR: report["totals"]["provider_error"]}
    if counts != expected:
        raise RuntimeError(f"fresh attempt-1 classification {counts} disagrees with the committed validation report {expected} -- refusing to build a completion manifest against stale/incorrect accounting")


def build_attempt_ledger(additional_attempt_rounds: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """additional_attempt_rounds: ordered list of REAL future rounds (attempt
    2, then attempt 3 if ever needed), each shaped as
    {"attempt_number": N, "custom_id_to_identity": {cid: identity, ...},
     "custom_id_status": {cid: status, ...}}. Supplied by the caller once
    that round has been submitted, retrieved, and classified -- never
    invented here. Omitted/None until then, so calling this with no
    argument reflects exactly today's real state (attempt 1 only)."""
    universe = load_intended_universe()
    attempt1 = classify_attempt1_responses()
    verify_attempt1_classification_matches_committed_report(attempt1)

    attempts_by_identity: dict[str, list[dict[str, Any]]] = {identity: [] for identity in universe}
    for cid, c in attempt1.items():
        attempts_by_identity[c["identity"]].append(
            {"attempt_number": 1, "custom_id": cid, "status": c["status"], "system_fingerprint": c.get("system_fingerprint"), "provider_batch_source": f"{c['stage']}/{c['part']}"}
        )

    for round_data in additional_attempt_rounds or []:
        n = round_data["attempt_number"]
        for cid, identity in round_data["custom_id_to_identity"].items():
            if identity not in attempts_by_identity:
                raise RuntimeError(f"attempt round {n} references an identity not in the intended universe: {identity}")
            attempts_by_identity[identity].append(
                {"attempt_number": n, "custom_id": cid, "status": round_data["custom_id_status"][cid], "system_fingerprint": round_data.get("custom_id_fingerprint", {}).get(cid), "provider_batch_source": round_data.get("source_label", f"attempt_{n}")}
            )

    ledger: dict[str, dict[str, Any]] = {}
    for identity, meta in universe.items():
        attempts = sorted(attempts_by_identity[identity], key=lambda a: a["attempt_number"])
        attempt_numbers = [a["attempt_number"] for a in attempts]
        if attempt_numbers != list(range(1, len(attempts) + 1)):
            raise RuntimeError(f"malformed attempt order for identity {identity}: {attempt_numbers} (must be contiguous starting at 1, no gaps, no repeats)")
        if len(attempts) > MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY:
            raise RuntimeError(f"identity {identity} has {len(attempts)} production attempts, exceeds MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY={MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY}")
        valid_attempts = [a["attempt_number"] for a in attempts if a["status"] == TERMINAL_VALID_STATUS]
        if len(valid_attempts) > 1:
            raise RuntimeError(f"identity {identity} has more than one schema-valid production response ({valid_attempts}) -- provenance ambiguity, first-valid selection cannot proceed")
        resolved = bool(valid_attempts)
        resolved_attempt = valid_attempts[0] if valid_attempts else None
        attempt_count = len(attempts)
        next_attempt_number = None if resolved or attempt_count >= MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY else attempt_count + 1
        ledger[identity] = {**meta, "attempts": attempts, "resolved": resolved, "resolved_attempt": resolved_attempt, "attempt_count": attempt_count, "next_attempt_number": next_attempt_number}
    return ledger


def identities_pending_next_attempt(ledger: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """Groups, by request_stage, the intended identities that need their
    next production attempt right now -- not resolved, attempt_count < MAX.
    Operates exclusively on the ledger's engineering fields (status/attempt
    count); never reads scientific response fields. This SAME function
    produces the current completion membership (called against the
    attempt-1-only ledger, where every pending identity's next_attempt_number
    is 1 or 2) and, later, the Attempt-3 fallback membership (called against
    a ledger that has real attempt-2 rounds merged in via
    build_attempt_ledger(additional_attempt_rounds=[...]))."""
    by_stage: dict[str, list[str]] = {stage: [] for stage in STAGES}
    for identity, entry in ledger.items():
        if entry["resolved"]:
            continue
        if entry["attempt_count"] >= MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY:
            continue
        by_stage[entry["request_stage"]].append(identity)
    for stage in by_stage:
        by_stage[stage].sort()
    return by_stage


def assemble_first_valid(ledger: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For each intended identity, the engineering provenance of its
    selected production response (first schema-valid response in
    increasing attempt order) -- or None if unresolved. Never exposes
    scientific response values; only provenance/status fields. Fails if any
    identity's attempt order/count is malformed (already enforced by
    build_attempt_ledger, re-checked here defensively)."""
    result: dict[str, dict[str, Any]] = {}
    for identity, entry in ledger.items():
        if entry["attempt_count"] > MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY:
            raise RuntimeError(f"identity {identity} exceeds MAX_PRODUCTION_ATTEMPTS_PER_IDENTITY")
        if entry["resolved"]:
            selected = next(a for a in entry["attempts"] if a["status"] == TERMINAL_VALID_STATUS)
            if selected["attempt_number"] != entry["resolved_attempt"]:
                raise RuntimeError(f"provenance ambiguity for identity {identity}: resolved_attempt mismatch")
            result[identity] = {
                "intended_identity": identity,
                "request_stage": entry["request_stage"],
                "resolved": True,
                "selected_attempt": selected["attempt_number"],
                "source_custom_id": selected["custom_id"],
                "provider_batch_source": selected["provider_batch_source"],
                "system_fingerprint": selected.get("system_fingerprint"),
            }
        else:
            result[identity] = {
                "intended_identity": identity,
                "request_stage": entry["request_stage"],
                "resolved": False,
                "selected_attempt": None,
                "attempt_count": entry["attempt_count"],
                "next_attempt_number": entry["next_attempt_number"],
            }
    return result


_G_MASTER_CACHE: pd.DataFrame | None = None


def _g_master_row(profile_id: str) -> pd.Series:
    global _G_MASTER_CACHE
    if _G_MASTER_CACHE is None:
        _G_MASTER_CACHE = pd.read_csv(G_MASTER_PATH).set_index("donor_key", drop=False)
    return _G_MASTER_CACHE.loc[profile_id]


def build_completion_requests(ledger: dict[str, dict[str, Any]], pending_by_stage: dict[str, list[str]], *, requested_model: str) -> dict[str, list[BatchRequest]]:
    """Builds fresh BatchRequest rows for exactly the pending identities,
    each at ITS OWN next_attempt_number (read from the ledger -- smoke-only
    identities' production attempt_number is 1, previously-failed
    identities' is 2). Always response_format_instruction_version="v2".
    Preserves donor, condition, persona, stimulus, questionnaire, item
    order (via order_replicate_id=1 for Consensus Stage-A), model, and
    sampling configuration exactly; only the wire-level replicate_id (hence
    request_key/seed/custom_id) is fresh.

    wire_replicate_id is NOT always literally equal to attempt_number: an
    identity with attempt_count==0 (smoke-only -- no REAL production
    attempt yet) already has its replicate_id=1 request_key/custom_id
    "spent" by the (non-production) engineering smoke, so its first
    production attempt uses wire_replicate_id=2 to get a genuinely fresh,
    never-before-submitted custom_id while its ledger attempt_number
    correctly stays 1 (smoke does not count as a production attempt)."""
    items = sc.load_items()
    out: dict[str, list[BatchRequest]] = {stage: [] for stage in STAGES}
    for stage, identities in pending_by_stage.items():
        for identity in identities:
            entry = ledger[identity]
            attempt_number = entry["next_attempt_number"]
            if attempt_number is None:
                raise RuntimeError(f"identity {identity} has no next_attempt_number but was included in pending_by_stage")
            wire_replicate_id = attempt_number + 1 if entry["attempt_count"] == 0 else attempt_number
            row = _g_master_row(entry["profile_id"])
            profile = _profile_dict(row)
            donor_key = entry["profile_id"]
            condition_id = entry["condition_id"]

            if stage == "standard":
                stimulus = sc.get_condition_stimulus(condition_id, state_abbr=row.get("state_abbr"), control_variant=1)
                render = build_g_prompt_render(profile, stimulus, items, donor_key=donor_key, condition_id=condition_id, replicate_id=wire_replicate_id, response_format_instruction_version="v2")
                request_stage = "standard"
                outcome_id = "full_questionnaire"
            elif stage == "consensus_stage_a":
                render = build_g_consensus_stage_a_prompt_render(profile, donor_key=donor_key, replicate_id=wire_replicate_id, order_replicate_id=1, response_format_instruction_version="v2")
                request_stage = "consensus_stage_a"
                outcome_id = CONSENSUS_STAGE_A_OUTCOME_ID
            else:  # pragma: no cover
                raise RuntimeError(f"unknown stage {stage!r}")

            key = render.request_key
            if intended_identity_from_request_key(key) != identity:
                raise RuntimeError(f"rebuilt request_key {key!r} does not map back to intended identity {identity!r}")
            custom_id = custom_id_from_request_key(key)
            out[stage].append(
                BatchRequest(
                    request_key=key,
                    custom_id=custom_id,
                    role="G",
                    study_id="target",
                    profile_id=donor_key,
                    condition_id=condition_id,
                    outcome_id=outcome_id,
                    replicate_id=wire_replicate_id,
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
                    request_stage=request_stage,
                    engine_config_hash=compute_engine_config_hash(requested_model),
                )
            )
    return out
