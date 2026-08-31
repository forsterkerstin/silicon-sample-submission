"""Fail-closed guard architecture for a FUTURE target-production submission.

This module is built and tested now, in advance, per the independent audit's
finding that the generic Together submission CLI can submit an arbitrary
JSONL that lives outside the two currently-guarded roots
(TINY_SMOKE_ROOT, SCIENTIFIC_BAKEOFF_ROOT) with zero allowlist/cost/
duplicate protection. It mirrors inference/scientific_bakeoff_guard.py's
shape, but with additional, STRICTER prerequisites specific to real target
inference: a target phase cannot even be DECLARED until the full frozen
method (F*, G*, R_F, and a usable_for_production calibration artifact) is
in place.

No target phase is declared or activated by this module on import or at any
other point in the current pipeline -- declare_target_phase() must be called
explicitly, and it hard-refuses unless every prerequisite below is already
met. As of this module's introduction, none of them are (R_F/calibration are
not yet frozen), so target production remains unauthorized.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ate.f_reliability import require_frozen_f_protocol
from inference.model_config import selected_model
from inference.together_batch import (
    SMOKE_MODEL_PRICES_PER_1M_TOKENS,
    TOGETHER_BATCH_MAX_INPUT_FILE_BYTES,
    TOGETHER_BATCH_MAX_REQUESTS,
    _read_jsonl_requests,
    _under_path,
)
from validation.holdout import FROZEN_METHOD_MANIFEST_PATH

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
TARGET_PRODUCTION_ROOT = PIPELINE_ROOT / "outputs" / "target_production"
TARGET_PRODUCTION_STATE_PATH = TARGET_PRODUCTION_ROOT / "target_production_submission_state.json"
TARGET_PRODUCTION_STUDY_ID = "target"
CALIBRATION_SELECTED_MODEL_PATH = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"


class TargetProductionNotAuthorized(RuntimeError):
    """The full frozen method is not yet in place; target inference cannot
    be declared or submitted. Never silently downgraded to a partial check."""


def is_target_production_jsonl(jsonl_path: Path | str) -> bool:
    return _under_path(Path(jsonl_path), TARGET_PRODUCTION_ROOT)


def assert_target_production_prerequisites_frozen() -> dict[str, Any]:
    """Hard-stop unless F*, G*, R_F, and a usable_for_production calibration
    artifact are ALL already frozen. Reused verbatim by both
    declare_target_phase (so an unauthorized phase can never even be
    registered) and by the CLI-level check, so there is exactly one place
    this is decided."""
    problems: list[str] = []

    f_star = g_star = None
    try:
        f_star = selected_model("f", require_frozen=True)
    except RuntimeError as exc:
        problems.append(f"F* is not frozen: {exc}")
    try:
        g_star = selected_model("g", require_frozen=True)
    except RuntimeError as exc:
        problems.append(f"G* is not frozen: {exc}")

    try:
        protocol = require_frozen_f_protocol()
        if "f_r_f" not in protocol:
            problems.append("R_F is not frozen: frozen F protocol missing f_r_f")
        if f_star and protocol.get("selected_f_model") and protocol.get("selected_f_model") != f_star:
            problems.append(f"frozen F protocol selected_f_model ({protocol.get('selected_f_model')!r}) does not match model_config selected_f_model ({f_star!r})")
    except RuntimeError as exc:
        problems.append(f"R_F is not frozen: {exc}")

    if not CALIBRATION_SELECTED_MODEL_PATH.exists():
        problems.append(f"calibration is not frozen: missing {CALIBRATION_SELECTED_MODEL_PATH}")
    else:
        try:
            calibration = json.loads(CALIBRATION_SELECTED_MODEL_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            problems.append(f"calibration artifact at {CALIBRATION_SELECTED_MODEL_PATH} could not be loaded: {exc}")
            calibration = {}
        if calibration.get("usable_for_production") is not True:
            problems.append(f"calibration artifact is not explicitly marked usable_for_production=True (got {calibration.get('usable_for_production')!r})")

    if not FROZEN_METHOD_MANIFEST_PATH.exists():
        problems.append(f"the final method/config/provenance manifest is not frozen: missing {FROZEN_METHOD_MANIFEST_PATH}")

    if problems:
        raise TargetProductionNotAuthorized("target production prerequisites are not met:\n- " + "\n- ".join(problems))

    return {"selected_f_model": f_star, "selected_g_model": g_star}


def _load_state(state_path: Path = TARGET_PRODUCTION_STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {"schema_version": "target_production_submission_state_v1", "phases": {}, "submitted_custom_ids": [], "successful_custom_ids": []}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != "target_production_submission_state_v1":
        raise RuntimeError(f"ambiguous existing target-production state: {state_path}")
    return state


def declare_target_phase(
    phase: str,
    *,
    approved_custom_ids: set[str],
    cost_cap_usd: float,
    request_stage: str | None = None,
    state_path: Path = TARGET_PRODUCTION_STATE_PATH,
) -> dict[str, Any]:
    """Register one target-production phase's exact approved allowlist and
    cost cap. Refuses outright unless assert_target_production_prerequisites_frozen()
    passes -- there is no way to declare a target phase against an unfrozen
    method. Re-declaring an existing phase with a different allowlist/cap/
    request_stage is refused, same as scientific_bakeoff_guard.declare_phase.

    request_stage defaults to `phase` itself (the original, only behavior
    before this parameter existed) -- every phase declared without it is
    unaffected. Passing an explicit request_stage lets a NEW phase name
    (e.g. "standard_g_v2") authorize rows whose manifest request_stage is a
    structural value that can't be renamed (e.g. "standard"), so a later
    generation (a format-only replacement, a rerun, ...) of the same
    request_stage can be declared under its own name/allowlist/cap without
    touching the immutable original phase."""
    assert_target_production_prerequisites_frozen()
    resolved_request_stage = phase if request_stage is None else request_stage

    state = _load_state(state_path)
    phases = state.setdefault("phases", {})
    existing = phases.get(phase)
    if existing is not None:
        if set(existing["approved_custom_ids"]) != set(approved_custom_ids):
            raise RuntimeError(f"target phase {phase!r} allowlist already declared and differs from the requested set")
        if float(existing["cost_cap_usd"]) != float(cost_cap_usd):
            raise RuntimeError(f"target phase {phase!r} cost cap already declared as ${existing['cost_cap_usd']}, not ${cost_cap_usd}")
        if existing.get("request_stage", phase) != resolved_request_stage:
            raise RuntimeError(f"target phase {phase!r} request_stage already declared as {existing.get('request_stage', phase)!r}, not {resolved_request_stage!r}")
        return existing
    phases[phase] = {
        "approved_custom_ids": sorted(approved_custom_ids),
        "cost_cap_usd": float(cost_cap_usd),
        "request_stage": resolved_request_stage,
        "cumulative_requests": 0,
        "cumulative_worst_case_cost_usd": 0.0,
        "submissions": [],
    }
    TARGET_PRODUCTION_ROOT.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return phases[phase]


def target_production_safety_guard(
    jsonl_path: Path | str,
    manifest_path: Path | str,
    *,
    phase: str,
    for_submit: bool = True,
    state_path: Path = TARGET_PRODUCTION_STATE_PATH,
) -> dict[str, Any]:
    """Analogous to scientific_bakeoff_guard.scientific_bakeoff_safety_guard,
    with target-specific tightening: EVERY manifest row must be study_id ==
    'target' (a development/calibration request can never enter this
    guard), and the model must be exactly the frozen F*/G* returned by
    assert_target_production_prerequisites_frozen -- no other model is ever
    accepted here, unlike the scientific bakeoff's two-candidate allowlist."""
    frozen = assert_target_production_prerequisites_frozen()
    allowed_models = {m for m in (frozen["selected_f_model"], frozen["selected_g_model"]) if m}

    path = Path(jsonl_path)
    manifest_path = Path(manifest_path)
    if not path.exists():
        raise RuntimeError(f"target-production input file is missing: {path}")
    if not manifest_path.exists():
        raise RuntimeError(f"target-production manifest is missing: {manifest_path}")
    if not _under_path(path, TARGET_PRODUCTION_ROOT):
        raise RuntimeError(f"input file is outside the target-production directory: {path}")

    file_size_bytes = path.stat().st_size
    if file_size_bytes > TOGETHER_BATCH_MAX_INPUT_FILE_BYTES:
        raise RuntimeError(f"input file is {file_size_bytes / (1024*1024):.2f} MB, exceeds the Together Batch API limit")

    state = _load_state(state_path)
    phase_state = state.get("phases", {}).get(phase)
    if phase_state is None:
        raise RuntimeError(f"target phase {phase!r} has not been declared; call declare_target_phase() first")
    approved = set(phase_state["approved_custom_ids"])
    cost_cap_usd = float(phase_state["cost_cap_usd"])
    expected_request_stage = phase_state.get("request_stage", phase)

    requests = _read_jsonl_requests(path)
    if len(requests) > TOGETHER_BATCH_MAX_REQUESTS:
        raise RuntimeError(f"batch has {len(requests)} requests, exceeds the Together Batch API limit of {TOGETHER_BATCH_MAX_REQUESTS}")
    manifest_rows = {row["custom_id"]: row for row in csv.DictReader(open(manifest_path, encoding="utf-8"))}

    custom_ids = [str(row.get("custom_id", "")) for row in requests]
    if not all(custom_ids) or len(custom_ids) != len(set(custom_ids)):
        raise RuntimeError("target-production JSONL custom_id values are missing or duplicated")

    missing_manifest = [cid for cid in custom_ids if cid not in manifest_rows]
    if missing_manifest:
        raise RuntimeError(f"custom_id(s) in JSONL have no matching manifest row: {missing_manifest[:5]}")
    non_target_ids = [cid for cid in custom_ids if manifest_rows[cid].get("study_id") != TARGET_PRODUCTION_STUDY_ID]
    if non_target_ids:
        raise RuntimeError(f"target-production guard refuses non-target request(s): {non_target_ids[:5]}")
    off_phase_ids = [cid for cid in custom_ids if manifest_rows[cid].get("request_stage") != expected_request_stage]
    if off_phase_ids:
        raise RuntimeError(f"custom_id(s) do not carry request_stage={expected_request_stage!r} (phase {phase!r}): {off_phase_ids[:5]}")

    bodies = [row.get("body") for row in requests]
    models = sorted({str(body.get("model", "")) for body in bodies if isinstance(body, dict)})
    unrecognized_models = [m for m in models if m not in allowed_models]
    if unrecognized_models:
        raise RuntimeError(f"target production only allows the frozen {sorted(allowed_models)}; observed {unrecognized_models}")
    if len(models) != 1:
        raise RuntimeError(f"target-production JSONL must target exactly one model per batch; observed {models}")
    declared_model = models[0]

    already = set(custom_ids) & (set(state.get("submitted_custom_ids", [])) | set(state.get("successful_custom_ids", [])))
    if already:
        raise RuntimeError(f"JSONL contains custom_id already recorded as submitted/successful: {sorted(already)[:5]}")

    if for_submit:
        not_approved = set(custom_ids) - approved
        if not_approved:
            raise RuntimeError(f"JSONL contains custom_id(s) not in target phase {phase!r}'s approved allowlist: {sorted(not_approved)[:5]}")

    max_tokens = [int(body.get("max_tokens", 0) or 0) for body in bodies if isinstance(body, dict)]
    if len(max_tokens) != len(requests) or min(max_tokens) <= 0:
        raise RuntimeError("target-production JSONL has missing/invalid maximum output-token budgets")

    estimated_input_tokens = int(sum(sum(len(m.get("content", "")) for m in body["messages"]) for body in bodies) / 4)
    maximum_output_tokens = int(sum(max_tokens))
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[declared_model]
    batch_cost = estimated_input_tokens * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000

    cumulative_requests = int(phase_state["cumulative_requests"]) + len(requests)
    cumulative_cost = float(phase_state["cumulative_worst_case_cost_usd"]) + batch_cost
    if cumulative_cost > cost_cap_usd:
        raise RuntimeError(f"target phase {phase!r} worst-case cost cap exceeded: ${cumulative_cost:.4f} > ${cost_cap_usd:.2f}")

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


def record_target_production_submission(
    guard: dict[str, Any],
    submit_result: dict[str, Any],
    *,
    state_path: Path = TARGET_PRODUCTION_STATE_PATH,
) -> dict[str, Any]:
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
