"""Build the standard_g_control_repair replacement request set.

Regenerates the G "control" condition request for exactly the donors whose
frozen g_control_randomization_assignment.csv variant (2=baseball,
3=dances) differs from the necktie (variant 1) filler every control
request used historically. The 334 donors assigned variant 1 keep their
existing necktie response and are NOT regenerated here.

Serving-format parity with final valid G production (both
wave1_g_v2_replacement and wave1_g_completion, which both build via
response_format_instruction_version="v2") is achieved by construction: this
reuses build_g_prompt_render(..., response_format_instruction_version="v2")
and together_batch._chat_body/compute_engine_config_hash directly, rather
than re-deriving the request body by hand. The ONLY scientific prompt
difference from the historical control request for the same donor is the
condition_stimulus filler block (get_condition_stimulus(..., control_variant
=assigned_variant)); everything else (profile, questionnaire, item order,
schema, model, sampling parameters, engine config) is identical.

replicate_id is chosen per donor as the smallest integer whose resulting
request_key/custom_id has never appeared in
target_production_submission_state.json's submitted_custom_ids -- this is
the same ledger target_production_safety_guard itself checks, so it cannot
silently collide with any historical attempt (original v1 wave, v2
replacement attempt 1, v2 completion attempt 2, or engineering smoke) no
matter which physical file that attempt lives in.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd  # noqa: E402

import survey_content as sc  # noqa: E402
from inference.prompts import G_PROMPT_PROTOCOL, PROMPT_COMPILER_VERSION, build_g_prompt_render, schema_hash  # noqa: E402
from inference.request_logging import seed_from_request_key  # noqa: E402
from inference.together_batch import (  # noqa: E402
    BatchRequest,
    G_MASTER_PATH,
    _chat_body,
    _profile_dict,
    _render_prompt_hash,
    compute_engine_config_hash,
    custom_id_from_request_key,
)

ASSIGNMENT_CSV = PIPELINE_ROOT / "outputs" / "target_production" / "g_control_randomization_assignment.csv"
STATE_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "target_production_submission_state.json"
OUT_DIR = PIPELINE_ROOT / "outputs" / "target_production" / "control_replacement_requests_v2"
JSONL_PATH = OUT_DIR / "replacement_batch_input.jsonl"
MANIFEST_PATH = OUT_DIR / "replacement_request_manifest.csv"

REQUESTED_MODEL = "google/gemma-4-31B-it"
FILLER_NAMES = {1: "neckties", 2: "baseball", 3: "dances"}
EXPECTED_TOTAL = 666
EXPECTED_BY_VARIANT = {2: 333, 3: 333}


def _already_submitted() -> set[str]:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return set(state.get("submitted_custom_ids", [])) | set(state.get("successful_custom_ids", []))


def _fresh_replicate(donor_key: str, taken: set[str]) -> tuple[int, str, str]:
    n = 1
    while True:
        request_key = f"G|{donor_key}|control|replicate_{n}|fmt_v2"
        custom_id = custom_id_from_request_key(request_key)
        if custom_id not in taken:
            return n, request_key, custom_id
        n += 1


def main() -> dict:
    assignment = pd.read_csv(ASSIGNMENT_CSV, dtype={"donor_key": str})
    replace_rows = assignment[assignment["assigned_variant"].astype(int) != 1].copy()
    if len(replace_rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} donors needing replacement, found {len(replace_rows)}")
    counts = replace_rows["assigned_variant"].astype(int).value_counts().to_dict()
    if counts != EXPECTED_BY_VARIANT:
        raise RuntimeError(f"replacement variant counts {counts} != expected {EXPECTED_BY_VARIANT}")

    submitted = _already_submitted()
    profiles = pd.read_csv(G_MASTER_PATH, dtype={"donor_key": str}).set_index("donor_key", drop=False)
    items = sc.load_items()

    engine_config_hash = compute_engine_config_hash(REQUESTED_MODEL)
    seen_custom_ids: set[str] = set()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_fields = [
        "donor_key", "request_key", "custom_id", "assigned_variant", "assigned_filler",
        "condition_id", "outcome_id", "requested_model", "seed", "prompt_hash", "schema_version",
        "prompt_protocol_id", "prompt_compiler_version", "engine_config_hash",
        "response_format_instruction_version", "study_id", "request_stage", "phase",
        "intended_logical_identity", "replacement_replicate_id",
    ]

    manifest_rows = []
    jsonl_lines = []
    for _, arow in replace_rows.sort_values("donor_key").iterrows():
        donor_key = str(arow["donor_key"])
        variant = int(arow["assigned_variant"])
        filler = str(arow["assigned_filler"])
        if FILLER_NAMES[variant] != filler:
            raise RuntimeError(f"assignment CSV filler name mismatch for {donor_key}: {filler} vs {FILLER_NAMES[variant]}")

        prow = profiles.loc[donor_key]
        profile = _profile_dict(prow)
        stimulus = sc.get_condition_stimulus("control", state_abbr=prow.get("state_abbr"), control_variant=variant)
        if filler not in stimulus.lower() and not (filler == "neckties" and "necktie" in stimulus.lower()):
            raise RuntimeError(f"rendered stimulus for {donor_key} does not mention assigned filler {filler!r}")

        replicate_id, request_key, custom_id = _fresh_replicate(donor_key, submitted | seen_custom_ids)
        seen_custom_ids.add(custom_id)

        render = build_g_prompt_render(
            profile,
            stimulus,
            items,
            donor_key=donor_key,
            condition_id="control",
            replicate_id=replicate_id,
            response_format_instruction_version="v2",
        )
        if render.request_key != request_key:
            raise RuntimeError(f"rebuilt request_key {render.request_key!r} != expected {request_key!r}")
        if render.protocol_id != G_PROMPT_PROTOCOL:
            raise RuntimeError(f"unexpected protocol_id for {donor_key}: {render.protocol_id!r}")

        seed = seed_from_request_key(request_key)
        req = BatchRequest(
            request_key=request_key,
            custom_id=custom_id,
            role="G",
            study_id="target",
            profile_id=donor_key,
            condition_id="control",
            outcome_id="full_questionnaire",
            replicate_id=replicate_id,
            requested_model=REQUESTED_MODEL,
            prompt_hash=_render_prompt_hash(render.messages),
            schema_version=schema_hash(render.response_schema),
            prompt_protocol_id=render.protocol_id,
            prompt_compiler_version=PROMPT_COMPILER_VERSION,
            seed=seed,
            status="pending",
            messages=render.messages,
            response_schema=render.response_schema,
            response_key_map=render.response_key_map or {},
            request_stage="standard",
            engine_config_hash=engine_config_hash,
        )
        body = _chat_body(req)
        jsonl_lines.append(json.dumps({"custom_id": custom_id, "body": body}, sort_keys=True))
        manifest_rows.append({
            "donor_key": donor_key,
            "request_key": request_key,
            "custom_id": custom_id,
            "assigned_variant": variant,
            "assigned_filler": filler,
            "condition_id": "control",
            "outcome_id": "full_questionnaire",
            "requested_model": REQUESTED_MODEL,
            "seed": seed,
            "prompt_hash": req.prompt_hash,
            "schema_version": req.schema_version,
            "prompt_protocol_id": req.prompt_protocol_id,
            "prompt_compiler_version": req.prompt_compiler_version,
            "engine_config_hash": engine_config_hash,
            "response_format_instruction_version": "v2",
            "study_id": "target",
            "request_stage": "standard",
            "phase": "standard_g_control_repair_v2",
            "intended_logical_identity": f"G|{donor_key}|control",
            "replacement_replicate_id": replicate_id,
        })

    if len(manifest_rows) != EXPECTED_TOTAL or len({r["custom_id"] for r in manifest_rows}) != EXPECTED_TOTAL:
        raise RuntimeError("did not produce exactly 666 unique replacement requests")
    overlap = {r["custom_id"] for r in manifest_rows} & submitted
    if overlap:
        raise RuntimeError(f"replacement custom_id(s) collide with already-submitted history: {sorted(overlap)[:5]}")

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(jsonl_lines) + "\n")
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "n_requests": len(manifest_rows),
        "by_variant": {str(k): int(v) for k, v in pd.Series([r["assigned_variant"] for r in manifest_rows]).value_counts().items()},
        "engine_config_hash": engine_config_hash,
        "jsonl_path": str(JSONL_PATH.relative_to(PIPELINE_ROOT)),
        "manifest_path": str(MANIFEST_PATH.relative_to(PIPELINE_ROOT)),
    }
    return summary


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
