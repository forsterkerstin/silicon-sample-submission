"""Build (never submit) the benchmark-exact Consensus STEP_3 manifest: the
third of four sequential, chained requests per Consensus donor, built from
STEP_1's AND STEP_2's REAL, already-retrieved, first-valid responses (see
inference/consensus_benchmark_exact.py's module docstring).

For every donor whose STEP_1 and STEP_2 responses are both locked
first-valid, this rebuilds STEP_1's and STEP_2's renders (byte-identical
prompt-compiler calls, same donor/order/attempt_id=1), reconstructs both
step records by pairing each render with its REAL retrieved response (this
is the one place besides the STEP_2 builder that reads actual model
response values -- purely mechanically, to embed as conversation history
for the next turn; never used to decide WHICH donors get a STEP_3 request,
since membership here is exactly "resolved in STEP_1 and STEP_2", an
engineering/status fact already established by score_consensus_exact_stage.py
before this script ever runs), then builds STEP_3's request via
build_step3_prompt_render.

No scientific response value is read, printed, or exposed by ANY OTHER
script in this pipeline -- this script's own output (the manifest/summary)
contains no response content either, only engineering fields (request_key,
custom_id, hashes, counts).

No LLM calls. No target requests submitted.
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
if str(PIPELINE_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT / "scripts"))

import pandas as pd  # noqa: E402

from ate.f_screen_validation import validate_response  # noqa: E402
from inference.consensus_benchmark_exact import (  # noqa: E402
    CONSENSUS_EXACT_PROTOCOL_ID,
    build_step1_prompt_render,
    build_step2_prompt_render,
    build_step3_prompt_render,
    step_record,
)
from inference.model_config import selected_model  # noqa: E402
from inference.prompts import PROMPT_COMPILER_VERSION, schema_hash  # noqa: E402
from inference.request_logging import seed_from_request_key  # noqa: E402
from inference.together_batch import (  # noqa: E402
    G_MASTER_PATH,
    SMOKE_MODEL_PRICES_PER_1M_TOKENS,
    BatchRequest,
    _chat_body,
    _profile_dict,
    _render_prompt_hash,
    compute_engine_config_hash,
    custom_id_from_request_key,
)

STEP1_DIR = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_exact" / "step1"
STEP2_DIR = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_exact" / "step2"
OUT_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "consensus_exact" / "step3"
MANIFEST_FIELDS = [
    "request_key", "custom_id", "role", "study_id", "profile_id", "condition_id",
    "outcome_id", "replicate_id", "requested_model", "prompt_hash", "schema_version",
    "prompt_protocol_id", "prompt_compiler_version", "seed", "status", "required_fields",
    "response_key_map", "request_stage", "engine_config_hash",
]
EXPECTED_DONOR_COUNT = 1000


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_schema_by_cid(jsonl_path: Path) -> dict:
    out = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[str(r["custom_id"])] = r["body"]["response_format"]["json_schema"]["schema"]
    return out


def _load_output_by_cid(jsonl_path: Path) -> dict:
    out = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r, dict) and "custom_id" in r:
                out[str(r["custom_id"])] = r
    return out


def _resolved_step_record(*, stage_dir: Path, donor_key: str, cid: str, expected_request_key: str, render, stage_label: str) -> dict[str, Any]:
    schema_by_cid = _load_schema_by_cid(stage_dir / "batch_input.jsonl")
    output_by_cid = _load_output_by_cid(stage_dir / "retrieved" / "batch_output.jsonl")
    if cid not in output_by_cid:
        raise RuntimeError(f"{stage_label} custom_id {cid} (donor {donor_key}) is not resolved in the retrieved output -- STEP_3 requires every donor to be first-valid in {stage_label}")
    if render.request_key != expected_request_key:
        raise RuntimeError(f"rebuilt {stage_label} request_key for donor {donor_key} does not match the frozen manifest -- refusing to chain from a mismatched render")

    raw_record = output_by_cid[cid]
    validation = validate_response(raw_record, schema_by_cid.get(cid))
    if not validation["valid"]:
        raise RuntimeError(f"{stage_label} response for donor {donor_key} is not schema-valid ({validation['reason']}) -- this contradicts the prior reconciliation report; refusing to chain from an invalid response")

    content = raw_record["response"]["body"]["choices"][0]["message"]["content"]
    response_dict = json.loads(content)
    return step_record(render, response_dict, donor_key=donor_key, attempt_id=1)


def build_requests(requested_model: str) -> list[BatchRequest]:
    with open(STEP1_DIR / "request_manifest.csv", newline="", encoding="utf-8") as f:
        step1_rows = {row["profile_id"]: row for row in csv.DictReader(f)}
    with open(STEP2_DIR / "request_manifest.csv", newline="", encoding="utf-8") as f:
        step2_rows = list(csv.DictReader(f))
    if len(step2_rows) != EXPECTED_DONOR_COUNT:
        raise RuntimeError(f"STEP_2 manifest has {len(step2_rows)} rows, expected exactly {EXPECTED_DONOR_COUNT}")

    step2_schema_by_cid = _load_schema_by_cid(STEP2_DIR / "batch_input.jsonl")
    step2_output_by_cid = _load_output_by_cid(STEP2_DIR / "retrieved" / "batch_output.jsonl")

    profiles = pd.read_csv(G_MASTER_PATH).set_index("donor_key", drop=False)

    requests: list[BatchRequest] = []
    for row2 in step2_rows:
        donor_key = row2["profile_id"]
        row1 = step1_rows.get(donor_key)
        if row1 is None:
            raise RuntimeError(f"STEP_2 donor {donor_key} has no corresponding STEP_1 manifest row")

        profile = _profile_dict(profiles.loc[donor_key])

        render1 = build_step1_prompt_render(profile, donor_key=donor_key, attempt_id=1)
        step1_record = _resolved_step_record(
            stage_dir=STEP1_DIR, donor_key=donor_key, cid=row1["custom_id"],
            expected_request_key=row1["request_key"], render=render1, stage_label="STEP_1",
        )

        render2 = build_step2_prompt_render(profile, donor_key=donor_key, step1_record=step1_record, attempt_id=1)
        cid2 = row2["custom_id"]
        if cid2 not in step2_output_by_cid:
            raise RuntimeError(f"STEP_2 custom_id {cid2} (donor {donor_key}) is not resolved in the retrieved output -- STEP_3 requires every donor to be first-valid in STEP_2")
        if render2.request_key != row2["request_key"]:
            raise RuntimeError(f"rebuilt STEP_2 request_key for donor {donor_key} does not match the frozen manifest -- refusing to chain from a mismatched render")
        raw_record2 = step2_output_by_cid[cid2]
        validation2 = validate_response(raw_record2, step2_schema_by_cid.get(cid2))
        if not validation2["valid"]:
            raise RuntimeError(f"STEP_2 response for donor {donor_key} is not schema-valid ({validation2['reason']}) -- this contradicts the prior reconciliation report; refusing to chain from an invalid response")
        content2 = raw_record2["response"]["body"]["choices"][0]["message"]["content"]
        step2_record = step_record(render2, json.loads(content2), donor_key=donor_key, attempt_id=1)

        render3 = build_step3_prompt_render(profile, donor_key=donor_key, step1_record=step1_record, step2_record=step2_record, attempt_id=1)
        key = render3.request_key
        custom_id = custom_id_from_request_key(key)
        requests.append(
            BatchRequest(
                request_key=key,
                custom_id=custom_id,
                role="G",
                study_id="target",
                profile_id=donor_key,
                condition_id="Consensus",
                outcome_id="consensus_exact_step3_estimate",
                replicate_id=1,
                requested_model=requested_model,
                prompt_hash=_render_prompt_hash(render3.messages),
                schema_version=schema_hash(render3.response_schema),
                prompt_protocol_id=render3.protocol_id,
                prompt_compiler_version=PROMPT_COMPILER_VERSION,
                seed=seed_from_request_key(key),
                status="pending",
                messages=render3.messages,
                response_schema=render3.response_schema,
                response_key_map=render3.response_key_map or {},
                request_stage="consensus_exact_step3",
                engine_config_hash=compute_engine_config_hash(requested_model),
            )
        )
    if len({r.custom_id for r in requests}) != len(requests):
        raise RuntimeError("duplicate custom_id in Consensus-exact STEP_3 manifest")
    return requests


def write_requests(requests: list[BatchRequest], out_dir: Path) -> dict[str, Any]:
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
    requests = build_requests(g_star)
    stats = write_requests(requests, OUT_ROOT)
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[g_star]
    stats["worst_case_cost_usd"] = round(stats["estimated_prompt_tokens_rough"] * prices["input"] / 1_000_000 + stats["maximum_output_tokens"] * prices["output"] / 1_000_000, 6)
    stats["model"] = g_star
    stats["protocol_id"] = CONSENSUS_EXACT_PROTOCOL_ID
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
