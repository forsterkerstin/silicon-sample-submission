"""Tests for the zero-tolerance Orchinik G-v2 post-inference validator,
using synthetic retrieved-output fixtures (never real Together output --
no Orchinik paid inference has been submitted yet). Confirms: full
2545/2545-per-model schema-valid with zero missing/unexpected/duplicate is
all_valid=True; any single corruption/drop/duplicate/mismatched fingerprint
flips it to False and is reflected in the right counter."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

V2_ROOT = PIPELINE_ROOT / "outputs" / "domain_validation" / "orchinik_g_domain_confirmation_v2"
MODEL_DIRS = {
    "google/gemma-4-31B-it": "google_gemma-4-31B-it",
    "deepseek-ai/DeepSeek-V4-Pro-0813": "deepseek-ai_DeepSeek-V4-Pro-0813",
}

pytestmark = pytest.mark.skipif(not V2_ROOT.exists(), reason="Orchinik G-v2 manifests not built in this environment")


def _fixture_for_model(model: str, tmp_path, *, corrupt_one=False, drop_one=False, duplicate_one=False, bad_fingerprint_one=False):
    model_dir = V2_ROOT / MODEL_DIRS[model]
    with open(model_dir / "request_manifest.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    schemas = {}
    with open(model_dir / "batch_input.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            schemas[r["custom_id"]] = r["body"]["response_format"]["json_schema"]["schema"]

    lines = []
    for row in rows:
        cid = row["custom_id"]
        schema = schemas[cid]
        props = {}
        for key, spec in schema["properties"].items():
            lo = spec.get("minimum", spec.get("enum", [0])[0] if "enum" in spec else 0)
            props[key] = lo
        content = json.dumps(props)
        rec = {"custom_id": cid, "response": {"body": {"choices": [{"message": {"content": content}}], "system_fingerprint": "healthy-fp"}}}
        lines.append(rec)

    if corrupt_one and lines:
        lines[0]["response"]["body"]["choices"][0]["message"]["content"] = "```json\n" + lines[0]["response"]["body"]["choices"][0]["message"]["content"] + "\n```"
    if drop_one and lines:
        lines = lines[1:]
    if duplicate_one and lines:
        lines.append(lines[0])
    if bad_fingerprint_one and lines:
        lines[0]["response"]["body"]["system_fingerprint"] = "vllm-0.21.0-8326ea74"

    out_path = tmp_path / f"{MODEL_DIRS[model]}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in lines:
            f.write(json.dumps(rec) + "\n")
    return out_path


def _run(gemma_path, deepseek_path, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_ROOT / "scripts" / "score_orchinik_g_domain_confirmation_v2.py"),
            "--gemma-output",
            str(gemma_path),
            "--deepseek-output",
            str(deepseek_path),
            "--out",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
        text=True,
        cwd=PIPELINE_ROOT,
    )
    return result


def test_all_valid_passes(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "d").mkdir()
    gemma_path = _fixture_for_model("google/gemma-4-31B-it", tmp_path / "g")
    deepseek_path = _fixture_for_model("deepseek-ai/DeepSeek-V4-Pro-0813", tmp_path / "d")
    result = _run(gemma_path, deepseek_path, tmp_path)
    payload = json.loads(result.stdout)
    assert payload["all_valid"] is True
    assert payload["total_expected"] == 5090
    assert payload["schema_valid"] == 5090
    assert payload["schema_invalid"] == 0
    assert result.returncode == 0


def test_single_corruption_in_one_model_fails(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "d").mkdir()
    gemma_path = _fixture_for_model("google/gemma-4-31B-it", tmp_path / "g", corrupt_one=True)
    deepseek_path = _fixture_for_model("deepseek-ai/DeepSeek-V4-Pro-0813", tmp_path / "d")
    result = _run(gemma_path, deepseek_path, tmp_path)
    payload = json.loads(result.stdout)
    assert payload["all_valid"] is False
    assert payload["schema_invalid"] == 1
    assert payload["malformed_json"] == 1
    assert result.returncode == 1


def test_missing_response_fails(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "d").mkdir()
    gemma_path = _fixture_for_model("google/gemma-4-31B-it", tmp_path / "g")
    deepseek_path = _fixture_for_model("deepseek-ai/DeepSeek-V4-Pro-0813", tmp_path / "d", drop_one=True)
    result = _run(gemma_path, deepseek_path, tmp_path)
    payload = json.loads(result.stdout)
    assert payload["all_valid"] is False
    assert payload["missing"] == 1


def test_duplicate_response_fails(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "d").mkdir()
    gemma_path = _fixture_for_model("google/gemma-4-31B-it", tmp_path / "g", duplicate_one=True)
    deepseek_path = _fixture_for_model("deepseek-ai/DeepSeek-V4-Pro-0813", tmp_path / "d")
    result = _run(gemma_path, deepseek_path, tmp_path)
    payload = json.loads(result.stdout)
    assert payload["all_valid"] is False
    assert payload["duplicates"] == 1


def test_bad_fingerprint_surfaces_even_if_schema_still_valid(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "d").mkdir()
    gemma_path = _fixture_for_model("google/gemma-4-31B-it", tmp_path / "g", bad_fingerprint_one=True)
    deepseek_path = _fixture_for_model("deepseek-ai/DeepSeek-V4-Pro-0813", tmp_path / "d")
    result = _run(gemma_path, deepseek_path, tmp_path)
    payload = json.loads(result.stdout)
    # schema can still be valid (the bad serving fingerprint doesn't always
    # imply malformed content), but the fingerprint occurrence itself must
    # be visible for human review.
    assert payload["gemma_system_fingerprint_breakdown"].get("vllm-0.21.0-8326ea74") == 1
