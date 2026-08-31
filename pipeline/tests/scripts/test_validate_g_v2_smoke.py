"""Tests for the zero-tolerance G-v2 smoke validator, using synthetic
retrieved-output fixtures (never real Together output -- none has been
submitted yet). Confirms the frozen acceptance rule: 10/10 schema-valid
with zero missing/unexpected/duplicate is PASS; anything else (including
9/10) is FAIL."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SMOKE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "g_v2_engineering_smoke"

pytestmark = pytest.mark.skipif(not SMOKE_ROOT.exists(), reason="G-v2 smoke manifest not built in this environment")


def _build_fixture(tmp_path, *, corrupt_one=False, drop_one=False, duplicate_one=False):
    import csv

    out_paths = {}
    for stage in ("standard", "consensus_stage_a"):
        with open(SMOKE_ROOT / stage / "request_manifest.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        with open(SMOKE_ROOT / stage / "batch_input.jsonl", encoding="utf-8") as f:
            schemas = {}
            for line in f:
                r = json.loads(line)
                schemas[r["custom_id"]] = r["body"]["response_format"]["json_schema"]["schema"]

        out_path = tmp_path / f"{stage}.jsonl"
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

        if corrupt_one and stage == "standard" and lines:
            lines[0]["response"]["body"]["choices"][0]["message"]["content"] = "```json\n" + lines[0]["response"]["body"]["choices"][0]["message"]["content"] + "\n```"
        if drop_one and stage == "standard" and lines:
            lines = lines[1:]
        if duplicate_one and stage == "standard" and lines:
            lines.append(lines[0])

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in lines:
                f.write(json.dumps(rec) + "\n")
        out_paths[stage] = out_path
    return out_paths


def _run(out_paths, tmp_path):
    # --out is redirected to tmp_path so this NEVER overwrites the real,
    # committed smoke_validation_result.json (which reflects the actual
    # Together submission's real result).
    result = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_ROOT / "scripts" / "validate_g_v2_smoke.py"),
            "--standard-output",
            str(out_paths["standard"]),
            "--consensus-a-output",
            str(out_paths["consensus_stage_a"]),
            "--out",
            str(tmp_path / "result.json"),
        ],
        capture_output=True,
        text=True,
        cwd=PIPELINE_ROOT,
    )
    return result


def test_10_of_10_valid_passes(tmp_path):
    out_paths = _build_fixture(tmp_path)
    result = _run(out_paths, tmp_path)
    payload = json.loads(result.stdout)
    assert payload["smoke_pass"] is True
    assert payload["schema_valid"] == 10
    assert payload["schema_invalid"] == 0
    assert result.returncode == 0


def test_9_of_10_valid_fails(tmp_path):
    out_paths = _build_fixture(tmp_path, corrupt_one=True)
    result = _run(out_paths, tmp_path)
    payload = json.loads(result.stdout)
    assert payload["smoke_pass"] is False
    assert payload["schema_valid"] == 9
    assert payload["malformed_json"] == 1
    assert result.returncode == 1


def test_missing_response_fails(tmp_path):
    out_paths = _build_fixture(tmp_path, drop_one=True)
    result = _run(out_paths, tmp_path)
    payload = json.loads(result.stdout)
    assert payload["smoke_pass"] is False
    assert payload["missing"] == 1


def test_duplicate_response_fails(tmp_path):
    out_paths = _build_fixture(tmp_path, duplicate_one=True)
    result = _run(out_paths, tmp_path)
    payload = json.loads(result.stdout)
    assert payload["smoke_pass"] is False
    assert payload["duplicates"] == 1


def test_never_authorizes_full_replacement_regardless_of_outcome(tmp_path):
    for kwargs in ({}, {"corrupt_one": True}):
        out_paths = _build_fixture(tmp_path, **kwargs)
        result = _run(out_paths, tmp_path)
        payload = json.loads(result.stdout)
        assert payload["full_replacement_automatically_authorized"] is False
