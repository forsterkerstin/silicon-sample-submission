"""Regression tests for the scientific-bakeoff safety guard: a separate ledger
from engineering smoke, with an explicit per-phase allowlist and cost cap.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from inference import scientific_bakeoff_guard as g
from inference.together_batch import DEEPSEEK_MODEL_ID, GEMMA_MODEL_ID, QWEN_MODEL_ID, custom_id_from_request_key

MANIFEST_FIELDS = ["custom_id", "study_id", "request_stage"]


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def _write_jsonl(path: Path, *, model: str, custom_ids: list[str], max_tokens: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid in custom_ids:
            f.write(json.dumps({"custom_id": cid, "body": {"model": model, "messages": [{"role": "user", "content": "hello world"}], "max_tokens": max_tokens}}) + "\n")


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "scientific_bakeoff"
    state = root / "scientific_bakeoff_submission_state.json"
    monkeypatch.setattr(g, "SCIENTIFIC_BAKEOFF_ROOT", root)
    monkeypatch.setattr(g, "SCIENTIFIC_BAKEOFF_STATE_PATH", state)
    return root, state


def test_guard_accepts_exactly_the_declared_allowlist(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    cids = [custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|P{i}|control|Study:y:hyp1|replicate_1") for i in range(3)]
    g.declare_phase("phase_x", approved_custom_ids=set(cids), cost_cap_usd=10.0, state_path=state)
    jsonl = root / "phase_x" / "deepseek" / "batch_input.jsonl"
    manifest = root / "phase_x" / "deepseek" / "request_manifest.csv"
    _write_jsonl(jsonl, model=DEEPSEEK_MODEL_ID, custom_ids=cids)
    _write_manifest(manifest, [{"custom_id": c, "study_id": "Study", "request_stage": "phase_x"} for c in cids])
    guard = g.scientific_bakeoff_safety_guard(jsonl, manifest, phase="phase_x", state_path=state)
    assert guard["request_count"] == 3
    assert guard["submission_allowed"] is True


def test_undeclared_phase_fails_closed(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    cids = [custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|P0|control|Study:y:hyp1|replicate_1")]
    jsonl = root / "phase_x" / "deepseek" / "batch_input.jsonl"
    manifest = root / "phase_x" / "deepseek" / "request_manifest.csv"
    _write_jsonl(jsonl, model=DEEPSEEK_MODEL_ID, custom_ids=cids)
    _write_manifest(manifest, [{"custom_id": c, "study_id": "Study", "request_stage": "phase_x"} for c in cids])
    with pytest.raises(RuntimeError, match="has not been declared"):
        g.scientific_bakeoff_safety_guard(jsonl, manifest, phase="phase_x", state_path=state)


def test_unapproved_custom_id_fails_even_under_cost_and_count_headroom(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    approved = [custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|P0|control|Study:y:hyp1|replicate_1")]
    g.declare_phase("phase_x", approved_custom_ids=set(approved), cost_cap_usd=1000.0, state_path=state)
    rogue = custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|ROGUE|control|Study:y:hyp1|replicate_1")
    jsonl = root / "phase_x" / "deepseek" / "batch_input.jsonl"
    manifest = root / "phase_x" / "deepseek" / "request_manifest.csv"
    _write_jsonl(jsonl, model=DEEPSEEK_MODEL_ID, custom_ids=[rogue])
    _write_manifest(manifest, [{"custom_id": rogue, "study_id": "Study", "request_stage": "phase_x"}])
    with pytest.raises(RuntimeError, match="not in phase"):
        g.scientific_bakeoff_safety_guard(jsonl, manifest, phase="phase_x", state_path=state)


def test_duplicate_previously_submitted_id_fails(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    cid = custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|P0|control|Study:y:hyp1|replicate_1")
    g.declare_phase("phase_x", approved_custom_ids={cid}, cost_cap_usd=1000.0, state_path=state)
    existing_state = json.loads(state.read_text())
    existing_state["submitted_custom_ids"] = [cid]
    state.write_text(json.dumps(existing_state))
    jsonl = root / "phase_x" / "deepseek" / "batch_input.jsonl"
    manifest = root / "phase_x" / "deepseek" / "request_manifest.csv"
    _write_jsonl(jsonl, model=DEEPSEEK_MODEL_ID, custom_ids=[cid])
    _write_manifest(manifest, [{"custom_id": cid, "study_id": "Study", "request_stage": "phase_x"}])
    with pytest.raises(RuntimeError, match="already recorded"):
        g.scientific_bakeoff_safety_guard(jsonl, manifest, phase="phase_x", state_path=state)


def test_qwen_model_refused_outright(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    cid = custom_id_from_request_key(f"{QWEN_MODEL_ID}|F|Study|P0|control|Study:y:hyp1|replicate_1")
    g.declare_phase("phase_x", approved_custom_ids={cid}, cost_cap_usd=1000.0, state_path=state)
    jsonl = root / "phase_x" / "qwen" / "batch_input.jsonl"
    manifest = root / "phase_x" / "qwen" / "request_manifest.csv"
    _write_jsonl(jsonl, model=QWEN_MODEL_ID, custom_ids=[cid])
    _write_manifest(manifest, [{"custom_id": cid, "study_id": "Study", "request_stage": "phase_x"}])
    with pytest.raises(RuntimeError, match="refuses these models outright"):
        g.scientific_bakeoff_safety_guard(jsonl, manifest, phase="phase_x", state_path=state)


def test_target_production_request_refused_during_development_phase(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    cid = custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|target|F0001|control|trust_post|replicate_1")
    g.declare_phase("phase_x", approved_custom_ids={cid}, cost_cap_usd=1000.0, state_path=state)
    jsonl = root / "phase_x" / "deepseek" / "batch_input.jsonl"
    manifest = root / "phase_x" / "deepseek" / "request_manifest.csv"
    _write_jsonl(jsonl, model=DEEPSEEK_MODEL_ID, custom_ids=[cid])
    _write_manifest(manifest, [{"custom_id": cid, "study_id": "target", "request_stage": "phase_x"}])
    with pytest.raises(RuntimeError, match="refuses target-production requests"):
        g.scientific_bakeoff_safety_guard(jsonl, manifest, phase="phase_x", state_path=state)


def test_phase_cost_cap_enforced(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    cids = [custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|P{i}|control|Study:y:hyp1|replicate_1") for i in range(5)]
    g.declare_phase("phase_x", approved_custom_ids=set(cids), cost_cap_usd=0.000001, state_path=state)
    jsonl = root / "phase_x" / "deepseek" / "batch_input.jsonl"
    manifest = root / "phase_x" / "deepseek" / "request_manifest.csv"
    _write_jsonl(jsonl, model=DEEPSEEK_MODEL_ID, custom_ids=cids, max_tokens=1024)
    _write_manifest(manifest, [{"custom_id": c, "study_id": "Study", "request_stage": "phase_x"} for c in cids])
    with pytest.raises(RuntimeError, match="cost cap exceeded"):
        g.scientific_bakeoff_safety_guard(jsonl, manifest, phase="phase_x", state_path=state)


def test_declaring_same_phase_twice_with_different_allowlist_fails(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    cid1 = custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|P0|control|Study:y:hyp1|replicate_1")
    cid2 = custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|P1|control|Study:y:hyp1|replicate_1")
    g.declare_phase("phase_x", approved_custom_ids={cid1}, cost_cap_usd=10.0, state_path=state)
    with pytest.raises(RuntimeError, match="already declared and differs"):
        g.declare_phase("phase_x", approved_custom_ids={cid2}, cost_cap_usd=10.0, state_path=state)


def test_off_phase_request_stage_fails(tmp_path, monkeypatch):
    root, state = _setup(tmp_path, monkeypatch)
    cid = custom_id_from_request_key(f"{DEEPSEEK_MODEL_ID}|F|Study|P0|control|Study:y:hyp1|replicate_1")
    g.declare_phase("phase_x", approved_custom_ids={cid}, cost_cap_usd=1000.0, state_path=state)
    jsonl = root / "phase_x" / "deepseek" / "batch_input.jsonl"
    manifest = root / "phase_x" / "deepseek" / "request_manifest.csv"
    _write_jsonl(jsonl, model=DEEPSEEK_MODEL_ID, custom_ids=[cid])
    _write_manifest(manifest, [{"custom_id": cid, "study_id": "Study", "request_stage": "some_other_phase"}])
    with pytest.raises(RuntimeError, match="do not carry request_stage"):
        g.scientific_bakeoff_safety_guard(jsonl, manifest, phase="phase_x", state_path=state)
