"""Tests for the Consensus-exact submission guard. Mirrors the established
pattern from test_target_g_completion_guard.py/test_orchinik_domain_confirmation_guard.py:
real STEP_1 manifest passes end to end; synthetic fixtures for mismatch
cases; the disabled legacy manifest is refused outright; a module-scoped
fixture asserts no real frozen artifact is mutated by this test suite."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import inference.consensus_exact_guard as guard_mod  # noqa: E402
from inference.consensus_exact_guard import ConsensusExactNotAuthorized  # noqa: E402

pytestmark = pytest.mark.skipif(not guard_mod.PHASES["consensus_exact_step1"]["jsonl_path"].exists(), reason="Consensus-exact STEP_1 manifest not built in this environment")

REAL_FROZEN_PATHS = [
    guard_mod.PHASES["consensus_exact_step1"]["manifest_path"],
    guard_mod.PHASES["consensus_exact_step1"]["jsonl_path"],
]


@pytest.fixture(scope="module", autouse=True)
def _verify_no_frozen_artifact_mutation():
    before = {p: p.read_bytes() for p in REAL_FROZEN_PATHS}
    yield
    for p in REAL_FROZEN_PATHS:
        assert p.read_bytes() == before[p], f"frozen artifact mutated by the Consensus-exact guard test suite: {p}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_step1_manifest_passes_guard_end_to_end(tmp_path):
    spec = guard_mod.PHASES["consensus_exact_step1"]
    guard_mod.declare_consensus_exact_phase("consensus_exact_step1", state_path=tmp_path / "state.json")
    result = guard_mod.consensus_exact_safety_guard(spec["jsonl_path"], phase="consensus_exact_step1", state_path=tmp_path / "state.json")
    assert result["submission_allowed"] is True
    assert result["request_count"] == 1000
    assert result["model"] == "google/gemma-4-31B-it"
    assert result["automatic_follow_on_inference_authorized"] is False


def test_legacy_manifest_refused_outright(tmp_path):
    guard_mod.declare_consensus_exact_phase("consensus_exact_step1", state_path=tmp_path / "state.json")
    with pytest.raises(ConsensusExactNotAuthorized, match="permanently disabled"):
        guard_mod.consensus_exact_safety_guard(guard_mod.LEGACY_CONSENSUS_A_COMPLETION_JSONL, phase="consensus_exact_step1", state_path=tmp_path / "state.json")


def test_unknown_phase_name_is_refused():
    with pytest.raises(ConsensusExactNotAuthorized):
        guard_mod.declare_consensus_exact_phase("consensus_exact_bogus_phase", state_path=Path("/tmp/unused.json"))


def test_all_four_phases_now_exist_and_declare_successfully(tmp_path):
    for phase in ("consensus_exact_step2", "consensus_exact_step3", "consensus_exact_outcomes"):
        guard_mod.declare_consensus_exact_phase(phase, state_path=tmp_path / "state.json")


def test_one_byte_manifest_alteration_fails(monkeypatch, tmp_path):
    manifest_path = tmp_path / "request_manifest.csv"
    jsonl_path = tmp_path / "batch_input.jsonl"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["custom_id", "study_id", "condition_id", "request_stage", "requested_model", "request_key"])
        writer.writeheader()
        writer.writerow({"custom_id": "c1", "study_id": "target", "condition_id": "Consensus", "request_stage": "consensus_exact_step1", "requested_model": "google/gemma-4-31B-it", "request_key": "k"})
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"custom_id": "c1", "body": {"model": "google/gemma-4-31B-it", "messages": [{"role": "user", "content": "x" * 40}], "max_tokens": 10}}) + "\n")

    spec = {
        "model": "google/gemma-4-31B-it",
        "request_stage": "consensus_exact_step1",
        "manifest_path": manifest_path,
        "jsonl_path": jsonl_path,
        "manifest_sha256": _sha256(manifest_path),
        "jsonl_sha256": _sha256(jsonl_path),
        "expected_request_count": 1,
        "cost_cap_usd": 1.0,
    }
    phases = dict(guard_mod.PHASES)
    phases["synthetic_phase"] = spec
    monkeypatch.setattr(guard_mod, "PHASES", phases)

    with open(manifest_path, "ab") as f:
        f.write(b"X")
    with pytest.raises(ConsensusExactNotAuthorized, match="manifest.*SHA256 mismatch"):
        guard_mod.declare_consensus_exact_phase("synthetic_phase", state_path=tmp_path / "state.json")
