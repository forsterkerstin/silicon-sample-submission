"""Tests for the least-privilege target G Wave-1 completion submission
guard. Mirrors tests/inference/test_orchinik_domain_confirmation_guard.py's
structure: two tests exercise the REAL frozen canonical files read-only end
to end; the rest use tiny synthetic fixtures with a monkeypatched PHASES
table, so no test ever needs to alter -- or even touch -- the real
canonical manifests/jsonl/amendment files. Every guard/declare call uses an
isolated tmp_path state file. No Together API calls are made anywhere in
this file."""

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

import inference.target_g_completion_guard as guard_mod  # noqa: E402
from inference.target_g_completion_guard import TargetGCompletionNotAuthorized  # noqa: E402
import together_batch as tb_mod  # noqa: E402

MANIFEST_FIELDS = ["request_key", "custom_id", "study_id", "requested_model", "request_stage"]

_disabled_consensus_a = guard_mod.DISABLED_LEGACY_CONSENSUS_A_COMPLETION_PHASE["target_g_wave1_completion_consensus_a"]
REAL_FROZEN_PATHS = [
    guard_mod.G_V2_AMENDMENT_PATH,
    guard_mod.PHASES["target_g_wave1_completion_standard"]["manifest_path"],
    guard_mod.PHASES["target_g_wave1_completion_standard"]["jsonl_path"],
    _disabled_consensus_a["manifest_path"],
    _disabled_consensus_a["jsonl_path"],
]

pytestmark = pytest.mark.skipif(not guard_mod.PHASES["target_g_wave1_completion_standard"]["jsonl_path"].exists(), reason="target G completion manifests not built in this environment")


@pytest.fixture(scope="module", autouse=True)
def _verify_no_frozen_artifact_mutation():
    before = {p: (p.read_bytes() if p.exists() else None) for p in REAL_FROZEN_PATHS}
    yield
    for p in REAL_FROZEN_PATHS:
        after = p.read_bytes() if p.exists() else None
        assert after == before[p], f"frozen artifact mutated by the completion guard test suite: {p}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path, *, manifest_model: str, body_model: str, request_stage: str, n: int, study_id="target", suffix="|fmt_v2", max_tokens=100):
    manifest_path = tmp_path / "request_manifest.csv"
    jsonl_path = tmp_path / "batch_input.jsonl"
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf, open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for i in range(n):
            request_key = f"G|LP{i:04d}|control|replicate_2{suffix}"
            custom_id = f"synthetic-{i}"
            writer.writerow({"request_key": request_key, "custom_id": custom_id, "study_id": study_id, "requested_model": manifest_model, "request_stage": request_stage})
            body = {"model": body_model, "messages": [{"role": "user", "content": "x" * 40}], "max_tokens": max_tokens}
            jf.write(json.dumps({"custom_id": custom_id, "body": body}) + "\n")
    return manifest_path, jsonl_path


def _install_synthetic_phase(monkeypatch, tmp_path, phase_name, *, n=3, expected_request_count=None, cost_cap_usd=1.0, request_stage="standard", manifest_model="google/gemma-4-31B-it", body_model=None, **fixture_kwargs):
    body_model = body_model or manifest_model
    manifest_path, jsonl_path = _write_fixture(tmp_path, manifest_model=manifest_model, body_model=body_model, request_stage=request_stage, n=n, **fixture_kwargs)
    spec = {
        "model": manifest_model,
        "request_stage": request_stage,
        "manifest_path": manifest_path,
        "jsonl_path": jsonl_path,
        "manifest_sha256": _sha256(manifest_path),
        "jsonl_sha256": _sha256(jsonl_path),
        "expected_request_count": expected_request_count if expected_request_count is not None else n,
        "cost_cap_usd": cost_cap_usd,
    }
    phases = dict(guard_mod.PHASES)
    phases[phase_name] = spec
    monkeypatch.setattr(guard_mod, "PHASES", phases)

    amendment_path = tmp_path / "amendment.json"
    amendment_path.write_text("{}", encoding="utf-8")
    amendment_sha = _sha256(amendment_path)
    monkeypatch.setattr(guard_mod, "G_V2_AMENDMENT_PATH", amendment_path)
    monkeypatch.setattr(guard_mod, "EXPECTED_G_V2_AMENDMENT_SHA256", amendment_sha)
    return spec


# real manifests pass their own guard end to end.


def test_correct_standard_manifest_passes_real_guard(tmp_path):
    spec = guard_mod.PHASES["target_g_wave1_completion_standard"]
    guard_mod.declare_target_g_completion_phase("target_g_wave1_completion_standard", state_path=tmp_path / "state.json")
    result = guard_mod.target_g_completion_safety_guard(spec["jsonl_path"], phase="target_g_wave1_completion_standard", state_path=tmp_path / "state.json")
    assert result["submission_allowed"] is True
    assert result["request_count"] == 1401
    assert result["model"] == "google/gemma-4-31B-it"
    assert result["attempt_3_authorized"] is False
    assert result["consensus_stage_b_authorized"] is False


# The legacy 82-request Consensus-A completion phase was DISABLED after the
# public-instrument audit found FAIL_MATERIAL_SEQUENCE_MISMATCH -- it must
# fail closed by unknown-phase-name, not merely be flagged/skipped.
def test_consensus_a_phase_is_permanently_disabled(tmp_path):
    with pytest.raises(TargetGCompletionNotAuthorized, match="unknown target G completion phase"):
        guard_mod.declare_target_g_completion_phase("target_g_wave1_completion_consensus_a", state_path=tmp_path / "state.json")
    with pytest.raises(TargetGCompletionNotAuthorized):
        guard_mod.target_g_completion_safety_guard(_disabled_consensus_a["jsonl_path"], phase="target_g_wave1_completion_consensus_a", state_path=tmp_path / "state.json")


def test_consensus_a_jsonl_refused_even_under_the_standard_phase(tmp_path):
    guard_mod.declare_target_g_completion_phase("target_g_wave1_completion_standard", state_path=tmp_path / "state.json")
    with pytest.raises(TargetGCompletionNotAuthorized, match="only accepts its own canonical jsonl"):
        guard_mod.target_g_completion_safety_guard(_disabled_consensus_a["jsonl_path"], phase="target_g_wave1_completion_standard", state_path=tmp_path / "state.json")


# 16. one-byte manifest alteration fails.


def test_one_byte_manifest_alteration_fails(monkeypatch, tmp_path):
    spec = _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase")
    with open(spec["manifest_path"], "ab") as f:
        f.write(b"X")
    with pytest.raises(TargetGCompletionNotAuthorized, match="manifest.*SHA256 mismatch"):
        guard_mod.declare_target_g_completion_phase("synthetic_phase", state_path=tmp_path / "state.json")


# 17. wrong count/model/cost all fail.


def test_wrong_model_fails(monkeypatch, tmp_path):
    _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase", manifest_model="google/gemma-4-31B-it", body_model="deepseek-ai/DeepSeek-V4-Pro-0813")
    guard_mod.declare_target_g_completion_phase("synthetic_phase", state_path=tmp_path / "state.json")
    jsonl_path = guard_mod.PHASES["synthetic_phase"]["jsonl_path"]
    with pytest.raises(TargetGCompletionNotAuthorized, match="only allows model"):
        guard_mod.target_g_completion_safety_guard(jsonl_path, phase="synthetic_phase", state_path=tmp_path / "state.json")


def test_wrong_request_count_fails(monkeypatch, tmp_path):
    _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase", n=3, expected_request_count=5)
    with pytest.raises(TargetGCompletionNotAuthorized, match="expected exactly 5"):
        guard_mod.declare_target_g_completion_phase("synthetic_phase", state_path=tmp_path / "state.json")


def test_cost_above_frozen_cap_fails(monkeypatch, tmp_path):
    _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase", n=3, cost_cap_usd=0.0000001, max_tokens=100)
    guard_mod.declare_target_g_completion_phase("synthetic_phase", state_path=tmp_path / "state.json")
    jsonl_path = guard_mod.PHASES["synthetic_phase"]["jsonl_path"]
    with pytest.raises(TargetGCompletionNotAuthorized, match="cost cap exceeded"):
        guard_mod.target_g_completion_safety_guard(jsonl_path, phase="synthetic_phase", state_path=tmp_path / "state.json")


# 19. attempt-3 cannot auto-submit -- no attempt-3 phase exists at all
# (fails closed by unknown-phase-name, not by a bypassable runtime check),
# and neither guard result ever authorizes it. Also confirms the disabled
# Consensus-A phase is gone from the live PHASES table entirely (not just
# unreachable).


def test_no_attempt_3_or_consensus_b_or_disabled_consensus_a_phase_exists():
    assert set(guard_mod.PHASES) == {"target_g_wave1_completion_standard"}
    with pytest.raises(TargetGCompletionNotAuthorized, match="unknown target G completion phase"):
        guard_mod.declare_target_g_completion_phase("target_g_wave1_attempt_3", state_path=Path("/tmp/unused_state.json"))


def test_generic_path_cannot_masquerade_as_guarded_completion_phase(tmp_path):
    unrelated = tmp_path / "unrelated" / "batch_input.jsonl"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text('{"custom_id": "x", "body": {}}\n', encoding="utf-8")
    assert tb_mod.is_target_g_completion_jsonl(unrelated) is False


def test_completion_root_is_routed_before_generic_target_production_check():
    # wave1_g_completion/ is nested under outputs/target_production/, so
    # is_target_production_jsonl also matches it -- the CLI dispatcher must
    # check is_target_g_completion_jsonl first (see scripts/together_batch.py's
    # elif ordering) or these requests would be wrongly routed through
    # target_production_safety_guard, which has no allowlist for them.
    standard_jsonl = guard_mod.PHASES["target_g_wave1_completion_standard"]["jsonl_path"]
    assert tb_mod.is_target_g_completion_jsonl(standard_jsonl) is True
    assert tb_mod.is_target_production_jsonl(standard_jsonl) is True


# 20. no frozen scientific G/S2 artifact is modified (covered by the
# module-scoped autouse fixture above; this test additionally confirms S2's
# own frozen artifact specifically, since the guard module never imports or
# touches it).


def test_s2_artifact_never_referenced_by_this_guard():
    import inspect

    source = inspect.getsource(guard_mod)
    assert "s2_" not in source.lower() and "mconst" not in source.lower() and "gshape" not in source.lower()
