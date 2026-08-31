"""Tests for the least-privilege Orchinik domain-confirmation v2 submission
guard (inference/orchinik_domain_confirmation_guard.py) and its CLI wiring
in scripts/together_batch.py.

Two tests (test_correct_*_manifest_passes_real_guard) exercise the REAL
frozen canonical files read-only, end to end, proving the actual production
wiring is correct; every other test uses a tiny synthetic fixture with a
monkeypatched PHASES table (auto-reverted by pytest at test teardown), so no
test ever needs to alter -- or even touch -- the real canonical manifests/
jsonl/serving-amendment files. Every guard/declare call uses an isolated
tmp_path state file; none writes to the real
orchinik_domain_confirmation_submission_state.json. A module-scoped autouse
fixture snapshots every real frozen artifact this module reads and asserts
byte-for-byte identity after the whole module has run -- the same
frozen-artifact-mutation-regression class of check already applied elsewhere
this session (see tests/scripts/test_orchinik_g_domain_confirmation_build.py).
No Together API calls are made anywhere in this file."""

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

import inference.orchinik_domain_confirmation_guard as guard_mod  # noqa: E402
from inference.orchinik_domain_confirmation_guard import OrchinikDomainConfirmationNotAuthorized  # noqa: E402
import together_batch as tb_mod  # noqa: E402

MANIFEST_FIELDS = ["request_key", "custom_id", "study_id", "requested_model", "request_stage"]

REAL_FROZEN_PATHS = [
    PIPELINE_ROOT / "outputs" / "domain_validation" / "frozen_orchinik_g_domain_confirmation.json",
    guard_mod.SERVING_AMENDMENT_PATH,
    guard_mod.SERVING_AMENDMENT_SHA256_PATH,
    guard_mod.PHASES["orchinik_g_domain_confirmation_v2_gemma"]["manifest_path"],
    guard_mod.PHASES["orchinik_g_domain_confirmation_v2_gemma"]["jsonl_path"],
    guard_mod.PHASES["orchinik_g_domain_confirmation_v2_deepseek"]["manifest_path"],
    guard_mod.PHASES["orchinik_g_domain_confirmation_v2_deepseek"]["jsonl_path"],
]


@pytest.fixture(scope="module", autouse=True)
def _verify_no_frozen_artifact_mutation():
    before = {p: (p.read_bytes() if p.exists() else None) for p in REAL_FROZEN_PATHS}
    yield
    for p in REAL_FROZEN_PATHS:
        after = p.read_bytes() if p.exists() else None
        assert after == before[p], f"frozen artifact mutated by the guard test suite: {p}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path, *, manifest_model: str, body_model: str, n: int, study_id="orchinik2024_bovitz", request_stage="domain_confirmation", suffix="|fmt_v2", max_tokens=100):
    manifest_path = tmp_path / "request_manifest.csv"
    jsonl_path = tmp_path / "batch_input.jsonl"
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf, open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for i in range(n):
            request_key = f"G_EXTERNAL|{study_id}|resp{i}|replicate_1{suffix}"
            custom_id = f"synthetic-{i}"
            writer.writerow({"request_key": request_key, "custom_id": custom_id, "study_id": study_id, "requested_model": manifest_model, "request_stage": request_stage})
            body = {"model": body_model, "messages": [{"role": "user", "content": "x" * 40}], "max_tokens": max_tokens}
            jf.write(json.dumps({"custom_id": custom_id, "body": body}) + "\n")
    return manifest_path, jsonl_path


def _install_synthetic_phase(monkeypatch, tmp_path, phase_name, *, n=3, expected_request_count=None, cost_cap_usd=1.0, manifest_model="google/gemma-4-31B-it", body_model=None, **fixture_kwargs):
    body_model = body_model or manifest_model
    manifest_path, jsonl_path = _write_fixture(tmp_path, manifest_model=manifest_model, body_model=body_model, n=n, **fixture_kwargs)
    spec = {
        "model": manifest_model,
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

    amendment_path = tmp_path / "serving_amendment.json"
    amendment_path.write_text("{}", encoding="utf-8")
    amendment_sha = _sha256(amendment_path)
    amendment_sha_path = tmp_path / "serving_amendment.sha256.txt"
    amendment_sha_path.write_text(amendment_sha + "\n", encoding="utf-8")
    monkeypatch.setattr(guard_mod, "SERVING_AMENDMENT_PATH", amendment_path)
    monkeypatch.setattr(guard_mod, "SERVING_AMENDMENT_SHA256_PATH", amendment_sha_path)
    monkeypatch.setattr(guard_mod, "EXPECTED_SERVING_AMENDMENT_SHA256", amendment_sha)
    return spec


# 1 & 2. correct real manifests pass their own guard end to end.


def test_correct_gemma_manifest_passes_real_guard(tmp_path):
    spec = guard_mod.PHASES["orchinik_g_domain_confirmation_v2_gemma"]
    if not spec["jsonl_path"].exists():
        pytest.skip("real Orchinik v2 gemma manifest not built in this environment")
    guard_mod.declare_orchinik_domain_confirmation_phase("orchinik_g_domain_confirmation_v2_gemma", state_path=tmp_path / "state.json")
    result = guard_mod.orchinik_domain_confirmation_safety_guard(spec["jsonl_path"], phase="orchinik_g_domain_confirmation_v2_gemma", state_path=tmp_path / "state.json")
    assert result["submission_allowed"] is True
    assert result["request_count"] == 2545
    assert result["model"] == "google/gemma-4-31B-it"
    assert result["automatic_follow_on_inference_authorized"] is False


def test_correct_deepseek_manifest_passes_real_guard(tmp_path):
    spec = guard_mod.PHASES["orchinik_g_domain_confirmation_v2_deepseek"]
    if not spec["jsonl_path"].exists():
        pytest.skip("real Orchinik v2 deepseek manifest not built in this environment")
    guard_mod.declare_orchinik_domain_confirmation_phase("orchinik_g_domain_confirmation_v2_deepseek", state_path=tmp_path / "state.json")
    result = guard_mod.orchinik_domain_confirmation_safety_guard(spec["jsonl_path"], phase="orchinik_g_domain_confirmation_v2_deepseek", state_path=tmp_path / "state.json")
    assert result["submission_allowed"] is True
    assert result["request_count"] == 2545
    assert result["model"] == "deepseek-ai/DeepSeek-V4-Pro-0813"


# 3. one-byte alteration of the canonical manifest fails.


def test_one_byte_manifest_alteration_fails(monkeypatch, tmp_path):
    spec = _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase")
    with open(spec["manifest_path"], "ab") as f:
        f.write(b"X")
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="manifest.*SHA256 mismatch"):
        guard_mod.declare_orchinik_domain_confirmation_phase("synthetic_phase", state_path=tmp_path / "state.json")


# 4. wrong model (request body model differs from the phase's frozen model) fails.


def test_wrong_model_fails(monkeypatch, tmp_path):
    _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase", manifest_model="google/gemma-4-31B-it", body_model="deepseek-ai/DeepSeek-V4-Pro-0813")
    guard_mod.declare_orchinik_domain_confirmation_phase("synthetic_phase", state_path=tmp_path / "state.json")
    jsonl_path = guard_mod.PHASES["synthetic_phase"]["jsonl_path"]
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="only allows model"):
        guard_mod.orchinik_domain_confirmation_safety_guard(jsonl_path, phase="synthetic_phase", state_path=tmp_path / "state.json")


# 5. wrong request count (canonical manifest doesn't have exactly the
# phase's declared expected_request_count) fails at declare time.


def test_wrong_request_count_fails(monkeypatch, tmp_path):
    _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase", n=3, expected_request_count=5)
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="expected exactly 5"):
        guard_mod.declare_orchinik_domain_confirmation_phase("synthetic_phase", state_path=tmp_path / "state.json")


# 6. worst-case cost above the frozen phase cap fails.


def test_cost_above_frozen_cap_fails(monkeypatch, tmp_path):
    _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase", n=3, cost_cap_usd=0.0000001, max_tokens=100)
    guard_mod.declare_orchinik_domain_confirmation_phase("synthetic_phase", state_path=tmp_path / "state.json")
    jsonl_path = guard_mod.PHASES["synthetic_phase"]["jsonl_path"]
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="cost cap exceeded"):
        guard_mod.orchinik_domain_confirmation_safety_guard(jsonl_path, phase="synthetic_phase", state_path=tmp_path / "state.json")


# 7 & 8. cross-model masquerading: the real Gemma manifest cannot be
# submitted under the DeepSeek phase, and vice versa.


def test_gemma_manifest_cannot_be_submitted_under_deepseek_phase(tmp_path):
    gemma_spec = guard_mod.PHASES["orchinik_g_domain_confirmation_v2_gemma"]
    deepseek_spec = guard_mod.PHASES["orchinik_g_domain_confirmation_v2_deepseek"]
    if not gemma_spec["jsonl_path"].exists() or not deepseek_spec["jsonl_path"].exists():
        pytest.skip("real Orchinik v2 manifests not built in this environment")
    guard_mod.declare_orchinik_domain_confirmation_phase("orchinik_g_domain_confirmation_v2_deepseek", state_path=tmp_path / "state.json")
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="only accepts its own canonical jsonl"):
        guard_mod.orchinik_domain_confirmation_safety_guard(gemma_spec["jsonl_path"], phase="orchinik_g_domain_confirmation_v2_deepseek", state_path=tmp_path / "state.json")


def test_deepseek_manifest_cannot_be_submitted_under_gemma_phase(tmp_path):
    gemma_spec = guard_mod.PHASES["orchinik_g_domain_confirmation_v2_gemma"]
    deepseek_spec = guard_mod.PHASES["orchinik_g_domain_confirmation_v2_deepseek"]
    if not gemma_spec["jsonl_path"].exists() or not deepseek_spec["jsonl_path"].exists():
        pytest.skip("real Orchinik v2 manifests not built in this environment")
    guard_mod.declare_orchinik_domain_confirmation_phase("orchinik_g_domain_confirmation_v2_gemma", state_path=tmp_path / "state.json")
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="only accepts its own canonical jsonl"):
        guard_mod.orchinik_domain_confirmation_safety_guard(deepseek_spec["jsonl_path"], phase="orchinik_g_domain_confirmation_v2_gemma", state_path=tmp_path / "state.json")


# 9. an arbitrary domain-validation JSONL (e.g. the v1 Orchinik manifest,
# which is scientifically real but not the v2-amended canonical file) cannot
# use either guarded phase.


def test_arbitrary_domain_validation_jsonl_cannot_use_these_phases(tmp_path):
    v1_jsonl = PIPELINE_ROOT / "outputs" / "domain_validation" / "orchinik_g_domain_confirmation" / "google_gemma-4-31B-it" / "batch_input.jsonl"
    if not v1_jsonl.exists():
        pytest.skip("real v1 Orchinik manifest not built in this environment")
    assert tb_mod.is_orchinik_domain_confirmation_v2_jsonl(v1_jsonl) is False
    guard_mod.declare_orchinik_domain_confirmation_phase("orchinik_g_domain_confirmation_v2_gemma", state_path=tmp_path / "state.json")
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="only accepts its own canonical jsonl"):
        guard_mod.orchinik_domain_confirmation_safety_guard(v1_jsonl, phase="orchinik_g_domain_confirmation_v2_gemma", state_path=tmp_path / "state.json")


# 10. a generic-path submission (a jsonl living outside every guarded root)
# cannot be routed through the Orchinik guard by together_batch.py's
# dispatcher -- is_orchinik_domain_confirmation_v2_jsonl must not claim it.


def test_generic_path_cannot_masquerade_as_guarded_orchinik_phase(tmp_path):
    unrelated = tmp_path / "unrelated" / "batch_input.jsonl"
    unrelated.parent.mkdir(parents=True, exist_ok=True)
    unrelated.write_text('{"custom_id": "x", "body": {}}\n', encoding="utf-8")
    assert tb_mod.is_orchinik_domain_confirmation_v2_jsonl(unrelated) is False
    assert tb_mod.is_target_production_jsonl(unrelated) is False
    assert tb_mod.is_scientific_bakeoff_jsonl(unrelated) is False


def test_unknown_phase_name_refused(tmp_path):
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="unknown Orchinik domain-confirmation phase"):
        guard_mod.declare_orchinik_domain_confirmation_phase("not_a_real_phase", state_path=tmp_path / "state.json")
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="unknown Orchinik domain-confirmation phase"):
        guard_mod.orchinik_domain_confirmation_safety_guard(tmp_path / "whatever.jsonl", phase="not_a_real_phase", state_path=tmp_path / "state.json")


def test_combined_cap_constants_match_frozen_scope():
    assert guard_mod.COMBINED_TOTAL_REQUEST_COUNT == 5090
    assert guard_mod.COMBINED_TOTAL_COST_CAP_USD == 29.835322
    assert guard_mod.PHASES["orchinik_g_domain_confirmation_v2_gemma"]["expected_request_count"] == 2545
    assert guard_mod.PHASES["orchinik_g_domain_confirmation_v2_deepseek"]["expected_request_count"] == 2545
    assert guard_mod.PHASES["orchinik_g_domain_confirmation_v2_gemma"]["cost_cap_usd"] == 3.699383
    assert guard_mod.PHASES["orchinik_g_domain_confirmation_v2_deepseek"]["cost_cap_usd"] == 26.135939


def test_combined_cap_refuses_a_third_synthetic_phase_once_scope_would_be_exceeded(monkeypatch, tmp_path):
    """The combined 5090/$29.835322 cap is checked against state shared
    across phases -- simulate exceeding it (via a synthetic phase, never the
    real ones) by pre-seeding state with the full combined budget already
    spent, then confirming even a 1-request synthetic phase is refused."""
    _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase", n=1, cost_cap_usd=1000.0, max_tokens=10)
    guard_mod.declare_orchinik_domain_confirmation_phase("synthetic_phase", state_path=tmp_path / "state.json")
    state_path = tmp_path / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["combined_cumulative_requests"] = guard_mod.COMBINED_TOTAL_REQUEST_COUNT
    state["combined_cumulative_worst_case_cost_usd"] = guard_mod.COMBINED_TOTAL_COST_CAP_USD
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    jsonl_path = guard_mod.PHASES["synthetic_phase"]["jsonl_path"]
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="combined Orchinik domain-confirmation"):
        guard_mod.orchinik_domain_confirmation_safety_guard(jsonl_path, phase="synthetic_phase", state_path=state_path)


def test_duplicate_submission_refused(monkeypatch, tmp_path):
    _install_synthetic_phase(monkeypatch, tmp_path, "synthetic_phase", n=2)
    guard_mod.declare_orchinik_domain_confirmation_phase("synthetic_phase", state_path=tmp_path / "state.json")
    jsonl_path = guard_mod.PHASES["synthetic_phase"]["jsonl_path"]
    state_path = tmp_path / "state.json"
    first = guard_mod.orchinik_domain_confirmation_safety_guard(jsonl_path, phase="synthetic_phase", state_path=state_path)
    guard_mod.record_orchinik_domain_confirmation_submission(first, {"id": "fake-batch-id"}, state_path=state_path)
    with pytest.raises(OrchinikDomainConfirmationNotAuthorized, match="already recorded as submitted"):
        guard_mod.orchinik_domain_confirmation_safety_guard(jsonl_path, phase="synthetic_phase", state_path=state_path)
