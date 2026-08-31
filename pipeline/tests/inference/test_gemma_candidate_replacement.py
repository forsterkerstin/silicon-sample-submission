"""Regression tests for the Qwen3.8 -> google/gemma-4-31B-it candidate swap.

All offline/synthetic -- none call a real model or submit anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import prepare_tiny_together_smoke_manifest as ptsm  # noqa: E402
from inference.model_config import load_model_config, model_candidates, model_engine_config  # noqa: E402
from inference import together_batch as tb  # noqa: E402
from inference.together_batch import (  # noqa: E402
    DEEPSEEK_MODEL_ID,
    GEMMA_MODEL_ID,
    GEMMA_PREFLIGHT_REQUEST_KEY,
    QWEN_MODEL_ID,
    TINY_SMOKE_FIRST_WAVE_REQUEST_KEYS,
    TINY_SMOKE_STAGE_B_REQUEST_KEYS,
    compute_engine_config_hash,
    smoke_scoped_custom_id,
    tiny_smoke_approved_pending_custom_ids,
)


def test_gemma_is_the_declared_g_and_f_candidate():
    assert model_candidates("g") == [DEEPSEEK_MODEL_ID, GEMMA_MODEL_ID]
    assert model_candidates("f") == [DEEPSEEK_MODEL_ID, GEMMA_MODEL_ID]


def test_qwen_retirement_recorded_and_engineering_only():
    cfg = load_model_config()
    history = cfg["model_selection"]["candidate_amendment_history"]
    assert len(history) == 1
    entry = history[0]
    assert entry["removed_candidate"] == QWEN_MODEL_ID
    assert entry["removed_reason_code"] == "ENGINEERING_SCREENED_OUT_PRE_SCIENTIFIC_EVALUATION"
    assert entry["added_candidate"] == GEMMA_MODEL_ID
    assert entry["model_selection_metrics_and_tie_break_rules_changed"] is False
    assert entry["g_and_f_still_selected_independently_via_frozen_external_validation_procedure"] is True
    assert entry["historical_qwen_smoke_artifacts_preserved"] is True


def test_model_selection_manifest_preserves_tie_break_rules_across_versions(tmp_path):
    import sys

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import final_offline_preinference_gate as gate

    result = gate.model_selection_manifest(tmp_path)
    payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert payload["g_candidate_selection"]["tie_break_rule"] == (
        "Prefer lower invalid-response rate, then lower cost, then deterministic lexical model id."
    )
    assert payload["f_candidate_selection"]["tie_break_rule"].startswith("Prefer lower LOSO RMSE")
    assert payload["g_candidate_selection"]["candidates"] == [DEEPSEEK_MODEL_ID, GEMMA_MODEL_ID]
    assert Path(result["versioned_path"]).exists()


def test_gemma_thinking_is_disabled_via_engine_config_not_prompt_text():
    engine_cfg = model_engine_config(GEMMA_MODEL_ID)
    assert engine_cfg["chat_template_kwargs"] == {"enable_thinking": False}
    # DeepSeek/Qwen must never pick up Gemma's engine config.
    assert "chat_template_kwargs" not in model_engine_config(DEEPSEEK_MODEL_ID)
    assert "chat_template_kwargs" not in model_engine_config(QWEN_MODEL_ID)


def test_gemma_chat_body_carries_chat_template_kwargs_deepseek_does_not():
    request = ptsm.build_engineering_capability_preflight_request(GEMMA_MODEL_ID)
    body = ptsm.chat_body(request)
    assert body["model"] == GEMMA_MODEL_ID
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    for message in request.messages:
        assert "<|think|>" not in message["content"]

    ds_request = ptsm.build_engineering_capability_preflight_request(DEEPSEEK_MODEL_ID)
    ds_body = ptsm.chat_body(ds_request)
    assert "chat_template_kwargs" not in ds_body


def test_gemma_logical_smoke_cases_match_deepseek_byte_for_byte(monkeypatch, tmp_path):
    # build_smoke_requests() -> build_external_pair() -> ptsm.rpv.export_archive_source_rows()
    # writes intermediate CSVs to ptsm.rpv.OUTPUT_DIR (a separate importlib-loaded
    # module instance, not the singleton `render_prompt_validation` import);
    # redirect it so this never touches outputs/prompt_validation/f_external_source_*.csv.
    monkeypatch.setattr(ptsm.rpv, "OUTPUT_DIR", tmp_path)
    gemma_first, _ = ptsm.build_smoke_requests(GEMMA_MODEL_ID)
    deepseek_first, _ = ptsm.build_smoke_requests(DEEPSEEK_MODEL_ID)
    assert len(gemma_first) == len(deepseek_first) == 12
    by_case_g = {(r.condition_id, r.outcome_id, r.role): r for r in gemma_first}
    by_case_d = {(r.condition_id, r.outcome_id, r.role): r for r in deepseek_first}
    assert set(by_case_g) == set(by_case_d)
    for case, g_req in by_case_g.items():
        d_req = by_case_d[case]
        assert g_req.messages == d_req.messages, f"prompt text diverged for {case}"
        assert g_req.response_schema == d_req.response_schema, f"schema diverged for {case}"
        assert g_req.seed == d_req.seed
        # Only the requested model and its engine-config hash may differ.
        assert g_req.requested_model != d_req.requested_model
    assert any(r.condition_id == "Extreme weather predictions" for r in gemma_first)
    extreme = next(r for r in gemma_first if r.condition_id == "Extreme weather predictions")
    assert "Texas" in extreme.messages[1]["content"]
    assert "flood" in extreme.messages[1]["content"].lower()


def test_gemma_custom_ids_have_zero_overlap_with_deepseek_and_qwen_first_wave(monkeypatch, tmp_path):
    monkeypatch.setattr(ptsm.rpv, "OUTPUT_DIR", tmp_path)
    gemma_first, gemma_second = ptsm.build_smoke_requests(GEMMA_MODEL_ID)
    gemma_ids = {r.custom_id for r in gemma_first} | {r.custom_id for r in gemma_second}
    deepseek_ids = {smoke_scoped_custom_id(DEEPSEEK_MODEL_ID, k) for k in TINY_SMOKE_FIRST_WAVE_REQUEST_KEYS}
    qwen_ids = {smoke_scoped_custom_id(QWEN_MODEL_ID, k) for k in TINY_SMOKE_FIRST_WAVE_REQUEST_KEYS}
    assert gemma_ids.isdisjoint(deepseek_ids)
    assert gemma_ids.isdisjoint(qwen_ids)
    assert len(gemma_ids) == 14  # 12 first-wave + 2 placeholder stage-b


def test_approved_pending_custom_ids_is_exactly_the_intended_17():
    approved = tiny_smoke_approved_pending_custom_ids()
    assert len(approved) == 17
    expected = {smoke_scoped_custom_id(GEMMA_MODEL_ID, key) for key in TINY_SMOKE_FIRST_WAVE_REQUEST_KEYS}
    for key in TINY_SMOKE_STAGE_B_REQUEST_KEYS:
        expected.add(smoke_scoped_custom_id(DEEPSEEK_MODEL_ID, key))
        expected.add(smoke_scoped_custom_id(GEMMA_MODEL_ID, key))
    expected.add(smoke_scoped_custom_id(GEMMA_MODEL_ID, GEMMA_PREFLIGHT_REQUEST_KEY))
    assert approved == expected


def test_qwen_stage_b_is_never_in_the_approved_allowlist():
    approved = tiny_smoke_approved_pending_custom_ids()
    qwen_stage_b_ids = {smoke_scoped_custom_id(QWEN_MODEL_ID, key) for key in TINY_SMOKE_STAGE_B_REQUEST_KEYS}
    assert qwen_stage_b_ids.isdisjoint(approved)


def test_qwen_stage_b_jsonl_is_rejected_by_the_guard_end_to_end(monkeypatch, tmp_path):
    root = tmp_path / "tiny_pre_api"
    state = root / "smoke_submission_state.json"
    monkeypatch.setattr(tb, "TINY_SMOKE_ROOT", root)
    monkeypatch.setattr(tb, "TINY_SMOKE_STATE_PATH", state)
    root.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "schema_version": "tiny_smoke_submission_state_v1",
                "smoke_cost_cap_usd": tb.SMOKE_COST_CAP_USD,
                "global_max_new_requests": tb.SMOKE_GLOBAL_MAX_NEW_REQUESTS,
                "submitted_custom_ids": [],
                "successful_custom_ids": [],
                "cumulative_new_requests": 0,
                "estimated_worst_case_cost_usd": 0.0,
            }
        ),
        encoding="utf-8",
    )
    qwen_ids = [smoke_scoped_custom_id(QWEN_MODEL_ID, key) for key in TINY_SMOKE_STAGE_B_REQUEST_KEYS]
    jsonl = root / "Qwen_Qwen3.8-2.4T-A95B" / "stage_b_real" / "batch_input.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl, "w", encoding="utf-8") as f:
        for cid in qwen_ids:
            f.write(json.dumps({"custom_id": cid, "body": {"model": QWEN_MODEL_ID, "messages": [{"role": "user", "content": "x"}], "max_tokens": 1024}}) + "\n")
    # Qwen is no longer a declared candidate -- rejected before the allowlist is even consulted.
    with pytest.raises(RuntimeError, match="does not identify one declared smoke candidate"):
        tb.tiny_smoke_safety_guard(jsonl, for_submit=True)


def test_arbitrary_unapproved_custom_id_fails_closed_even_under_the_request_cap(monkeypatch, tmp_path):
    root = tmp_path / "tiny_pre_api"
    state = root / "smoke_submission_state.json"
    monkeypatch.setattr(tb, "TINY_SMOKE_ROOT", root)
    monkeypatch.setattr(tb, "TINY_SMOKE_STATE_PATH", state)
    root.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "schema_version": "tiny_smoke_submission_state_v1",
                "smoke_cost_cap_usd": tb.SMOKE_COST_CAP_USD,
                "global_max_new_requests": tb.SMOKE_GLOBAL_MAX_NEW_REQUESTS,
                "submitted_custom_ids": [],
                "successful_custom_ids": [],
                "cumulative_new_requests": 0,
                "estimated_worst_case_cost_usd": 0.0,
            }
        ),
        encoding="utf-8",
    )
    # A single-request preflight batch, well under SMOKE_GLOBAL_MAX_NEW_REQUESTS
    # and matching the expected-count for its wave, but for a custom_id that was
    # never approved -- the count/cost caps alone would let this through, so
    # this proves the allowlist is the actual gate.
    jsonl = root / "google_gemma-4-31B-it" / "preflight" / "batch_input.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "custom_id": "not-an-approved-id",
                    "body": {"model": GEMMA_MODEL_ID, "messages": [{"role": "user", "content": "x"}], "max_tokens": 128},
                }
            )
        )
    with pytest.raises(RuntimeError, match="not in the approved pending allowlist"):
        tb.tiny_smoke_safety_guard(jsonl, for_submit=True)


def test_cumulative_worst_case_cost_across_all_completed_smoke_stays_under_cap():
    """All engineering smoke (DeepSeek/Qwen/Gemma first-wave, Gemma preflight,
    both real Stage-Bs) has been submitted; this asserts on the final ledger
    rather than re-deriving projections from now-already-submitted jsonl files
    (which the guard's own duplicate check would now correctly refuse)."""
    state = json.loads(tb.TINY_SMOKE_STATE_PATH.read_text(encoding="utf-8"))
    total_requests = int(state["cumulative_new_requests"])
    total_cost = float(state["estimated_worst_case_cost_usd"])
    assert total_requests == 41
    assert total_requests == tb.SMOKE_GLOBAL_MAX_NEW_REQUESTS
    assert len(state["submitted_custom_ids"]) == 41
    assert len(state["submitted_custom_ids"]) == len(set(state["submitted_custom_ids"]))
    assert total_cost <= tb.SMOKE_COST_CAP_USD
    assert total_cost < 0.25  # comfortable margin, not razor-thin


def test_engine_config_hash_distinguishes_gemma_from_deepseek():
    assert compute_engine_config_hash(DEEPSEEK_MODEL_ID) == ""
    assert compute_engine_config_hash(QWEN_MODEL_ID) == ""
    gemma_hash = compute_engine_config_hash(GEMMA_MODEL_ID)
    assert gemma_hash != ""
    assert compute_engine_config_hash(GEMMA_MODEL_ID) == gemma_hash  # deterministic
