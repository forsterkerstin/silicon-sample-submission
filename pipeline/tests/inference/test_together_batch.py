from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import survey_content as sc  # noqa: E402
from inference.prompts import build_g_prompt_render  # noqa: E402
import pytest

from inference import together_batch as tb  # noqa: E402
from inference.together_batch import custom_id_from_request_key, iter_f_requests, iter_g_requests, load_consensus_stage_a_success, parse_batch_results, prepare_batch, split_jsonl_file  # noqa: E402


def test_custom_id_is_deterministic_and_short():
    key = "F|target|F0001|Corporate reliance|trust_multidimensional|replicate_1"
    assert custom_id_from_request_key(key) == custom_id_from_request_key(key)
    assert len(custom_id_from_request_key(key)) <= 64


def test_together_batch_prepare_dry_run_cli_writes_manifest_and_jsonl(tmp_path):
    out = tmp_path / "batch"
    result = subprocess.run(
        [
            sys.executable,
            str(PIPELINE_ROOT / "scripts" / "together_batch.py"),
            "prepare",
            "--role",
            "G",
            "--requested-model",
            "test/model",
            "--out-dir",
            str(out),
            "--max-requests",
            "3",
            "--dry-run",
        ],
        cwd=PIPELINE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    manifest = pd.read_csv(out / "request_manifest.csv")
    assert len(manifest) == 3
    assert manifest["custom_id"].is_unique
    assert "response_key_map" in manifest.columns
    assert "prompt_protocol_id" in manifest.columns
    assert "prompt_compiler_version" in manifest.columns
    assert set(json.loads(manifest["response_key_map"].iloc[0]).values()) >= {"trust_competence_1", "newsletter_signup"}
    lines = [json.loads(line) for line in (out / "batch_input.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 3
    assert {"custom_id", "body"} == set(lines[0])
    assert lines[0]["body"]["response_format"]["type"] == "json_schema"


def test_resume_excludes_already_successful_requests_from_jsonl(tmp_path):
    first = prepare_batch(role="G", requested_model="test/model", output_dir=tmp_path / "first", max_requests=3)
    manifest = pd.read_csv(first["manifest"])
    success_path = tmp_path / "success.jsonl"
    success_path.write_text(json.dumps({"custom_id": manifest["custom_id"].iloc[0], "response": {"body": {"choices": []}}}) + "\n", encoding="utf-8")

    second = prepare_batch(
        role="G",
        requested_model="test/model",
        output_dir=tmp_path / "second",
        max_requests=3,
        successful_result_paths=[success_path],
    )
    manifest2 = pd.read_csv(second["manifest"])
    assert (manifest2["status"] == "already_successful").sum() == 1
    assert sum(1 for _ in open(second["jsonl"], encoding="utf-8")) == 2


def test_no_duplicate_batch_requests(tmp_path):
    result = prepare_batch(role="F", requested_model="test/model", output_dir=tmp_path, max_requests=10)
    manifest = pd.read_csv(result["manifest"])
    assert manifest["request_key"].is_unique
    assert manifest["custom_id"].is_unique


def test_jsonl_batch_splitter_is_deterministic(tmp_path):
    result = prepare_batch(role="F", requested_model="test/model", output_dir=tmp_path / "batch", max_requests=5)
    shards = split_jsonl_file(result["jsonl"], max_lines_per_shard=2, output_dir=tmp_path / "shards")
    assert [path.name for path in shards] == [
        "batch_input_part0001.jsonl",
        "batch_input_part0002.jsonl",
        "batch_input_part0003.jsonl",
    ]
    assert [sum(1 for _ in open(path, encoding="utf-8")) for path in shards] == [2, 2, 1]


def _write_smoke_state(path: Path, *, submitted: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "tiny_smoke_submission_state_v1",
                "smoke_cost_cap_usd": tb.SMOKE_COST_CAP_USD,
                "global_max_new_requests": tb.SMOKE_GLOBAL_MAX_NEW_REQUESTS,
                "submitted_custom_ids": submitted or [],
                "successful_custom_ids": [],
                "cumulative_new_requests": 0,
                "estimated_worst_case_cost_usd": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_smoke_jsonl(path: Path, *, model: str, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "custom_id": f"c{i}",
                        "body": {
                            "model": model,
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 1024,
                        },
                    }
                )
                + "\n"
            )


def test_tiny_smoke_guard_enforces_first_wave_count_and_cost(monkeypatch, tmp_path):
    root = tmp_path / "tiny_pre_api"
    state = root / "smoke_submission_state.json"
    monkeypatch.setattr(tb, "TINY_SMOKE_ROOT", root)
    monkeypatch.setattr(tb, "TINY_SMOKE_STATE_PATH", state)
    _write_smoke_state(state)
    jsonl = root / "deepseek-ai_DeepSeek-V4-Pro-0813" / "first_wave" / "batch_input.jsonl"
    _write_smoke_jsonl(jsonl, model="deepseek-ai/DeepSeek-V4-Pro-0813", n=12)
    # for_submit=False: this test exercises count/cost accounting in isolation from the
    # exact-custom_id allowlist gate (see test_tiny_smoke_guard_* allowlist tests below).
    guard = tb.tiny_smoke_safety_guard(jsonl, for_submit=False)
    assert guard["request_count"] == 12
    assert guard["cumulative_new_requests"] == 12


def test_tiny_smoke_guard_rejects_resubmitted_custom_id(monkeypatch, tmp_path):
    root = tmp_path / "tiny_pre_api"
    state = root / "smoke_submission_state.json"
    monkeypatch.setattr(tb, "TINY_SMOKE_ROOT", root)
    monkeypatch.setattr(tb, "TINY_SMOKE_STATE_PATH", state)
    _write_smoke_state(state, submitted=["c0"])
    jsonl = root / "deepseek-ai_DeepSeek-V4-Pro-0813" / "first_wave" / "batch_input.jsonl"
    _write_smoke_jsonl(jsonl, model="deepseek-ai/DeepSeek-V4-Pro-0813", n=12)
    with pytest.raises(RuntimeError, match="already recorded"):
        tb.tiny_smoke_safety_guard(jsonl)


def test_tiny_smoke_guard_rejects_placeholder_stage_b(monkeypatch, tmp_path):
    root = tmp_path / "tiny_pre_api"
    state = root / "smoke_submission_state.json"
    monkeypatch.setattr(tb, "TINY_SMOKE_ROOT", root)
    monkeypatch.setattr(tb, "TINY_SMOKE_STATE_PATH", state)
    _write_smoke_state(state)
    jsonl = root / "Qwen_Qwen3.8-2.4T-A95B" / "stage_b_placeholder_validate_only" / "batch_input.jsonl"
    _write_smoke_jsonl(jsonl, model="Qwen/Qwen3.8-2.4T-A95B", n=2)
    with pytest.raises(RuntimeError, match="placeholder Stage-B"):
        tb.tiny_smoke_safety_guard(jsonl)


def test_json_safe_preserves_nested_datetimes_as_iso_strings():
    payload = {
        "created_at": datetime(2026, 8, 25, 12, 58, 11, tzinfo=timezone.utc),
        "nested": [{"deadline": date(2026, 8, 26)}],
    }
    safe = tb.json_safe(payload)
    assert safe == {
        "created_at": "2026-08-25T12:58:11+00:00",
        "nested": [{"deadline": "2026-08-26"}],
    }
    json.dumps(safe)


def test_batch_generation_uses_consensus_stage_a_instead_of_one_shot():
    g_req = next(req for req in iter_g_requests(requested_model="test/model") if req.condition_id == "Consensus")
    f_reqs = [req for req in iter_f_requests(requested_model="test/model", max_requests=250) if req.condition_id == "Consensus"]

    assert g_req.request_stage == "consensus_stage_a"
    assert g_req.outcome_id == "consensus_stage_a_estimates"
    assert g_req.required_fields == "Q001|Q002|Q003"
    assert len(f_reqs) == 1
    assert f_reqs[0].request_stage == "consensus_stage_a"
    assert f_reqs[0].outcome_id == "consensus_stage_a_estimates"


def test_consensus_second_wave_uses_verified_stage_a_success(tmp_path):
    stage_a = next(req for req in iter_f_requests(requested_model="test/model", max_requests=250) if req.condition_id == "Consensus")
    key_map = json.loads(json.dumps(stage_a.response_key_map))
    parsed_target = {target: 50 for target in key_map.values()}
    success_path = tmp_path / "stage_a_success.csv"
    pd.DataFrame(
        [
            {
                "profile_id": stage_a.profile_id,
                "outcome_id": stage_a.outcome_id,
                "parsed_output": json.dumps(parsed_target, sort_keys=True),
                "response_key_map": json.dumps(key_map, sort_keys=True),
                "consensus_stage_a_prompt_hash": stage_a.consensus_stage_a_prompt_hash,
                "consensus_stage_a_schema_hash": stage_a.consensus_stage_a_schema_hash,
            }
        ]
    ).to_csv(success_path, index=False)

    successes = load_consensus_stage_a_success(success_path)
    stage_b = next(
        req
        for req in iter_f_requests(requested_model="test/model", max_requests=250, consensus_stage_a_success=successes)
        if req.condition_id == "Consensus"
    )

    assert stage_b.request_stage == "consensus_stage_b"
    assert stage_b.outcome_id in sc.OUTCOME_COMPOSITES
    assert stage_b.consensus_stage_a_request_key == stage_a.request_key
    assert any(message["role"] == "assistant" and message["content"] == '{"Q001":50,"Q002":50,"Q003":50}' for message in stage_b.messages)
    _assert_v2_closing_instruction(stage_b)


def _assert_v2_closing_instruction(req) -> None:
    """The R1 root-cause v2 closing instruction is a fixed, recognizable
    marker (see inference.prompts._f_response_format_instruction_v2) --
    present iff response_format_instruction_version="v2" was used. Uses the
    LAST user turn, since a Consensus Stage B request's messages include
    Stage A's original (differently-worded) user turn as prior history."""
    user_text = [m["content"] for m in req.messages if m["role"] == "user"][-1]
    assert "Do NOT use a survey item's own label" in user_text
    assert "Respond ONLY with one JSON object" in user_text


def test_target_f_non_consensus_requests_use_v2_format_instruction():
    """Target F (the 111,000-request production path) must use the same
    R1 root-cause format-only fix already proven necessary at this request
    volume for structured-output reliability -- not the v1 default."""
    non_consensus = next(req for req in iter_f_requests(requested_model="test/model", max_requests=50) if req.condition_id != "Consensus")
    _assert_v2_closing_instruction(non_consensus)


def test_target_f_consensus_stage_a_untouched_by_v2():
    """Consensus Stage A is a structurally different elicitation protocol
    (3 anchor estimates, not a scored outcome item) -- it was never part of
    the R1 remediation and must not be silently rewritten."""
    stage_a = next(req for req in iter_f_requests(requested_model="test/model", max_requests=250) if req.condition_id == "Consensus")
    user_text = next(m["content"] for m in stage_a.messages if m["role"] == "user")
    assert "Return a single JSON object with one integer value per question key" in user_text


def test_parse_results_writes_retry_manifest_for_malformed_output(tmp_path):
    result = prepare_batch(role="G", requested_model="test/model", output_dir=tmp_path / "batch", max_requests=2)
    manifest = pd.read_csv(result["manifest"])
    required = manifest["required_fields"].iloc[0].split("|")
    valid_payload = {field: 50 for field in required}
    output = tmp_path / "output.jsonl"
    output.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": manifest["custom_id"].iloc[0],
                        "response": {
                            "body": {
                                "choices": [{"message": {"content": json.dumps(valid_payload)}}],
                                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                            }
                        },
                    }
                ),
                json.dumps(
                    {
                        "custom_id": manifest["custom_id"].iloc[1],
                        "response": {"body": {"choices": [{"message": {"content": "not json"}}]}},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    parsed = parse_batch_results(manifest_path=Path(result["manifest"]), results_jsonl=output, output_dir=tmp_path / "parsed")
    assert parsed["successful"] == 1
    assert parsed["malformed"] == 1
    success = pd.read_csv(parsed["success"])
    parsed_output = json.loads(success["parsed_output"].iloc[0])
    assert "Q001" not in parsed_output
    assert "trust_competence_1" in parsed_output
    retry = pd.read_csv(parsed["retry_manifest"])
    assert len(retry) == 1
    assert retry["custom_id"].iloc[0] == manifest["custom_id"].iloc[1]


def test_g_q_mapping_round_trips_per_request_for_different_donor_orders(tmp_path):
    profile = {
        "age": 40,
        "gender": "Female",
        "race": "White / Caucasian",
        "education": "Bachelor's degree",
        "income": "$56,000 to $99,999",
        "party": "Independent",
        "state": "Texas",
        "state_abbr": "TX",
    }
    items = sc.load_items()
    render_a = build_g_prompt_render(profile, "Stimulus", items, donor_key="D1", condition_id="control")
    donor_b = next(
        f"D{i}"
        for i in range(2, 200)
        if build_g_prompt_render(profile, "Stimulus", items, donor_key=f"D{i}", condition_id="control").response_key_map
        != render_a.response_key_map
    )
    render_b = build_g_prompt_render(profile, "Stimulus", items, donor_key=donor_b, condition_id="control")
    assert render_a.response_key_map != render_b.response_key_map

    manifest = pd.DataFrame(
        [
            {
                "request_key": "G|D1|control|replicate_1",
                "custom_id": "a",
                "role": "G",
                "study_id": "target",
                "profile_id": "D1",
                "condition_id": "control",
                "outcome_id": "full_questionnaire",
                "replicate_id": 1,
                "requested_model": "test/model",
                "prompt_hash": "h",
                "schema_version": "s",
                "seed": 1,
                "status": "pending",
                "required_fields": "|".join(render_a.response_schema["required"]),
                "response_key_map": json.dumps(render_a.response_key_map, sort_keys=True),
            },
            {
                "request_key": f"G|{donor_b}|control|replicate_1",
                "custom_id": "b",
                "role": "G",
                "study_id": "target",
                "profile_id": donor_b,
                "condition_id": "control",
                "outcome_id": "full_questionnaire",
                "replicate_id": 1,
                "requested_model": "test/model",
                "prompt_hash": "h",
                "schema_version": "s",
                "seed": 2,
                "status": "pending",
                "required_fields": "|".join(render_b.response_schema["required"]),
                "response_key_map": json.dumps(render_b.response_key_map, sort_keys=True),
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    output_path = tmp_path / "results.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps({"custom_id": "a", "response": {"body": {"choices": [{"message": {"content": json.dumps({key: i for i, key in enumerate(render_a.response_schema["required"], start=1)})}}]}}}),
                json.dumps({"custom_id": "b", "response": {"body": {"choices": [{"message": {"content": json.dumps({key: i for i, key in enumerate(render_b.response_schema["required"], start=1)})}}]}}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = parse_batch_results(manifest_path=manifest_path, results_jsonl=output_path, output_dir=tmp_path / "parsed")
    success = pd.read_csv(parsed["success"])
    by_profile = {row["profile_id"]: json.loads(row["parsed_output"]) for _, row in success.iterrows()}

    assert by_profile["D1"]["donation_ams"] == next(i for i, key in enumerate(render_a.response_schema["required"], start=1) if render_a.response_key_map[key] == "donation_ams")
    assert by_profile[donor_b]["donation_ams"] == next(i for i, key in enumerate(render_b.response_schema["required"], start=1) if render_b.response_key_map[key] == "donation_ams")
