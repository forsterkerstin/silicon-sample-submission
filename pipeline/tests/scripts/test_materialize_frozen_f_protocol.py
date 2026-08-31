"""Tests for scripts/materialize_frozen_f_protocol.py -- bookkeeping only,
proves the materialized artifact is genuinely valid against the frozen
consumer (ate.f_reliability.require_frozen_f_protocol), not just
well-formed JSON."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ate.f_reliability import require_frozen_f_protocol  # noqa: E402
from materialize_frozen_f_protocol import OUT_PATH, build_payload  # noqa: E402

pytestmark = pytest.mark.skipif(not OUT_PATH.exists(), reason="frozen_f_protocol.json not materialized in this environment")


def test_frozen_f_protocol_passes_the_real_consumer():
    verified = require_frozen_f_protocol()
    assert verified["selected_f_model"] == "google/gemma-4-31B-it"
    assert verified["n_f"] == 500
    assert verified["f_num_draws"] == 1


def test_frozen_f_protocol_records_v2_and_frozen_parameters():
    payload = build_payload()
    assert payload["response_format_instruction_version"] == "v2"
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["n"] == 1
    assert payload["presence_penalty"] == 0
    assert payload["frequency_penalty"] == 0
    assert payload["model_engine_config"]["chat_template_kwargs"]["enable_thinking"] is False
    assert payload["f_r_f"] == 1


def test_frozen_f_protocol_cites_real_provenance_commits():
    payload = build_payload()
    prov = payload["provenance"]
    assert prov["f_star_selection"]["commit"] == "6a25d3b73bd278c4134cb88a067cfe2805b80f5b"
    assert prov["r_f_freeze"]["commit"] == "3fff154"
    assert prov["response_format_v2_extended_to_target_f"]["commit"] == "3a47541"


def test_build_payload_deterministic_except_timestamp():
    a = build_payload()
    b = build_payload()
    a.pop("frozen_at")
    b.pop("frozen_at")
    a.pop("f_inference_config_hash")
    b.pop("f_inference_config_hash")
    assert a == b
