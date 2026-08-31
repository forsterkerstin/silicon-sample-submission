#!/usr/bin/env python3
"""scripts/materialize_frozen_f_protocol.py

Materializes outputs/f_reliability/frozen_f_protocol.json -- pure
bookkeeping/provenance, not a new scientific decision. Every value here was
already frozen by an earlier commit; this script only writes them down in
the shape ate.f_reliability.require_frozen_f_protocol() expects.

Does NOT call ate.f_reliability.write_frozen_f_protocol(), which is a
DIFFERENT, still-valid constructor for the (never executed) pilot-based
convergence route (it requires a pilot_manifest_path/convergence_summary/
stochastic_reliability_summary this repo's actual frozen path -- the F
mini-screen + replacement-R1 sequential reliability decision -- never
produced). Writing this artifact directly, with explicit provenance to the
three commits that actually froze each value, avoids fabricating pilot
inputs that were never generated. Only ONE frozen_f_protocol.json can exist
at outputs/f_reliability/frozen_f_protocol.json; nothing in this script
changes which route future work uses.

No paid inference. No LLM calls.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from ate.f_reliability import DEFAULT_N_F, f_inference_config_hash, f_protocol_config, require_frozen_f_protocol  # noqa: E402
from inference.model_config import inference_parameters, load_model_config, model_engine_config, selected_model  # noqa: E402

OUT_PATH = PIPELINE_ROOT / "outputs" / "f_reliability" / "frozen_f_protocol.json"


def build_payload() -> dict:
    model = selected_model("f", require_frozen=True)
    if not model:
        raise RuntimeError("F* is not frozen; cannot materialize frozen_f_protocol.json")
    cfg = f_protocol_config()
    params = inference_parameters()
    engine_cfg = model_engine_config(model)
    f_r_f = int(cfg["f_num_draws"])

    payload = {
        "selected_f_model": model,
        "n_f": DEFAULT_N_F,
        "f_num_draws": f_r_f,
        "f_r_f": f_r_f,
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "presence_penalty": params.get("presence_penalty", 0),
        "frequency_penalty": params.get("frequency_penalty", 0),
        "n": params.get("n", 1),
        "reasoning_configuration": {"reasoning_effort": params.get("reasoning_effort"), "use_reasoning_effort_when_supported": params.get("use_reasoning_effort_when_supported")},
        "structured_output": params.get("structured_output"),
        "response_format_instruction_version": "v2",
        "model_engine_config": engine_cfg,
        "prompt_version": load_model_config()["prompting"]["f_prompt_protocol"],
        "provenance": {
            "f_star_selection": {
                "description": "F mini-screen: both candidates retrieved, reconciled, and scored under frozen v4 rules; Gemma selected.",
                "commit": "6a25d3b73bd278c4134cb88a067cfe2805b80f5b",
            },
            "r_f_freeze": {
                "description": "Replacement F* R1 (24,000 requests, fresh draws 5/6, format-only v2 remediation): all three frozen gates passed, R_F=1 frozen.",
                "commit": "3fff154",
                "amendment_type": "REPLACEMENT_R1_PASSED_R_F_1_FROZEN",
            },
            "response_format_v2_extended_to_target_f": {
                "description": "build_f_prompt_render/build_f_consensus_stage_b_prompt_render (target F's actual request builders) explicitly opted into response_format_instruction_version=v2, matching the R1 root-cause remediation, before any target F request is built.",
                "commit": "3a47541",
            },
        },
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["f_inference_config_hash"] = f_inference_config_hash(payload)
    return payload


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    payload = build_payload()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Round-trip through the exact frozen consumer to prove this artifact is genuinely valid, not just well-formed.
    verified = require_frozen_f_protocol()
    assert verified["selected_f_model"] == payload["selected_f_model"]

    print(f"FROZEN_F_PROTOCOL_MATERIALIZED = YES")
    print(f"FROZEN_F_PROTOCOL_SHA256 = {sha256_file(OUT_PATH)}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
