"""Build (never submit) the G-v2 format-only replacement for the Orchinik
G-vs-DeepSeek domain-confirmation manifests.

Motivation: the v1 Orchinik manifests (outputs/domain_validation/
orchinik_g_domain_confirmation/) were audited and found to lack the
anti-markdown-fence serving instruction -- the exact gap that caused the
92-98% invalid rate on the original target G Wave-1 batch (system_fingerprint
vllm-0.21.0-8326ea74). This script does NOT modify or overwrite the v1
manifests; it builds a completely separate, new-versioned set under
outputs/domain_validation/orchinik_g_domain_confirmation_v2/, using
response_format_instruction_version="v2" (G_FORMAT_INSTRUCTION_V2, the same
format-only closing-instruction addition already used for the target G-v2
replacement) on the SAME frozen personas/donor universe/conditions/
materials/battery/items -- scientific equivalence to v1 is verified
programmatically before this script accepts its own output.

No LLM calls. No target requests submitted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ate.orchinik_g_domain_confirmation import build_25_items, load_eligible_respondents  # noqa: E402
from inference.prompts import G_FORMAT_INSTRUCTION_V2, text_hash  # noqa: E402
import build_orchinik_g_domain_confirmation_manifest as v1_mod  # noqa: E402

OUT_ROOT_V2 = PIPELINE_ROOT / "outputs" / "domain_validation" / "orchinik_g_domain_confirmation_v2"


def verify_scientific_equivalence(v1_requests, v2_requests) -> None:
    v1_by_id = {r.profile_id: r for r in v1_requests}
    v2_by_id = {r.profile_id: r for r in v2_requests}
    if set(v1_by_id) != set(v2_by_id):
        raise RuntimeError("v1/v2 donor universes differ")
    for pid, v1_req in v1_by_id.items():
        v2_req = v2_by_id[pid]
        if v1_req.response_schema != v2_req.response_schema:
            raise RuntimeError(f"schema changed for {pid}")
        if v1_req.condition_id != v2_req.condition_id:
            raise RuntimeError(f"condition changed for {pid}")
        v1_user = v1_req.messages[-1]["content"]
        v2_user = v2_req.messages[-1]["content"]
        if v1_req.messages[0]["content"] != v2_req.messages[0]["content"]:
            raise RuntimeError(f"system prompt changed for {pid}")
        if not v2_user.startswith(v1_user):
            raise RuntimeError(f"v2 user prompt is not v1 + suffix for {pid}")
        if v2_user[len(v1_user) :] != f" {G_FORMAT_INSTRUCTION_V2}":
            raise RuntimeError(f"v2 suffix is not exactly G_FORMAT_INSTRUCTION_V2 for {pid}")


def main() -> dict:
    respondents = load_eligible_respondents()
    items = build_25_items()

    all_custom_ids: set[str] = set()
    summary: dict = {"eligible_respondents": len(respondents), "items_per_respondent": len(items), "models": {}, "scientific_equivalence_verified": True}
    for model in v1_mod.MODELS:
        v1_requests = v1_mod.build_requests_for_model(model, respondents, items, response_format_instruction_version="v1")
        v2_requests = v1_mod.build_requests_for_model(model, respondents, items, response_format_instruction_version="v2")
        verify_scientific_equivalence(v1_requests, v2_requests)

        model_dir_name = model.replace("/", "_")
        stats = v1_mod.write_requests(v2_requests, OUT_ROOT_V2 / model_dir_name)
        stats["worst_case_cost_usd"] = round(v1_mod.worst_case_cost(model, stats["estimated_prompt_tokens_rough"], stats["maximum_output_tokens"]), 6)
        summary["models"][model] = stats

        overlap = all_custom_ids & {r.custom_id for r in v2_requests}
        if overlap:
            raise RuntimeError(f"custom_id collision across models: {sorted(overlap)[:5]}")
        all_custom_ids |= {r.custom_id for r in v2_requests}

        v1_ids = {r.custom_id for r in v1_requests}
        v2_ids = {r.custom_id for r in v2_requests}
        if v1_ids & v2_ids:
            raise RuntimeError(f"v1/v2 custom_id collision for {model}: {sorted(v1_ids & v2_ids)[:5]}")

    summary["total_requests"] = sum(m["requests"] for m in summary["models"].values())
    summary["total_worst_case_cost_usd"] = round(sum(m["worst_case_cost_usd"] for m in summary["models"].values()), 6)

    OUT_ROOT_V2.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT_V2 / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
