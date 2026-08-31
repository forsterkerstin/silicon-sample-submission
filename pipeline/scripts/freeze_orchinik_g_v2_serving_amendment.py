"""Freeze the Orchinik domain-confirmation G-v2 serving-only amendment
(OFFLINE ONLY). Writes outputs/domain_validation/
orchinik_g_domain_confirmation_v2/serving_amendment.json.

Motivated solely by the independently observed provider serving-format
defect (system_fingerprint vllm-0.21.0-8326ea74, markdown-fenced JSON
despite response_format.json_schema strict) that caused the original
target G Wave-1 batch's 92-98% invalid rate. Does not change any scientific
aspect of the frozen Orchinik protocol (outputs/domain_validation/
frozen_orchinik_g_domain_confirmation.json, untouched by this script) --
personas, donor universe, randomized conditions, experimental material,
response battery, supports, item ordering, models, temperature, top_p,
reasoning settings, primary metric, cell weighting, and the confirmation
rule are all unchanged; only an explicit anti-markdown-fence closing
instruction is appended.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
DOMAIN_DIR = PIPELINE_ROOT / "outputs" / "domain_validation"
V1_PROTOCOL_PATH = DOMAIN_DIR / "frozen_orchinik_g_domain_confirmation.json"
V1_ROOT = DOMAIN_DIR / "orchinik_g_domain_confirmation"
V2_ROOT = DOMAIN_DIR / "orchinik_g_domain_confirmation_v2"
G_V2_TARGET_AMENDMENT_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "g_wave1_v1_format_failure_amendment.json"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> dict:
    if not V1_PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"v1 Orchinik protocol missing (must not be altered): {V1_PROTOCOL_PATH}")
    if not G_V2_TARGET_AMENDMENT_PATH.exists():
        raise FileNotFoundError(f"target G-v2 amendment missing -- this Orchinik amendment is modeled on it: {G_V2_TARGET_AMENDMENT_PATH}")
    v2_summary = json.loads((V2_ROOT / "summary.json").read_text(encoding="utf-8"))

    v1_protocol_sha256_before = _sha256_file(V1_PROTOCOL_PATH)

    amendment = {
        "amendment_type": "ORCHINIK_DOMAIN_CONFIRMATION_SERVING_FORMAT_ONLY_V2",
        "motivation": "audit found the v1 Orchinik manifests lacked the anti-markdown-fence closing instruction -- the exact gap independently diagnosed as the root cause of the 92-98% invalid rate on the original target G Wave-1 batch (system_fingerprint vllm-0.21.0-8326ea74)",
        "analogous_to": {
            "path": str(G_V2_TARGET_AMENDMENT_PATH.relative_to(PIPELINE_ROOT)),
            "sha256": _sha256_file(G_V2_TARGET_AMENDMENT_PATH),
        },
        "v1_protocol_path": str(V1_PROTOCOL_PATH.relative_to(PIPELINE_ROOT)),
        "v1_protocol_sha256": v1_protocol_sha256_before,
        "v1_protocol_unmodified_by_this_amendment": True,
        "v1_manifests_preserved_unmodified": True,
        "format_only_change": {
            "description": "appends G_FORMAT_INSTRUCTION_V2 verbatim after the v1 closing instruction sentence -- never replaces or removes it",
            "instruction_text": (
                "Return ONLY the raw JSON object matching the supplied schema. Do not use Markdown. "
                "Do not use ```json or any other code fences. Do not place text before or after the JSON. "
                "The response must begin with { and end with }."
            ),
            "scientific_equivalence_programmatically_verified": v2_summary["scientific_equivalence_verified"],
        },
        "no_change_to": ["personas", "donor_universe", "randomized_condition", "experimental_material", "response_battery", "response_supports", "item_ordering", "model", "temperature", "top_p", "reasoning_settings", "stochastic_draws", "primary_metric", "cell_weighting", "confirmation_rule", "scientific_consequence"],
        "gemma_manifest_sha256": v2_summary["models"]["google/gemma-4-31B-it"]["manifest_sha256"],
        "gemma_jsonl_sha256": v2_summary["models"]["google/gemma-4-31B-it"]["jsonl_sha256"],
        "gemma_request_count": v2_summary["models"]["google/gemma-4-31B-it"]["requests"],
        "gemma_worst_case_cost_usd": v2_summary["models"]["google/gemma-4-31B-it"]["worst_case_cost_usd"],
        "deepseek_manifest_sha256": v2_summary["models"]["deepseek-ai/DeepSeek-V4-Pro-0813"]["manifest_sha256"],
        "deepseek_jsonl_sha256": v2_summary["models"]["deepseek-ai/DeepSeek-V4-Pro-0813"]["jsonl_sha256"],
        "deepseek_request_count": v2_summary["models"]["deepseek-ai/DeepSeek-V4-Pro-0813"]["requests"],
        "deepseek_worst_case_cost_usd": v2_summary["models"]["deepseek-ai/DeepSeek-V4-Pro-0813"]["worst_case_cost_usd"],
        "total_requests": v2_summary["total_requests"],
        "total_worst_case_cost_usd": v2_summary["total_worst_case_cost_usd"],
        "eligible_donors_per_model": v2_summary["eligible_respondents"],
        "target_g_scientific_outputs_accessed": False,
        "target_human_outcomes_used": False,
        "new_paid_inference_performed": False,
    }

    V2_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = V2_ROOT / "serving_amendment.json"
    out_path.write_text(json.dumps(amendment, indent=2) + "\n", encoding="utf-8")
    sha = _sha256_file(out_path)
    (V2_ROOT / "serving_amendment.sha256.txt").write_text(sha + "\n", encoding="utf-8")

    v1_protocol_sha256_after = _sha256_file(V1_PROTOCOL_PATH)
    if v1_protocol_sha256_before != v1_protocol_sha256_after:
        raise RuntimeError("v1 protocol artifact was unexpectedly modified while freezing this amendment")

    amendment["amendment_sha256"] = sha
    amendment["v1_protocol_unchanged_verified"] = v1_protocol_sha256_before == v1_protocol_sha256_after
    return amendment


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
