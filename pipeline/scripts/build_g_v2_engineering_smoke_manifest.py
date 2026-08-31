"""Build a tiny engineering-smoke manifest for the G-v2 format-only fix
(schema/serving validation ONLY -- never used for scientific analysis or
model selection). Extracts a small, deterministic subset (first 5 rows by
custom_id, ascending) of each request_stage from the already-built full
G-v2 replacement manifest (outputs/target_production/wave1_g_v2_replacement/
by_stage/) -- no new prompt compilation, no LLM calls, not submitted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
STAGE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "by_stage"
SMOKE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "g_v2_engineering_smoke"
SMOKE_N_PER_STAGE = 5


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl_lines_by_custom_id(path: Path) -> dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            cid = json.loads(stripped)["custom_id"]
            out[cid] = line if line.endswith("\n") else line + "\n"
    return out


def main() -> dict:
    summary = {}
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    all_ids = []
    for stage in ("standard", "consensus_stage_a"):
        manifest = pd.read_csv(STAGE_ROOT / stage / "request_manifest.csv")
        subset = manifest.sort_values("custom_id").head(SMOKE_N_PER_STAGE).reset_index(drop=True)
        lines_by_cid = _load_jsonl_lines_by_custom_id(STAGE_ROOT / stage / "batch_input.jsonl")

        out_dir = SMOKE_ROOT / stage
        out_dir.mkdir(parents=True, exist_ok=True)
        subset.to_csv(out_dir / "request_manifest.csv", index=False)
        jsonl_path = out_dir / "batch_input.jsonl"
        jsonl_path.write_text("".join(lines_by_cid[cid] for cid in subset["custom_id"]), encoding="utf-8")

        ids = subset["custom_id"].tolist()
        all_ids.extend(ids)
        summary[stage] = {"requests": len(subset), "manifest_sha256": _sha256_file(out_dir / "request_manifest.csv"), "jsonl_sha256": _sha256_file(jsonl_path)}

    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("duplicate custom_id across smoke stages")

    result = {
        "note": "SCHEMA/SERVING VALIDATION ONLY -- never used for scientific analysis or model selection. Not submitted.",
        "per_stage": summary,
        "total_requests": len(all_ids),
    }
    (SMOKE_ROOT / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
