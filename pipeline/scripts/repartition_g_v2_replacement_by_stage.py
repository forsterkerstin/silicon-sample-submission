"""Re-partition the G-v2 (PROVIDER_SERVING_FORMAT_FAILURE replacement)
target Wave-1 manifest by request_stage ("standard" / "consensus_stage_a"),
same reason and pattern as scripts/repartition_target_wave1_by_stage.py.
Re-slices the already-built outputs/target_production/wave1_g_v2_replacement/
manifest (built via inference.together_batch.prepare_batch(...,
response_format_instruction_version="v2"), no new prompt compilation here)
into stage-pure, size-bounded partitions.

No LLM calls. No target requests submitted.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

from inference.together_batch import TOGETHER_BATCH_MAX_REQUESTS  # noqa: E402

REPLACEMENT_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement"
STAGE_ROOT = REPLACEMENT_ROOT / "by_stage"
OPERATIONAL_FILE_SIZE_CEILING_BYTES = int(95 * 1024 * 1024)
CANDIDATE_PARTITION_COUNTS = (1, 2, 4, 5, 8, 10, 16, 20, 25)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl_lines_by_custom_id(jsonl_path: Path) -> dict[str, str]:
    out = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            cid = json.loads(stripped)["custom_id"]
            out[cid] = line if line.endswith("\n") else line + "\n"
    return out


def choose_k(n_rows: int, line_bytes: list[int]) -> int:
    for k in CANDIDATE_PARTITION_COUNTS:
        if n_rows % k != 0 and k != 1:
            continue
        chunk = -(-n_rows // k)
        max_bytes = 0
        max_requests = 0
        for p in range(k):
            part = line_bytes[p * chunk : (p + 1) * chunk]
            if not part:
                continue
            max_bytes = max(max_bytes, sum(part))
            max_requests = max(max_requests, len(part))
        if max_bytes < OPERATIONAL_FILE_SIZE_CEILING_BYTES and max_requests < TOGETHER_BATCH_MAX_REQUESTS:
            return k
    raise RuntimeError(f"no candidate K in {CANDIDATE_PARTITION_COUNTS} keeps every partition under the size/request ceilings for {n_rows} rows")


def build_stage_subset(stage: str, lines_by_cid: dict[str, str], manifest_df: pd.DataFrame) -> dict[str, Any]:
    subset = manifest_df[manifest_df["request_stage"] == stage].reset_index(drop=True)
    out_dir = STAGE_ROOT / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_dir / "request_manifest.csv", index=False)
    lines = [lines_by_cid[cid] for cid in subset["custom_id"]]
    (out_dir / "batch_input.jsonl").write_text("".join(lines), encoding="utf-8")
    line_bytes = [len(line.encode("utf-8")) for line in lines]
    k = choose_k(len(lines), line_bytes)
    chunk = -(-len(lines) // k)
    partitions = []
    for p in range(k):
        lo, hi = p * chunk, min((p + 1) * chunk, len(lines))
        if lo >= hi:
            continue
        part_dir = out_dir / f"part{p + 1}"
        part_dir.mkdir(parents=True, exist_ok=True)
        part_jsonl = part_dir / "batch_input.jsonl"
        part_jsonl.write_text("".join(lines[lo:hi]), encoding="utf-8")
        subset.iloc[lo:hi].to_csv(part_dir / "request_manifest.csv", index=False)
        partitions.append(
            {
                "partition": f"part{p + 1}",
                "requests": hi - lo,
                "jsonl_size_mb": round(part_jsonl.stat().st_size / (1024 * 1024), 2),
                "sha256": sha256_file(part_jsonl),
            }
        )
    return {
        "stage": stage,
        "requests": len(subset),
        "manifest_sha256": sha256_file(out_dir / "request_manifest.csv"),
        "jsonl_sha256": sha256_file(out_dir / "batch_input.jsonl"),
        "partition_count": k,
        "partitions": partitions,
        "custom_ids": subset["custom_id"].tolist(),
    }


def main() -> int:
    STAGE_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_df = pd.read_csv(REPLACEMENT_ROOT / "request_manifest.csv")
    lines_by_cid = load_jsonl_lines_by_custom_id(REPLACEMENT_ROOT / "batch_input.jsonl")

    summary: dict[str, Any] = {}
    for stage in ("standard", "consensus_stage_a"):
        result = build_stage_subset(stage, lines_by_cid, manifest_df)
        summary[stage] = result
        print(f"{stage}: {result['requests']} requests, K={result['partition_count']}")

    combined_ids = sorted(set(summary["standard"]["custom_ids"]) | set(summary["consensus_stage_a"]["custom_ids"]))
    print(f"combined G-v2 replacement: {len(combined_ids)} candidate custom_ids (NOT approved/declared -- no submission authorized)")

    for key in summary:
        summary[key].pop("custom_ids", None)
    (STAGE_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (STAGE_ROOT / "combined_ids.json").write_text(json.dumps({"all": combined_ids}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
