"""Build the ACTUAL submission-ready G-v2 full-replacement partitions:
the frozen 17,000-row by-stage manifests
(outputs/target_production/wave1_g_v2_replacement/by_stage/) MINUS the 10
custom_ids already submitted (and already recorded in the real ledger's
submitted_custom_ids) via the engineering smoke -- submitting those again
would either double-pay for already-obtained results or be refused outright
by target_production_safety_guard's own submitted/successful dedup check.

This is a purely mechanical re-slice of the SAME already-hashed rows (no
new prompt compilation, no scientific change) into fresh, appropriately
sized partitions under outputs/target_production/wave1_g_v2_replacement/
submission/.

No LLM calls. No target requests submitted.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPLACEMENT_STAGE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "by_stage"
SMOKE_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "g_v2_engineering_smoke"
SUBMISSION_ROOT = PIPELINE_ROOT / "outputs" / "target_production" / "wave1_g_v2_replacement" / "submission"
LEDGER_PATH = PIPELINE_ROOT / "outputs" / "target_production" / "target_production_submission_state.json"

OPERATIONAL_FILE_SIZE_CEILING_BYTES = int(95 * 1024 * 1024)
CANDIDATE_PARTITION_COUNTS = (1, 2, 4, 5, 8, 10, 16, 20, 25)
TOGETHER_BATCH_MAX_REQUESTS = 50_000


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_ids(path: Path) -> set[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["custom_id"] for row in csv.DictReader(f)}


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


def _choose_k(n_rows: int, line_bytes: list[int]) -> int:
    for k in CANDIDATE_PARTITION_COUNTS:
        if n_rows % k != 0 and k != 1:
            continue
        chunk = -(-n_rows // k)
        max_bytes = max_requests = 0
        for p in range(k):
            part = line_bytes[p * chunk : (p + 1) * chunk]
            if not part:
                continue
            max_bytes = max(max_bytes, sum(part))
            max_requests = max(max_requests, len(part))
        if max_bytes < OPERATIONAL_FILE_SIZE_CEILING_BYTES and max_requests < TOGETHER_BATCH_MAX_REQUESTS:
            return k
    raise RuntimeError(f"no candidate K in {CANDIDATE_PARTITION_COUNTS} keeps every partition under the size/request ceilings for {n_rows} rows")


def build_stage(stage: str, exclude_ids: set[str]) -> dict:
    import pandas as pd

    manifest_df = pd.read_csv(REPLACEMENT_STAGE_ROOT / stage / "request_manifest.csv")
    manifest_df = manifest_df[~manifest_df["custom_id"].isin(exclude_ids)].reset_index(drop=True)
    lines_by_cid = _load_jsonl_lines_by_custom_id(REPLACEMENT_STAGE_ROOT / stage / "batch_input.jsonl")

    out_dir = SUBMISSION_ROOT / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(out_dir / "request_manifest.csv", index=False)
    lines = [lines_by_cid[cid] for cid in manifest_df["custom_id"]]
    (out_dir / "batch_input.jsonl").write_text("".join(lines), encoding="utf-8")

    line_bytes = [len(line.encode("utf-8")) for line in lines]
    k = _choose_k(len(lines), line_bytes)
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
        manifest_df.iloc[lo:hi].to_csv(part_dir / "request_manifest.csv", index=False)
        partitions.append({"partition": f"part{p + 1}", "requests": hi - lo, "jsonl_size_mb": round(part_jsonl.stat().st_size / (1024 * 1024), 2), "sha256": _sha256_file(part_jsonl)})

    return {"stage": stage, "requests": len(manifest_df), "excluded_already_submitted": len(exclude_ids), "partition_count": k, "partitions": partitions, "manifest_sha256": _sha256_file(out_dir / "request_manifest.csv")}


def main() -> dict:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    already_submitted = set(ledger.get("submitted_custom_ids", [])) | set(ledger.get("successful_custom_ids", []))

    summary = {}
    for stage in ("standard", "consensus_stage_a"):
        full_ids = _load_ids(REPLACEMENT_STAGE_ROOT / stage / "request_manifest.csv")
        exclude = full_ids & already_submitted
        summary[stage] = build_stage(stage, exclude)
        print(f"{stage}: {summary[stage]['requests']} requests to submit (excluded {summary[stage]['excluded_already_submitted']} already-submitted), K={summary[stage]['partition_count']}")

    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    (SUBMISSION_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = main()
    print(json.dumps(out, indent=2))
