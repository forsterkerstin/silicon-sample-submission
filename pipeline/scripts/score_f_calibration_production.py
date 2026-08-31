"""Score the retrieved, complete 136,000-request F* calibration-production
batch (5 partitions, commit 43f9778) under the already-frozen scorer
(ate.f_calibration_validation.score_f_calibration_production_from_raw,
built and synthetic-tested before any calibration inference was submitted)
and the already-frozen LOSO M0/M1/M2 selection
(ate.calibrate_lambda.fit_calibration_model_comparison, unmodified,
pre-existing).

Applies ONLY frozen machinery. No coercion, clipping, imputation, or retry
of malformed responses. No new invalid-rate gate is invented for this
phase (none was prospectively frozen). human_ate (already the frozen
percent-of-range theta_H column in data/ate_archive.csv, precomputed
independently of this run) is read verbatim, never touched.

Does NOT write back into data/ate_archive.csv's per-effect prediction
columns (a separate, larger decision assert_external_f_predictions_production_ready
gates and this task does not ask for) -- writes its own standalone
136-effect calibration table instead.
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
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd  # noqa: E402

from ate.calibrate_lambda import fit_calibration_model_comparison  # noqa: E402
from ate.f_calibration_validation import IntegrityFailure, score_f_calibration_production_from_raw  # noqa: E402

CALIB_ROOT = PIPELINE_ROOT / "outputs" / "scientific_bakeoff" / "f_calibration_production"
CANONICAL_DIR = CALIB_ROOT / "google_gemma-4-31B-it"
PARTITIONS = ("part1", "part2", "part3", "part4", "part5")
ARCHIVE_PATH = PIPELINE_ROOT / "data" / "ate_archive.csv"
OUT_DIR = PIPELINE_ROOT / "outputs" / "calibration_production"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw_by_custom_id(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            out[str(record["custom_id"])] = record
    return out


def load_schema_by_custom_id(jsonl_path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            schema = record["body"]["response_format"]["json_schema"]["schema"]
            out[str(record["custom_id"])] = schema
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canonical_manifest = pd.read_csv(CANONICAL_DIR / "request_manifest.csv")
    expected_total = len(canonical_manifest)
    print(f"EXPECTED_REQUESTS = {expected_total}")

    per_partition: dict[str, Any] = {}
    merged_raw: dict[str, dict[str, Any]] = {}
    for part in PARTITIONS:
        part_dir = CALIB_ROOT / part / "google_gemma-4-31B-it"
        retrieved_dir = part_dir / "retrieved"
        status = json.loads((retrieved_dir / "batch_status.json").read_text(encoding="utf-8"))
        raw = load_raw_by_custom_id(retrieved_dir / "batch_output.jsonl")
        overlap = set(raw) & set(merged_raw)
        if overlap:
            raise RuntimeError(f"{part}: {len(overlap)} custom_id(s) already seen in an earlier partition -- refusing to merge")
        merged_raw.update(raw)
        per_partition[part] = {
            "batch_id": status.get("id"),
            "status": status.get("status"),
            "expected": 27200,
            "retrieved_lines": len(raw),
            "batch_output_sha256": sha256_file(retrieved_dir / "batch_output.jsonl"),
        }
        print(f"{part}: status={status.get('status')} retrieved_lines={len(raw)}")

    print(f"RETURNED_PROVIDER_RECORDS = {len(merged_raw)}")

    schema_by_custom_id = load_schema_by_custom_id(CANONICAL_DIR / "batch_input.jsonl")

    archive = pd.read_csv(ARCHIVE_PATH)
    primary = archive[archive["included_primary_calibration"] == True].copy()  # noqa: E712
    manifest_effects = set(canonical_manifest["outcome_id"])
    primary = primary[primary["effect_id"].isin(manifest_effects)]
    print(f"eligible effects in manifest: {len(primary)} across {primary['study_id'].nunique()} studies")

    effect_native_bounds = {row.effect_id: (row.outcome_min, row.outcome_max) for row in primary.itertuples()}
    effect_response_field = {eid: "response" for eid in effect_native_bounds}
    study_id_by_effect = {row.effect_id: row.study_id for row in primary.itertuples()}

    try:
        result = score_f_calibration_production_from_raw(
            manifest=canonical_manifest,
            raw_by_custom_id=merged_raw,
            schema_by_custom_id=schema_by_custom_id,
            effect_native_bounds=effect_native_bounds,
            effect_response_field=effect_response_field,
            study_id_by_effect=study_id_by_effect,
            expected_total=expected_total,
            replicate_id=7,
        )
    except IntegrityFailure as exc:
        out = {"per_partition": per_partition, "integrity_failure": str(exc)}
        (OUT_DIR / "calibration_scoring_result.json").write_text(json.dumps(out, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2, default=str))
        return 1

    reconciliation = result["reconciliation"]
    print(f"MISSING_IDS = {len(reconciliation['missing_entirely'])}")
    print(f"UNEXPECTED_IDS = {len(reconciliation['unexpected'])}")
    print(f"DUPLICATE_IDS = {len(reconciliation['duplicate'])}")
    print(f"INVALID_RATE = {result['invalid_rate']}")

    per_effect_df = pd.DataFrame(result["per_effect_ate"])
    per_effect_df = per_effect_df.merge(primary[["effect_id", "human_ate"]], on="effect_id", how="left")
    per_effect_df = per_effect_df.rename(columns={"theta_l_pp": "theta_L_pp", "human_ate": "theta_H_pp"})
    per_effect_df.to_csv(OUT_DIR / "frozen_136_effect_calibration_table.csv", index=False)

    accounting_rows = []
    for effect_id, acc in result["per_effect_accounting"].items():
        accounting_rows.append({"effect_id": effect_id, **acc})
    pd.DataFrame(accounting_rows).to_csv(OUT_DIR / "per_effect_accounting.csv", index=False)

    output = {"per_partition": per_partition, "reconciliation": reconciliation, "invalid_rate": result["invalid_rate"]}
    (OUT_DIR / "calibration_scoring_result.json").write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    print(f"scored {len(per_effect_df)} effects")
    print(per_effect_df[["study_id", "effect_id", "theta_L_pp", "theta_H_pp", "paired_n"]].to_string())

    if per_effect_df["theta_H_pp"].isna().any():
        missing_human = per_effect_df[per_effect_df["theta_H_pp"].isna()]["effect_id"].tolist()
        raise RuntimeError(f"missing human_ate (theta_H) for effect(s): {missing_human}")

    fit_result = fit_calibration_model_comparison(
        per_effect_df["theta_L_pp"].tolist(),
        per_effect_df["theta_H_pp"].tolist(),
        per_effect_df["study_id"].tolist(),
        outputs_dir=OUT_DIR,
    )
    print(json.dumps({k: v for k, v in fit_result.items() if k not in ("loso_predictions", "comparison_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
