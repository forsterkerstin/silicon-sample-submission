"""Materialize (never submit) the full 136-effect external F calibration
production manifest.

Offline preparation only -- this script never calls the Together API. It
builds the genuinely fresh (replicate_id=7, never used by any prior F
request) request set for every effect flagged `included_primary_calibration`
in data/ate_archive.csv, at the frozen N_F=500 population panel
(data/generated/external_primary_f_panels.csv, already built and frozen by
scripts/build_external_calibration_panels.py) and R_F=1 (the frozen
replacement-R1 reliability decision, config/model_config.yaml
f_protocol.target_f_num_draws / external_calibration_f_num_draws == 1).

Reuses, unmodified: the R1-root-cause format-only v2 prompt instruction
(inference.prompts.build_f_prompt_render_from_items,
response_format_instruction_version="v2"), the same deterministic profile
ordering used for R1's partitioning (ate.f_reliability.deterministic_profile_order),
and the same material/item extraction machinery already frozen for the F
mini-screen and F reliability pilot/R1 (scripts/render_prompt_validation.py).

Unlike the 12-effect pilot/R1 manifests, condition-pair/material extraction
is NOT guaranteed to succeed for every one of the 136 primary effects; any
effect that cannot be materialized by the existing frozen extraction code is
EXCLUDED and reported with a reason -- never silently substituted or forced.

Partitioning: control/treatment pairs for the same profile always stay in
the same partition (partition assigned once per (study, profile) via
deterministic_profile_order, independent of condition). The partition count
K is chosen as the smallest candidate in CANDIDATE_PARTITION_COUNTS (in
order) for which every partition's exact JSONL byte size stays under
OPERATIONAL_FILE_SIZE_CEILING_BYTES and its request count stays under
Together's documented per-batch limit -- computed analytically from the
already-serialized request bodies (no repeated disk writes per candidate K);
only the chosen K's partitions are actually written to disk.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from ate.f_reliability import deterministic_profile_order  # noqa: E402
from calibration.study_population import archive_profile_to_prompt_profile  # noqa: E402
from inference.prompts import build_f_prompt_render_from_items  # noqa: E402
from inference.together_batch import (  # noqa: E402
    TOGETHER_BATCH_MAX_INPUT_FILE_BYTES,
    TOGETHER_BATCH_MAX_REQUESTS,
    BatchRequest,
    _chat_body,
)
from prepare_f_reliability_pilot_manifest import (  # noqa: E402
    ATE_ARCHIVE_PATH,
    EXTERNAL_F_PANEL_PATH,
    request_from_render,
    worst_case_cost,
    write_requests,
)
import render_prompt_validation as rpv  # noqa: E402

OUT_ROOT = PIPELINE_ROOT / "outputs" / "scientific_bakeoff" / "f_calibration_production"
REQUESTED_MODEL = "google/gemma-4-31B-it"
REPLICATES = (7,)  # fresh: never used by F-screen(1), pilot(1,2, never submitted), R1(3,4), replacement R1(5,6)
CONDITIONS = ("control", "treatment")
REQUEST_STAGE = "f_calibration_production"
N_F = 500
OPERATIONAL_FILE_SIZE_CEILING_BYTES = int(95 * 1024 * 1024)
CANDIDATE_PARTITION_COUNTS = (4, 5, 10, 20, 25, 50, 100, 125, 250, 500)  # divisors of 500, ascending


def eligible_effects() -> pd.DataFrame:
    archive = pd.read_csv(ATE_ARCHIVE_PATH)
    primary = archive[archive["included_primary_calibration"] == True].copy()  # noqa: E712
    return primary[["study_id", "effect_id", "outcome_type", "population_type", "outcome_min", "outcome_max", "outcome_range", "finite_range"]]


def materials_for_effects(effects: pd.DataFrame) -> tuple[dict[tuple[str, str], tuple[str, dict[str, Any]]], dict[str, str]]:
    """{(effect_id, condition_id): (material_text, item)} for every effect that
    the existing frozen extraction code CAN materialize, plus {effect_id: reason}
    for every effect it cannot -- unlike materials_for_pilot_effects (which never
    needed to handle a condition-pair lookup failure across only 12 hand-picked
    effects), this wraps BOTH the condition-pair lookup and the material/item
    extraction in per-effect exception handling, since across the full 136
    primary effects at least one archived hypothesis/condition mapping is known
    to fail lookup (Haaland874:Affirmative action: Assistance:hyp1, verified
    live before this function was written)."""
    hypotheses = pd.read_csv(rpv.ARCHIVE_HYPOTHESES_PATH)
    per_effect: dict[str, tuple[str, str, str, str, str]] = {}
    excluded: dict[str, str] = {}
    selection_rows = []
    for effect_id in effects["effect_id"]:
        try:
            study, outcome_name, hyp = rpv._archive_outcome_name(effect_id)
            control_condition, treatment_condition = rpv._condition_pair_for_effect(effect_id, hypotheses)
        except Exception as exc:  # noqa: BLE001
            excluded[effect_id] = f"condition_pair_lookup: {exc}"
            continue
        per_effect[effect_id] = (study, outcome_name, control_condition, treatment_condition, hyp)
        selection_rows.append({"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": control_condition, "arm": "control"})
        selection_rows.append({"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": treatment_condition, "arm": "treatment"})

    if not selection_rows:
        return {}, excluded
    source_rows = rpv.export_archive_source_rows(pd.DataFrame(selection_rows))
    source_key = {(str(src["study"]), str(src["outcome.name"]), str(src["condition.name"])): src for _, src in source_rows.iterrows()}

    materials: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for effect_id, (study, outcome_name, control_condition, treatment_condition, _hyp) in per_effect.items():
        try:
            control_row = source_key[(study, outcome_name, control_condition)]
            treatment_row = source_key[(study, outcome_name, treatment_condition)]
            control_material, control_item = rpv.archive_material_and_item(control_row, effect_id=effect_id, is_control_arm=True)
            treatment_material, treatment_item = rpv.archive_material_and_item(treatment_row, effect_id=effect_id)
            if control_item["question_text"] != treatment_item["question_text"]:
                raise ValueError(f"{effect_id}: control/treatment outcome wording mismatch")
        except (KeyError, ValueError) as exc:
            excluded[effect_id] = f"material_extraction: {exc}"
            continue
        materials[(effect_id, "control")] = (control_material, control_item)
        materials[(effect_id, "treatment")] = (treatment_material, treatment_item)
    return materials, excluded


def build_requests(effects: pd.DataFrame, materials: dict, panel: pd.DataFrame) -> tuple[list[BatchRequest], dict[str, int]]:
    """Returns (requests, profile_rank_by_pid) -- profile_rank_by_pid maps
    f_profile_id -> its deterministic 0..499 rank within its (study, effect),
    assigned once per profile independent of condition/replicate, so a
    profile's control/treatment requests always share a partition regardless
    of which K is ultimately chosen."""
    requests: list[BatchRequest] = []
    profile_rank_by_pid: dict[str, int] = {}
    materialized_effects = sorted({eid for (eid, _cond) in materials})
    for effect_id in materialized_effects:
        study = effects.loc[effects["effect_id"] == effect_id, "study_id"].iloc[0]
        rows = panel[panel["effect_id"] == effect_id]
        if len(rows) != N_F:
            raise RuntimeError(f"{effect_id}: expected {N_F} panel profiles, found {len(rows)}")
        profile_rows = rows.to_dict("records")
        all_profile_ids = [str(r["f_profile_id"]) for r in profile_rows]
        ordered = deterministic_profile_order(study, all_profile_ids)
        for rank, pid in enumerate(ordered):
            profile_rank_by_pid.setdefault(pid, rank)

        for condition_id in CONDITIONS:
            material, item = materials[(effect_id, condition_id)]
            for replicate_id in REPLICATES:
                for prow in profile_rows:
                    f_profile_id = str(prow["f_profile_id"])
                    profile = archive_profile_to_prompt_profile(prow)
                    render = build_f_prompt_render_from_items(
                        profile,
                        material,
                        [item],
                        study_id=study,
                        f_profile_id=f_profile_id,
                        outcome_id=effect_id,
                        replicate_id=replicate_id,
                        condition_id=condition_id,
                        study_setting="This is an online survey shown to adult respondents.",
                        intentional_no_material_control=not material,
                        response_format_instruction_version="v2",
                    )
                    req = request_from_render(
                        render=render,
                        requested_model=REQUESTED_MODEL,
                        study_id=study,
                        profile_id=f_profile_id,
                        condition_id=condition_id,
                        outcome_id=effect_id,
                        replicate_id=replicate_id,
                        request_stage=REQUEST_STAGE,
                    )
                    requests.append(req)
    if len({r.custom_id for r in requests}) != len(requests):
        raise RuntimeError("duplicate custom_id in calibration production manifest")
    return requests, profile_rank_by_pid


def request_line_bytes(req: BatchRequest) -> int:
    body = _chat_body(req)
    line = json.dumps({"custom_id": req.custom_id, "body": body}, sort_keys=True) + "\n"
    return len(line.encode("utf-8"))


def choose_partition_count(requests: list[BatchRequest], manifest_df: pd.DataFrame, profile_rank_by_pid: dict[str, int]) -> tuple[int, dict[int, dict[str, Any]]]:
    """Analytically (no disk writes) find the smallest K in
    CANDIDATE_PARTITION_COUNTS for which every partition's exact serialized
    byte size stays under the operational ceiling and its request count stays
    under Together's documented per-batch limit. Returns (K, trial_report) --
    trial_report records every K actually tested and why it passed/failed, so
    a rejected K is disclosed, not silently skipped."""
    line_bytes = {req.custom_id: request_line_bytes(req) for req in requests}
    ranks = manifest_df["profile_id"].astype(str).map(profile_rank_by_pid)
    if ranks.isna().any():
        raise RuntimeError("some requests could not be assigned a deterministic profile rank")
    ranks = ranks.astype(int)

    trial_report: dict[int, dict[str, Any]] = {}
    for k in CANDIDATE_PARTITION_COUNTS:
        partition_of = ranks % k
        byte_sums = [0] * k
        req_counts = [0] * k
        for cid, part in zip(manifest_df["custom_id"], partition_of):
            byte_sums[part] += line_bytes[cid]
            req_counts[part] += 1
        max_bytes = max(byte_sums)
        max_requests = max(req_counts)
        fits = max_bytes < OPERATIONAL_FILE_SIZE_CEILING_BYTES and max_requests < TOGETHER_BATCH_MAX_REQUESTS
        trial_report[k] = {
            "max_partition_bytes": max_bytes,
            "max_partition_mb": round(max_bytes / (1024 * 1024), 2),
            "max_partition_requests": max_requests,
            "fits": fits,
        }
        if fits:
            return k, trial_report
    raise RuntimeError(f"no candidate partition count in {CANDIDATE_PARTITION_COUNTS} satisfies the size/request ceilings: {trial_report}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from calibration.external_prediction_provenance import current_external_f_panel_provenance

    effects = eligible_effects()
    if len(effects) != 136:
        print(f"NOTE: expected 136 primary effects from ate_archive.csv, found {len(effects)} -- proceeding with the real count, not assuming 136.")

    materials, excluded_effects = materials_for_effects(effects)
    materialized_effect_ids = sorted({eid for (eid, _cond) in materials})
    print(f"materialized {len(materialized_effect_ids)}/{len(effects)} primary effects; excluded {len(excluded_effects)}:")
    for eid, reason in excluded_effects.items():
        print(f"  - {eid}: {reason[:200]}")

    panel = pd.read_csv(EXTERNAL_F_PANEL_PATH)
    panel = panel[panel["effect_id"].isin(set(materialized_effect_ids))]
    panel_prov = current_external_f_panel_provenance(EXTERNAL_F_PANEL_PATH)
    f_prompt_protocol_id = yaml.safe_load((PIPELINE_ROOT / "config" / "model_config.yaml").read_text())["prompting"]["f_prompt_protocol"]

    requests, profile_rank_by_pid = build_requests(effects, materials, panel)
    expected = len(materialized_effect_ids) * N_F * len(CONDITIONS) * len(REPLICATES)
    if len(requests) != expected:
        raise RuntimeError(f"expected {expected} calibration production requests ({len(materialized_effect_ids)} effects x {N_F} profiles x {len(CONDITIONS)} conditions x {len(REPLICATES)} draw), got {len(requests)}")

    canonical_dir = OUT_ROOT / "google_gemma-4-31B-it"
    written = write_requests(requests, canonical_dir, panel_prov, f_prompt_protocol_id)
    cost = worst_case_cost(REQUESTED_MODEL, written["estimated_prompt_tokens_rough"], written["maximum_output_tokens"])
    written["worst_case_cost_usd"] = round(cost, 4)
    written["canonical_manifest_sha256"] = sha256_file(canonical_dir / "request_manifest.csv")
    written["canonical_jsonl_sha256"] = sha256_file(canonical_dir / "batch_input.jsonl")

    manifest_df = pd.read_csv(canonical_dir / "request_manifest.csv")
    k, partition_trial_report = choose_partition_count(requests, manifest_df, profile_rank_by_pid)
    print(f"chosen partition count K={k}; trial report:")
    for cand, report in partition_trial_report.items():
        print(f"  K={cand}: {report}")
        if cand == k:
            break

    ranks = manifest_df["profile_id"].astype(str).map(profile_rank_by_pid).astype(int)
    manifest_df["_partition"] = ranks % k

    partition_summaries = []
    requests_by_cid = {r.custom_id: r for r in requests}
    for p in range(k):
        part_rows = manifest_df[manifest_df["_partition"] == p]
        part_requests = [requests_by_cid[cid] for cid in part_rows["custom_id"]]
        part_dir = OUT_ROOT / f"part{p + 1}" / "google_gemma-4-31B-it"
        part_written = write_requests(part_requests, part_dir, panel_prov, f_prompt_protocol_id)
        part_cost = worst_case_cost(REQUESTED_MODEL, part_written["estimated_prompt_tokens_rough"], part_written["maximum_output_tokens"])
        part_written["worst_case_cost_usd"] = round(part_cost, 4)
        part_jsonl_path = part_dir / "batch_input.jsonl"
        part_written["sha256"] = sha256_file(part_jsonl_path)
        part_written["jsonl_size_mb"] = round(part_jsonl_path.stat().st_size / (1024 * 1024), 2)

        effects_covered = part_rows["outcome_id"].nunique()
        profiles_per_effect = part_rows.groupby("outcome_id")["profile_id"].nunique()
        control_n = int((part_rows["condition_id"] == "control").sum())
        treatment_n = int((part_rows["condition_id"] == "treatment").sum())

        partition_summaries.append(
            {
                "partition": f"part{p + 1}",
                "requests": len(part_rows),
                "effects": int(effects_covered),
                "profiles_per_effect_min": int(profiles_per_effect.min()),
                "profiles_per_effect_max": int(profiles_per_effect.max()),
                "control_requests": control_n,
                "treatment_requests": treatment_n,
                "jsonl_size_mb": part_written["jsonl_size_mb"],
                "sha256": part_written["sha256"],
                "worst_case_cost_usd": part_written["worst_case_cost_usd"],
            }
        )

    reconstructed: set[str] = set()
    for p in range(k):
        reconstructed |= set(manifest_df.loc[manifest_df["_partition"] == p, "custom_id"])
    canonical_ids = set(manifest_df["custom_id"])
    reconstruction_ok = reconstructed == canonical_ids and len(reconstructed) == expected

    summary: dict[str, Any] = {
        "purpose": "F* (Gemma) calibration-PRODUCTION manifest (replicate 7, R_F=1, format-v2); not submitted",
        "eligible_primary_effects": len(effects),
        "materialized_effects": len(materialized_effect_ids),
        "excluded_effects": excluded_effects,
        "n_profiles_per_effect": N_F,
        "conditions": list(CONDITIONS),
        "replicates": list(REPLICATES),
        "response_format_instruction_version": "v2",
        "request_stage": REQUEST_STAGE,
        "canonical": written,
        "partition_count": k,
        "partition_trial_report": partition_trial_report,
        "partitions": partition_summaries,
        "request_set_reconstruction_ok": reconstruction_ok,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"canonical": written, "partition_count": k, "request_set_reconstruction_ok": reconstruction_ok}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
