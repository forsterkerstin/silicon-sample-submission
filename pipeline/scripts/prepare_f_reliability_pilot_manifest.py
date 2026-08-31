"""Materialize (never submit) the frozen F reliability/convergence pilot manifest.

Implements the R_F freeze_r_f_rule from
outputs/final_offline_gate/model_selection_r_f_rule_manifest.json: before any
full-scale F candidate bakeoff, run a small deterministic-effect pilot at
nested profile sizes N = 50, 100, 250, 500 (convergence) with two independent
replicates per profile (stochastic reliability), using ONLY the already-frozen
effect/profile selection machinery in pipeline/ate/f_reliability.py and the
already-frozen F prompt compiler. This script builds request manifests only;
it never calls the Together API.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import survey_content as sc  # noqa: E402
from ate.f_reliability import NESTED_SIZES, create_pilot_manifest  # noqa: E402
from calibration.study_population import archive_profile_to_prompt_profile  # noqa: E402
from inference.prompts import (  # noqa: E402
    PROMPT_COMPILER_VERSION,
    build_f_prompt_render_from_items,
    schema_hash,
    text_hash,
)
from inference.request_logging import seed_from_request_key  # noqa: E402
from inference.together_batch import (  # noqa: E402
    BatchRequest,
    SMOKE_MODEL_PRICES_PER_1M_TOKENS,
    _chat_body,
    compute_engine_config_hash,
    custom_id_from_request_key,
)

import render_prompt_validation as rpv  # noqa: E402

OUT_ROOT = PIPELINE_ROOT / "outputs" / "scientific_bakeoff" / "f_reliability_pilot"
# The frozen pilot_manifest_path in config/model_config.yaml -- this is where
# the 12-effect selection itself is written, matching what require_frozen_f_protocol()
# and downstream code expect to find. Request-batch JSONL/CSVs (per candidate) are a
# separate, non-path-designated artifact and live under OUT_ROOT.
FROZEN_PILOT_MANIFEST_DIR = PIPELINE_ROOT / "outputs" / "f_reliability"
ATE_ARCHIVE_PATH = PIPELINE_ROOT / "data" / "ate_archive.csv"
EXTERNAL_F_PANEL_PATH = PIPELINE_ROOT / "data" / "generated" / "external_primary_f_panels.csv"
REPLICATES = (1, 2)
CONDITIONS = ("control", "treatment")


def render_messages_hash(messages: list[dict[str, str]]) -> str:
    return text_hash("\n".join(f"{m['role']}:{m['content']}" for m in messages))


def eligible_effects() -> pd.DataFrame:
    archive = pd.read_csv(ATE_ARCHIVE_PATH)
    primary = archive[archive["included_primary_calibration"] == True].copy()  # noqa: E712
    return primary[["study_id", "effect_id", "outcome_type", "population_type"]]


def select_pilot_effects() -> pd.DataFrame:
    return create_pilot_manifest(eligible_effects(), outputs_dir=FROZEN_PILOT_MANIFEST_DIR)


def materials_for_pilot_effects(pilot_effects: pd.DataFrame) -> tuple[dict[tuple[str, str], tuple[str, dict[str, Any]]], dict[str, str]]:
    """{(effect_id, condition_id): (material_text, item)} for all pilot effects x conditions,
    plus {effect_id: reason} for any effect the existing frozen extraction code cannot
    materialize (e.g. a bare "no narrative" control arm with zero non-demographic material
    pages -- an existing gap in render_prompt_validation.archive_material_and_item, not
    something this script invents a workaround for)."""
    hypotheses = pd.read_csv(rpv.ARCHIVE_HYPOTHESES_PATH)
    selection_rows = []
    per_effect: dict[str, tuple[str, str, str, str, str]] = {}
    for effect_id in pilot_effects["effect_id"]:
        study, outcome_name, _hyp = rpv._archive_outcome_name(effect_id)
        control_condition, treatment_condition = rpv._condition_pair_for_effect(effect_id, hypotheses)
        per_effect[effect_id] = (study, outcome_name, control_condition, treatment_condition, _hyp)
        selection_rows.append({"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": control_condition, "arm": "control"})
        selection_rows.append({"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": treatment_condition, "arm": "treatment"})
    source_rows = rpv.export_archive_source_rows(pd.DataFrame(selection_rows))
    source_key = {
        (str(src["study"]), str(src["outcome.name"]), str(src["condition.name"])): src for _, src in source_rows.iterrows()
    }
    materials: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    excluded: dict[str, str] = {}
    for effect_id, (study, outcome_name, control_condition, treatment_condition, _hyp) in per_effect.items():
        control_row = source_key[(study, outcome_name, control_condition)]
        treatment_row = source_key[(study, outcome_name, treatment_condition)]
        try:
            control_material, control_item = rpv.archive_material_and_item(control_row, effect_id=effect_id, is_control_arm=True)
            treatment_material, treatment_item = rpv.archive_material_and_item(treatment_row, effect_id=effect_id)
            if control_item["question_text"] != treatment_item["question_text"]:
                raise RuntimeError(f"{effect_id}: control/treatment outcome wording mismatch")
        except ValueError as exc:
            excluded[effect_id] = str(exc)
            continue
        materials[(effect_id, "control")] = (control_material, control_item)
        materials[(effect_id, "treatment")] = (treatment_material, treatment_item)
    return materials, excluded


def request_from_render(
    *,
    render,
    requested_model: str,
    study_id: str,
    profile_id: str,
    condition_id: str,
    outcome_id: str,
    replicate_id: int,
    request_stage: str = "f_reliability_pilot",
) -> BatchRequest:
    return BatchRequest(
        request_key=render.request_key,
        custom_id=custom_id_from_request_key(f"{requested_model}|{render.request_key}"),
        role="F",
        study_id=study_id,
        profile_id=profile_id,
        condition_id=condition_id,
        outcome_id=outcome_id,
        replicate_id=replicate_id,
        requested_model=requested_model,
        prompt_hash=render_messages_hash(render.messages),
        schema_version=schema_hash(render.response_schema),
        prompt_protocol_id=render.protocol_id,
        prompt_compiler_version=PROMPT_COMPILER_VERSION,
        seed=seed_from_request_key(render.request_key),
        status="pending",
        messages=render.messages,
        response_schema=render.response_schema,
        response_key_map=render.response_key_map or {},
        request_stage=request_stage,
        engine_config_hash=compute_engine_config_hash(requested_model),
    )


MANIFEST_FIELDS = [
    "request_key", "custom_id", "role", "study_id", "profile_id", "condition_id",
    "outcome_id", "replicate_id", "requested_model", "prompt_hash", "schema_version",
    "prompt_protocol_id", "prompt_compiler_version", "seed", "status", "required_fields",
    "response_key_map", "request_stage", "engine_config_hash",
    "f_prompt_protocol_id", "external_f_population_panel_version", "external_f_population_panel_sha256",
]


def build_requests_for_model(requested_model: str, pilot_effects: pd.DataFrame, materials: dict, panel: pd.DataFrame) -> list[BatchRequest]:
    from calibration.external_prediction_provenance import current_external_f_panel_provenance

    panel_prov = current_external_f_panel_provenance(EXTERNAL_F_PANEL_PATH)
    f_prompt_protocol_id = yaml.safe_load((PIPELINE_ROOT / "config" / "model_config.yaml").read_text())["prompting"]["f_prompt_protocol"]

    requests: list[BatchRequest] = []
    for effect_id in pilot_effects["effect_id"]:
        if (effect_id, "control") not in materials:
            continue
        study, outcome_name, _hyp = rpv._archive_outcome_name(effect_id)
        rows = panel[panel["effect_id"] == effect_id]
        if len(rows) != 500:
            raise RuntimeError(f"{effect_id}: expected 500 panel profiles, found {len(rows)}")
        profile_rows = rows.to_dict("records")
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
                        # `material` can only be "" here because archive_material_and_item
                        # accepted it for the frozen-designated control arm (is_control_arm=True
                        # in materials_for_pilot_effects); a treatment arm with empty material
                        # would have raised there already, so this check is safe.
                        intentional_no_material_control=not material,
                    )
                    req = request_from_render(
                        render=render,
                        requested_model=requested_model,
                        study_id=study,
                        profile_id=f_profile_id,
                        condition_id=condition_id,
                        outcome_id=effect_id,
                        replicate_id=replicate_id,
                    )
                    requests.append(req)
    if len({r.custom_id for r in requests}) != len(requests):
        raise RuntimeError("duplicate custom_id in F reliability pilot manifest")
    return requests


def write_requests(requests: list[BatchRequest], out_dir: Path, panel_prov: dict, f_prompt_protocol_id: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "request_manifest.csv"
    jsonl_path = out_dir / "batch_input.jsonl"
    prompt_chars = 0
    max_tokens_total = 0
    with open(manifest_path, "w", newline="", encoding="utf-8") as mf, open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for req in requests:
            row = asdict(req)
            row["required_fields"] = req.required_fields
            row["response_key_map"] = json.dumps(req.response_key_map, sort_keys=True)
            row["f_prompt_protocol_id"] = f_prompt_protocol_id
            row["external_f_population_panel_version"] = panel_prov["external_f_population_panel_version"]
            row["external_f_population_panel_sha256"] = panel_prov["external_f_population_panel_sha256"]
            writer.writerow({field: row[field] for field in MANIFEST_FIELDS})
            body = _chat_body(req)
            prompt_chars += sum(len(m["content"]) for m in req.messages)
            max_tokens_total += int(body["max_tokens"])
            jf.write(json.dumps({"custom_id": req.custom_id, "body": body}, sort_keys=True) + "\n")
    return {
        "manifest": str(manifest_path),
        "jsonl": str(jsonl_path),
        "requests": len(requests),
        "estimated_prompt_characters": prompt_chars,
        "estimated_prompt_tokens_rough": int(prompt_chars / 4),
        "maximum_output_tokens": max_tokens_total,
    }


def worst_case_cost(model: str, estimated_prompt_tokens_rough: int, maximum_output_tokens: int) -> float:
    prices = SMOKE_MODEL_PRICES_PER_1M_TOKENS[model]
    return estimated_prompt_tokens_rough * prices["input"] / 1_000_000 + maximum_output_tokens * prices["output"] / 1_000_000


def main() -> int:
    from calibration.external_prediction_provenance import current_external_f_panel_provenance

    pilot_effects = select_pilot_effects()
    materials, excluded_effects = materials_for_pilot_effects(pilot_effects)
    if excluded_effects:
        print(f"WARNING: {len(excluded_effects)}/{len(pilot_effects)} deterministically-selected pilot effect(s) "
              f"could not be materialized by the existing frozen extraction code and are EXCLUDED, not substituted:")
        for eid, reason in excluded_effects.items():
            print(f"  - {eid}: {reason}")
    panel = pd.read_csv(EXTERNAL_F_PANEL_PATH)
    panel = panel[panel["effect_id"].isin(set(pilot_effects["effect_id"]))]
    panel_prov = current_external_f_panel_provenance(EXTERNAL_F_PANEL_PATH)
    f_prompt_protocol_id = yaml.safe_load((PIPELINE_ROOT / "config" / "model_config.yaml").read_text())["prompting"]["f_prompt_protocol"]

    summary: dict[str, Any] = {
        "purpose": "F R_F reliability/convergence pilot manifest only; not submitted",
        "pilot_effects_selected": pilot_effects["effect_id"].tolist(),
        "pilot_effects_materialized": [e for e in pilot_effects["effect_id"] if (e, "control") in materials],
        "pilot_effects_excluded": excluded_effects,
        "nested_sizes": list(NESTED_SIZES),
        "replicates": list(REPLICATES),
        "conditions": list(CONDITIONS),
        "models": {},
    }
    for model in ["deepseek-ai/DeepSeek-V4-Pro-0813", "google/gemma-4-31B-it"]:
        requests = build_requests_for_model(model, pilot_effects, materials, panel)
        model_dir = OUT_ROOT / re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")
        written = write_requests(requests, model_dir, panel_prov, f_prompt_protocol_id)
        cost = worst_case_cost(model, written["estimated_prompt_tokens_rough"], written["maximum_output_tokens"])
        written["worst_case_cost_usd"] = round(cost, 4)
        summary["models"][model] = written
        print(model, "->", json.dumps(written, indent=2))

    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
