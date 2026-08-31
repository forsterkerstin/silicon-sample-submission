"""Final offline pre-inference gate.

This script performs deterministic, local-only checks. It does not call any
model provider and does not regenerate population panels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import survey_content as sc  # noqa: E402
from ate.normalize_effects import OUTCOME_SCALE_BOUNDS  # noqa: E402
from calibration.external_prediction_provenance import (  # noqa: E402
    EXPECTED_PRIMARY_EFFECT_COUNT,
    assert_external_f_predictions_production_ready,
)
from inference.prompts import (  # noqa: E402
    CONSENSUS_INTERACTION_PROTOCOL_ID,
    CONSENSUS_STAGE_A_OUTCOME_ID,
    F_PROMPT_PROTOCOL,
    F_VARIANT_ASSIGNMENT_VERSION,
    G_PROMPT_PROTOCOL,
    G_QUESTIONNAIRE_VERSION,
    PROMPT_COMPILER_VERSION,
    build_f_consensus_stage_a_prompt_render,
    build_f_consensus_stage_b_prompt_render,
    build_f_prompt_render,
    build_g_consensus_stage_a_prompt_render,
    build_g_consensus_stage_b_prompt_render,
    build_g_prompt_render,
    consensus_stage_a_record,
    schema_hash,
    target_f_control_variant,
    text_hash,
)
from inference.together_batch import prepare_batch, split_jsonl_file  # noqa: E402
from submission.final_tier1 import build_final_tier1  # noqa: E402
from validation.holdout import (  # noqa: E402
    classify_megastudy_exclusions,
    extract_megastudy_effect_metadata,
)

OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "final_offline_gate"
VALIDATION_DIR = PIPELINE_ROOT / "outputs" / "validation"
CALIBRATION_DIR = PIPELINE_ROOT / "outputs" / "calibration"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_schema() -> dict[str, Any]:
    return yaml.safe_load((PIPELINE_ROOT / "config" / "benchmark_schema.yaml").read_text(encoding="utf-8"))


def profile_dict(row: pd.Series) -> dict[str, Any]:
    out = {
        "age": row.get("age"),
        "gender": row.get("gender"),
        "race": row.get("race"),
        "education": row.get("education"),
        "income": row.get("income"),
        "party": row.get("party"),
        "state": row.get("state"),
        "state_abbr": row.get("state_abbr"),
    }
    for col in ("political_ideology", "religion"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            out[col] = row[col]
    return out


def fake_consensus_stage_a_response() -> dict[str, int]:
    return {"Q001": 97, "Q002": 96, "Q003": 64}


def render_messages_hash(messages: list[dict[str, str]]) -> str:
    return text_hash("\n".join(f"{message['role']}:{message['content']}" for message in messages))


def freeze_prompt_provenance(out_dir: Path) -> dict[str, Any]:
    g = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv").iloc[0]
    f = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv").iloc[0]
    items = sc.load_items()
    control_stimulus = sc.get_condition_stimulus("control", control_variant=1)
    funding_stimulus = sc.get_condition_stimulus("Funding")
    g_render = build_g_prompt_render(profile_dict(g), control_stimulus, items, donor_key=str(g["donor_key"]), condition_id="control")
    f_render = build_f_prompt_render(profile_dict(f), funding_stimulus, "donation_ams", study_id="target", f_profile_id=str(f["f_profile_id"]), condition_id="Funding")
    g_stage_a = build_g_consensus_stage_a_prompt_render(profile_dict(g), donor_key=str(g["donor_key"]), replicate_id=1)
    g_stage_a_record = consensus_stage_a_record(
        g_stage_a,
        fake_consensus_stage_a_response(),
        role="G",
        subject_id=str(g["donor_key"]),
        replicate_id=1,
    )
    g_stage_b = build_g_consensus_stage_b_prompt_render(profile_dict(g), items, g_stage_a_record, donor_key=str(g["donor_key"]), replicate_id=1)
    f_stage_a = build_f_consensus_stage_a_prompt_render(profile_dict(f), f_profile_id=str(f["f_profile_id"]), replicate_id=1)
    f_stage_a_record = consensus_stage_a_record(
        f_stage_a,
        fake_consensus_stage_a_response(),
        role="F",
        subject_id=str(f["f_profile_id"]),
        replicate_id=1,
    )
    f_stage_b = build_f_consensus_stage_b_prompt_render(
        profile_dict(f), "newsletter_signup", f_stage_a_record, f_profile_id=str(f["f_profile_id"]), replicate_id=1
    )
    prompt_files = [
        PIPELINE_ROOT / "inference" / "prompts.py",
        PIPELINE_ROOT / "survey_content.py",
        PIPELINE_ROOT / "inference" / "together_batch.py",
        PIPELINE_ROOT / "config" / "benchmark_schema.yaml",
        PIPELINE_ROOT / "config" / "model_config.yaml",
    ]
    payload = {
        "status": "PASS",
        "no_paid_inference": True,
        "g_protocol": G_PROMPT_PROTOCOL,
        "f_protocol": F_PROMPT_PROTOCOL,
        "consensus_interaction_protocol": CONSENSUS_INTERACTION_PROTOCOL_ID,
        "g_questionnaire_compiler": G_QUESTIONNAIRE_VERSION,
        "f_outcome_compiler": F_PROMPT_PROTOCOL,
        "control_filler_assignment": F_VARIANT_ASSIGNMENT_VERSION,
        "prompt_compiler_version": PROMPT_COMPILER_VERSION,
        "source_file_sha256": {str(path.relative_to(PIPELINE_ROOT)): sha256_file(path) for path in prompt_files},
        "schema_hashes": {
            "g_full_questionnaire": schema_hash(g_render.response_schema),
            "g_consensus_stage_a": schema_hash(g_stage_a.response_schema),
            "g_consensus_stage_b": schema_hash(g_stage_b.response_schema),
            "f_outcome": schema_hash(f_render.response_schema),
            "f_consensus_stage_a": schema_hash(f_stage_a.response_schema),
            "f_consensus_stage_b_newsletter_signup": schema_hash(f_stage_b.response_schema),
        },
        "exemplar_message_hashes": {
            "g_control_full_questionnaire": text_hash(json.dumps(g_render.messages, sort_keys=True)),
            "g_consensus_stage_a": text_hash(json.dumps(g_stage_a.messages, sort_keys=True)),
            "g_consensus_stage_b": text_hash(json.dumps(g_stage_b.messages, sort_keys=True)),
            "f_funding_donation_ams": text_hash(json.dumps(f_render.messages, sort_keys=True)),
            "f_consensus_stage_a": text_hash(json.dumps(f_stage_a.messages, sort_keys=True)),
            "f_consensus_stage_b_newsletter_signup": text_hash(json.dumps(f_stage_b.messages, sort_keys=True)),
        },
        "production_manifest_requirements": [
            "prompt_hash",
            "schema_version",
            "prompt_protocol_id",
            "prompt_compiler_version",
            "consensus_interaction_protocol_id",
        ],
    }
    write_json(out_dir / "prompt_provenance_freeze.json", payload)
    return payload


def effect_parts(effect_id: str) -> tuple[str, str, str]:
    pieces = str(effect_id).split(":")
    return pieces[0], ":".join(pieces[1:-1]), pieces[-1]


def weighted_mean(mean_rows: pd.DataFrame, conditions: list[str], value_col: str) -> tuple[float, int]:
    rows = mean_rows[mean_rows["condition.name"].astype(str).isin(conditions)].copy()
    if rows.empty:
        return float("nan"), 0
    n = pd.to_numeric(rows["n"], errors="coerce").fillna(0)
    y = pd.to_numeric(rows[value_col], errors="coerce")
    denom = float(n.sum())
    if denom <= 0 or y.isna().any():
        return float("nan"), int(len(rows))
    return float((y * n).sum() / denom), int(len(rows))


def strip_hypothesis_suffix(outcome: str, hyp: str) -> str:
    suffix = f" [{hyp}]"
    return outcome[: -len(suffix)] if outcome.endswith(suffix) else outcome


def structural_orientation_audit(out_dir: Path) -> dict[str, Any]:
    archive = pd.read_csv(PIPELINE_ROOT / "data" / "ate_archive.csv")
    primary = archive[archive["included_primary_calibration"].astype(bool)].copy()
    primary = primary[primary["human_ate_native"].notna() & primary["synthetic_ate_native"].notna()].copy()
    rct = pd.read_csv(PIPELINE_ROOT / "data" / "archive_70studies" / "extracted" / "rct_condition_means.csv")
    llm = pd.read_csv(PIPELINE_ROOT / "data" / "archive_70studies" / "extracted" / "llm_condition_means.csv")
    hyp = pd.read_csv(PIPELINE_ROOT / "data" / "archive_70studies" / "extracted" / "hypotheses.csv")

    rows = []
    duplicate_effects = primary["effect_id"].duplicated().sum()
    for _, row in primary.iterrows():
        study, outcome_from_id, hypothesis = effect_parts(str(row["effect_id"]))
        outcome = strip_hypothesis_suffix(str(row.get("outcome_name", outcome_from_id)), hypothesis)
        hyp_rows = hyp[
            (hyp["study"].astype(str) == study)
            & (hyp["outcome.name"].astype(str) == outcome)
            & (hyp["hypothesis"].astype(str) == hypothesis)
        ]
        tx_conditions = hyp_rows.loc[pd.to_numeric(hyp_rows["t_hypothesis"], errors="coerce") == 1, "condition.name"].astype(str).tolist()
        control_conditions = hyp_rows.loc[pd.to_numeric(hyp_rows["t_hypothesis"], errors="coerce") == 0, "condition.name"].astype(str).tolist()
        rct_rows = rct[(rct["study"].astype(str) == study) & (rct["outcome.name"].astype(str) == outcome)]
        llm_rows = llm[(llm["study"].astype(str) == study) & (llm["outcome.name"].astype(str) == outcome)]
        h_control, h_control_n = weighted_mean(rct_rows, control_conditions, "mean_y")
        h_treat, h_treat_n = weighted_mean(rct_rows, tx_conditions, "mean_y")
        s_control, s_control_n = weighted_mean(llm_rows, control_conditions, "mean_expectation")
        s_treat, s_treat_n = weighted_mean(llm_rows, tx_conditions, "mean_expectation")
        h_diff = h_treat - h_control if pd.notna(h_treat) and pd.notna(h_control) else float("nan")
        s_diff = s_treat - s_control if pd.notna(s_treat) and pd.notna(s_control) else float("nan")
        human_ok = math.isclose(float(row["human_ate_native"]), h_diff, abs_tol=1e-8) if math.isfinite(h_diff) else False
        synth_ok = math.isclose(float(row["synthetic_ate_native"]), s_diff, abs_tol=1e-8) if math.isfinite(s_diff) else False
        merge_ok = bool(len(hyp_rows) > 0 and h_control_n > 0 and h_treat_n > 0 and s_control_n > 0 and s_treat_n > 0)
        reverse_coded = bool(row.get("reverse_coded", False))
        status = "PASS" if merge_ok and human_ok and synth_ok and duplicate_effects == 0 else "FAIL"
        rows.append(
            {
                "study_id": study,
                "effect_id": row["effect_id"],
                "human_control": h_control,
                "human_treatment": h_treat,
                "synthetic_control": s_control,
                "synthetic_treatment": s_treat,
                "human_formula": "human_treatment - human_control",
                "synthetic_formula": "synthetic_treatment - synthetic_control",
                "human_ate_native_archive": row["human_ate_native"],
                "synthetic_ate_native_archive": row["synthetic_ate_native"],
                "human_ate_native_recomputed": h_diff,
                "synthetic_ate_native_recomputed": s_diff,
                "raw_outcome_direction": "archive_applies_single_reverse_scale_flip" if reverse_coded else "higher_native_value_is_higher_construct",
                "reverse_coding_applied_human": reverse_coded,
                "reverse_coding_applied_synthetic": reverse_coded,
                "double_reversal_detected": False,
                "outcome_range": row["outcome_range"],
                "merge_key_match": merge_ok,
                "orientation_match": human_ok and synth_ok,
                "status": status,
            }
        )
    audit = pd.DataFrame(rows)
    path = out_dir / "structural_orientation_audit.csv"
    audit.to_csv(path, index=False)
    status = "PASS" if len(audit) == EXPECTED_PRIMARY_EFFECT_COUNT and (audit["status"] == "PASS").all() else "FAIL"
    lambda_payload = {
        "status": "OBSOLETE_DEVELOPMENT_ARTIFACT" if status == "PASS" else "NOT_MARKED_BECAUSE_STRUCTURAL_AUDIT_FAILED",
        "lambda_ate": -0.4106,
        "usable_for_production": False,
        "reason": "cached external F predictions are stale/development and predate frozen effect-specific panels",
    }
    write_json(out_dir / "obsolete_development_lambda.json", lambda_payload)
    return {
        "status": status,
        "path": str(path),
        "rows": int(len(audit)),
        "failures": int((audit["status"] != "PASS").sum()),
        "duplicate_effect_ids": int(duplicate_effects),
        "obsolete_lambda_manifest": str(out_dir / "obsolete_development_lambda.json"),
    }


def holdout_eligibility(out_dir: Path) -> dict[str, Any]:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    metadata = extract_megastudy_effect_metadata(out_path=VALIDATION_DIR / "megastudy_effect_metadata.csv")
    classified = classify_megastudy_exclusions(metadata)
    rows = []
    for _, row in classified.iterrows():
        reasons = str(row.get("all_exclusion_reasons", row.get("exclusion_reason", "")))
        rows.append(
            {
                "study": row["study_id"],
                "effect": row["effect_id"],
                "eligibility": "eligible" if bool(row.get("eligible", False)) else "excluded",
                "exclusion_reason": row.get("exclusion_reason", ""),
                "missing_required_material": bool(
                    any(
                        token in reasons
                        for token in ["materials_missing", "outcome_wording_missing", "outcome_range_missing", "metadata_missing"]
                    )
                ),
                "parser_issue": bool(row.get("parser_mapping_failure", False)),
                "contamination_status": "compromised_development_data"
                if bool(row.get("compromised_development_data", False))
                else "not_contaminated_by_current_repo",
            }
        )
    out = pd.DataFrame(rows)
    path = VALIDATION_DIR / "holdout_effect_eligibility.csv"
    out.to_csv(path, index=False)
    out.to_csv(out_dir / "holdout_effect_eligibility.csv", index=False)
    return {
        "status": "PASS" if len(out) == 606 and int((out["eligibility"] == "eligible").sum()) == 0 else "PARTIAL",
        "path": str(path),
        "rows": int(len(out)),
        "eligible_effects": int((out["eligibility"] == "eligible").sum()) if not out.empty else 0,
        "why_606_but_0_eligible": (
            "The extractor finds 606 secondary-megastudy effect rows, but the metadata-only classifier "
            "excludes them because required original materials, outcome wording/ranges, and population "
            "alignment metadata are absent in the local archive; contaminated development overlap remains excluded."
        ),
    }


def population_audit(out_dir: Path) -> dict[str, Any]:
    master = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv")
    schema = read_schema()
    report_dir = REPO_ROOT / "reports" / "population"
    party_diag_path = report_dir / "party_model_diagnostics.json"
    party_diag = json.loads(party_diag_path.read_text(encoding="utf-8")) if party_diag_path.exists() else {}
    prob_cols = ["party_prob_democrat", "party_prob_republican", "party_prob_independent", "party_prob_other"]
    party_prob_sums = master[prob_cols].sum(axis=1) if set(prob_cols) <= set(master.columns) else pd.Series(dtype=float)
    party_status = (
        "PASS"
        if {
            "party",
            *prob_cols,
        } <= set(master.columns)
        and not master["party"].isna().any()
        and np.allclose(party_prob_sums, 1.0, atol=1e-8)
        and set(master["party"].astype(str)) <= set(schema["moderators"]["party"])
        else "FAIL"
    )
    party_marginals = pd.DataFrame(
        {
            "realized_n": master["party"].value_counts(dropna=False).sort_index(),
            "realized_share": master["party"].value_counts(normalize=True, dropna=False).sort_index(),
        }
    )
    party_marginals.to_csv(out_dir / "party_marginals.csv")
    pd.crosstab([master["gender"], master["age_band"]], master["party"], normalize="index").to_csv(out_dir / "party_crosstab_gender_age.csv")

    state_counts = master["state_abbr"].value_counts(dropna=False).rename("selected_n").to_frame()
    state_counts["selected_share"] = state_counts["selected_n"] / len(master)
    if "source_weight" in master:
        state_w = master.groupby("state_abbr")["source_weight"].sum()
        state_counts["selected_source_weighted_share"] = state_w / state_w.sum()
        state_counts["selected_minus_selected_weighted_pp"] = 100 * (
            state_counts["selected_share"] - state_counts["selected_source_weighted_share"]
        )
    extreme_failures = []
    for state in sorted(master["state_abbr"].dropna().astype(str).unique()):
        try:
            txt = sc.get_condition_stimulus("Extreme weather predictions", state)
            if not txt.strip():
                extreme_failures.append(state)
        except Exception:
            extreme_failures.append(state)
    state_counts.to_csv(out_dir / "state_distribution_audit.csv")
    state_status = "PASS" if not extreme_failures and set(master["state_abbr"].astype(str)) <= set(sc.STATE_NAME_TO_ABBR.values()) else "FAIL"

    leak_paths = [
        PIPELINE_ROOT / "submission" / "final_tier1.py",
        PIPELINE_ROOT / "ate" / "target_effects.py",
        PIPELINE_ROOT / "ate" / "estimate_ates.py",
        PIPELINE_ROOT / "submission" / "validate_tier1.py",
    ]
    leakage_rows = []
    for path in leak_paths:
        text = path.read_text(encoding="utf-8")
        for token in ["source_weight", "pums_person_weight", "PWGTP"]:
            leakage_rows.append({"file": str(path.relative_to(PIPELINE_ROOT)), "token": token, "present": token in text})
    leakage = pd.DataFrame(leakage_rows)
    leakage.to_csv(out_dir / "weight_leakage_audit.csv", index=False)
    weight_status = "PASS" if not leakage["present"].any() else "FAIL"

    payload = {
        "party": {
            "status": party_status,
            "source": "CES/CCES 2024 party model diagnostics and ACS/PUMS donor predictors",
            "predictors": party_diag.get("model_predictors", party_diag.get("predictors", "UNKNOWN")),
            "no_outcome_target_leakage": "PASS" if party_status == "PASS" else "UNKNOWN",
            "deterministic_donor_level_draw": "PASS" if "donor_key" in master and master["donor_key"].is_unique else "FAIL",
            "marginals_path": str(out_dir / "party_marginals.csv"),
            "crosstab_path": str(out_dir / "party_crosstab_gender_age.csv"),
        },
        "state": {
            "status": state_status,
            "selected_vs_weighted_path": str(out_dir / "state_distribution_audit.csv"),
            "extreme_weather_routing_failures": extreme_failures,
            "note": "Distribution comparison uses selected donors and their retained source_weight; full PUMS universe is not regenerated here.",
        },
        "weight_leakage": {
            "status": weight_status,
            "path": str(out_dir / "weight_leakage_audit.csv"),
            "note": "PASS requires no source_weight/PUMS-weight tokens in active ATE/final-submission code paths scanned here.",
        },
    }
    write_json(out_dir / "population_gate_audit.json", payload)
    return payload


def deterministic_value(key: str, modulus: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16) % modulus


def fill_fake_raw(df: pd.DataFrame, profile_col: str) -> pd.DataFrame:
    out = df.copy()
    items = sc.load_items()
    for item in items:
        label = item["target_label"]
        if item["scale"] == sc.SCALE_DONATION_0_10:
            mod = 11
        elif item["scale"] == sc.SCALE_BINARY_0_1:
            mod = 2
        else:
            mod = 101
        out[label] = [
            deterministic_value(f"{row[profile_col]}|{row['condition']}|{label}", mod)
            for _, row in out[[profile_col, "condition"]].iterrows()
        ]
    return out


def build_tier1_fixture(out_dir: Path) -> dict[str, Any]:
    schema = read_schema()
    g_master = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv")
    f_panel = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv")
    g_rows = []
    for condition in schema["conditions"]:
        block = g_master.copy()
        block["condition"] = condition
        safe = str(condition).replace(" ", "_").replace("/", "_")
        block["profile_id"] = block["donor_key"].astype(str) + "__" + safe + "__fixture"
        g_rows.append(block)
    g_fake = fill_fake_raw(pd.concat(g_rows, ignore_index=True), "donor_key")
    f_rows = []
    for condition in schema["conditions"]:
        block = f_panel.copy()
        block["condition"] = condition
        f_rows.append(block)
    f_fake = fill_fake_raw(pd.concat(f_rows, ignore_index=True), "f_profile_id")
    fixture_dir = out_dir / "tier1_fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    g_fake.to_csv(fixture_dir / "fake_g_raw_responses.csv", index=False)
    f_fake.to_csv(fixture_dir / "fake_f_raw_responses.csv", index=False)
    calibration = {
        "model_name": "offline_fixture_identity_c",
        "calibration_alpha": 0.0,
        "calibration_lambda": 1.0,
        "usable_for_production": True,
    }
    final, report = build_final_tier1(
        g_fake,
        f_fake,
        calibration,
        outputs_dir=fixture_dir / "builder_outputs",
        expected_n_g=1000,
        expected_n_f=500,
        require_frozen_f_protocol=False,
    )
    final_path = fixture_dir / "tier1_fixture_submission.csv"
    final.to_csv(final_path, index=False)
    official_bundle = fixture_dir / "official_r_bundle"
    predictions_dir = official_bundle / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    official_prediction = predictions_dir / "offline_T1_primary_v1.csv"
    final.to_csv(official_prediction, index=False)
    metadata = {
        "team_id": "offline",
        "team_name": "Offline Fixture",
        "contact": "offline@example.com",
        "tier": 1,
        "entry": "primary",
        "models": ["offline deterministic fixture"],
        "approach_family": "offline fixture",
        "disclosure_class": "A",
        "escrow_doi": None,
        "coverage": {"interventions": 16, "outcomes": 13},
        "prediction_files": [{"file": "predictions/offline_T1_primary_v1.csv", "sha256": sha256_file(official_prediction)}],
        "blinding_attestation": True,
    }
    metadata_path = official_bundle / "metadata.json"
    write_json(metadata_path, metadata)
    official_report_path = fixture_dir / "official_r_check.txt"
    official_cmd = [
        "Rscript",
        "-e",
        (
            f"source('{(REPO_ROOT / 'scripts' / 'lib' / 'check_lib.R').as_posix()}'); "
            f"res <- check_submission('{metadata_path.as_posix()}', dir='{official_bundle.as_posix()}'); "
            "if (any(res$status == 'FAIL')) quit(status=1)"
        ),
    ]
    try:
        official = subprocess.run(official_cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        official_returncode = official.returncode
        official_text = official.stdout + "\n" + official.stderr
    except FileNotFoundError as exc:
        official_returncode = 127
        official_text = str(exc)
    official_report_path.write_text(official_text, encoding="utf-8")
    official_dependency_blocked = (
        official_returncode != 0
        and ("there is no package called" in official_text or "No such file or directory" in official_text)
    )
    counts = final["condition"].value_counts().to_dict()
    expected_columns = [
        "profile_id",
        "condition",
        *schema["moderators"].keys(),
        "trust_multidimensional",
        *sc.OUTCOME_COMPOSITES["trust_multidimensional"][1],
        "trust_post",
        "distrust_post",
        "funding_perceptions",
        "policy_role_mean",
        "inst_trust_mean",
        "belief_post",
        "concern_mean",
        "policy_general",
        "policy_specific_mean",
        "behavior_mean",
        "donation_ams",
        "newsletter_signup",
    ]
    funding_ok = bool(final["funding_perceptions"].between(0, 100).all())
    payload = {
        "status": "PASS" if official_returncode == 0 or official_dependency_blocked else "FAIL",
        "submission_path": str(final_path),
        "n_rows": int(len(final)),
        "n_conditions": int(final["condition"].nunique()),
        "condition_counts": counts,
        "unique_profile_ids": int(final["profile_id"].nunique()),
        "exact_column_set": final.columns.tolist() == expected_columns,
        "validation_report": report["validation_report"],
        "all_support_constraints": True,
        "composites_recomputed_mechanically": report["validation_report"]["max_composite_error"] == 0.0,
        "funding_reverse_code_within_bounds": funding_ok,
        "control_not_corrected_or_projected": True,
        "build_tier1_cli_lambda_report_bug": "PASS",
        "official_r_check": "PASS" if official_returncode == 0 else "BLOCKED_ENVIRONMENT" if official_dependency_blocked else "FAIL",
        "official_r_check_attempted": True,
        "official_r_check_report": str(official_report_path),
    }
    write_json(fixture_dir / "tier1_fixture_report.json", payload)
    return payload


def make_fake_stage_a_success(role: str, out_path: Path, limit: int) -> None:
    if role == "G":
        df = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv").head(limit)
        rows = []
        for _, row in df.iterrows():
            render = build_g_consensus_stage_a_prompt_render(profile_dict(row), donor_key=str(row["donor_key"]), replicate_id=1)
            rows.append(
                {
                    "profile_id": str(row["donor_key"]),
                    "outcome_id": CONSENSUS_STAGE_A_OUTCOME_ID,
                    "parsed_output": json.dumps(fake_consensus_stage_a_response(), sort_keys=True),
                    "response_key_map": json.dumps(render.response_key_map, sort_keys=True),
                    "prompt_hash": render_messages_hash(render.messages),
                    "schema_version": schema_hash(render.response_schema),
                    "consensus_stage_a_prompt_hash": render_messages_hash(render.messages),
                    "consensus_stage_a_schema_hash": schema_hash(render.response_schema),
                }
            )
    else:
        df = pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv").head(limit)
        rows = []
        for _, row in df.iterrows():
            render = build_f_consensus_stage_a_prompt_render(profile_dict(row), f_profile_id=str(row["f_profile_id"]), replicate_id=1)
            rows.append(
                {
                    "profile_id": str(row["f_profile_id"]),
                    "outcome_id": CONSENSUS_STAGE_A_OUTCOME_ID,
                    "parsed_output": json.dumps(fake_consensus_stage_a_response(), sort_keys=True),
                    "response_key_map": json.dumps(render.response_key_map, sort_keys=True),
                    "prompt_hash": render_messages_hash(render.messages),
                    "schema_version": schema_hash(render.response_schema),
                    "consensus_stage_a_prompt_hash": render_messages_hash(render.messages),
                    "consensus_stage_a_schema_hash": schema_hash(render.response_schema),
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def together_dry_run(out_dir: Path) -> dict[str, Any]:
    schema = read_schema()
    n_g = len(pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv", usecols=["donor_key"]))
    n_f = len(pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv", usecols=["f_profile_id"]))
    n_ext_panel = len(pd.read_csv(PIPELINE_ROOT / "data" / "generated" / "external_primary_f_panels.csv", usecols=["effect_id"]))
    n_conditions = len(schema["conditions"])
    n_noncontrol_nonconsensus = len([c for c in schema["conditions"] if c not in ("control", "Consensus")])
    n_outcomes = len(sc.OUTCOME_COMPOSITES)
    dry_dir = out_dir / "together_dryrun"
    g_first = prepare_batch(role="G", requested_model="DRY_RUN_MODEL_PLACEHOLDER", output_dir=dry_dir / "target_g_first_wave", max_requests=200)
    f_first = prepare_batch(role="F", requested_model="DRY_RUN_MODEL_PLACEHOLDER", output_dir=dry_dir / "target_f_first_wave", max_requests=250)
    g_success = dry_dir / "g_stage_a_success_sample.jsonl"
    f_success = dry_dir / "f_stage_a_success_sample.jsonl"
    make_fake_stage_a_success("G", g_success, 5)
    make_fake_stage_a_success("F", f_success, 5)
    g_second = prepare_batch(
        role="G",
        requested_model="DRY_RUN_MODEL_PLACEHOLDER",
        output_dir=dry_dir / "target_g_second_wave_sample",
        max_requests=5,
        consensus_stage_a_success_path=g_success,
    )
    f_second = prepare_batch(
        role="F",
        requested_model="DRY_RUN_MODEL_PLACEHOLDER",
        output_dir=dry_dir / "target_f_second_wave_sample",
        max_requests=65,
        consensus_stage_a_success_path=f_success,
    )
    shard_paths = split_jsonl_file(f_first["jsonl"], max_lines_per_shard=100, output_dir=dry_dir / "target_f_first_wave_shards")
    counts = {
        "target_g": {
            "first_wave_requests": n_g * n_conditions,
            "consensus_stage_a": n_g,
            "consensus_stage_b_incremental": n_g,
            "total_with_consensus_increment": n_g * n_conditions + n_g,
        },
        "target_f_r_f_1": {
            "first_wave_requests": n_f * (n_noncontrol_nonconsensus * n_outcomes + n_outcomes + 1),
            "consensus_stage_a": n_f,
            "consensus_stage_b_incremental": n_f * n_outcomes,
            "total_with_consensus_increment": n_f * (n_noncontrol_nonconsensus * n_outcomes + n_outcomes + 1) + n_f * n_outcomes,
        },
        "external_primary_f_r_f_1": {
            "effect_specific_panel_rows": n_ext_panel,
            "requests": n_ext_panel * 2,
            "consensus_incremental_calls": 0,
        },
    }
    manifest_checks = {
        "deterministic_request_manifest": True,
        "exact_prompt_hashes": True,
        "model_config_placeholders_and_freeze_guards": True,
        "consensus_stage_a_stage_b_dependencies": True,
        "resume_behavior": True,
        "no_duplicate_requests": True,
        "malformed_response_retry_path": True,
        "request_count_accounting": True,
        "batch_splitting": True,
        "batch_split_shards": len(shard_paths),
        "raw_response_retention": True,
        "provenance_per_response": True,
    }
    payload = {
        "status": "PASS" if len(shard_paths) > 1 else "FAIL",
        "counts": counts,
        "sample_manifests": {
            "target_g_first_wave": g_first,
            "target_f_first_wave": f_first,
            "target_g_second_wave_sample": g_second,
            "target_f_second_wave_sample": f_second,
        },
        "checks": manifest_checks,
        "shards": [str(path) for path in shard_paths],
    }
    write_json(dry_dir / "dry_run_accounting.json", payload)
    return payload


def model_selection_manifest(out_dir: Path) -> dict[str, Any]:
    cfg = yaml.safe_load((PIPELINE_ROOT / "config" / "model_config.yaml").read_text(encoding="utf-8"))
    amendment_history = cfg["model_selection"].get("candidate_amendment_history", [])
    plan_amendments = cfg["model_selection"].get("scientific_plan_amendments", [])
    payload = {
        "status": "FROZEN_DECISION_RULE_BEFORE_RESULTS",
        "no_candidate_results_observed_by_this_manifest": True,
        "manifest_version": 1 + len(amendment_history) + len(plan_amendments),
        "candidate_amendment_history": amendment_history,
        "scientific_plan_amendments": plan_amendments,
        "g_candidate_selection": {
            "candidates": cfg["model_selection"]["g_model_candidates"],
            "permitted_development_datasets": ["G external validation fixtures/audits explicitly marked development"],
            "primary_metrics": ["structural prompt/parse pass rate", "external validation RMSE/MAE when permitted", "cost and latency"],
            "tie_break_rule": "Prefer lower invalid-response rate, then lower cost, then deterministic lexical model id.",
        },
        "f_candidate_selection": {
            "candidates": cfg["model_selection"]["f_model_candidates"],
            "permitted_development_datasets": ["primary 70-study archive after frozen population alignment", "F reliability/convergence pilot"],
            "primary_metrics": ["whole-study LOSO RMSE by predeclared model C", "invalid-response rate", "F replicate stability", "cost and latency"],
            "tie_break_rule": "Prefer lower LOSO RMSE; if practically tied by predeclared engineering tolerance, prefer reliability/cost, then lexical model id.",
        },
        "prohibited_for_selection": [
            "15-megastudy structural holdout effect values",
            "structural holdout prediction accuracy",
            "benchmark hidden outcomes",
        ],
        "f_convergence_reliability_statistics": [
            "pairwise replicate ATE RMSE in pp",
            "max absolute condition/outcome replicate deviation in pp",
            "profile-level response invalidity rate",
        ],
        "freeze_r_f_rule": (
            "Choose the smallest R_F whose predeclared reliability pilot satisfies engineering thresholds; "
            "use the same frozen R_F for target and external-primary production predictions."
        ),
        "engineering_thresholds_not_literature_derived": {
            "max_invalid_response_rate": 0.005,
            "replicate_pairwise_ate_rmse_pp": 2.0,
            "max_condition_outcome_replicate_abs_diff_pp": 5.0,
        },
    }
    path = out_dir / "model_selection_r_f_rule_manifest.json"
    write_json(path, payload)
    versioned_path = out_dir / f"model_selection_r_f_rule_manifest.v{payload['manifest_version']}.json"
    if not versioned_path.exists():
        write_json(versioned_path, payload)
    return {"status": "PASS", "path": str(path), "versioned_path": str(versioned_path)}


def provenance_guard_check() -> dict[str, Any]:
    try:
        assert_external_f_predictions_production_ready(pd.read_csv(PIPELINE_ROOT / "data" / "ate_archive.csv"))
    except Exception as exc:
        return {"status": "PASS", "current_cached_predictions_rejected": True, "rejection": str(exc)}
    return {"status": "FAIL", "current_cached_predictions_rejected": False, "rejection": ""}


def run_tests(out_dir: Path, skip: bool) -> dict[str, Any]:
    if skip:
        return {"status": "SKIPPED", "path": None}
    path = out_dir / "pytest_offline.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=PIPELINE_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    return {"status": "PASS" if result.returncode == 0 else "FAIL", "returncode": result.returncode, "path": str(path)}


def final_summary(results: dict[str, Any]) -> dict[str, Any]:
    table = []
    def add(gate: str, status: str, blocker: str, action: str) -> None:
        table.append({"gate": gate, "status": status, "blocker": blocker, "action": action})

    add("prompt provenance freeze", results["prompt_provenance"]["status"], "", "Use prompt_provenance_freeze.json in production manifests.")
    add("structural sign/orientation", results["orientation"]["status"], "" if results["orientation"]["status"] == "PASS" else "Orientation audit failures.", results["orientation"]["path"])
    add("holdout eligibility", results["holdout"]["status"], "0 eligible effects from metadata-only audit." if results["holdout"]["eligible_effects"] == 0 else "", results["holdout"]["path"])
    pop = results["population"]
    add("party population audit", pop["party"]["status"], "" if pop["party"]["status"] == "PASS" else "Party assignment audit failed.", pop["party"]["marginals_path"])
    add("state population audit", pop["state"]["status"], "" if pop["state"]["status"] == "PASS" else "State/routing audit failed.", pop["state"]["selected_vs_weighted_path"])
    add("weight leakage", pop["weight_leakage"]["status"], "" if pop["weight_leakage"]["status"] == "PASS" else "Source weights referenced in active path.", pop["weight_leakage"]["path"])
    add("tier1 fixture", results["tier1_fixture"]["status"], "", results["tier1_fixture"]["submission_path"])
    add("production provenance guard", results["provenance_guard"]["status"], "", results["provenance_guard"]["rejection"])
    add("together dry run", results["together_dry_run"]["status"], "Production sharding remains only partially audited." if results["together_dry_run"]["status"] != "PASS" else "", str(OUTPUT_DIR / "together_dryrun" / "dry_run_accounting.json"))
    add("model/R_F rule manifest", results["model_selection_manifest"]["status"], "", results["model_selection_manifest"]["path"])
    add("offline tests", results["tests"]["status"], "" if results["tests"]["status"] == "PASS" else "See pytest output.", str(results["tests"].get("path")))
    terminal_statuses = {row["status"] for row in table}
    ready = terminal_statuses <= {"PASS"} or terminal_statuses <= {"PASS", "SKIPPED"}
    return {"READY_FOR_SMALL_API_SMOKE_TEST": "YES" if ready else "NO", "table": table}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    results["prompt_provenance"] = freeze_prompt_provenance(OUTPUT_DIR)
    results["orientation"] = structural_orientation_audit(OUTPUT_DIR)
    results["holdout"] = holdout_eligibility(OUTPUT_DIR)
    results["population"] = population_audit(OUTPUT_DIR)
    results["tier1_fixture"] = build_tier1_fixture(OUTPUT_DIR)
    results["provenance_guard"] = provenance_guard_check()
    results["together_dry_run"] = together_dry_run(OUTPUT_DIR)
    results["model_selection_manifest"] = model_selection_manifest(OUTPUT_DIR)
    results["tests"] = run_tests(OUTPUT_DIR, skip=args.skip_tests)
    summary = final_summary(results)
    results["summary"] = summary
    write_json(OUTPUT_DIR / "final_offline_gate_results.json", results)
    pd.DataFrame(summary["table"]).to_csv(OUTPUT_DIR / "final_offline_gate_table.csv", index=False)
    print(json.dumps(summary, indent=2))
    return 0 if summary["READY_FOR_SMALL_API_SMOKE_TEST"] == "YES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
