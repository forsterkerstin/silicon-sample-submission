"""Structural-holdout validation utilities.

The functions in this module deliberately separate three phases:

1. metadata audits and split manifests,
2. method freezing,
3. holdout opening/evaluation.

Only phase 3 consumes effect-level holdout outcomes. Calibration fitting and
model selection are restricted to rows explicitly assigned to the development
archive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
VALIDATION_DIR = PIPELINE_ROOT / "outputs" / "validation"
PLOTS_DIR = VALIDATION_DIR / "plots"
ATE_ARCHIVE_PATH = PIPELINE_ROOT / "data" / "ate_archive.csv"
MEGASTUDIES_RDS_PATH = PIPELINE_ROOT / "data" / "archive_70studies" / "megastudies.RDS"
PRIMARY_STUDY_FEATURES_PATH = PIPELINE_ROOT / "data" / "archive_70studies" / "RA_study_features.csv"
MODEL_CONFIG_PATH = PIPELINE_ROOT / "config" / "model_config.yaml"
CALIBRATION_MODEL_PATH = PIPELINE_ROOT / "outputs" / "calibration_selected_model.json"
FROZEN_METHOD_MANIFEST_PATH = VALIDATION_DIR / "frozen_method_manifest.json"
HOLDOUT_STATUS_PATH = VALIDATION_DIR / "holdout_status.json"

ASSIGNED_ROLES = {"development_calibration", "structural_holdout", "excluded", "compromised_holdout"}
BENCHMARK_REFERENCE_ROWS = [
    {
        "reference": "Ashokkumar et al. survey-experiment megastudies",
        "pearson_r": 0.43,
        "adjusted_pearson_r": 0.52,
        "usage": "context_only_not_a_target",
    },
    {
        "reference": "Ashokkumar et al. text-treatment megastudies",
        "pearson_r": 0.45,
        "adjusted_pearson_r": 0.54,
        "usage": "context_only_not_a_target",
    },
]


class HoldoutIntegrityError(RuntimeError):
    """Raised when a validation action would compromise the holdout."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_model_config(path: Path = MODEL_CONFIG_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def base_megastudy_id(dataset: str) -> str:
    if dataset.startswith("Tappin-"):
        return "Tappin"
    if dataset.startswith("Broockman-"):
        return "Broockman"
    return dataset


def extract_megastudy_effect_metadata(
    rds_path: Path = MEGASTUDIES_RDS_PATH,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Extract effect labels/shape metadata from megastudies.RDS.

    The output intentionally excludes `estimate.rct` and model-prediction
    columns; it is for eligibility/split auditing, not performance evaluation.
    """
    if out_path is None:
        out_path = VALIDATION_DIR / "megastudy_effect_metadata.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rds_path.exists():
        return pd.DataFrame()

    r_code = f"""
    x <- readRDS({json.dumps(str(rds_path))})
    rows <- list()
    k <- 1
    base_id <- function(dataset) {{
      if (grepl('^Tappin-', dataset)) return('Tappin')
      if (grepl('^Broockman-', dataset)) return('Broockman')
      dataset
    }}
    clean <- function(value) {{
      value <- as.character(value)
      value[is.na(value)] <- ''
      gsub('[^A-Za-z0-9_.-]+', '_', value)
    }}
    for (i in seq_len(nrow(x))) {{
      df <- x$df[[i]]
      dataset <- as.character(x$dataset[[i]])
      outcome <- as.character(x$outcome[[i]])
      issue <- as.character(x$issue[[i]])
      side <- as.character(x$side[[i]])
      if (is.na(outcome)) outcome <- ''
      if (is.na(issue)) issue <- ''
      if (is.na(side)) side <- ''
      for (j in seq_len(nrow(df))) {{
        condition <- as.character(df$condition.name[[j]])
        rows[[k]] <- data.frame(
          archive_row_id = i,
          study_id = base_id(dataset),
          dataset = dataset,
          outcome = outcome,
          issue = issue,
          side = side,
          condition_name = condition,
          effect_id = paste(dataset, clean(outcome), clean(issue), clean(side), sprintf('E%03d', j), sep=':'),
          number_effects_in_archive_row = nrow(df),
          n_human_reported = as.numeric(x$N[[i]]),
          df_columns = paste(names(df), collapse='|'),
          human_estimate_column_present = 'estimate.rct' %in% names(df),
          gpt4_prediction_column_present = 'prediction.gpt-4' %in% names(df),
          stringsAsFactors = FALSE
        )
        k <- k + 1
      }}
    }}
    write.csv(do.call(rbind, rows), {json.dumps(str(out_path))}, row.names=FALSE)
    """
    result = subprocess.run(["Rscript", "-e", r_code], cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise HoldoutIntegrityError(f"failed to extract megastudy metadata: {result.stderr.strip()}")
    return pd.read_csv(out_path)


def megastudy_study_summary(effect_metadata: pd.DataFrame) -> pd.DataFrame:
    if effect_metadata.empty:
        return pd.DataFrame(columns=["study_id", "number_effects", "archive_rows", "datasets", "outcomes", "n_human_reported"])
    return (
        effect_metadata.groupby("study_id", dropna=False)
        .agg(
            number_effects=("effect_id", "count"),
            archive_rows=("archive_row_id", "nunique"),
            datasets=("dataset", lambda s: "; ".join(sorted(set(map(str, s))))),
            outcomes=("outcome", lambda s: "; ".join([v for v in sorted(set(map(str, s))) if v and v != "nan"])),
            n_human_reported=("n_human_reported", "max"),
        )
        .reset_index()
        .sort_values("study_id")
    )


def current_secondary_contamination() -> dict[str, dict[str, Any]]:
    """Pre-existing repo usage of secondary/domain holdout material."""
    contamination: dict[str, dict[str, Any]] = {}
    climate_data = PIPELINE_ROOT / "data" / "climate_advocacy_megastudy" / "data" / "advocacy_data.csv"
    climate_ate = PIPELINE_ROOT / "data" / "validation_advocacy_ate.csv"
    baseline_refs = PIPELINE_ROOT / "data" / "baseline_references.json"
    if climate_data.exists() or climate_ate.exists() or baseline_refs.exists():
        contamination["Voelkel2025"] = {
            "human_outcomes_previously_opened": True,
            "used_for_g_selection": False,
            "used_for_f_selection": False,
            "used_for_prompt_tuning": False,
            "used_for_protocol_tuning": True,
            "used_for_calibration": False,
            "used_for_debugging": True,
            "notes": (
                "Local climate_advocacy_megastudy files, validation_advocacy_ate.csv, "
                "and/or baseline_references.json are present; treat Voelkel climate "
                "megastudy as development/compromised, not an independent second holdout."
            ),
        }

    # megastudies.RDS's "Doell" dataset and data/data63.xlsx (Vlasceanu et al.
    # 2024, "Addressing climate change with behavioral science: A global
    # intervention tournament in 63 countries") are the same underlying
    # study: their condition names match 1:1 by abbreviation (BindMorals/
    # BindingMoral, CollAct/CollectAction, DynNorms/DynamicNorm, FtrSelf/
    # FutureSelfCont, WorkTogNorms/WorkTogetherNorm, Let2Ftr/LetterFutureGen,
    # NegEmo/NegativeEmotions, PluralIg/PluralIgnorance, PsychDist/
    # PsychDistance, SciCons/SciConsens, SysJust/SystemJust -- verified
    # against data63.xlsx's actual condName values, not assumed).
    # scripts/build_vlasceanu_validation.py reads data63.xlsx, computes real
    # human_ate_pp effect sizes, prints them, and persists them to
    # data/validation_vlasceanu_us.csv -- that output's presence on disk is
    # direct evidence this study's human outcomes have already been opened.
    vlasceanu_output = PIPELINE_ROOT / "data" / "validation_vlasceanu_us.csv"
    vlasceanu_source = PIPELINE_ROOT / "data" / "data63.xlsx"
    if vlasceanu_output.exists() or vlasceanu_source.exists():
        contamination["Doell"] = {
            "human_outcomes_previously_opened": True,
            "used_for_g_selection": False,
            "used_for_f_selection": False,
            "used_for_prompt_tuning": False,
            "used_for_protocol_tuning": False,
            "used_for_calibration": False,
            "used_for_debugging": True,
            "notes": (
                "data/data63.xlsx / data/validation_vlasceanu_us.csv are present "
                "(scripts/build_vlasceanu_validation.py reads and computes real "
                "human effect sizes from this study); megastudies.RDS's 'Doell' "
                "dataset is the same underlying study by condition-name match. "
                "Treat as development/compromised, not a pristine independent "
                "holdout; excluded from calibration fitting as it always was."
            ),
        }
    return contamination


def build_data_usage_audit(
    ate_archive: pd.DataFrame,
    primary_features: pd.DataFrame,
    megastudy_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    archive_primary_studies = set(ate_archive.loc[ate_archive.get("included_primary_calibration", False) == True, "study_id"].astype(str))
    all_archive_studies = set(ate_archive["study_id"].astype(str)) if not ate_archive.empty else set()
    contamination = current_secondary_contamination()

    for _, study in primary_features.iterrows():
        study_id = str(study["study"])
        is_used = study_id in archive_primary_studies
        rows.append(
            {
                "dataset": study.get("study_title", study_id),
                "study_id": study_id,
                "archive": "Ashokkumar primary 70-study archive",
                "human_outcomes_available": True,
                "human_outcomes_previously_opened": study_id in all_archive_studies,
                "used_for_g_selection": False,
                "used_for_f_selection": False,
                "used_for_prompt_tuning": False,
                "used_for_protocol_tuning": False,
                "used_for_calibration": is_used,
                "used_for_debugging": False,
                "eligible_as_pristine_holdout": False,
                "notes": "development/calibration archive; never label as independent holdout",
            }
        )

    for _, study in megastudy_summary.iterrows():
        study_id = str(study["study_id"])
        flags = contamination.get(study_id, {})
        opened = bool(flags.get("human_outcomes_previously_opened", False))
        rows.append(
            {
                "dataset": study.get("datasets", study_id),
                "study_id": study_id,
                "archive": "Ashokkumar secondary megastudy archive",
                "human_outcomes_available": True,
                "human_outcomes_previously_opened": opened,
                "used_for_g_selection": bool(flags.get("used_for_g_selection", False)),
                "used_for_f_selection": bool(flags.get("used_for_f_selection", False)),
                "used_for_prompt_tuning": bool(flags.get("used_for_prompt_tuning", False)),
                "used_for_protocol_tuning": bool(flags.get("used_for_protocol_tuning", False)),
                "used_for_calibration": bool(flags.get("used_for_calibration", False)),
                "used_for_debugging": bool(flags.get("used_for_debugging", False)),
                "eligible_as_pristine_holdout": not opened,
                "notes": flags.get("notes", "no pre-existing repo code/output reference to megastudies.RDS performance found"),
            }
        )
    return pd.DataFrame(rows)


def build_validation_split_manifest(primary_features: pd.DataFrame, ate_archive: pd.DataFrame, megastudy_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary_counts = (
        ate_archive.groupby("study_id")
        .agg(
            number_effects=("effect_id", "count"),
            primary_effects=("included_primary_calibration", lambda s: int((s == True).sum())),
        )
        .to_dict("index")
        if not ate_archive.empty
        else {}
    )
    for _, study in primary_features.iterrows():
        study_id = str(study["study"])
        counts = primary_counts.get(study_id, {"number_effects": 0, "primary_effects": 0})
        assigned = "development_calibration" if counts["primary_effects"] else "excluded"
        rows.append(
            {
                "archive": "primary_70_study_archive",
                "study_id": study_id,
                "study_name": study.get("study_title", ""),
                "DOI_if_available": study.get("link", ""),
                "population": "general U.S. adults (TESS sample)" if bool(study.get("study_is_tess", False)) else "not primary-eligible under current population metadata",
                "design_type": "survey experiment",
                "treatment_type": "text/survey treatment metadata from archive",
                "shared_control": "",
                "number_effects": int(counts["number_effects"]),
                "assigned_role": assigned,
                "reason": "eligible primary rows feed M0/M1/M2 selection and final C fit" if assigned == "development_calibration" else "no primary-eligible effects in current ate_archive.csv",
                "holdout_integrity_status": "development_data_not_holdout",
            }
        )

    contamination = current_secondary_contamination()
    for _, study in megastudy_summary.iterrows():
        study_id = str(study["study_id"])
        compromised = study_id in contamination
        rows.append(
            {
                "archive": "secondary_15_megastudy_archive",
                "study_id": study_id,
                "study_name": study.get("datasets", study_id),
                "DOI_if_available": "",
                "population": "metadata not yet mapped to F population contract",
                "design_type": "megastudy/shared-control structural analogue",
                "treatment_type": "mixed; must be checked before F reproduction",
                "shared_control": True,
                "number_effects": int(study["number_effects"]),
                "assigned_role": "compromised_holdout" if compromised else "structural_holdout",
                "reason": contamination[study_id]["notes"] if compromised else "reserved for structural holdout; eligibility pending metadata-only checks",
                "holdout_integrity_status": "compromised_development_data" if compromised else "reserved_not_opened_by_current_repo",
            }
        )
    out = pd.DataFrame(rows)
    bad_roles = sorted(set(out["assigned_role"]) - ASSIGNED_ROLES)
    if bad_roles:
        raise HoldoutIntegrityError(f"invalid assigned_role values: {bad_roles}")
    return out


def build_megastudy_holdout_eligibility(effect_metadata: pd.DataFrame) -> pd.DataFrame:
    classified = classify_megastudy_exclusions(effect_metadata)
    if classified.empty:
        return pd.DataFrame(
            columns=[
                "study_id",
                "effect_id",
                "population",
                "treatment_type",
                "outcome",
                "outcome_type",
                "outcome_range",
                "shared_control",
                "materials_available",
                "population_match_possible",
                "estimand_compatible",
                "eligible",
                "exclusion_reason",
            ]
        )
    out = classified[
        [
            "study_id",
            "effect_id",
            "population",
            "treatment_type",
            "outcome",
            "outcome_type",
            "outcome_range",
            "shared_control",
            "materials_available",
            "population_match_possible",
            "estimand_compatible",
            "eligible",
            "exclusion_reason",
        ]
    ].copy()
    return out


def _voelkel_materials_available() -> bool:
    return (PIPELINE_ROOT / "data" / "climate_advocacy_megastudy" / "materials" / "intervention_docx").exists()


def classify_megastudy_exclusions(effect_metadata: pd.DataFrame) -> pd.DataFrame:
    """Metadata-only holdout eligibility/exclusion classifier.

    This intentionally does not read `estimate.rct` values or calculate any
    holdout prediction accuracy. It classifies why an effect cannot currently
    be used as a pristine structural holdout under the F/C protocol.
    """
    rows: list[dict[str, Any]] = []
    contamination = current_secondary_contamination()
    if effect_metadata.empty:
        return pd.DataFrame()
    for _, row in effect_metadata.iterrows():
        study_id = str(row["study_id"])
        outcome = "" if pd.isna(row.get("outcome", "")) else str(row.get("outcome", ""))
        condition_name = "" if pd.isna(row.get("condition_name", "")) else str(row.get("condition_name", ""))
        is_compromised = study_id in contamination
        materials_available = bool(study_id == "Voelkel2025" and _voelkel_materials_available())
        parser_mapping_failure = not bool(condition_name.strip())
        shared_control = int(row.get("number_effects_in_archive_row", 0) or 0) > 1
        outcome_wording_available = False
        outcome_range_available = False
        population_metadata_available = False
        population_match_possible = False

        reasons: list[str] = []
        if is_compromised:
            reasons.append("compromised_development_data")
        if parser_mapping_failure:
            reasons.append("parser_mapping_failure")
        if not materials_available:
            reasons.append("materials_missing")
        if not outcome_wording_available:
            reasons.append("outcome_wording_missing")
        if not outcome_range_available:
            reasons.append("outcome_range_missing")
        if not population_metadata_available:
            reasons.append("metadata_missing")
        if not shared_control:
            reasons.append("no_shared_control")

        primary_reason_order = [
            "compromised_development_data",
            "parser_mapping_failure",
            "materials_missing",
            "outcome_wording_missing",
            "outcome_range_missing",
            "metadata_missing",
            "no_shared_control",
        ]
        primary_reason = next((reason for reason in primary_reason_order if reason in reasons), "other")
        if primary_reason == "compromised_development_data":
            problem_type = "holdout_integrity"
        elif primary_reason in {"materials_missing", "outcome_wording_missing", "outcome_range_missing", "metadata_missing", "parser_mapping_failure"}:
            problem_type = "data_or_metadata_gap"
        else:
            problem_type = "substantive_or_design_gap"

        rows.append(
            {
                "study_id": study_id,
                "effect_id": row["effect_id"],
                "dataset": row.get("dataset", ""),
                "outcome": outcome,
                "condition_name": condition_name,
                "population": "not established in repo metadata",
                "treatment_type": "metadata pending",
                "outcome_type": "metadata pending",
                "outcome_range": pd.NA,
                "shared_control": shared_control,
                "materials_available": materials_available,
                "outcome_wording_available": outcome_wording_available,
                "outcome_range_available": outcome_range_available,
                "population_metadata_available": population_metadata_available,
                "population_match_possible": population_match_possible,
                "estimand_compatible": True,
                "parser_mapping_failure": parser_mapping_failure,
                "compromised_development_data": is_compromised,
                "eligible": False,
                "primary_exclusion_reason": primary_reason,
                "all_exclusion_reasons": ";".join(reasons),
                "problem_type": problem_type,
                "exclusion_reason": (
                    contamination[study_id]["notes"]
                    if is_compromised
                    else "metadata/material gap: " + "; ".join(reasons)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_megastudy_exclusions(classified: pd.DataFrame) -> pd.DataFrame:
    if classified.empty:
        return pd.DataFrame(columns=["summary_level", "study_id", "primary_exclusion_reason", "problem_type", "n_effects"])
    by_study_reason = (
        classified.groupby(["study_id", "primary_exclusion_reason", "problem_type"], dropna=False)
        .size()
        .reset_index(name="n_effects")
    )
    by_study_reason.insert(0, "summary_level", "study_reason")
    by_reason = (
        classified.groupby(["primary_exclusion_reason", "problem_type"], dropna=False)
        .size()
        .reset_index(name="n_effects")
    )
    by_reason.insert(0, "study_id", "ALL")
    by_reason.insert(0, "summary_level", "reason_total")
    by_problem = classified.groupby(["problem_type"], dropna=False).size().reset_index(name="n_effects")
    by_problem["primary_exclusion_reason"] = "ALL"
    by_problem["study_id"] = "ALL"
    by_problem.insert(0, "summary_level", "problem_type_total")
    cols = ["summary_level", "study_id", "primary_exclusion_reason", "problem_type", "n_effects"]
    return pd.concat([by_reason[cols], by_problem[cols], by_study_reason[cols]], ignore_index=True)


def climate_holdout_overlap_audit(effect_metadata: pd.DataFrame) -> dict[str, Any]:
    vo_rows = effect_metadata[effect_metadata["study_id"].astype(str) == "Voelkel2025"] if not effect_metadata.empty else pd.DataFrame()
    climate_paths = {
        "data": PIPELINE_ROOT / "data" / "climate_advocacy_megastudy" / "data" / "advocacy_data.csv",
        "materials": PIPELINE_ROOT / "data" / "climate_advocacy_megastudy" / "materials",
        "validation_ate": PIPELINE_ROOT / "data" / "validation_advocacy_ate.csv",
        "baseline_references": PIPELINE_ROOT / "data" / "baseline_references.json",
    }
    local_present = {key: path.exists() for key, path in climate_paths.items()}
    contained = not vo_rows.empty
    compromised = any(local_present.values())
    if contained:
        status = "contained in structural holdout"
    elif compromised:
        status = "development / compromised holdout"
    else:
        status = "not found in repo or secondary metadata"
    return {
        "query_title": "A Registered Report Megastudy on the Persuasiveness of the Most-Cited Climate Messages",
        "query_authors": "Voelkel et al.",
        "query_doi": "10.1038/s41558-025-02536-2",
        "secondary_archive_match": contained,
        "matched_study_id": "Voelkel2025" if contained else None,
        "matched_effect_rows": int(len(vo_rows)),
        "local_climate_dataset_present": local_present,
        "human_outcomes_already_used_in_repo": compromised,
        "status": status,
        "independent_domain_holdout_eligible": bool(not contained and not compromised),
        "notes": (
            "Do not count as a second independent validation dataset when contained in the 15-study structural holdout. "
            "Local climate_advocacy_megastudy outputs also indicate development use."
        ),
    }


def assert_no_holdout_studies(
    rows: pd.DataFrame,
    split_manifest: pd.DataFrame,
    *,
    context: str,
    study_col: str = "study_id",
) -> None:
    if rows.empty:
        return
    holdout = set(
        split_manifest.loc[
            split_manifest["assigned_role"].isin(["structural_holdout", "compromised_holdout"]),
            "study_id",
        ].astype(str)
    )
    observed = set(rows[study_col].dropna().astype(str))
    bad = sorted(observed & holdout)
    if bad:
        raise HoldoutIntegrityError(f"{context} includes structural-holdout study/studies: {bad}")


def assert_no_holdout_in_calibration_archive(ate_archive: pd.DataFrame, split_manifest: pd.DataFrame) -> None:
    calibration_rows = ate_archive[ate_archive["included_primary_calibration"] == True]
    assert_no_holdout_studies(calibration_rows, split_manifest, context="C fitting archive")


def write_initial_holdout_status(method_hash: str | None = None, *, notes: str = "") -> dict[str, Any]:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "frozen_method_hash": method_hash,
        "holdout_opened_at": None,
        "method_changed_after_holdout": False,
        "holdout_still_pristine": True,
        "holdout_consumed": False,
        "notes": notes,
    }
    HOLDOUT_STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_frozen_method_manifest(path: Path = FROZEN_METHOD_MANIFEST_PATH) -> dict[str, Any]:
    if not path.exists():
        raise HoldoutIntegrityError(f"structural holdout evaluation requires frozen method manifest at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["method_hash", "selected_calibration_model", "final_alpha", "final_lambda", "N_G", "N_F"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise HoldoutIntegrityError(f"frozen method manifest missing key(s): {missing}")
    expected_hash = payload["method_hash"]
    copy = dict(payload)
    copy.pop("method_hash", None)
    if canonical_hash(copy) != expected_hash:
        raise HoldoutIntegrityError("frozen method manifest hash does not match manifest contents")
    return payload


def build_frozen_method_manifest(*, output_path: Path = FROZEN_METHOD_MANIFEST_PATH, allow_unselected_models: bool = False) -> dict[str, Any]:
    cfg = load_model_config()
    model_selection = cfg.get("model_selection", {})
    selected_g = model_selection.get("selected_g_model")
    selected_f = model_selection.get("selected_f_model")
    if not allow_unselected_models and (not selected_g or not selected_f):
        raise HoldoutIntegrityError(
            "cannot freeze final method: selected_g_model and selected_f_model must be set in pipeline/config/model_config.yaml"
        )
    if not CALIBRATION_MODEL_PATH.exists():
        raise HoldoutIntegrityError(f"cannot freeze final method: missing {CALIBRATION_MODEL_PATH}")
    calibration = json.loads(CALIBRATION_MODEL_PATH.read_text(encoding="utf-8"))
    payload = {
        "selected_G_model": selected_g,
        "selected_F_model": selected_f,
        "G_prompt_version": cfg.get("prompting", {}).get("g_prompt_protocol"),
        "F_prompt_version": cfg.get("prompting", {}).get("f_prompt_protocol"),
        "G_inference_params": cfg.get("inference_parameters", {}),
        "F_inference_params": cfg.get("inference_parameters", {}),
        "N_G": 1000,
        "N_F": int(cfg.get("f_protocol", {}).get("n_f", 500)),
        "F_stochastic_draw_count": int(cfg.get("f_protocol", {}).get("f_num_draws", 1)),
        "persona_generation_version": sha256_file(PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv"),
        "F_population_construction_version": sha256_file(PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv"),
        "calibration_eligibility_rules": "primary archive only; included_primary_calibration metadata; population-compatible effects only",
        "calibration_weighting_rule": calibration.get("weighting_rule"),
        "selected_calibration_model": calibration["model_name"],
        "final_alpha": float(calibration.get("calibration_alpha", calibration.get("alpha", 0.0))),
        "final_lambda": float(calibration.get("calibration_lambda", calibration.get("lambda", 1.0))),
        "date_time_frozen": now_utc(),
        "git_commit": git_commit(),
        "config_hashes": {
            "model_config.yaml": sha256_file(MODEL_CONFIG_PATH),
            "population.yaml": sha256_file(PIPELINE_ROOT / "config" / "population.yaml"),
            "benchmark_schema.yaml": sha256_file(PIPELINE_ROOT / "config" / "benchmark_schema.yaml"),
            "ate_archive.csv": sha256_file(ATE_ARCHIVE_PATH),
            "calibration_selected_model.json": sha256_file(CALIBRATION_MODEL_PATH),
        },
        "holdout_lock_rule": "do not modify model, prompts, N_F/R_F, calibration model, alpha/lambda, personas, or projection logic after opening holdout",
    }
    payload["method_hash"] = canonical_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_initial_holdout_status(payload["method_hash"], notes="frozen method created; structural holdout not opened")
    return payload


def percent_of_range(ate_native: Sequence[float] | pd.Series | np.ndarray, outcome_range: Sequence[float] | pd.Series | np.ndarray) -> np.ndarray:
    ate = np.asarray(ate_native, dtype=float)
    rng = np.asarray(outcome_range, dtype=float)
    if np.any(~np.isfinite(rng)) or np.any(rng <= 0):
        raise ValueError("outcome_range must be finite and positive")
    return 100 * ate / rng


def sign_label(value: float) -> str:
    if pd.isna(value):
        return "missing"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def apply_frozen_calibration(holdout: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    required = {"study_id", "effect_id", "outcome_range", "human_ate_native", "raw_f_ate_native"}
    missing = required - set(holdout.columns)
    if missing:
        raise ValueError(f"holdout predictions missing column(s): {sorted(missing)}")
    out = holdout.copy()
    out["human_ate_pp"] = percent_of_range(out["human_ate_native"], out["outcome_range"])
    out["raw_f_ate_pp"] = percent_of_range(out["raw_f_ate_native"], out["outcome_range"])
    if "human_se_native" in out:
        out["human_se_pp"] = percent_of_range(out["human_se_native"], out["outcome_range"])
    else:
        out["human_se_native"] = pd.NA
        out["human_se_pp"] = pd.NA
    alpha = float(manifest["final_alpha"])
    lam = float(manifest["final_lambda"])
    out["calibrated_f_ate_pp"] = alpha + lam * out["raw_f_ate_pp"].astype(float)
    out["calibrated_f_ate_native"] = out["calibrated_f_ate_pp"] / 100 * out["outcome_range"].astype(float)
    out["raw_error_pp"] = out["raw_f_ate_pp"] - out["human_ate_pp"]
    out["calibrated_error_pp"] = out["calibrated_f_ate_pp"] - out["human_ate_pp"]
    out["human_sign"] = out["human_ate_pp"].map(sign_label)
    out["raw_sign"] = out["raw_f_ate_pp"].map(sign_label)
    out["calibrated_sign"] = out["calibrated_f_ate_pp"].map(sign_label)
    return out


def _corr_or_nan(x: pd.Series, y: pd.Series, kind: str) -> tuple[float, str]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return math.nan, "too_few_effects"
    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return math.nan, "constant_input"
    if kind == "pearson":
        return float(pearsonr(valid["x"], valid["y"]).statistic), ""
    return float(spearmanr(valid["x"], valid["y"]).statistic), ""


def prediction_metrics(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    valid = df[["human_ate_pp", pred_col, "human_se_pp"]].copy() if "human_se_pp" in df else df[["human_ate_pp", pred_col]].copy()
    valid = valid.dropna(subset=["human_ate_pp", pred_col])
    err = valid[pred_col].astype(float) - valid["human_ate_pp"].astype(float)
    pearson, pearson_reason = _corr_or_nan(valid["human_ate_pp"], valid[pred_col], "pearson")
    spearman, spearman_reason = _corr_or_nan(valid["human_ate_pp"], valid[pred_col], "spearman")
    metrics = {
        "n_effects": int(len(valid)),
        "pearson": pearson,
        "pearson_na_reason": pearson_reason,
        "spearman": spearman,
        "spearman_na_reason": spearman_reason,
        "rmse": float(np.sqrt(np.mean(err**2))) if len(valid) else math.nan,
        "mae": float(np.mean(np.abs(err))) if len(valid) else math.nan,
        "sign_accuracy": float((valid[pred_col].map(sign_label) == valid["human_ate_pp"].map(sign_label)).mean()) if len(valid) else math.nan,
        "adjusted_rmse": math.nan,
        "adjusted_pearson": math.nan,
        "adjusted_metrics_status": "human_se_unavailable",
    }
    if "human_se_pp" in valid and valid["human_se_pp"].notna().any():
        se = valid["human_se_pp"].astype(float)
        mse_adj = np.mean(err**2 - se.fillna(0.0) ** 2)
        metrics["adjusted_rmse"] = float(np.sqrt(max(mse_adj, 0.0)))
        y_var_adj = float(np.var(valid["human_ate_pp"], ddof=0) - np.mean(se.fillna(0.0) ** 2))
        x_var = float(np.var(valid[pred_col], ddof=0))
        if x_var > 0 and y_var_adj > 0:
            cov = float(np.cov(valid[pred_col], valid["human_ate_pp"], ddof=0)[0, 1])
            metrics["adjusted_pearson"] = float(np.clip(cov / math.sqrt(x_var * y_var_adj), -1, 1))
            metrics["adjusted_metrics_status"] = "approximation_uses_human_se"
        else:
            metrics["adjusted_metrics_status"] = "human_se_available_but_variance_not_positive"
    return metrics


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, pred_col in [("raw_F", "raw_f_ate_pp"), ("calibrated_FC", "calibrated_f_ate_pp")]:
        row = {"prediction": label, **prediction_metrics(predictions, pred_col)}
        rows.append(row)
    return pd.DataFrame(rows)


def study_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study_id, group in predictions.groupby("study_id"):
        for label, pred_col in [("raw_F", "raw_f_ate_pp"), ("calibrated_FC", "calibrated_f_ate_pp")]:
            rows.append({"study_id": study_id, "prediction": label, **prediction_metrics(group, pred_col)})
    return pd.DataFrame(rows)


def equal_study_summary(study_level_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in study_level_metrics.groupby("prediction"):
        mse = group["rmse"].astype(float) ** 2
        rows.append(
            {
                "prediction": label,
                "study_equal_rmse": float(np.sqrt(np.nanmean(mse))),
                "study_equal_mae": float(np.nanmean(group["mae"].astype(float))),
                "study_equal_sign_accuracy": float(np.nanmean(group["sign_accuracy"].astype(float))),
                "median_within_study_pearson": float(np.nanmedian(group["pearson"].astype(float))),
                "median_within_study_spearman": float(np.nanmedian(group["spearman"].astype(float))),
                "n_studies": int(group["study_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def raw_vs_calibrated_comparison(pooled: pd.DataFrame) -> pd.DataFrame:
    raw = pooled.loc[pooled["prediction"] == "raw_F"].iloc[0]
    cal = pooled.loc[pooled["prediction"] == "calibrated_FC"].iloc[0]
    rows = []
    for metric in ["rmse", "mae", "pearson", "spearman", "sign_accuracy", "adjusted_rmse", "adjusted_pearson"]:
        rows.append({"metric": metric, "raw_F": raw[metric], "calibrated_FC": cal[metric], "change": cal[metric] - raw[metric]})
    return pd.DataFrame(rows)


def diagnostic_calibration_regression(predictions: pd.DataFrame) -> dict[str, Any]:
    valid = predictions[["human_ate_pp", "calibrated_f_ate_pp"]].dropna()
    if len(valid) < 2 or valid["calibrated_f_ate_pp"].nunique() < 2:
        return {
            "label": "HOLDOUT_DIAGNOSTIC_NOT_USED_FOR_CALIBRATION",
            "diagnostic_intercept": math.nan,
            "diagnostic_slope": math.nan,
            "n_effects": int(len(valid)),
            "na_reason": "too_few_or_constant_predictions",
        }
    x = valid["calibrated_f_ate_pp"].astype(float).to_numpy()
    y = valid["human_ate_pp"].astype(float).to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "label": "HOLDOUT_DIAGNOSTIC_NOT_USED_FOR_CALIBRATION",
        "diagnostic_intercept": float(intercept),
        "diagnostic_slope": float(slope),
        "n_effects": int(len(valid)),
        "na_reason": "",
    }


def write_holdout_status_opened(method_hash: str) -> dict[str, Any]:
    status = {
        "frozen_method_hash": method_hash,
        "holdout_opened_at": now_utc(),
        "method_changed_after_holdout": False,
        "holdout_still_pristine": True,
        "holdout_consumed": False,
        "notes": "structural holdout performance generated; do not retune without marking holdout_consumed=true",
    }
    HOLDOUT_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOLDOUT_STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def mark_holdout_consumed(reason: str, *, path: Path = HOLDOUT_STATUS_PATH) -> dict[str, Any]:
    if not path.exists():
        raise HoldoutIntegrityError(f"cannot mark holdout consumed; missing {path}")
    status = json.loads(path.read_text(encoding="utf-8"))
    if not status.get("holdout_opened_at"):
        raise HoldoutIntegrityError("cannot mark unopened holdout as consumed")
    status["method_changed_after_holdout"] = True
    status["holdout_still_pristine"] = False
    status["holdout_consumed"] = True
    status["notes"] = reason
    path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def write_holdout_outputs(predictions: pd.DataFrame, outputs_dir: Path = VALIDATION_DIR) -> dict[str, pd.DataFrame | dict[str, Any]]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    pooled = pooled_metrics(predictions)
    by_study = study_metrics(predictions)
    equal = equal_study_summary(by_study)
    comparison = raw_vs_calibrated_comparison(pooled)
    diagnostic = diagnostic_calibration_regression(predictions)
    reference = pd.DataFrame(BENCHMARK_REFERENCE_ROWS)

    predictions.to_csv(outputs_dir / "megastudy_holdout_predictions.csv", index=False)
    pooled.to_csv(outputs_dir / "megastudy_holdout_pooled_metrics.csv", index=False)
    by_study.to_csv(outputs_dir / "megastudy_holdout_study_metrics.csv", index=False)
    equal.to_csv(outputs_dir / "megastudy_holdout_equal_study_metrics.csv", index=False)
    comparison.to_csv(outputs_dir / "raw_vs_calibrated_holdout.csv", index=False)
    pd.DataFrame([diagnostic]).to_csv(outputs_dir / "holdout_diagnostic_calibration_regression.csv", index=False)
    reference.to_csv(outputs_dir / "benchmark_reference_metrics.csv", index=False)
    return {
        "predictions": predictions,
        "pooled": pooled,
        "study": by_study,
        "equal": equal,
        "comparison": comparison,
        "diagnostic": diagnostic,
    }


def evaluate_structural_holdout(predictions_input: Path, manifest_path: Path = FROZEN_METHOD_MANIFEST_PATH, outputs_dir: Path = VALIDATION_DIR) -> dict[str, Any]:
    manifest = load_frozen_method_manifest(manifest_path)
    raw = pd.read_csv(predictions_input)
    predictions = apply_frozen_calibration(raw, manifest)
    written = write_holdout_outputs(predictions, outputs_dir=outputs_dir)
    write_holdout_status_opened(manifest["method_hash"])
    return {"manifest": manifest, **written}


def write_placeholder_g_validation_status(outputs_dir: Path = VALIDATION_DIR / "g_validation") -> pd.DataFrame:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "validation_layer": "G respondent-level validation",
            "status": "infrastructure_present_pending_frozen_g_run",
            "required_metrics": "means; variance ratio; KS; W1; subgroup mean RMSE; demographic coefficient error; demographic R2 discrepancy",
            "implementation": "submission.g_validation.validate_g_against_human",
            "notes": "G validation remains separate from F/C structural holdout performance",
        }
    ]
    df = pd.DataFrame(rows)
    df.to_csv(outputs_dir / "g_validation_status.csv", index=False)
    return df
