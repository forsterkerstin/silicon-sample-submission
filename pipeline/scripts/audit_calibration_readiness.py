#!/usr/bin/env python3
"""Offline calibration readiness audits on cached/development data only."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "calibration"
ATE_ARCHIVE_PATH = PIPELINE_ROOT / "data" / "ate_archive.csv"
HYPOTHESES_PATH = PIPELINE_ROOT / "data" / "archive_70studies" / "extracted" / "hypotheses.csv"
RCT_RESPONSES_RDS = PIPELINE_ROOT / "data" / "archive_70studies" / "rct_responses.RDS"
LLM_RESPONSES_RDS = PIPELINE_ROOT / "data" / "archive_70studies" / "llm_responses.RDS"


def _bool_mode(series: pd.Series) -> bool:
    values = series.dropna().astype(bool)
    return bool(values.mode().iloc[0]) if not values.empty else False


def _str_mode(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    return values.mode().iloc[0] if not values.empty else ""


def rct_data_column_manifest() -> pd.DataFrame:
    r_code = f"""
    x <- readRDS({json.dumps(str(RCT_RESPONSES_RDS))})
    rows <- list()
    for (i in seq_len(nrow(x))) {{
      d <- x$data[[i]]
      rows[[i]] <- data.frame(
        study=as.character(x$study[[i]]),
        outcome=as.character(x$outcome.name[[i]]),
        n_rows=if (is.data.frame(d)) nrow(d) else NA_integer_,
        cols=if (is.data.frame(d)) paste(names(d), collapse='|') else '',
        stringsAsFactors=FALSE
      )
    }}
    write.csv(do.call(rbind, rows), row.names=FALSE)
    """
    result = subprocess.run(["Rscript", "-e", r_code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return pd.read_csv(StringIO(result.stdout))


def llm_scale_flip_manifest() -> pd.DataFrame:
    r_code = f"""
    x <- readRDS({json.dumps(str(LLM_RESPONSES_RDS))})
    y <- unique(x[, c('study', 'outcome.name', 'scale_flip')])
    write.csv(y, row.names=FALSE)
    """
    result = subprocess.run(["Rscript", "-e", r_code], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return pd.read_csv(StringIO(result.stdout))


def write_population_alignment_availability(archive: pd.DataFrame) -> pd.DataFrame:
    manifest = rct_data_column_manifest()
    primary = archive[archive["included_primary_calibration"] == True].copy()
    demographic_terms = {"GENDER", "race_4", "pid_3", "age_5", "EDUC4", "ideo_3"}
    rows = []
    for study_id, group in primary.groupby("study_id", sort=True):
        study_cols = "|".join(manifest.loc[manifest["study"].astype(str) == str(study_id), "cols"].dropna().astype(str))
        cols = set(study_cols.split("|")) if study_cols else set()
        respondent_demographics_available = bool(demographic_terms & cols)
        survey_weights_available = any("weight" in col.lower() or col.lower() in {"wt", "wgt"} for col in cols)
        published_population_margins_available = False
        current = _str_mode(group["population_matching_method"])
        if respondent_demographics_available and survey_weights_available:
            preferred = "study_respondent_weighted"
        elif respondent_demographics_available:
            preferred = "study_effect_analytic_profile_distribution_unweighted_largest_remainder"
        elif published_population_margins_available:
            preferred = "raked_to_study_margins"
        else:
            preferred = "representative_us_fallback"
        if current == preferred:
            action = "none"
        elif respondent_demographics_available and not survey_weights_available:
            action = "regenerate_primary_archive_synthetic_predictions_with_effect_specific_analytic_panels_before_final_calibration"
        else:
            action = "review_population_alignment_metadata"
        rows.append(
            {
                "study_id": study_id,
                "respondent_demographics_available": respondent_demographics_available,
                "survey_weights_available": survey_weights_available,
                "published_population_margins_available": published_population_margins_available,
                "currently_used_method": current,
                "preferred_method_under_protocol": preferred,
                "action_needed": action,
                "notes": (
                    f"rct_responses.RDS columns: {study_cols}"
                    if study_cols
                    else "study not found in rct_responses.RDS column manifest"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "population_alignment_availability.csv", index=False)
    return out


def hypothesis_labels() -> dict[tuple[str, str], dict[str, str]]:
    if not HYPOTHESES_PATH.exists():
        return {}
    hyp = pd.read_csv(HYPOTHESES_PATH)
    labels = {}
    for (study, hypothesis), group in hyp.groupby(["study", "hypothesis"], dropna=False):
        treatment = " | ".join(group.loc[group["t_hypothesis"] == 1, "condition.name"].astype(str))
        control = " | ".join(group.loc[group["t_hypothesis"] == 0, "condition.name"].astype(str))
        labels[(str(study), str(hypothesis))] = {"treatment_label": treatment, "control_label": control}
    return labels


def effect_hypothesis(effect_id: str) -> str:
    parts = str(effect_id).split(":")
    return parts[-1] if parts else ""


def write_sign_alignment_audit(archive: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    rows = archive[(archive["included_primary_calibration"] == True) & archive["model_ate"].notna() & archive["human_ate"].notna()].copy()
    labels = hypothesis_labels()
    scale_flip = llm_scale_flip_manifest()
    flip_lookup = {
        (str(row["study"]), str(row["outcome.name"])): bool(row["scale_flip"])
        for _, row in scale_flip.iterrows()
        if pd.notna(row.get("scale_flip"))
    }
    audit_rows = []
    for _, row in rows.iterrows():
        hyp = effect_hypothesis(str(row["effect_id"]))
        label = labels.get((str(row["study_id"]), hyp), {"treatment_label": "", "control_label": ""})
        human_pp_calc = 100 * float(row["human_ate_native"]) / float(row["outcome_range"])
        synthetic_pp_calc = 100 * float(row["synthetic_ate_native"]) / float(row["outcome_range"])
        problems = []
        if not math.isclose(human_pp_calc, float(row["human_ate"]), rel_tol=1e-7, abs_tol=1e-7):
            problems.append("human_percent_conversion_mismatch")
        if not math.isclose(synthetic_pp_calc, float(row["model_ate"]), rel_tol=1e-7, abs_tol=1e-7):
            problems.append("synthetic_percent_conversion_mismatch")
        if not label["treatment_label"] or not label["control_label"]:
            problems.append("condition_label_missing")
        if label["treatment_label"] == label["control_label"] and label["treatment_label"]:
            problems.append("duplicated_control_treatment_label")
        if np.sign(float(row["human_ate"])) != np.sign(float(row["model_ate"])) and float(row["human_ate"]) != 0 and float(row["model_ate"]) != 0:
            problems.append("sign_disagreement_review")
        if not problems:
            status = "ok"
        elif set(problems) == {"sign_disagreement_review"}:
            status = "sign_disagreement_not_implementation_error_by_itself"
        else:
            status = "review"
        audit_rows.append(
            {
                "study_id": row["study_id"],
                "effect_id": row["effect_id"],
                "human_ate_native": float(row["human_ate_native"]),
                "synthetic_ate_native": float(row["synthetic_ate_native"]),
                "outcome_range": float(row["outcome_range"]),
                "human_ate_pp": float(row["human_ate"]),
                "synthetic_ate_pp": float(row["model_ate"]),
                "treatment_label": label["treatment_label"],
                "control_label": label["control_label"],
                "reverse_coded": flip_lookup.get((str(row["study_id"]), str(row["outcome"]).replace(f" [{hyp}]", "")), pd.NA),
                "alignment_status": status,
                "notes": ";".join(problems) if problems else "percent conversion and condition labels pass cached-data checks",
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUTPUT_DIR / "sign_alignment_audit.csv", index=False)

    x = audit["synthetic_ate_pp"].to_numpy(dtype=float)
    y = audit["human_ate_pp"].to_numpy(dtype=float)
    sign_agreement = float(np.mean(np.sign(x) == np.sign(y)))
    slope_origin = float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) else math.nan
    slope_intercept = np.polyfit(x, y, 1) if len(audit) >= 2 else [math.nan, math.nan]
    pearson = float(pearsonr(x, y).statistic) if len(audit) >= 3 and len(set(x)) > 1 and len(set(y)) > 1 else math.nan
    spearman = float(spearmanr(x, y).correlation) if len(audit) >= 3 and len(set(x)) > 1 and len(set(y)) > 1 else math.nan
    summary = {
        "n_effects": int(len(audit)),
        "pearson": pearson,
        "spearman": spearman,
        "sign_agreement": sign_agreement,
        "slope_through_origin": slope_origin,
        "intercept_plus_slope_intercept": float(slope_intercept[1]),
        "intercept_plus_slope_slope": float(slope_intercept[0]),
        "n_review_rows": int((audit["alignment_status"] == "review").sum()),
        "n_sign_disagreement_rows": int(audit["notes"].str.contains("sign_disagreement_review", regex=False).sum()),
    }
    (OUTPUT_DIR / "sign_alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_scatter_plot(audit)
    return audit, summary


def write_scatter_plot(audit: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    x = audit["synthetic_ate_pp"].astype(float)
    y = audit["human_ate_pp"].astype(float)
    plt.figure(figsize=(6, 5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.scatter(x, y, s=18, alpha=0.75)
    plt.xlabel("Cached synthetic effect (percent of range)")
    plt.ylabel("Human effect (percent of range)")
    plt.title("Current cached primary calibration effects")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "current_raw_f_vs_human.png", dpi=160)
    plt.close()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive = pd.read_csv(ATE_ARCHIVE_PATH)
    availability = write_population_alignment_availability(archive)
    _, sign_summary = write_sign_alignment_audit(archive)
    payload = {
        "population_alignment_rows": int(len(availability)),
        "studies_with_respondent_demographics": int(availability["respondent_demographics_available"].sum()),
        "studies_with_survey_weights": int(availability["survey_weights_available"].sum()),
        "studies_needing_population_recompute": int((availability["action_needed"] != "none").sum()),
        "sign_summary": sign_summary,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
