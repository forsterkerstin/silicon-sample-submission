#!/usr/bin/env python3
"""Build effect-specific F population panels for primary external calibration.

Offline only. For every eligible primary `study_id x effect_id`, the panel
targets the pooled human analytic respondents that contributed observed `y`
to that effect's unweighted treatment/control condition means. Missing
demographic fields are preserved by omitting them from profile signatures;
they are not imputed and respondents are not dropped for demographic
missingness.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

from calibration.study_population import (  # noqa: E402
    ALLOWED_PRETREATMENT_DEMOGRAPHICS,
    DEFAULT_EXTERNAL_N_F,
    effect_panel_from_analytic_sample,
)

ATE_ARCHIVE_PATH = PIPELINE_ROOT / "data" / "ate_archive.csv"
EXTRACTED_DIR = PIPELINE_ROOT / "data" / "archive_70studies" / "extracted"
HYPOTHESES_PATH = EXTRACTED_DIR / "hypotheses.csv"
RESPONSE_DEMOGRAPHICS_PATH = EXTRACTED_DIR / "rct_response_demographics_by_outcome.csv"
COLUMN_MANIFEST_PATH = EXTRACTED_DIR / "rct_study_column_manifest.csv"
PANELS_PATH = PIPELINE_ROOT / "data" / "generated" / "external_primary_f_panels.csv"
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "calibration"
EFFECT_AUDIT_PATH = OUTPUT_DIR / "external_f_effect_panels_audit.csv"
MISSINGNESS_PATH = OUTPUT_DIR / "external_f_missingness_by_effect.csv"
ARM_COMPOSITION_PATH = OUTPUT_DIR / "external_f_arm_composition_audit.csv"
WEIGHT_AUDIT_PATH = OUTPUT_DIR / "human_estimand_weight_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "external_primary_population_alignment_summary.json"
POPULATION_METHOD = "study_effect_analytic_profile_distribution_unweighted_largest_remainder"


def ensure_extracted_inputs() -> None:
    needed = [HYPOTHESES_PATH, RESPONSE_DEMOGRAPHICS_PATH, COLUMN_MANIFEST_PATH]
    if all(path.exists() for path in needed):
        return
    subprocess.run(["Rscript", "scripts/extract_archive_rds.R"], cwd=PIPELINE_ROOT, check=True)


def primary_effects() -> pd.DataFrame:
    archive = pd.read_csv(ATE_ARCHIVE_PATH)
    mask = archive["included_primary_calibration"].astype(str).str.lower().isin({"true", "1", "yes"})
    return archive[mask].copy()


def parse_effect_id(effect_id: str) -> tuple[str, str, str]:
    parts = str(effect_id).split(":")
    if len(parts) < 3:
        raise ValueError(f"cannot parse effect_id {effect_id!r}")
    return parts[0], ":".join(parts[1:-1]), parts[-1]


def condition_sets(hypotheses: pd.DataFrame, *, study_id: str, outcome: str, hypothesis: str) -> tuple[set[str], set[str]]:
    rows = hypotheses[
        (hypotheses["study"].astype(str) == study_id)
        & (hypotheses["outcome.name"].astype(str) == outcome)
        & (hypotheses["hypothesis"].astype(str) == hypothesis)
    ]
    treatment = set(rows.loc[rows["t_hypothesis"].astype(str).isin({"1", "1.0", "True", "TRUE"}), "condition.name"].astype(str))
    control = set(rows.loc[rows["t_hypothesis"].astype(str).isin({"0", "0.0", "False", "FALSE"}), "condition.name"].astype(str))
    if not treatment or not control:
        raise ValueError(f"{study_id}:{outcome}:{hypothesis} has no two-sided hypothesis condition mapping")
    return control, treatment


def study_available_fields(responses: pd.DataFrame, study_id: str) -> list[str]:
    study = responses[responses["study"].astype(str) == study_id]
    return [field for field in ALLOWED_PRETREATMENT_DEMOGRAPHICS if field in study.columns and study[field].notna().any()]


def build_human_weight_audit(studies: list[str], manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_study = manifest.set_index("study").to_dict("index")
    for study_id in studies:
        meta = by_study.get(study_id, {})
        raw_weight_var = meta.get("weight_variables_detected", "")
        weight_var = "" if pd.isna(raw_weight_var) else str(raw_weight_var).strip()
        weighted = bool(weight_var)
        rows.append(
            {
                "study_id": study_id,
                "human_estimator_source": "rct_condition_means.csv from extract_archive_rds.R: mean(y, na.rm=TRUE), n=sum(!is.na(y))",
                "human_weighted": weighted,
                "weight_variable": weight_var,
                "alignment_compatible": not weighted,
                "notes": (
                    "raw respondent data include detected weight variable(s); verify human ATE estimator before production"
                    if weighted
                    else "no weight variable detected; archived human ATE is treated as the unweighted study-sample estimand"
                ),
            }
        )
    return pd.DataFrame(rows)


def _distribution(series: pd.Series) -> dict[str, float]:
    if len(series) == 0:
        return {}
    labels = series.astype(object).where(series.notna(), "__MISSING__").astype(str)
    return (labels.value_counts(normalize=True, dropna=False) * 100).to_dict()


def arm_composition_rows(study_id: str, effect_id: str, control: pd.DataFrame, treatment: pd.DataFrame, fields: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for field in fields:
        control_dist = _distribution(control[field])
        treatment_dist = _distribution(treatment[field])
        for category in sorted(set(control_dist) | set(treatment_dist)):
            rows.append(
                {
                    "study_id": study_id,
                    "effect_id": effect_id,
                    "demographic_variable": field,
                    "category": category,
                    "control_pct": control_dist.get(category, 0.0),
                    "treatment_pct": treatment_dist.get(category, 0.0),
                    "abs_pp_difference": abs(control_dist.get(category, 0.0) - treatment_dist.get(category, 0.0)),
                    "diagnostic_only": True,
                }
            )
    return rows


def main() -> int:
    ensure_extracted_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PANELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    effects = primary_effects().sort_values(["study_id", "effect_id"])
    hypotheses = pd.read_csv(HYPOTHESES_PATH)
    responses = pd.read_csv(RESPONSE_DEMOGRAPHICS_PATH, low_memory=False)
    manifest = pd.read_csv(COLUMN_MANIFEST_PATH)

    raw_by_key = {(str(study), str(outcome)): group.copy() for (study, outcome), group in responses.groupby(["study", "outcome.name"], dropna=False)}
    panels: list[pd.DataFrame] = []
    effect_audits: list[dict[str, object]] = []
    missingness_rows: list[dict[str, object]] = []
    arm_rows: list[dict[str, object]] = []
    unconstructed: list[dict[str, str]] = []

    for _, effect in effects.iterrows():
        effect_id = str(effect["effect_id"])
        study_id, outcome, hypothesis = parse_effect_id(effect_id)
        try:
            fields = study_available_fields(responses, study_id)
            control_conditions, treatment_conditions = condition_sets(hypotheses, study_id=study_id, outcome=outcome, hypothesis=hypothesis)
            raw = raw_by_key[(study_id, outcome)]
            observed = raw[raw["y"].notna()].copy()
            control = observed[observed["condition.name"].astype(str).isin(control_conditions)].copy()
            treatment = observed[observed["condition.name"].astype(str).isin(treatment_conditions)].copy()
            analytic = observed[observed["condition.name"].astype(str).isin(control_conditions | treatment_conditions)].copy()
            panel, audit = effect_panel_from_analytic_sample(analytic, study_id=study_id, effect_id=effect_id, fields=fields, n_f=DEFAULT_EXTERNAL_N_F)
        except Exception as exc:  # noqa: BLE001
            unconstructed.append({"study_id": study_id, "effect_id": effect_id, "reason": str(exc)})
            continue

        audit.update(
            {
                "control_observed_n": int(len(control)),
                "treatment_observed_n": int(len(treatment)),
                "human_control_conditions": "|".join(sorted(control_conditions)),
                "human_treatment_conditions": "|".join(sorted(treatment_conditions)),
                "outcome": outcome,
                "analytic_sample_definition": "pooled respondents in human control/treatment comparison with observed y",
                "human_estimator": "unweighted difference in condition means",
                "population_matching_method": POPULATION_METHOD,
            }
        )
        panels.append(panel)
        effect_audits.append(audit)
        arm_rows.extend(arm_composition_rows(study_id, effect_id, control, treatment, fields))
        for field in fields:
            observed_n = int(analytic[field].notna().sum())
            missing_n = int(analytic[field].isna().sum())
            missingness_rows.append(
                {
                    "study_id": study_id,
                    "effect_id": effect_id,
                    "demographic_variable": field,
                    "observed_n": observed_n,
                    "missing_n": missing_n,
                    "missing_pct": float(100 * missing_n / len(analytic)) if len(analytic) else float("nan"),
                }
            )

    panel_df = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    panel_df.to_csv(PANELS_PATH, index=False)
    pd.DataFrame(effect_audits).to_csv(EFFECT_AUDIT_PATH, index=False)
    pd.DataFrame(missingness_rows).to_csv(MISSINGNESS_PATH, index=False)
    pd.DataFrame(arm_rows).to_csv(ARM_COMPOSITION_PATH, index=False)
    build_human_weight_audit(sorted(effects["study_id"].dropna().astype(str).unique()), manifest).to_csv(WEIGHT_AUDIT_PATH, index=False)

    summary = {
        "n_primary_effects": int(len(effects)),
        "effects_built": int(len(effect_audits)),
        "total_panel_rows": int(len(panel_df)),
        "expected_panel_rows": int(len(effects) * DEFAULT_EXTERNAL_N_F),
        "effects_with_any_demographic_missingness": int(sum(a["profiles_with_any_missing_demographic"] > 0 for a in effect_audits)),
        "maximum_allocation_discrepancy_pp": float(max((a["max_abs_signature_share_error_pp"] for a in effect_audits), default=float("nan"))),
        "unconstructed_effects": unconstructed,
        "n_f_per_effect": DEFAULT_EXTERNAL_N_F,
        "population_matching_method": POPULATION_METHOD,
        "panels_path": str(PANELS_PATH.relative_to(PIPELINE_ROOT)),
        "effect_audit_path": str(EFFECT_AUDIT_PATH.relative_to(PIPELINE_ROOT)),
        "missingness_path": str(MISSINGNESS_PATH.relative_to(PIPELINE_ROOT)),
        "arm_composition_path": str(ARM_COMPOSITION_PATH.relative_to(PIPELINE_ROOT)),
        "human_estimand_weight_audit_path": str(WEIGHT_AUDIT_PATH.relative_to(PIPELINE_ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 1 if unconstructed else 0


if __name__ == "__main__":
    raise SystemExit(main())
