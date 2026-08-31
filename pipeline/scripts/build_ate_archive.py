#!/usr/bin/env python3
"""scripts/build_ate_archive.py

Builds the real pipeline/data/ate_archive.csv from the 70-experiment archive
(Ashokkumar, Hewitt, Ghezae & Willer 2026, Nature) -- the actual archive [5]
this approach's shrinkage step (§5) calls for, supplied by the user as
data/capsule-9843791-data.zip and extracted (read-only) into
data/archive_70studies/. Run scripts/extract_archive_rds.R first to produce
the flat CSVs this script reads (data/archive_70studies/extracted/*.csv).

For each (study, outcome, hypothesis) triple in the archive's own RA-coded
hypotheses.csv (the authors' own treatment-vs-control contrast, not a guess
from condition-name text), computes:

  human_ate = (n-weighted mean of real RCT respondents' y on the
               treatment-side conditions) - (... on the reference/control-side conditions)
  model_ate = the same contrast, computed identically from gpt-4's own
               elicited response ("expectation") in the archive's
               llm_responses.RDS -- i.e. the archive's own pre-computed GPT-4
               predictions, NOT a fresh rerun of this repo's own elicitation
               pipeline on the archive's 70 studies (that would cost the same
               order of many-hours-to-days of local-model compute already
               measured for the primary pipeline -- see
               reports/elicitation_ablation_report.md -- so this is a
               disclosed, real substitute, not a fabrication: it is the
               archive's own genuine GPT-4 elicitation, just not one we ran
               ourselves for every one of the 70 studies).

Both human_ate and model_ate are converted to PERCENTAGE-OF-RANGE (0-100
scale) right here, using each row's own real outcome.min/outcome.max
(verified to agree between the RCT and LLM sources for every row used) --
via ate.normalize_effects.to_unit_scale(). This is done at build time,
not downstream, because the archive's ~70 TESS studies use outcome names and
native scales (Likert 1-5, 1-7, 1-2 binary, ...) that don't correspond to
this benchmark's own 13 outcomes at all -- looking up a scale by matching the
archive's outcome name against this benchmark's OUTCOME_SCALE_BOUNDS would
silently fail for every single row. Converting once here, with the scale
bounds actually observed alongside each ATE, means every consumer of
ate_archive.csv can treat model_ate/human_ate as already comparable
percentage-point quantities, regardless of source.

`treatment_family` is left blank: the archive's 70 TESS studies span far too
many distinct research domains (partisan animosity, vaccination, workplace
attitudes, ...) to honestly assign this benchmark's 6 SSB-specific
categories (Collaboration and peer-review / Scientific methods and results /
... climate-trust intervention families) -- doing so would be invented, not
verified. `outcome_family` uses a simple, disclosed heuristic from the
outcome's own scale width (max - min == 1 -> "binary_behavior", else
"attitude" -- no archive outcome resembles this benchmark's dollar-valued
"donation" family).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import ate.normalize_effects as dc  # noqa: E402
from calibration.external_prediction_provenance import (  # noqa: E402
    LEGACY_NO_PANEL_PROVENANCE,
    PREDICTION_F_INFERENCE_CONFIG_HASH_COL,
    PREDICTION_F_MODEL_ID_COL,
    PREDICTION_F_PROMPT_PROTOCOL_ID_COL,
    PREDICTION_F_R_F_COL,
    PREDICTION_PANEL_SHA256_COL,
    PREDICTION_PANEL_VERSION_COL,
)

EXTRACTED_DIR = PIPELINE_ROOT / "data" / "archive_70studies" / "extracted"
STUDY_FEATURES_PATH = PIPELINE_ROOT / "data" / "archive_70studies" / "RA_study_features.csv"
RCT_STUDY_DEMOGRAPHICS_PATH = EXTRACTED_DIR / "rct_study_demographics.csv"
OUT_PATH = PIPELINE_ROOT / "data" / "ate_archive.csv"
AUDIT_PATH = PIPELINE_ROOT / "data" / "ate_archive_audit.csv"
OUTPUT_AUDIT_PATH = PIPELINE_ROOT / "outputs" / "calibration_study_audit.csv"
STUDY_RESPONDENT_ALIGNMENT_METHOD = "study_effect_analytic_profile_distribution_unweighted_largest_remainder"
STALE_SYNTHETIC_STATUS = "DEVELOPMENT_STALE_REQUIRES_REGENERATION_WITH_STUDY_SPECIFIC_F_PANEL"
ALLOWED_PROFILE_FIELDS = ("GENDER", "race_4", "pid_3", "age_5", "EDUC4", "ideo_3")

ARCHIVE_FIELDS = [
    "study_id",
    "effect_id",
    "outcome",
    "model_ate",
    "human_ate",
    "treatment_family",
    "outcome_family",
    "target_population",
    "synthetic_target_population",
    "population_type",
    "is_general_us_adult",
    "is_specialized_population",
    "study_weights_available",
    "profile_variables_available",
    "population_matching_method",
    "profile_fields_available",
    "weights_used",
    "num_profiles",
    PREDICTION_F_MODEL_ID_COL,
    PREDICTION_F_PROMPT_PROTOCOL_ID_COL,
    PREDICTION_F_INFERENCE_CONFIG_HASH_COL,
    PREDICTION_F_R_F_COL,
    PREDICTION_PANEL_VERSION_COL,
    PREDICTION_PANEL_SHA256_COL,
    "synthetic_prediction_status",
    "requires_synthetic_regeneration",
    "treatment_type",
    "randomized_between_subjects",
    "materials_available",
    "outcome_type",
    "outcome_min",
    "outcome_max",
    "outcome_range",
    "finite_range",
    "main_effect_compatible",
    "human_ate_native",
    "synthetic_ate_native",
    "included_primary_calibration",
    "included_secondary_sensitivity",
    "exclusion_reason",
]

AUDIT_FIELDS = [
    "study_id",
    "effect_id",
    "target_population",
    "population_type",
    "population_matching_method",
    "profile_fields_available",
    "weights_used",
    "num_profiles",
    PREDICTION_F_MODEL_ID_COL,
    PREDICTION_F_PROMPT_PROTOCOL_ID_COL,
    PREDICTION_F_INFERENCE_CONFIG_HASH_COL,
    PREDICTION_F_R_F_COL,
    PREDICTION_PANEL_VERSION_COL,
    PREDICTION_PANEL_SHA256_COL,
    "synthetic_prediction_status",
    "requires_synthetic_regeneration",
    "outcome_name",
    "outcome_type",
    "outcome_range",
    "human_ate_native",
    "synthetic_ate_native",
    "human_effect_pp",
    "synthetic_effect_pp",
    "included_primary_calibration",
    "included_secondary_sensitivity",
    "exclusion_reason",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def weighted_mean(rows: list[dict], mean_col: str, condition_names: set[str]) -> tuple[float, int] | None:
    matched = [r for r in rows if r["condition.name"] in condition_names]
    if not matched:
        return None
    total_n = sum(int(r["n"]) for r in matched)
    if total_n == 0:
        return None
    weighted = sum(float(r[mean_col]) * int(r["n"]) for r in matched) / total_n
    return weighted, total_n


def outcome_family_for(scale_min: float, scale_max: float) -> str:
    return "binary_behavior" if (scale_max - scale_min) == 1 else "attitude"


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().upper() == "TRUE"


def load_profile_fields_by_study(path: Path = RCT_STUDY_DEMOGRAPHICS_PATH) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    rows = load_csv(path)
    fields: dict[str, set[str]] = {}
    for row in rows:
        study = row["study"]
        for field in ALLOWED_PROFILE_FIELDS:
            if str(row.get(field, "")).strip() not in {"", "NA", "nan"}:
                fields.setdefault(study, set()).add(field)
    return {study: [field for field in ALLOWED_PROFILE_FIELDS if field in present] for study, present in fields.items()}


def population_metadata(study: str, study_features: dict[str, dict[str, str]], profile_fields_by_study: dict[str, list[str]] | None = None) -> dict[str, object]:
    """Pre-declared transportability metadata. TESS rows are treated as the
    primary calibration source. For external primary calibration, synthetic F
    panels target the unweighted empirical respondent demographic distribution
    for each study_id x effect_id human analytic sample, not the benchmark
    target's representative-U.S. F panel."""
    feature = study_features.get(study, {})
    if _is_true(feature.get("study_is_tess")):
        fields = (profile_fields_by_study or {}).get(study, [])
        if not fields:
            return {
                "target_population": "general U.S. adults (TESS study sample)",
                "synthetic_target_population": "not transport-compatible until respondent demographics are extracted",
                "population_type": "general_us_adult",
                "is_general_us_adult": True,
                "is_specialized_population": False,
                "study_weights_available": False,
                "profile_variables_available": False,
                "population_matching_method": "not_matched",
                "profile_fields_available": "",
                "weights_used": False,
                PREDICTION_F_MODEL_ID_COL: "",
                PREDICTION_F_PROMPT_PROTOCOL_ID_COL: "",
                PREDICTION_F_INFERENCE_CONFIG_HASH_COL: "",
                PREDICTION_F_R_F_COL: "",
                PREDICTION_PANEL_VERSION_COL: "",
                PREDICTION_PANEL_SHA256_COL: "",
                "synthetic_prediction_status": "UNALIGNED_MISSING_STUDY_RESPONDENT_DEMOGRAPHICS",
                "requires_synthetic_regeneration": True,
                "included_primary_calibration": False,
                "included_secondary_sensitivity": False,
                "exclusion_reason": "study respondent demographics unavailable",
            }
        population = f"{study} effect-specific empirical human analytic respondent pretreatment demographic distribution (unweighted)"
        return {
            "target_population": population,
            "synthetic_target_population": population,
            "population_type": "general_us_adult",
            "is_general_us_adult": True,
            "is_specialized_population": False,
            "study_weights_available": False,
            "profile_variables_available": True,
            "population_matching_method": STUDY_RESPONDENT_ALIGNMENT_METHOD,
            "profile_fields_available": "|".join(fields),
            "weights_used": False,
            "num_profiles": 500,
            PREDICTION_F_MODEL_ID_COL: "LEGACY_CACHED_NO_F_MODEL_PROVENANCE",
            PREDICTION_F_PROMPT_PROTOCOL_ID_COL: "LEGACY_CACHED_NO_F_PROMPT_PROTOCOL_PROVENANCE",
            PREDICTION_F_INFERENCE_CONFIG_HASH_COL: "LEGACY_CACHED_NO_F_INFERENCE_CONFIG_HASH",
            PREDICTION_F_R_F_COL: "",
            PREDICTION_PANEL_VERSION_COL: LEGACY_NO_PANEL_PROVENANCE,
            PREDICTION_PANEL_SHA256_COL: "",
            "synthetic_prediction_status": STALE_SYNTHETIC_STATUS,
            "requires_synthetic_regeneration": True,
            "included_primary_calibration": True,
            "included_secondary_sensitivity": False,
            "exclusion_reason": "",
        }
    return {
        "target_population": "secondary archive population not established as general U.S. adults",
        "synthetic_target_population": "representative U.S. fallback not transport-compatible with secondary archive population",
        "population_type": "secondary_population_sensitivity",
        "is_general_us_adult": False,
        "is_specialized_population": True,
        "study_weights_available": False,
        "profile_variables_available": False,
        "population_matching_method": "not_matched",
        "profile_fields_available": "",
        "weights_used": False,
        PREDICTION_F_MODEL_ID_COL: "",
        PREDICTION_F_PROMPT_PROTOCOL_ID_COL: "",
        PREDICTION_F_INFERENCE_CONFIG_HASH_COL: "",
        PREDICTION_F_R_F_COL: "",
        PREDICTION_PANEL_VERSION_COL: "",
        PREDICTION_PANEL_SHA256_COL: "",
        "synthetic_prediction_status": "",
        "requires_synthetic_regeneration": False,
        "included_primary_calibration": False,
        "included_secondary_sensitivity": True,
        "exclusion_reason": "target population not transport-compatible",
    }


def archive_row(
    *,
    study: str,
    effect_id: str,
    outcome: str,
    model_ate_pp: float | None = None,
    human_ate_pp: float | None = None,
    treatment_family: str = "",
    outcome_type: str = "",
    outcome_min: float | None = None,
    outcome_max: float | None = None,
    human_ate_native: float | None = None,
    synthetic_ate_native: float | None = None,
    num_profiles: int | None = None,
    eligibility: dict[str, object] | None = None,
    exclusion_reason: str = "",
) -> dict[str, object]:
    eligibility = eligibility or {}
    outcome_range = None if outcome_min is None or outcome_max is None else outcome_max - outcome_min
    finite_range = outcome_range is not None and outcome_range > 0
    row = {
        "study_id": study,
        "effect_id": effect_id,
        "outcome": outcome,
        "model_ate": "" if model_ate_pp is None else model_ate_pp,
        "human_ate": "" if human_ate_pp is None else human_ate_pp,
        "treatment_family": treatment_family,
        "outcome_family": outcome_type,
        "treatment_type": "survey_experiment",
        "randomized_between_subjects": True,
        "materials_available": True,
        "outcome_type": outcome_type,
        "outcome_min": "" if outcome_min is None else outcome_min,
        "outcome_max": "" if outcome_max is None else outcome_max,
        "outcome_range": "" if outcome_range is None else outcome_range,
        "finite_range": finite_range,
        "main_effect_compatible": True,
        "human_ate_native": "" if human_ate_native is None else human_ate_native,
        "synthetic_ate_native": "" if synthetic_ate_native is None else synthetic_ate_native,
        "num_profiles": "" if num_profiles is None else num_profiles,
        **eligibility,
    }
    if exclusion_reason:
        row["included_primary_calibration"] = False
        row["exclusion_reason"] = exclusion_reason
    return row


def audit_row_from_archive(row: dict[str, object]) -> dict[str, object]:
    return {
        "study_id": row["study_id"],
        "effect_id": row["effect_id"],
        "target_population": row.get("target_population", ""),
        "population_type": row.get("population_type", ""),
        "population_matching_method": row.get("population_matching_method", ""),
        "profile_fields_available": row.get("profile_fields_available", ""),
        "weights_used": row.get("weights_used", ""),
        "num_profiles": row.get("num_profiles", ""),
        PREDICTION_F_MODEL_ID_COL: row.get(PREDICTION_F_MODEL_ID_COL, ""),
        PREDICTION_F_PROMPT_PROTOCOL_ID_COL: row.get(PREDICTION_F_PROMPT_PROTOCOL_ID_COL, ""),
        PREDICTION_F_INFERENCE_CONFIG_HASH_COL: row.get(PREDICTION_F_INFERENCE_CONFIG_HASH_COL, ""),
        PREDICTION_F_R_F_COL: row.get(PREDICTION_F_R_F_COL, ""),
        PREDICTION_PANEL_VERSION_COL: row.get(PREDICTION_PANEL_VERSION_COL, ""),
        PREDICTION_PANEL_SHA256_COL: row.get(PREDICTION_PANEL_SHA256_COL, ""),
        "outcome_name": row["outcome"],
        "outcome_type": row.get("outcome_type", ""),
        "outcome_range": row.get("outcome_range", ""),
        "human_ate_native": row.get("human_ate_native", ""),
        "synthetic_ate_native": row.get("synthetic_ate_native", ""),
        "human_effect_pp": row.get("human_ate", ""),
        "synthetic_effect_pp": row.get("model_ate", ""),
        "included_primary_calibration": row.get("included_primary_calibration", ""),
        "included_secondary_sensitivity": row.get("included_secondary_sensitivity", ""),
        "exclusion_reason": row.get("exclusion_reason", ""),
    }


def main() -> int:
    for name in ("hypotheses.csv", "rct_condition_means.csv", "llm_condition_means.csv", "rct_study_demographics.csv"):
        if not (EXTRACTED_DIR / name).exists():
            print(f"missing {EXTRACTED_DIR / name} -- run: Rscript scripts/extract_archive_rds.R", file=sys.stderr)
            return 1
    if not STUDY_FEATURES_PATH.exists():
        print(f"missing {STUDY_FEATURES_PATH}", file=sys.stderr)
        return 1

    hyp_rows = load_csv(EXTRACTED_DIR / "hypotheses.csv")
    rct_rows = load_csv(EXTRACTED_DIR / "rct_condition_means.csv")
    llm_rows = load_csv(EXTRACTED_DIR / "llm_condition_means.csv")
    study_features = {r["study"]: r for r in load_csv(STUDY_FEATURES_PATH)}
    profile_fields_by_study = load_profile_fields_by_study()

    rct_by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rct_rows:
        rct_by_key.setdefault((r["study"], r["outcome.name"]), []).append(r)
    llm_by_key: dict[tuple[str, str], list[dict]] = {}
    for r in llm_rows:
        llm_by_key.setdefault((r["study"], r["outcome.name"]), []).append(r)

    hypotheses: dict[tuple[str, str, str], list[dict]] = {}
    for r in hyp_rows:
        hypotheses.setdefault((r["study"], r["outcome.name"], r["hypothesis"]), []).append(r)

    # disambiguate outcome names when a study has more than one hypothesis for the same outcome
    outcome_hyp_counts: dict[tuple[str, str], int] = {}
    for study, outcome, _hyp in hypotheses:
        outcome_hyp_counts[(study, outcome)] = outcome_hyp_counts.get((study, outcome), 0) + 1

    out_rows = []
    audit_rows = []
    skipped = {"no_rct_match": 0, "no_llm_match": 0, "scale_mismatch": 0, "single_sided": 0}

    for (study, outcome, hyp), condition_rows in hypotheses.items():
        effect_id = f"{study}:{outcome}:{hyp}"
        eligibility = population_metadata(study, study_features, profile_fields_by_study)
        treat_conditions = {r["condition.name"] for r in condition_rows if r["t_hypothesis"] == "1"}
        control_conditions = {r["condition.name"] for r in condition_rows if r["t_hypothesis"] == "0"}
        if not treat_conditions or not control_conditions:
            skipped["single_sided"] += 1
            row = archive_row(study=study, effect_id=effect_id, outcome=outcome, eligibility=eligibility, exclusion_reason="single_sided_hypothesis")
            out_rows.append(row)
            audit_rows.append(audit_row_from_archive(row))
            continue

        rct_candidates = rct_by_key.get((study, outcome), [])
        llm_candidates = llm_by_key.get((study, outcome), [])
        if not rct_candidates:
            skipped["no_rct_match"] += 1
            row = archive_row(study=study, effect_id=effect_id, outcome=outcome, eligibility=eligibility, exclusion_reason="no_rct_match")
            out_rows.append(row)
            audit_rows.append(audit_row_from_archive(row))
            continue
        if not llm_candidates:
            skipped["no_llm_match"] += 1
            row = archive_row(study=study, effect_id=effect_id, outcome=outcome, eligibility=eligibility, exclusion_reason="no_llm_match")
            out_rows.append(row)
            audit_rows.append(audit_row_from_archive(row))
            continue

        rct_treat = weighted_mean(rct_candidates, "mean_y", treat_conditions)
        rct_control = weighted_mean(rct_candidates, "mean_y", control_conditions)
        llm_treat = weighted_mean(llm_candidates, "mean_expectation", treat_conditions)
        llm_control = weighted_mean(llm_candidates, "mean_expectation", control_conditions)
        if not (rct_treat and rct_control and llm_treat and llm_control):
            skipped["no_rct_match" if not (rct_treat and rct_control) else "no_llm_match"] += 1
            reason = "no_rct_condition_contrast" if not (rct_treat and rct_control) else "no_llm_condition_contrast"
            row = archive_row(study=study, effect_id=effect_id, outcome=outcome, eligibility=eligibility, exclusion_reason=reason)
            out_rows.append(row)
            audit_rows.append(audit_row_from_archive(row))
            continue

        rct_scale = (float(rct_candidates[0]["outcome.min"]), float(rct_candidates[0]["outcome.max"]))
        llm_scale = (float(llm_candidates[0]["outcome_scale_min"]), float(llm_candidates[0]["outcome_scale_max"]))
        if rct_scale != llm_scale:
            skipped["scale_mismatch"] += 1
            row = archive_row(study=study, effect_id=effect_id, outcome=outcome, eligibility=eligibility, exclusion_reason="scale_mismatch")
            out_rows.append(row)
            audit_rows.append(audit_row_from_archive(row))
            continue

        scale_low, scale_high = rct_scale
        if scale_high <= scale_low:
            skipped["scale_mismatch"] += 1
            row = archive_row(study=study, effect_id=effect_id, outcome=outcome, outcome_min=scale_low, outcome_max=scale_high, eligibility=eligibility, exclusion_reason="invalid_or_unbounded_range")
            out_rows.append(row)
            audit_rows.append(audit_row_from_archive(row))
            continue
        human_native = rct_treat[0] - rct_control[0]
        model_native = llm_treat[0] - llm_control[0]
        human_ate_pp = 100 * (dc.to_unit_scale(rct_treat[0], scale_low, scale_high) - dc.to_unit_scale(rct_control[0], scale_low, scale_high))
        model_ate_pp = 100 * (dc.to_unit_scale(llm_treat[0], scale_low, scale_high) - dc.to_unit_scale(llm_control[0], scale_low, scale_high))
        outcome_label = outcome if outcome_hyp_counts[(study, outcome)] == 1 else f"{outcome} [{hyp}]"
        outcome_type = outcome_family_for(scale_low, scale_high)

        row = archive_row(
                study=study,
                effect_id=effect_id,
                outcome=outcome_label,
                outcome_type=outcome_type,
                outcome_min=scale_low,
                outcome_max=scale_high,
                human_ate_native=human_native,
                synthetic_ate_native=model_native,
                model_ate_pp=model_ate_pp,
                human_ate_pp=human_ate_pp,
                num_profiles=llm_treat[1] + llm_control[1],
                eligibility=eligibility,
        )
        out_rows.append(row)
        audit_rows.append(audit_row_from_archive(row))

    usable_rows = [r for r in out_rows if r.get("model_ate") != "" and r.get("human_ate") != ""]
    if not usable_rows:
        print("no usable archive rows were produced -- check the extracted CSVs", file=sys.stderr)
        return 1

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ARCHIVE_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    with open(AUDIT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit_rows)
    OUTPUT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_AUDIT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit_rows)

    n_studies = len({r["study_id"] for r in usable_rows})
    n_primary = len([r for r in usable_rows if r.get("included_primary_calibration") is True])
    n_secondary = len([r for r in usable_rows if r.get("included_secondary_sensitivity") is True])
    print(f"wrote {OUT_PATH}: {len(usable_rows)} usable rows across {n_studies} studies ({n_primary} primary, {n_secondary} secondary)")
    print(f"wrote {AUDIT_PATH}: {len(audit_rows)} eligibility audit rows")
    print(f"wrote {OUTPUT_AUDIT_PATH}: {len(audit_rows)} calibration study audit rows")
    print(f"skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
