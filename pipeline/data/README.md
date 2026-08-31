# pipeline/data/

## `census_cells.csv` -- superseded, kept only as a historical artifact

This was the original 40-cell (2 sex x 4 age_band x 5 race) template for
`build_profile_panel()`/`load_census_cells()`, the reference implementation
in the now-deleted `regularized_probability_elicitation.py` (removed in the
native-response refactor along with the rest of that module -- see
`reports/archive/` for its own historical report). **The actual profile
panel was never built this way in the real pipeline**: `src/population/`
(real 2024 ACS PUMS + CES data, IPF raking, controlled integerization)
replaces it with a far more thoroughly verified construction -- see
`reports/population/existing_census_cells_audit.md` and
`reports/population/population_report.md`. This CSV remains in the repo as
a historical artifact only; nothing currently reads it.

## `ate_archive.csv` -- real, built from the 70-study archive

Columns: `study_id, outcome, model_ate, human_ate, treatment_family,
outcome_family` plus explicit population-transportability metadata. One row
per attempted (study, outcome, hypothesis) contrast from the 70-experiment
US archive in Ashokkumar, Hewitt, Ghezae & Willer (2026, *Nature*) --
supplied as `data/capsule-9843791-data.zip`, extracted (read-only) into
`data/archive_70studies/`. Rows enter the primary `lambda_ate` fit only when
`included_primary_calibration == True`; non-TESS/secondary archive rows are
kept out of the primary fit unless a population-compatible target can be
declared.

**Built by, in order:**
1. `Rscript scripts/extract_archive_rds.R` -- flattens the archive's R-native
   RDS files (`RA_hypotheses.RDS`, `rct_responses.RDS`, `llm_responses.RDS`)
   into plain CSVs under `data/archive_70studies/extracted/`.
2. `python scripts/build_ate_archive.py` -- joins them into this file.

Rerun both after re-extracting the zip (e.g. if the archive is updated).

- `model_ate` / `human_ate` -- **already percentage-of-scale-range (pp)**,
  not the outcome's original units: computed from the archive's own
  `RA_hypotheses.RDS` contrast (which conditions are the treatment side vs.
  the reference/control side, per the original authors' own coding -- not
  guessed from condition-name text) applied identically to real human RCT
  responses (`rct_responses.RDS`) and the archive's own real `gpt-4`
  elicitations (`llm_responses.RDS`), then divided by each row's own
  documented scale width and converted to a percentage
  (`ate.normalize_effects.to_unit_scale()`). This conversion happens
  once, at build time, using the *archive's own* scale bounds -- not by
  matching the archive's outcome name against this benchmark's own 13
  outcomes, which would silently fail for every row (the ~70 TESS studies
  measure entirely different things, on entirely different native scales,
  from this benchmark's 13 outcomes).
- `target_population`, `synthetic_target_population`, `population_type`,
  `population_matching_method`, `weights_used`,
  `included_primary_calibration`, `included_secondary_sensitivity`, and
  `exclusion_reason` make population alignment explicit. TESS rows are the
  primary general-U.S.-adult calibration source and use the documented
  `representative_us_fallback`; non-TESS/secondary rows are retained only as
  sensitivity candidates with
  `exclusion_reason = "target population not transport-compatible"` unless
  future metadata provides a defensible population match.
- `data/ate_archive_audit.csv` and
  `outputs/calibration_study_audit.csv` are machine-readable eligibility
  audit tables with native and percentage-of-range effects, outcome ranges,
  population matching method, weight use, and primary/secondary inclusion
  flags.
- `model_ate` is the archive's own already-computed `gpt-4` elicitation, not
  a fresh rerun of this repo's own native-response inference
  (`inference/client.py`) on the archive's 70 studies -- that would cost
  real additional compute on top of what's already measured for the
  primary pipeline. This is disclosed, not hidden: it is a genuine GPT-4
  elicitation the archive's own authors ran, just not one this repo ran
  itself.
- `treatment_family` is left blank for every row: the archive's ~70 TESS
  studies span far too many distinct research domains (partisan animosity,
  vaccination uptake, workplace attitudes, ...) to honestly assign this
  benchmark's 6 SSB-specific intervention-family tags -- doing so would be
  invented, not verified. Hierarchical shrinkage therefore has no real
  `treatment_family` signal to fit on for this archive and falls back to the
  global estimator (see `submission/build_tier1.py::resolve_lambda_ate`).
- `outcome_family` uses a simple, disclosed heuristic from the outcome's own
  scale width (`max - min == 1` -> `binary_behavior`, else `attitude`; no
  archive outcome resembles this benchmark's dollar-valued `donation`
  family).

Consumed by `ate.calibrate_lambda.load_ate_archive()`, which validates the
population metadata and returns only primary-eligible rows by default, then
by `ate.calibrate_lambda.fit_calibration_model_comparison()`
(`study_id`/`model_ate`/`human_ate` on the already-aligned
percentage-of-range scale). Whole-study LOSO compares:

- `M0`: identity, `y_hat = x`
- `M1`: slope-only, `y_hat = lambda * x`
- `M2`: intercept plus slope, `y_hat = alpha + lambda * x`

Training weights give every study equal total weight, held-out diagnostics
are averaged at the study level, and lambda is not clipped by default.
The selected model and fold diagnostics are written to
`outputs/calibration_production/calibration_model_comparison.csv`,
`outputs/calibration_production/calibration_loso_predictions.csv`, and
`outputs/calibration_selected_model.json` (the last of these lives at the
top level of `outputs/`, not under `calibration_production/`, because it is
also read directly by `scripts/freeze_final_submission_manifest_s2.py` and
`scripts/freeze_calibration_artifact.py`).

## `climate_advocacy_megastudy/` -- external baseline-calibration source

`data/wv7c3-osfstorage-archive.zip` (an OSF archive; see `readme.txt`),
extracted (read-only) into `data/climate_advocacy_megastudy/`: the anonymized
respondent-level data behind a real megastudy of behavioral interventions to
catalyze public, political, and financial climate advocacy. Used as the
closely domain-matched external human baseline data for §1's control-
distribution calibration (`baseline_calibration.py`) -- see
`scripts/build_baseline_reference.py` for how `BaselineReference` objects are
derived from its control-condition respondents, per outcome.
