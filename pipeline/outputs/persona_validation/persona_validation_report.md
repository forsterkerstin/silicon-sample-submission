# Persona Validation Report

## Status

- Hard failures: 0
- Warnings: 1
- G donors: 1000
- G design rows: 17000
- F target profiles: 500

## Authoritative population design

- G: 1000 unique latent U.S.-adult donors, reused across control plus 16 interventions for 17000 rows.
- F: 500 unique forecasting profiles drawn from the same U.S.-adult target population as a deterministic subset of G matching the F cross-quota files.
- Quotas used: gender x age and gender x race; no full gender x age x race quota cube was found or used.
- State comes from ACS PUMS state FIPS recoded to USPS abbreviation and full state name.

## Warnings

- data/quotas_18000.csv not found; using pipeline/config 1000 and 500 cross-quota CSVs as the operative benchmark quota sources.

## Failures

- None

## Generated artifacts

- `pipeline/data/generated/g_personas_master.csv`
- `pipeline/data/generated/f_target_panel.csv`
- `pipeline/data/generated/tier1_design_skeleton.csv`
- `pipeline/data/generated/tier1_submission_skeleton.csv`
- `pipeline/outputs/persona_validation/category_mapping_audit.csv`
- `pipeline/outputs/persona_validation/quota_diagnostics_gender_age.csv`
- `pipeline/outputs/persona_validation/quota_diagnostics_gender_race.csv`
- `pipeline/outputs/persona_validation/f_quota_diagnostics_gender_age.csv`
- `pipeline/outputs/persona_validation/f_quota_diagnostics_gender_race.csv`
- `pipeline/outputs/persona_validation/source_nonquota_distribution_audit.csv`
- `pipeline/outputs/persona_validation/joint_distribution_audit.csv`
- `pipeline/outputs/persona_validation/state_audit.csv`
- `pipeline/outputs/persona_validation/duplication_audit.csv`
- `pipeline/outputs/persona_validation/condition_balance_audit.csv`
- `pipeline/outputs/persona_validation/existing_persona_files.csv`

## Plots

- `pipeline/outputs/persona_validation/target_vs_actual_gender_age.png`
- `pipeline/outputs/persona_validation/target_vs_actual_gender_race.png`
- `pipeline/outputs/persona_validation/f_target_vs_actual_gender_age.png`
- `pipeline/outputs/persona_validation/f_target_vs_actual_gender_race.png`
- `pipeline/outputs/persona_validation/error_gender_age.png`
- `pipeline/outputs/persona_validation/error_gender_race.png`
- `pipeline/outputs/persona_validation/moderator_marginals.png`
- `pipeline/outputs/persona_validation/state_distribution.png`
- `pipeline/outputs/persona_validation/condition_balance.png`
- `pipeline/outputs/persona_validation/joint_distribution_checks.png`
- `pipeline/outputs/persona_validation/source_vs_selected_nonquota_demographics.png`

## Five-minute pre-inference checklist

- Confirm `g_personas_master.csv` has 1000 unique donor_key values and no outcome columns.
- Confirm `f_target_panel.csv` has 500 unique donor_key values and uses the same U.S.-adult target population.
- Confirm the two G quota diagnostics have zero absolute percentage-point error.
- Confirm condition balance has zero demographic/state drift across all 17 conditions.
- Confirm the skeleton outcome/raw-item fields are blank, not zeros or imputed values.
- Confirm no stale 500-profile G or 18000-row roster file is selected by any active script/config.

## Self-validation command

```bash
python pipeline/scripts/validate_personas.py
```
