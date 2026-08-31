# Validation Report

## Development Data

- Primary archive studies listed: 70
- Development/calibration studies: 31
- Uses: M0/M1/M2 selection and final C estimation only for primary-eligible effects.

## Structural Holdout

- Secondary megastudies listed: 15
- Secondary effect rows enumerated: 606
- Eligible effect rows under current documented metadata: 0
- Holdout opened: False
- Holdout pristine: True

## Raw F Performance

Pending: structural holdout has not been opened.

## Calibrated F/C Performance

Pending: structural holdout has not been opened.

## Holdout Calibration Diagnostic

Pending: diagnostic regression not run.

## Contextual Benchmark Reference

| reference                                       |   pearson_r |   adjusted_pearson_r | usage                     |
|:------------------------------------------------|------------:|---------------------:|:--------------------------|
| Ashokkumar et al. survey-experiment megastudies |        0.43 |                 0.52 | context_only_not_a_target |
| Ashokkumar et al. text-treatment megastudies    |        0.45 |                 0.54 | context_only_not_a_target |

## Climate-Domain Holdout

- Status: contained in structural holdout
- Secondary archive match: True
- Matched study id: Voelkel2025
- Human outcomes already used in repo: True

## G Validation Status

- G validation remains separate under `outputs/validation/g_validation/`.
- Existing implementation: `submission.g_validation.validate_g_against_human`.

## Holdout Integrity

- Frozen method hash: NOT_FROZEN
- Git commit in frozen manifest: NOT_FROZEN
- Method changed after holdout: False

## Final Status

- Calibration archive pipeline: PASS if `run_primary_calibration.py` completes.
- Structural holdout integrity: PASS if holdout is unopened or frozen manifest exists before opening.
- F raw ranking: numeric only after holdout opening; no arbitrary pass threshold.
- F/C absolute calibration: numeric only after holdout opening; no arbitrary pass threshold.
- Climate-domain validation: WARNING if contained in structural holdout or development-contaminated.
- G external validation: PENDING until frozen G respondent-level validation datasets are run.

## Plots

Pending: plots are generated after structural holdout predictions exist.
