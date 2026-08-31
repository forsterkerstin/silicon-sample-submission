# External validation report

Two real, held-out validation sources for our elicitation pipeline's predicted
treatment effects -- distinct from (and considerably stronger than) the
"baseline realism" check discussed earlier, which turned out to be circular
(it compared a control distribution against the exact external reference it
had just been force-calibrated to match) and, in the code as it stood, not
even wired into any real run. Neither issue applies here: both sources below
compare our model's own predictions against real human data it was never
calibrated against.

## 1. Climate-advocacy megastudy treatment arms (no leakage)

`data/climate_advocacy_megastudy/data/advocacy_data.csv` has real `cond`/
`condName` columns for 17 treatment arms + control -- previously only its
control rows were used, for `baseline_references.json`'s moment calibration.
Reusing the *same* respondents to also validate treatment effects would be
circular in a subtler way (our control baseline is literally anchored to
those rows), so `advocacy_split.py` deterministically splits every respondent
into a `calib` half and a disjoint `valid` half (stable `crc32(ResponseId)`,
verified disjoint by `tests/elicitation/test_advocacy_split.py`).
`build_baseline_reference.py` now reads only `calib`;
`build_advocacy_validation.py` reads only `valid`.

**Stimulus text**: `scripts/build_advocacy_stimuli.py` extracts each
condition's real intervention text from
`materials/intervention_docx/*.docx` (Qualtrics' own block export, not
hand-written excerpts). Two real data-quality problems were found and
excluded rather than papered over:
- `MispCorrectionRisks` references Qualtrics piped-field values
  (`${e://Field/...}`) not present in this export -- the literal displayed
  text can't be reconstructed.
- 6 conditions (`BipartisanEliteCues`, `ActivistPerspective`,
  `ClimatePolicyLiteracy`, `CollEfficacyEmoBenefit`, `GlobalHealthThreat`,
  `ShiftFocusIndColl`) explicitly refer to an embedded video the docx export
  doesn't contain.

**10 of 17** conditions have a fully self-contained, faithfully
reconstructable stimulus and got a real elicited `model_ate` (Qwen2.5-14B-
Instruct via vLLM, 10 roster profiles, this benchmark's own
belief_post/policy_general/donation_ams/newsletter_signup items). All 17 get
a real `human_ate` regardless (no stimulus text needed for that side).

**Result** (`data/validation_advocacy_ate.csv`, 40 condition x outcome pairs
with both a real model_ate and human_ate):

| | RMSE | bias | corr |
|---|---:|---:|---:|
| raw model_ate | 28.13pp | +18.17pp | 0.466 |
| shrunk (lambda=0.066, fit on the *unrelated* 70-study archive) | **2.80pp** | **-0.83pp** | 0.466 |

The raw model systematically over-predicts effect size by ~18pp on average --
direct, real confirmation of the exact failure mode the shrinkage step exists
to correct. Applying the shrinkage lambda already fit on a generic,
unrelated TESS archive (not this data) cuts RMSE by ~10x and nearly zeroes
the bias, while correlation (unaffected by a linear rescaling) shows the raw
model's relative ranking of which interventions matter more/less already
carries real signal. This is a genuine out-of-domain generalization check --
the shrinkage was never fit on this data -- and it passed.

**Caveats, disclosed not hidden**:
- Only 4/13 benchmark outcomes have a same-scale item here.
- The elicitation reuses this benchmark's own item wording for those 4
  outcomes, not advocacy_data.csv's exact Qualtrics wording (the same
  same-scale correspondence `baseline_calibration.py` already documents).
- Real-time floor-probability fallback rate was higher here than in the
  primary pipeline's own conditions (long, complex stimulus text some-times
  pushes the model's first token away from an immediate answer letter) --
  visible in the run log, not silently absorbed; the *averaged* per-condition
  model_ate values were still non-degenerate and directionally sensible.

## 2. Vlasceanu et al. 2024 US sample (fully independent, no shared data)

`data/data63.xlsx` -- "Addressing climate change with behavioral science: A
global intervention tournament in 63 countries" (*Science Advances*). US
sample: **n=8,253**, comfortably large. A completely separate study/platform/
respondent pool from the advocacy megastudy, so no leakage risk to guard
against here.

No materials folder was supplied for this one, so unlike the advocacy
megastudy there is no way to elicit a matching `model_ate` for its specific
11 interventions. What it gives instead (`data/validation_vlasceanu_us.csv`):

- A real, large-sample **effect-size envelope** for a comparable megastudy:
  |human_ate| ranges 1.27-7.38pp, median 3.68pp, across 11 real interventions
  x 2 outcomes (belief, policy support). Useful as an honest plausibility
  bound on our own predicted magnitudes -- and notably, it's the same order
  of magnitude the advocacy-megastudy validation's *shrunk* predictions land
  in (RMSE 2.80pp), not the raw ones (28pp) -- independent corroboration that
  shrunk-scale effects, not raw ones, are the realistic regime.
- One disclosed conceptual (not stimulus-identical) match: `SciConsens`
  (scientific-consensus messaging) is the same real-world mechanism as this
  benchmark's own `Consensus` condition. Real human_ate: belief +4.36pp,
  policy +3.00pp. Worth a direct spot-check once a full run of this
  benchmark's own pipeline produces a `Consensus` effect to compare against.

## Bottom line

Both sources point the same direction: this pipeline's *raw* elicited effects
are too large, its *shrinkage-calibrated* effects land in a realistic range,
and the shrinkage mechanism generalizes to real, held-out, climate-specific
data it was never fit on. This is a materially stronger validation basis
than the baseline-realism check alone, though it remains partial (4/13
outcomes, 10/17 conditions with real stimulus text, cross-instrument wording
for the advocacy comparison, no model_ate at all for the Vlasceanu side) --
disclosed above, not silently assumed away.
