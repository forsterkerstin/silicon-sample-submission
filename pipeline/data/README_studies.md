# Replication Archive for _Predicting Results of Social Science Experiments Using Large Language Models_

## Running the code ocean bucket

There are **two ways** to use this code: 

- By default, this code runs `code/00_run_all_analyses.R`. This will launch the full suite of analyses from the paper, and takes multiple hours. All paper figures are saved to `results/plots/` and statistics are saved to `results/variables/`

- To improve understandability and reproducibility, we have also created a minimal analysis script at `code/0_minimal_example.R`. This script takes only 2 minutes to run, and can edited/configured to reproduce any of the key estimates from Figure 1. To use this method, edit `code/main.sh` to set `0_minimal_example.R` as the main analysis, and then edit the latter file to choose your desired configuration.

Note that, because the analysis is stochastic, results may differ slightly from those in the paper.

## Output source data files (`results/plot_data/`)

Per-figure CSV files containing the data plotted in each main / Extended Data figure. One CSV per panel for multi-panel figures (e.g. `fig1_panelA.csv`, `fig1_panelB.csv`, `fig1_panelC.csv`). See PLOT_DATA_CODEBOOK.docx for per-file, per-column descriptions.

## Data

All data used in our analysis is stored in the `data/` directory:

### Survey of Social Scientists
- Data is located in `survey_data/`

### Archive 1 (TESS; Coppock et al)
- `rct_responses.RDS` - All original human responses
- `LLM_responses.RDS` - Prompts and responses for all LLM calls
- `forecasting_responses.RDS` - Data from layperson forecasters
- Research Assistant coding files:
  - `RA_study_features.csv` - Study-level features, publication dates, etc.
  - `RA_outcome_features.csv` - Features of the dependent variables in each study
  - `RA_hypotheses.RDS` - Contrasts that were hypothesized by the original authors
  - `gpt_author_recognition.csv` - Results from robustness check on whether GPT-4 can guess paper authors.

### Archive 2 (Megastudies)
- `megastudies.rds` - All effects and GPT-4 predictions (calculated by the same method as Archive 1)
  - These are preprocessed rather than raw data, as raw data isn't available for many studies
  - Some studies are unpublished, and so condition names have been changed to avoid pre-empting authors' original publications.
- `individual_expert_predictions.rds` - Expert forecasts (used in the paper section: _Identifying Effective Interventions_).




## Running outside of code ocean
1. Install all requirements using `00_requirements.R`
2. If required, modify `config.R` (see below)
3. Generate all paper figures by running `00_run_all_analyses.R`
   - Results are saved in `results/`


# Notes on code
- Intermediate results are saved in `output/processed_data/` and loaded by default, so subsequent runs will be faster
  - To run the analysis from scratch, you can delete this directory and they will be recreated.
  - The analysis is _slightly stochastic_ due to sampling outcomes and conditions.
- Analysis is computationally intensive due to running many replications and averaging results
  - To generate all paper figures takes **approximately 4 hours** on a 2024 MacBook Pro
  - For faster execution: modify `config.R` and decrease `n_runs = 32` to e.g. `n_runs = 2`. Results will be similar but noisier
  - By default, `config.R` uses `future::plan("multisession", workers = n_workers)` to parallelize over cores. If you experience issues with this setup, we suggest changing this to use the future.callr package (`future::plan(future.callr::callr, workers = n_workers)`) or otherwise disabling parallelization altogether (`future::plan("sequential")`)
  - By default, `config.R` sets `n_workers` to half of the computer's cores, but this can be changed.