#!/usr/bin/env Rscript
## pipeline/scripts/extract_archive_rds.R
##
## Flattens the 70-experiment archive's R-native (.RDS) files -- from
## Ashokkumar, Hewitt, Ghezae & Willer (2026, Nature), extracted from the
## user-supplied capsule-9843791-data.zip into pipeline/data/archive_70studies/
## -- into plain CSVs that scripts/build_ate_archive.py (pure Python) can
## consume without an R dependency at that stage.
##
## Reads (read-only): RA_hypotheses.RDS, rct_responses.RDS, llm_responses.RDS.
## Writes: pipeline/data/archive_70studies/extracted/{hypotheses,rct_condition_means,
## llm_condition_means,rct_study_demographics,rct_response_demographics_by_outcome,
## rct_study_column_manifest}.csv
##
## This is OUR pipeline code, not organizer code -- it never touches
## anything under survey/, codebook.csv, scripts/ (repo root), etc.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("^--file=", "", args[grep("^--file=", args)])
pipeline_dir <- dirname(dirname(normalizePath(script_path)))
archive_dir <- file.path(pipeline_dir, "data", "archive_70studies")
out_dir <- file.path(archive_dir, "extracted")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

## --- 1. RA_hypotheses.RDS: the authoritative treatment/control contrasts ---
## Long format: one row per (study, outcome, hypothesis, condition), with
## t_hypothesis (1 = treatment side, 0 = reference/control side) -- the
## paper's own RA-coded contrast, not a guess from condition-name text.
hyp <- readRDS(file.path(archive_dir, "RA_hypotheses.RDS"))
hyp_long <- hyp %>%
  select(study, outcome.name, hypothesis, hypothesis_data) %>%
  tidyr::unnest(hypothesis_data)
write.csv(hyp_long, file.path(out_dir, "hypotheses.csv"), row.names = FALSE)
cat("wrote hypotheses.csv:", nrow(hyp_long), "rows\n")

## --- 2. rct_responses.RDS: real human per-respondent data -> condition means ---
## Aggregated to (study, outcome, condition) means here (not exported at
## per-respondent grain) -- only the condition mean/n/scale bounds are
## needed to compute an ATE, and per-respondent export would be much larger
## for no benefit at this stage.
rct <- readRDS(file.path(archive_dir, "rct_responses.RDS"))
rct_means <- rct %>%
  select(study, outcome.name, outcome.min, outcome.max, data) %>%
  mutate(row_id = row_number()) %>%
  tidyr::unnest(data) %>%
  group_by(study, outcome.name, outcome.min, outcome.max, condition.name) %>%
  summarise(mean_y = mean(y, na.rm = TRUE), n = sum(!is.na(y)), .groups = "drop")
write.csv(rct_means, file.path(out_dir, "rct_condition_means.csv"), row.names = FALSE)
cat("wrote rct_condition_means.csv:", nrow(rct_means), "rows\n")

## Respondent-level pretreatment demographics for study-specific external F
## calibration panels. One data frame per study is used (the nested data repeat
## by outcome), and only allowed pretreatment demographic fields are exported.
allowed_demographics <- c("GENDER", "race_4", "pid_3", "age_5", "EDUC4", "ideo_3")
study_ids <- unique(as.character(rct$study))
demo_rows <- list()
response_demo_rows <- list()
manifest_rows <- list()
for (i in seq_along(study_ids)) {
  s <- study_ids[[i]]
  idx <- which(as.character(rct$study) == s)[1]
  d <- rct$data[[idx]]
  cols <- intersect(allowed_demographics, names(d))
  weight_cols <- names(d)[grepl("weight", names(d), ignore.case = TRUE) | tolower(names(d)) %in% c("wt", "wgt")]
  manifest_rows[[i]] <- data.frame(
    study = s,
    n_rows = if (is.data.frame(d)) nrow(d) else NA_integer_,
    cols = if (is.data.frame(d)) paste(names(d), collapse = "|") else "",
    allowed_demographic_fields = paste(cols, collapse = "|"),
    weight_variables_detected = paste(weight_cols, collapse = "|"),
    stringsAsFactors = FALSE
  )
  if (is.data.frame(d) && length(cols) > 0) {
    out <- as.data.frame(d[, cols, drop = FALSE])
    out <- data.frame(study = s, respondent_row_id = seq_len(nrow(out)), out, check.names = FALSE)
    demo_rows[[length(demo_rows) + 1]] <- out
  }
}
demo <- dplyr::bind_rows(demo_rows)
for (i in seq_len(nrow(rct))) {
  d <- as.data.frame(rct$data[[i]])
  cols <- intersect(allowed_demographics, names(d))
  out <- data.frame(
    study = as.character(rct$study[[i]]),
    outcome.name = as.character(rct$outcome.name[[i]]),
    respondent_row_id = seq_len(nrow(d)),
    y = if ("y" %in% names(d)) d$y else NA,
    condition.name = if ("condition.name" %in% names(d)) d$condition.name else NA,
    check.names = FALSE
  )
  for (col in cols) out[[col]] <- d[[col]]
  response_demo_rows[[i]] <- out
}
response_demo <- dplyr::bind_rows(response_demo_rows)
manifest <- dplyr::bind_rows(manifest_rows)
write.csv(demo, file.path(out_dir, "rct_study_demographics.csv"), row.names = FALSE)
write.csv(response_demo, file.path(out_dir, "rct_response_demographics_by_outcome.csv"), row.names = FALSE)
write.csv(manifest, file.path(out_dir, "rct_study_column_manifest.csv"), row.names = FALSE)
cat("wrote rct_study_demographics.csv:", nrow(demo), "rows\n")
cat("wrote rct_response_demographics_by_outcome.csv:", nrow(response_demo), "rows\n")
cat("wrote rct_study_column_manifest.csv:", nrow(manifest), "rows\n")

## --- 3. llm_responses.RDS: model predictions -> condition means, gpt-4 only ---
## Restricted to model == "gpt-4" (the model this approach's citation [5]
## and rationale refer to); the archive also contains several other models
## (gemma-3, deepseek, gpt-3.5-turbo, babbage-002, davinci-002), not used here.
llm <- readRDS(file.path(archive_dir, "llm_responses.RDS"))
llm_gpt4 <- llm %>% filter(model == "gpt-4")
llm_means <- llm_gpt4 %>%
  group_by(study, outcome.name, condition.name, outcome_scale_min, outcome_scale_max) %>%
  summarise(mean_expectation = mean(expectation, na.rm = TRUE), n = sum(!is.na(expectation)), .groups = "drop")
write.csv(llm_means, file.path(out_dir, "llm_condition_means.csv"), row.names = FALSE)
cat("wrote llm_condition_means.csv:", nrow(llm_means), "rows (model = gpt-4 only)\n")
