"""Population and simulation-roster construction for the Silicon Sample
Benchmark Tier-1 submission.

Scope: builds a quota-aligned synthetic-respondent panel (ACS PUMS donors,
raked to the benchmark's published quota margins; CES-imputed partisan
identity) and the 17,000-row simulation roster. Does not generate LLM survey
responses and does not estimate treatment effects -- see
inference/simulate_response.py / ate/estimate_ates.py for that, which this
package does not import from or modify.
"""
