# `pipeline/data/domain_validation/orchinik/`

Source data for the post-freeze external diagnostic validation against
Orchinik et al. (2024, *PNAS Nexus*, doi:10.1093/pnasnexus/pgae485) — see
`registration.md` sections I.2(iii) and J.1, and
`pipeline/outputs/domain_validation/orchinik_g_domain_confirmation_v2/`.

## `final_bovitz_raw.csv` is not distributed in this repository

This directory does **not** ship `final_bovitz_raw.csv`. That file is the
Bovitz-sample raw survey export underlying Orchinik et al. (2024), and it
contains at least one respondent's free-text open-ended `comments` response
with personal identifying information (an email address a participant typed
into a comment box). To avoid redistributing third-party respondent PII,
this repository does not include a copy of it; `final_bovitz_raw.csv` is
listed in `.gitignore` and, if present at all, exists only as an untracked
local file for offline test/reproduction convenience (see below).

**Retrieve the authoritative original directly from its source** for exact
reproduction:

- **Source**: OSF project "Learning from and about scientists"
- **DOI**: `10.17605/OSF.IO/JYNQH`
- **URL**: https://osf.io/jynqh/
- **Path within the project**: `data/final_bovitz_raw.csv`
- **SHA-256**: `cbae3d4a7faf1027e0434d1af62b527dc82a8bf6b9b7935a469fb058a0ea65f6`

This hash is exactly what this repository's frozen protocol artifacts
(`pipeline/outputs/domain_validation/frozen_domain_validation_protocol.json`,
`pipeline/outputs/domain_validation/frozen_orchinik_g_domain_confirmation.json`)
already record for this file — verified byte-for-byte identical against the
OSF deposit. Nothing about the frozen protocol or its recorded hash has been
changed on account of not redistributing this file; the hash was already
computed from, and matches, the public OSF original.

To reproduce the diagnostic offline: download `data/final_bovitz_raw.csv`
from the OSF project above, verify its SHA-256 matches the value recorded
here, and place it at `pipeline/data/domain_validation/orchinik/final_bovitz_raw.csv`
(gitignored — it will not be tracked or re-committed).

## Everything else in this directory ships as-is

`final_clean.csv`, `Bovitz_qualtrics.qsf`, `Bovitz_qualtrics.docx`,
`bovitz_data_clean.R`, and `analysis.Rmd` remain tracked and public — all
four data/instrument files are independently verified byte-for-byte
identical to their OSF originals (same project, DOI above) and none carry
the free-text PII issue that applies specifically to `final_bovitz_raw.csv`.
