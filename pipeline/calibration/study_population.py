"""Study-specific F population panels for external primary calibration.

Primary external calibration studies use their own respondent-level
pretreatment demographic distribution, not the target benchmark's fixed U.S.
adult F panel. The helpers here intentionally operate only on allowed
pretreatment demographic fields and sample observed joint rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Sequence

import numpy as np
import pandas as pd

ALLOWED_PRETREATMENT_DEMOGRAPHICS = ("GENDER", "race_4", "pid_3", "age_5", "EDUC4", "ideo_3")
FORBIDDEN_PROFILE_FIELDS = {"y", "condition.name", "condition", "treatment", "outcome", "outcome.name"}
DEFAULT_EXTERNAL_N_F = 500
ARCHIVE_PROFILE_FIELD_MAP = {
    "GENDER": "gender",
    "race_4": "race",
    "pid_3": "party",
    "age_5": "age",
    "EDUC4": "education",
    "ideo_3": "political_ideology",
}


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def available_profile_fields(respondents: pd.DataFrame, allowed_fields: Sequence[str] = ALLOWED_PRETREATMENT_DEMOGRAPHICS) -> list[str]:
    return [field for field in allowed_fields if field in respondents.columns and respondents[field].notna().any()]


def validate_profile_fields(panel: pd.DataFrame) -> None:
    forbidden = sorted(FORBIDDEN_PROFILE_FIELDS & set(panel.columns))
    if forbidden:
        raise ValueError(f"forbidden non-pretreatment field(s) in F profile panel: {forbidden}")


def archive_profile_to_prompt_profile(row: pd.Series | dict[str, object]) -> dict[str, object]:
    """Map archive demographic fields to the prompt compiler's profile keys.

    Missing values are omitted. They are never imputed and never rendered as
    "unknown" unless the source data literally contain that string.
    """
    profile: dict[str, object] = {}
    for source, target in ARCHIVE_PROFILE_FIELD_MAP.items():
        value = row.get(source) if isinstance(row, dict) else row.get(source)
        if pd.isna(value) or str(value).strip() == "":
            continue
        profile[target] = value
    return profile


def signature_payload(row: pd.Series, fields: Sequence[str]) -> tuple[tuple[str, object], ...]:
    parts: list[tuple[str, object]] = []
    for field in fields:
        value = row.get(field)
        if pd.isna(value):
            parts.append((field, None))
        else:
            parts.append((field, str(value)))
    return tuple(parts)


def signature_id(payload: tuple[tuple[str, object], ...]) -> str:
    return hashlib.sha256(signature_key(payload).encode("utf-8")).hexdigest()[:16]


def signature_key(payload: tuple[tuple[str, object], ...]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def largest_remainder_allocations(counts: pd.Series, *, n_f: int, tie_key: str) -> pd.Series:
    if counts.empty:
        raise ValueError("cannot allocate an empty profile-signature distribution")
    total = float(counts.sum())
    quotas = counts.astype(float) * n_f / total
    base = np.floor(quotas).astype(int)
    remaining = int(n_f - base.sum())
    if remaining:
        order = sorted(
            counts.index,
            key=lambda idx: (
                -(float(quotas.loc[idx]) - int(base.loc[idx])),
                stable_seed(tie_key, idx),
                str(idx),
            ),
        )
        for idx in order[:remaining]:
            base.loc[idx] += 1
    if int(base.sum()) != n_f:
        raise RuntimeError(f"largest-remainder allocation produced {int(base.sum())} slots, expected {n_f}")
    return base.astype(int)


def effect_panel_from_analytic_sample(
    analytic: pd.DataFrame,
    *,
    study_id: str,
    effect_id: str,
    fields: Sequence[str],
    n_f: int = DEFAULT_EXTERNAL_N_F,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build one deterministic F panel for a study_id x effect_id target.

    The source rows are the pooled human analytic sample for the relevant
    control/treatment comparison. Rows with missing demographic fields are
    retained; their missing-field pattern is part of the profile signature.
    """
    if analytic.empty:
        raise ValueError(f"{effect_id}: no human analytic rows with observed y")
    fields = [field for field in fields if field in analytic.columns and analytic[field].notna().any()]
    if not fields:
        raise ValueError(f"{effect_id}: no observed allowed demographic fields")
    working = analytic.reset_index(drop=True).copy()
    signatures = working.apply(lambda row: signature_payload(row, fields), axis=1)
    payload_by_key = {signature_key(payload): payload for payload in signatures}
    counts = signatures.map(signature_key).value_counts(sort=False)
    allocations = largest_remainder_allocations(counts, n_f=n_f, tie_key=f"{study_id}|{effect_id}")
    rows: list[dict[str, object]] = []
    errors = []
    for key, source_n in counts.items():
        payload = payload_by_key[key]
        allocated = int(allocations.loc[key])
        sid = signature_id(payload)
        empirical_share = float(source_n) / len(working)
        panel_share = allocated / n_f
        errors.append(abs(panel_share - empirical_share) * 100)
        values = {field: value for field, value in payload if value is not None}
        missing_fields = [field for field, value in payload if value is None]
        for copy_idx in range(1, allocated + 1):
            rows.append(
                {
                    "study_id": study_id,
                    "effect_id": effect_id,
                    "f_profile_id": f"{study_id}__{signature_id((('effect_id', effect_id),))}__F{len(rows) + 1:03d}",
                    "profile_signature_id": sid,
                    "signature_source_n": int(source_n),
                    "signature_allocated_n": allocated,
                    "signature_empirical_share": empirical_share,
                    "signature_panel_share": panel_share,
                    "missing_demographic_fields": "|".join(missing_fields),
                    **values,
                }
            )
    panel = pd.DataFrame(rows)
    validate_profile_fields(panel)
    profile_cols = [field for field in fields if field in panel.columns]
    profiles_with_any_missing = int(working[fields].isna().any(axis=1).sum())
    audit = {
        "study_id": study_id,
        "effect_id": effect_id,
        "analytic_n": int(len(working)),
        "distinct_profile_signatures": int(len(counts)),
        "profiles_with_any_missing_demographic": profiles_with_any_missing,
        "pct_profiles_with_any_missing_demographic": float(100 * profiles_with_any_missing / len(working)),
        "n_f": int(len(panel)),
        "max_signature_multiplicity_in_panel": int(panel["profile_signature_id"].value_counts().max()),
        "max_abs_signature_share_error_pp": float(max(errors) if errors else math.nan),
        "same_panel_control_treatment": True,
        "profile_fields_available": "|".join(profile_cols),
        "status": "PASS" if len(panel) == n_f else "FAIL",
    }
    return panel, audit


def deterministic_study_panel(
    respondents: pd.DataFrame,
    *,
    study_id: str,
    n_f: int = DEFAULT_EXTERNAL_N_F,
    allowed_fields: Sequence[str] = ALLOWED_PRETREATMENT_DEMOGRAPHICS,
) -> pd.DataFrame:
    """Return a deterministic N_F panel sampled from observed joint rows.

    Sampling is from the pooled study respondent demographic distribution,
    never condition-specific. If the study has fewer than N_F respondents,
    observed rows are sampled with replacement; otherwise, a deterministic
    representative sample of N_F observed rows is selected without replacement.
    """
    fields = available_profile_fields(respondents, allowed_fields)
    if not fields:
        raise ValueError(f"{study_id}: no allowed pretreatment demographic fields available")
    source = respondents.dropna(subset=fields).copy()
    if source.empty:
        raise ValueError(f"{study_id}: no complete pretreatment demographic rows available")
    if "respondent_row_id" in source.columns and source["respondent_row_id"].notna().all() and source["respondent_row_id"].is_unique:
        source_ids = source["respondent_row_id"].astype(int).reset_index(drop=True)
    else:
        source_ids = pd.Series(source.index + 1).astype(int).reset_index(drop=True)
    source_profiles = source[fields].reset_index(drop=True)
    rng = np.random.default_rng(stable_seed("external_primary_f_panel", study_id, "v1"))
    replace = len(source_profiles) < n_f
    sampled_idx = rng.choice(np.arange(len(source_profiles)), size=n_f, replace=replace)
    panel = source_profiles.iloc[sampled_idx].reset_index(drop=True).copy()
    panel.insert(0, "source_respondent_row_id", source_ids.iloc[sampled_idx].to_numpy())
    panel.insert(0, "f_profile_id", [f"{study_id}__F{i:03d}" for i in range(1, n_f + 1)])
    panel.insert(0, "study_id", study_id)
    panel["profile_weight"] = 1.0
    validate_profile_fields(panel)
    return panel


def profile_field_summary(respondents_by_study: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for study_id, respondents in sorted(respondents_by_study.items()):
        fields = available_profile_fields(respondents)
        complete = respondents.dropna(subset=fields) if fields else respondents.iloc[0:0]
        weight_fields = [c for c in respondents.columns if "weight" in c.lower() or c.lower() in {"wt", "wgt"}]
        rows.append(
            {
                "study_id": study_id,
                "n_respondents": len(respondents),
                "original_rows_with_complete_usable_demographics": len(complete),
                "distinct_demographic_tuples": int(complete[fields].drop_duplicates().shape[0]) if fields else 0,
                "profile_fields_available": "|".join(fields),
                "n_profile_fields_available": len(fields),
                "weight_variables_detected": "|".join(weight_fields),
            }
        )
    return pd.DataFrame(rows)
