"""Production provenance checks for external-primary F predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ate.f_reliability import DEFAULT_N_F, require_frozen_f_protocol
from inference.model_config import selected_model

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_PRIMARY_F_PANELS_PATH = PIPELINE_ROOT / "data" / "generated" / "external_primary_f_panels.csv"
EXTERNAL_F_PANEL_VERSION = "external_primary_f_effect_analytic_panels_v1"
EXTERNAL_F_POPULATION_METHOD = "study_effect_analytic_profile_distribution_unweighted_largest_remainder"
EXPECTED_PRIMARY_EFFECT_COUNT = 136
PREDICTION_F_MODEL_ID_COL = "f_model_id"
PREDICTION_F_PROMPT_PROTOCOL_ID_COL = "f_prompt_protocol_id"
PREDICTION_F_INFERENCE_CONFIG_HASH_COL = "f_inference_config_hash"
PREDICTION_F_R_F_COL = "f_r_f"
PREDICTION_PANEL_VERSION_COL = "external_f_population_panel_version"
PREDICTION_PANEL_SHA256_COL = "external_f_population_panel_sha256"
PRODUCTION_READY_SYNTHETIC_STATUS = "PRODUCTION_READY"
LEGACY_NO_PANEL_PROVENANCE = "LEGACY_CACHED_NO_EFFECT_SPECIFIC_PANEL_PROVENANCE"
REQUIRED_PREDICTION_PROVENANCE_COLUMNS = (
    PREDICTION_F_MODEL_ID_COL,
    PREDICTION_F_PROMPT_PROTOCOL_ID_COL,
    PREDICTION_F_INFERENCE_CONFIG_HASH_COL,
    PREDICTION_F_R_F_COL,
    PREDICTION_PANEL_VERSION_COL,
    PREDICTION_PANEL_SHA256_COL,
)


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_external_f_panel_provenance(panels_path: Path | str = EXTERNAL_PRIMARY_F_PANELS_PATH) -> dict[str, str]:
    path = Path(panels_path)
    if not path.exists():
        raise RuntimeError(f"frozen external F panel is absent: {path}")
    return {
        "external_f_population_panel_version": EXTERNAL_F_PANEL_VERSION,
        "external_f_population_panel_sha256": file_sha256(path),
        "external_f_population_panel_path": str(path),
        "population_matching_method": EXTERNAL_F_POPULATION_METHOD,
    }


def f_inference_config_hash(protocol: dict[str, Any]) -> str:
    payload = {
        "temperature": protocol.get("temperature"),
        "top_p": protocol.get("top_p"),
        "reasoning_configuration": protocol.get("reasoning_configuration", {}),
        "structured_output": protocol.get("structured_output"),
        "f_r_f": int(protocol["f_r_f"]),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frozen_f_prediction_provenance(protocol_path: Path | str | None = None) -> dict[str, str]:
    protocol = require_frozen_f_protocol(protocol_path)
    if "f_r_f" not in protocol:
        raise RuntimeError("R_F is unfrozen: frozen F protocol missing f_r_f")
    if int(protocol["f_r_f"]) != int(protocol["f_num_draws"]):
        raise RuntimeError("R_F is unfrozen: f_r_f differs from f_num_draws in frozen F protocol")
    expected_hash = f_inference_config_hash(protocol)
    if protocol.get("f_inference_config_hash") not in (None, "", expected_hash):
        raise RuntimeError("frozen F protocol f_inference_config_hash does not match frozen inference configuration")
    return {
        PREDICTION_F_MODEL_ID_COL: str(protocol["selected_f_model"]),
        PREDICTION_F_PROMPT_PROTOCOL_ID_COL: str(protocol["prompt_version"]),
        PREDICTION_F_INFERENCE_CONFIG_HASH_COL: expected_hash,
        PREDICTION_F_R_F_COL: str(int(protocol["f_r_f"])),
    }


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _primary_rows(archive: pd.DataFrame) -> pd.DataFrame:
    if "included_primary_calibration" not in archive.columns:
        raise RuntimeError("calibration archive missing included_primary_calibration")
    mask = archive["included_primary_calibration"].astype(str).str.lower().isin({"true", "1", "yes"})
    return archive[mask].copy()


def _expected_effect_ids(panels_path: Path | str) -> set[str]:
    path = Path(panels_path)
    if not path.exists():
        raise RuntimeError(f"frozen external F panel is absent: {path}")
    panel_effects = pd.read_csv(path, usecols=["effect_id"])["effect_id"].dropna().astype(str)
    return set(panel_effects.unique())


def _observed_unique_strings(frame: pd.DataFrame, col: str) -> list[str]:
    return sorted(frame[col].fillna("").map(lambda value: str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)).unique().tolist())


def assert_external_f_predictions_production_ready(
    archive: pd.DataFrame,
    *,
    panels_path: Path | str = EXTERNAL_PRIMARY_F_PANELS_PATH,
    frozen_protocol_path: Path | str | None = None,
    require_frozen_model_protocol: bool = True,
    expected_primary_effect_count: int = EXPECTED_PRIMARY_EFFECT_COUNT,
) -> dict[str, str]:
    """Hard-stop if primary external F predictions cannot support production C."""
    problems: list[str] = []
    primary = _primary_rows(archive)
    if primary.empty:
        problems.append("no primary calibration rows are available")

    for col in ("model_ate", "synthetic_ate_native"):
        if col not in primary.columns:
            problems.append(f"predictions are absent: missing {col}")
        elif primary[col].isna().any():
            missing = int(primary[col].isna().sum())
            problems.append(f"predictions are absent: {missing} primary row(s) missing {col}")

    if "synthetic_prediction_status" not in primary.columns:
        problems.append("prediction freshness is absent: missing synthetic_prediction_status")
    else:
        statuses = primary["synthetic_prediction_status"].fillna("").astype(str)
        bad_status = statuses.str.contains("STALE|DEVELOPMENT|LEGACY|UNALIGNED", case=False, regex=True) | (
            statuses != PRODUCTION_READY_SYNTHETIC_STATUS
        )
        if bad_status.any():
            observed = sorted(statuses[bad_status].unique().tolist())
            problems.append(f"predictions are marked stale/development/not production-ready: {observed}")

    if "requires_synthetic_regeneration" in primary.columns and primary["requires_synthetic_regeneration"].map(_is_true).any():
        problems.append("predictions require synthetic regeneration")

    try:
        expected = current_external_f_panel_provenance(panels_path)
    except RuntimeError as exc:
        expected = {}
        problems.append(str(exc))

    try:
        expected_effects = _expected_effect_ids(panels_path)
        observed_effects = primary["effect_id"].dropna().astype(str)
        duplicates = int(observed_effects.duplicated().sum())
        missing = sorted(expected_effects - set(observed_effects))
        unexpected = sorted(set(observed_effects) - expected_effects)
        if len(expected_effects) != expected_primary_effect_count:
            problems.append(f"expected {expected_primary_effect_count} frozen primary effects, panel defines {len(expected_effects)}")
        if len(observed_effects) != expected_primary_effect_count:
            problems.append(f"expected {expected_primary_effect_count} primary prediction effects, present {len(observed_effects)}")
        if duplicates:
            problems.append(f"primary prediction effects contain {duplicates} duplicate row(s)")
        if missing:
            problems.append(f"primary prediction effects missing {len(missing)} effect(s): {missing[:10]}")
        if unexpected:
            problems.append(f"primary prediction effects contain {len(unexpected)} unexpected effect(s): {unexpected[:10]}")
    except (RuntimeError, ValueError) as exc:
        problems.append(str(exc))

    if require_frozen_model_protocol:
        try:
            model = selected_model("f", require_frozen=True)
        except RuntimeError as exc:
            problems.append(str(exc))
            model = None
        try:
            f_expected = frozen_f_prediction_provenance(frozen_protocol_path)
            if model and f_expected.get(PREDICTION_F_MODEL_ID_COL) != model:
                problems.append("frozen F protocol selected_f_model does not match model_config selected_f_model")
            expected.update(f_expected)
        except RuntimeError as exc:
            problems.append(str(exc))

    for col in REQUIRED_PREDICTION_PROVENANCE_COLUMNS:
        expected_value = expected.get(col)
        if col not in primary.columns:
            problems.append(f"prediction provenance is absent: missing {col}")
            continue
        observed = _observed_unique_strings(primary, col)
        if expected_value is not None and observed != [expected_value]:
            problems.append(f"{col} does not match frozen configuration; expected {expected_value!r}, observed {observed!r}")

    if "population_matching_method" in primary.columns:
        methods = sorted(primary["population_matching_method"].fillna("").astype(str).unique().tolist())
        if methods != [EXTERNAL_F_POPULATION_METHOD]:
            problems.append(f"population_matching_method is not frozen effect-specific method: {methods!r}")

    if "num_profiles" in primary.columns:
        n_values = sorted(primary["num_profiles"].dropna().astype(int).unique().tolist())
        if n_values != [DEFAULT_N_F]:
            problems.append(f"primary predictions do not use frozen N_F={DEFAULT_N_F}: observed {n_values!r}")

    if problems:
        raise RuntimeError("external F predictions are not production-ready:\n- " + "\n- ".join(problems))
    return expected
