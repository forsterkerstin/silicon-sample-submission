"""F-panel protocol freezing, convergence, and stochastic reliability."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from inference.model_config import inference_parameters, load_model_config, selected_model

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "f_reliability"
DEFAULT_N_F = 500
DEFAULT_F_NUM_DRAWS = 1
NESTED_SIZES = (50, 100, 250, 500)


def f_protocol_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg = load_model_config()["f_protocol"]
    if int(cfg["n_f"]) != DEFAULT_N_F:
        raise RuntimeError(f"production n_f must be {DEFAULT_N_F}, got {cfg['n_f']}")
    if int(cfg["f_num_draws"]) < 1:
        raise RuntimeError("f_num_draws must be >= 1")
    if cfg.get("external_calibration_f_num_draws") not in (None, "") and cfg.get("target_f_num_draws") not in (None, ""):
        if int(cfg["external_calibration_f_num_draws"]) != int(cfg["target_f_num_draws"]):
            raise RuntimeError("external_calibration_f_num_draws and target_f_num_draws differ")
    return cfg


def frozen_protocol_path() -> Path:
    return PIPELINE_ROOT / f_protocol_config()["frozen_protocol_path"]


def require_frozen_f_protocol(path: Path | str | None = None) -> dict[str, Any]:
    protocol_path = Path(path) if path is not None else frozen_protocol_path()
    if not protocol_path.exists():
        raise RuntimeError(f"target F production requires frozen F protocol at {protocol_path}")
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    required = ["selected_f_model", "n_f", "f_num_draws", "temperature", "top_p", "frozen_at"]
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError(f"frozen F protocol missing key(s): {missing}")
    if int(payload["n_f"]) != DEFAULT_N_F:
        raise RuntimeError(f"frozen F protocol n_f must be {DEFAULT_N_F}")
    if int(payload["f_num_draws"]) != int(f_protocol_config()["f_num_draws"]):
        raise RuntimeError("frozen F protocol f_num_draws differs from config")
    return payload


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


def validate_external_target_protocol(external: dict[str, Any], target: dict[str, Any]) -> None:
    keys = ["selected_f_model", "n_f", "f_num_draws", "temperature", "top_p", "reasoning_configuration", "prompt_version"]
    mismatched = [key for key in keys if external.get(key) != target.get(key)]
    if mismatched:
        raise RuntimeError(f"external and target F protocols differ: {mismatched}")


def write_frozen_f_protocol(
    *,
    pilot_manifest_path: Path,
    convergence_summary: dict[str, Any],
    stochastic_reliability_summary: dict[str, Any],
    selected_f_model_value: str | None = None,
    outputs_dir: Path | str = OUTPUT_DIR,
) -> Path:
    params = inference_parameters()
    model = selected_f_model_value or selected_model("f", require_frozen=True)
    manifest_hash = hashlib.sha256(Path(pilot_manifest_path).read_bytes()).hexdigest()
    payload = {
        "selected_f_model": model,
        "n_f": DEFAULT_N_F,
        "f_num_draws": int(f_protocol_config()["f_num_draws"]),
        "f_r_f": int(f_protocol_config()["f_num_draws"]),
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "reasoning_configuration": {"reasoning_effort": params.get("reasoning_effort")},
        "structured_output": params.get("structured_output"),
        "prompt_version": load_model_config()["prompting"]["f_prompt_protocol"],
        "pilot_manifest_hash": manifest_hash,
        "convergence_summary": convergence_summary,
        "stochastic_reliability_summary": stochastic_reliability_summary,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    payload["f_inference_config_hash"] = f_inference_config_hash(payload)
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "frozen_f_protocol.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def create_pilot_manifest(
    eligible_effects: pd.DataFrame,
    *,
    n_effects: int = 12,
    seed: int = 20260824,
    outputs_dir: Path | str = OUTPUT_DIR,
) -> pd.DataFrame:
    required = {"study_id", "effect_id", "outcome_type", "population_type"}
    missing = required - set(eligible_effects.columns)
    if missing:
        raise ValueError(f"eligible_effects missing column(s): {sorted(missing)}")
    candidates = eligible_effects.drop_duplicates(["study_id", "effect_id"]).copy()
    candidates["deterministic_selection_key"] = [
        hashlib.sha256(f"{seed}|{sid}|{eid}".encode("utf-8")).hexdigest()
        for sid, eid in zip(candidates["study_id"], candidates["effect_id"])
    ]
    out = candidates.sort_values("deterministic_selection_key").head(n_effects).copy()
    out["selected_reason"] = "seeded_metadata_sample_not_performance_based"
    cols = ["study_id", "effect_id", "outcome_type", "population_type", "selected_reason", "deterministic_selection_key"]
    out = out[cols]
    path = Path(outputs_dir)
    path.mkdir(parents=True, exist_ok=True)
    out.to_csv(path / "pilot_manifest.csv", index=False)
    return out


def deterministic_profile_order(study_id: str, profile_ids: Sequence[str]) -> list[str]:
    return sorted(profile_ids, key=lambda pid: hashlib.sha256(f"{study_id}|{pid}".encode("utf-8")).hexdigest())


def nested_profile_subsets(study_id: str, profile_ids: Sequence[str], sizes: Sequence[int] = NESTED_SIZES) -> dict[int, list[str]]:
    unique = deterministic_profile_order(study_id, sorted(set(map(str, profile_ids))))
    if len(unique) < max(sizes):
        raise ValueError(f"need at least {max(sizes)} unique F profiles")
    return {n: unique[:n] for n in sizes}


def profile_level_summary(delta: Sequence[float], outcome_range: float) -> dict[str, float]:
    values = np.asarray(delta, dtype=float)
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    se = sd / float(np.sqrt(len(values)))
    ate = float(np.mean(values))
    return {
        "n_f": int(len(values)),
        "raw_ate_native": ate,
        "raw_ate_pp": 100 * ate / outcome_range,
        "profile_delta_sd": sd,
        "profile_ate_se_native": se,
        "profile_ate_se_pp": 100 * se / outcome_range,
    }


def convergence_by_effect(delta_df: pd.DataFrame, *, outputs_dir: Path | str = OUTPUT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"study_id", "effect_id", "f_profile_id", "delta_native", "outcome_range"}
    missing = required - set(delta_df.columns)
    if missing:
        raise ValueError(f"delta_df missing column(s): {sorted(missing)}")
    rows = []
    for (study_id, effect_id), group in delta_df.groupby(["study_id", "effect_id"], sort=True):
        subsets = nested_profile_subsets(str(study_id), group["f_profile_id"])
        by_profile = group.set_index(group["f_profile_id"].astype(str))
        row = {"study_id": study_id, "effect_id": effect_id}
        outcome_range = float(group["outcome_range"].iloc[0])
        for n, ids in subsets.items():
            z = float(by_profile.loc[ids, "delta_native"].mean())
            row[f"z_{n}_native"] = z
            row[f"z_{n}_pp"] = 100 * z / outcome_range
        rows.append(row)
    by_effect = pd.DataFrame(rows)
    summary_rows = []
    for n in (50, 100, 250):
        diff = by_effect[f"z_{n}_pp"] - by_effect["z_500_pp"]
        summary_rows.append(
            {
                "n": n,
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "median_abs_diff": float(np.median(np.abs(diff))),
                "max_abs_diff": float(np.max(np.abs(diff))),
                "pearson": float(np.corrcoef(by_effect[f"z_{n}_pp"], by_effect["z_500_pp"])[0, 1]),
                "spearman": float(spearmanr(by_effect[f"z_{n}_pp"], by_effect["z_500_pp"]).correlation),
                "sign_agreement": float(np.mean(np.sign(by_effect[f"z_{n}_pp"]) == np.sign(by_effect["z_500_pp"]))),
            }
        )
    summary = pd.DataFrame(summary_rows)
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_effect.to_csv(out / "convergence_by_effect.csv", index=False)
    summary.to_csv(out / "convergence_summary.csv", index=False)
    return by_effect, summary


def stochastic_reliability_by_effect(replicate_effects: pd.DataFrame, *, outputs_dir: Path | str = OUTPUT_DIR) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {"study_id", "effect_id", "replicate", "z_native", "z_pp"}
    missing = required - set(replicate_effects.columns)
    if missing:
        raise ValueError(f"replicate_effects missing column(s): {sorted(missing)}")
    wide = replicate_effects.pivot(index=["study_id", "effect_id"], columns="replicate", values="z_pp")
    if not {"replicate_1", "replicate_2"} <= set(wide.columns):
        raise ValueError("need replicate_1 and replicate_2")
    diff = wide["replicate_1"] - wide["replicate_2"]
    by_effect = wide.reset_index().rename(columns={"replicate_1": "z_r1_pp", "replicate_2": "z_r2_pp"})
    by_effect["z_r1_minus_z_r2_pp"] = diff.to_numpy()
    summary = {
        "pearson": float(np.corrcoef(wide["replicate_1"], wide["replicate_2"])[0, 1]),
        "spearman": float(spearmanr(wide["replicate_1"], wide["replicate_2"]).correlation),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "median_abs_diff": float(np.median(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "sign_agreement": float(np.mean(np.sign(wide["replicate_1"]) == np.sign(wide["replicate_2"]))),
    }
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_effect.to_csv(out / "stochastic_reliability_by_effect.csv", index=False)
    (out / "stochastic_reliability_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return by_effect, summary
