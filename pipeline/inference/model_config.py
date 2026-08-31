"""Central model and frozen-protocol configuration."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG_PATH = PIPELINE_ROOT / "config" / "model_config.yaml"
MODEL_SELECTION_DIR = PIPELINE_ROOT / "outputs" / "model_selection"


@lru_cache(maxsize=32)
def _load_model_config_cached(path_str: str, mtime_ns: int, size: int) -> dict[str, Any]:
    return yaml.safe_load(Path(path_str).read_text(encoding="utf-8"))


def load_model_config(path: Path | str = MODEL_CONFIG_PATH) -> dict[str, Any]:
    """Cached on (path, mtime, size) -- a config file edited on disk (e.g. a
    test writing a fresh tmp_path fixture, or a later amendment committed to
    config/model_config.yaml) always invalidates the cache automatically, so
    this is purely a performance change: this call is on the hot path of
    every per-request engine-config hash (compute_engine_config_hash), which
    made building large (100k+) request manifests dominated by repeated
    whole-file YAML re-parses. A fresh deep copy is returned on every call
    (matching the prior always-reparse behavior) so no caller can mutate a
    shared cached object."""
    p = Path(path)
    stat = p.stat()
    return copy.deepcopy(_load_model_config_cached(str(p), stat.st_mtime_ns, stat.st_size))


def inference_parameters(path: Path | str = MODEL_CONFIG_PATH) -> dict[str, Any]:
    return dict(load_model_config(path)["inference_parameters"])


def model_engine_config(model: str, path: Path | str = MODEL_CONFIG_PATH) -> dict[str, Any]:
    """Model-engine configuration (e.g. chat_template_kwargs) for one exact model id.

    Returns {} for any model without a declared entry -- this must never change
    the request body for models that do not explicitly opt in."""
    cfg = load_model_config(path).get("model_engine_config", {}) or {}
    return dict(cfg.get(model, {}))


def model_candidates(role: str, path: Path | str = MODEL_CONFIG_PATH) -> list[str]:
    cfg = load_model_config(path)["model_selection"]
    key = f"{role}_model_candidates"
    if key not in cfg:
        raise ValueError("role must be 'g' or 'f'")
    return list(cfg[key])


def selected_model(role: str, *, require_frozen: bool = True, path: Path | str = MODEL_CONFIG_PATH) -> str | None:
    cfg = load_model_config(path)["model_selection"]
    key = f"selected_{role}_model"
    if key not in cfg:
        raise ValueError("role must be 'g' or 'f'")
    selected = cfg.get(key)
    if require_frozen and cfg.get("candidate_bakeoff_required") and not selected:
        raise RuntimeError(f"{key} is not frozen; run/record candidate bakeoff before production")
    if selected and selected not in cfg[f"{role}_model_candidates"]:
        raise RuntimeError(f"{key}={selected!r} is not in declared candidate set")
    return selected


def write_selected_model_metadata(
    *,
    model_role: str,
    requested_model: str,
    returned_model_identifier: str,
    selection_dataset_ids: list[str],
    selection_metric: str,
    inference_parameters_used: dict[str, Any],
    outputs_dir: Path | str = MODEL_SELECTION_DIR,
) -> Path:
    if requested_model not in model_candidates(model_role):
        raise ValueError(f"requested_model {requested_model!r} is not declared for role {model_role!r}")
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "requested_model": requested_model,
        "returned_model_identifier": returned_model_identifier,
        "model_role": model_role,
        "selection_dataset_ids": selection_dataset_ids,
        "selection_metric": selection_metric,
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "inference_parameters": inference_parameters_used,
    }
    path = out / f"{model_role}_selected_model.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def provider_parameters(*, supports_reasoning_effort: bool, path: Path | str = MODEL_CONFIG_PATH) -> dict[str, Any]:
    params = inference_parameters(path)
    if not supports_reasoning_effort:
        params.pop("reasoning_effort", None)
        params["reasoning_effort_omitted_reason"] = "provider/model does not support reasoning_effort"
    return params
