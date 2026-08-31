"""Deterministic request keys, seeds, and raw request logs."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REQUEST_LOG_PATH = PIPELINE_ROOT / "outputs" / "request_logs.csv"


def request_key_g(*, donor_key: str, condition: str, replicate: int = 1) -> str:
    return f"G|{donor_key}|{condition}|replicate_{replicate}"


def request_key_f(*, study_id: str, f_profile_id: str, condition: str, outcome: str, replicate: int = 1) -> str:
    return f"F|{study_id}|{f_profile_id}|{condition}|{outcome}|replicate_{replicate}"


def seed_from_request_key(request_key: str) -> int:
    return int(hashlib.sha256(request_key.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)


def log_request(
    *,
    request_key: str,
    requested_model: str,
    provider_returned_model: str | None,
    temperature: float,
    top_p: float,
    reasoning_setting: str | None,
    seed: int | None,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict[str, Any],
    parsed_output: dict[str, Any] | None,
    raw_provider_response: str | None,
    retry_count: int,
    path: Path | str = REQUEST_LOG_PATH,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "request_key": request_key,
        "requested_model": requested_model,
        "provider_returned_model": provider_returned_model or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_setting": reasoning_setting or "",
        "seed": "" if seed is None else seed,
        "exact_system_prompt": system_prompt,
        "exact_user_prompt": user_prompt,
        "response_schema_version": hashlib.sha256(json.dumps(response_schema, sort_keys=True).encode("utf-8")).hexdigest(),
        "response_schema": json.dumps(response_schema, sort_keys=True),
        "parsed_output": json.dumps(parsed_output, sort_keys=True) if parsed_output is not None else "",
        "raw_provider_response": raw_provider_response or "",
        "retry_count": retry_count,
    }
    write_header = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
