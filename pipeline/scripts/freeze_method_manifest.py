#!/usr/bin/env python3
"""Freeze the F/G/C method before structural-holdout evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from validation.holdout import HoldoutIntegrityError, build_frozen_method_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-unselected-models",
        action="store_true",
        help="write a non-production draft even when selected_g_model/selected_f_model are unset",
    )
    args = parser.parse_args()
    try:
        payload = build_frozen_method_manifest(allow_unselected_models=args.allow_unselected_models)
    except HoldoutIntegrityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"method_hash": payload["method_hash"], "selected_calibration_model": payload["selected_calibration_model"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
