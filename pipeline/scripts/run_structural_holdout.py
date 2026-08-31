#!/usr/bin/env python3
"""Open/evaluate the structural holdout using an already frozen F/G/C method.

This command does not run LLM inference. It expects a CSV containing raw F
holdout ATEs produced by the frozen F protocol, then applies the frozen C
parameters exactly once and writes prediction/metric outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from validation.holdout import FROZEN_METHOD_MANIFEST_PATH, HoldoutIntegrityError, VALIDATION_DIR, evaluate_structural_holdout  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-input", type=Path, required=True, help="CSV with human_ate_native/raw_f_ate_native/outcome_range")
    parser.add_argument("--manifest", type=Path, default=FROZEN_METHOD_MANIFEST_PATH)
    parser.add_argument("--outputs-dir", type=Path, default=VALIDATION_DIR)
    args = parser.parse_args()
    try:
        result = evaluate_structural_holdout(args.predictions_input, manifest_path=args.manifest, outputs_dir=args.outputs_dir)
    except HoldoutIntegrityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "method_hash": result["manifest"]["method_hash"],
                "n_effects": int(len(result["predictions"])),
                "pooled_metrics": str(args.outputs_dir / "megastudy_holdout_pooled_metrics.csv"),
                "effect_predictions": str(args.outputs_dir / "megastudy_holdout_predictions.csv"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
