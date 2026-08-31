#!/usr/bin/env python3
"""scripts/generate_responses.py

Phase A of the native-response architecture: ties the simulation roster to
pipeline/inference/simulate_response.py and produces real, native
per-respondent survey answers -- no token probabilities, no answer labels,
no per-item calls. For each roster row (one real respondent, one
condition):

  1. Build that respondent's persona + the exact condition/control stimulus
     they'd have read (survey_content.get_condition_stimulus; control rows
     rotate through the three neutral filler texts).
  2. ONE inference call answers every applicable raw item together
     (inference.simulate_response.simulate_response), so within-person
     response dependence is retained -- not separately elicited per item.
  3. Every replicate row gets its OWN call (not one call per (profile,
     condition) copied across replicates): two "respondents" sharing a
     demographic cell are different real people in the actual study, and
     collapsing them to an identical value would erase real within-cell
     response variance that a genuine human sample has. The model's own
     sampling stochasticity (temperature=1.0) is what produces that
     variance here.

This script writes RAW items only -- no outcome composites, no ATE
calibration. That's Phase B (pipeline/submission/build_tier1.py), which
needs the FULL set of profiles/conditions raw responses before it can
estimate arm-level ATEs, so it can't run per-profile like this script does.

Reads survey/survey.json, survey/condition_codenames.csv, and codebook.csv
strictly read-only (via survey_content.py) -- never modifies any organizer
file. Writes its output under pipeline/data/, never into predictions/.

Usage (small, fast, real demonstration slice, CPU default backend):
    python scripts/generate_responses.py --profiles 2 --interventions "Interview Prof. Maraun" \\
        --out data/raw_responses_demo.csv

Usage (same slice, GPU backend -- see inference/client.py's docstring for the required
CUDA torch + vLLM environment; CUDA_VISIBLE_DEVICES=0 pins it to one GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_responses.py --profiles 2 \\
        --backend vllm --out data/raw_responses_demo_vllm.csv

Usage (full scope -- see the printed compute-cost estimate before running this):
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_responses.py --backend vllm --out data/raw_responses_full.csv

Checkpointing: output rows are written (and flushed) after each latent
profile completes, not all at once at the end. A companion
"<out>.meta.json" file records the exact run configuration; re-running the
SAME command after a crash detects it, skips whichever profiles are already
in <out>, and continues -- no special flag needed. Running a DIFFERENT
configuration against an existing --out is refused, not silently mixed in;
pass --overwrite to intentionally start that file over.

Usage (splitting the full run across 2 GPUs -- see --profile-offset; each
worker gets its own --out, merge with pandas.concat or `csv` after both finish):
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_responses.py --backend vllm \\
        --profile-offset 0   --profiles 250 --out data/raw_responses_full_a.csv &
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_responses.py --backend vllm \\
        --profile-offset 250 --profiles 250 --out data/raw_responses_full_b.csv &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

import pandas as pd  # noqa: E402

import survey_content as sc  # noqa: E402
from inference.client import make_native_client  # noqa: E402
from inference.prompts import ROSTER_PROFILE_COLUMNS  # noqa: E402
from inference.simulate_response import simulate_response  # noqa: E402


def build_profile(row: pd.Series) -> dict[str, Any]:
    return {col: row[col] for col in ROSTER_PROFILE_COLUMNS if col in row and pd.notna(row[col])}


def simulate_profile(latent_profile_id: str, profile_rows: pd.DataFrame, items: list[dict[str, str]], client) -> list[dict[str, Any]]:
    """One inference call per roster ROW (see module docstring for why
    replicate rows aren't collapsed to a single shared call)."""
    results = []
    for _, row in profile_rows.iterrows():
        condition = row["condition"]
        stimulus = sc.get_condition_stimulus(
            condition,
            row.get("state_abbr"),
            control_variant=int(row.get("condition_replicate", 1)) if condition == "control" else None,
        )
        profile = build_profile(row)
        response = simulate_response(
            profile,
            stimulus,
            items,
            client,
            donor_key=str(row.get("donor_key", latent_profile_id)),
            condition_id=str(condition),
        )
        results.append({"latent_profile_id": latent_profile_id, "profile_id": row["profile_id"], "condition": condition, **profile, **dict(response.items())})
    return results


def _run_config(args: argparse.Namespace, resolved_item_labels: list[str]) -> dict[str, Any]:
    """Everything that determines the CONTENT of --out (deliberately NOT
    --profile-offset/--profiles: two workers covering different profile
    slices of the same roster under the same methodology are still
    perfectly mergeable). If any of this differs between the invocation
    that started a checkpoint file and the one resuming it, the two halves
    would not be comparable, so a mismatch refuses to resume rather than
    silently mixing methodologies -- see _check_resumable().
    """
    return {
        "roster": str(args.roster),
        "items": sorted(resolved_item_labels),
        "interventions": sorted(args.interventions.split(",")) if args.interventions else None,
        "backend": args.backend,
        "model_name": args.model_name,
        "seed": args.seed,
    }


def _meta_path(out: Path) -> Path:
    return out.with_suffix(out.suffix + ".meta.json")


def _check_resumable(out: Path, run_config: dict[str, Any]) -> tuple[set[str], int]:
    """Returns (latent_profile_ids already completed in `out`, how many rows
    that is) -- (set(), 0) for a fresh file. Refuses (SystemExit) to resume a
    file whose recorded run configuration doesn't match this invocation's,
    or that has no recorded configuration at all -- silently appending
    mismatched rows would produce an output that looks whole but mixes
    methodologies. --overwrite is the deliberate escape hatch.
    """
    if not out.exists() or out.stat().st_size == 0:
        return set(), 0
    meta_path = _meta_path(out)
    if not meta_path.exists():
        raise SystemExit(f"{out} already exists with no {meta_path.name} -- refusing to guess whether it's safe to resume. Pass --overwrite to start it over.")
    recorded = json.loads(meta_path.read_text(encoding="utf-8"))
    if recorded != run_config:
        diff = {k: (recorded.get(k), run_config.get(k)) for k in run_config if recorded.get(k) != run_config.get(k)}
        raise SystemExit(f"{out} was started with a different configuration (recorded vs. current): {diff} -- refusing to mix. Pass --overwrite to start it over.")
    existing_ids = pd.read_csv(out, usecols=["latent_profile_id"])["latent_profile_id"]
    completed = set(existing_ids.unique())
    print(f"resuming {out}: {len(completed)} profile(s) ({len(existing_ids)} rows) already completed")
    return completed, len(existing_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roster", type=Path, default=None, help="defaults to the newest data/processed/population/simulation_roster_*.csv")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-offset", type=int, default=0, help="skip this many latent profiles from the start, before applying --profiles -- for splitting a run across workers/GPUs")
    parser.add_argument("--profiles", type=int, default=None, help="limit to N latent profiles after --profile-offset (omit for all remaining -- see the compute-cost estimate)")
    parser.add_argument("--overwrite", action="store_true", help="if --out already exists, wipe it (and its .meta.json) and start fresh instead of resuming/refusing")
    parser.add_argument("--interventions", type=str, default=None, help="comma-separated intervention titles to include besides control (omit for all 16)")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf", help="native-response backend: 'hf' (CPU/transformers, default) or 'vllm' (GPU, guided JSON decoding -- see inference/client.py)")
    parser.add_argument("--model-name", type=str, default=None, help="override the backend's default model")
    parser.add_argument("--seed", type=int, default=20260831, help="not currently used for RNG (native responses come from the model's own sampling, not a seeded draw) -- recorded in .meta.json for provenance")
    args = parser.parse_args()

    if args.roster is None:
        preferred = PIPELINE_ROOT / "data/processed/population" / "simulation_roster_17000.csv"
        candidates = sorted((PIPELINE_ROOT / "data/processed/population").glob("simulation_roster_*.csv"))
        if not candidates:
            raise SystemExit("no data/processed/population/simulation_roster_*.csv found -- run scripts/build_population.py first, or pass --roster")
        args.roster = preferred if preferred.exists() else max(candidates, key=lambda p: p.stat().st_mtime)
    roster = pd.read_csv(args.roster)
    print(f"using roster: {args.roster}")

    all_profile_ids = roster["latent_profile_id"].drop_duplicates()
    if args.profile_offset:
        all_profile_ids = all_profile_ids.iloc[args.profile_offset :]
    if args.profiles is not None:
        all_profile_ids = all_profile_ids.head(args.profiles)
    roster = roster[roster["latent_profile_id"].isin(all_profile_ids)]
    if args.interventions is not None:
        wanted = set(args.interventions.split(",")) | {"control"}
        roster = roster[roster["condition"].isin(wanted)]

    items = sc.load_items()
    item_labels = [it["target_label"] for it in items]

    n_profiles = roster["latent_profile_id"].nunique()
    n_rows = len(roster)
    print(f"Simulating {n_profiles} profile(s), {n_rows} roster row(s) total -- ONE inference call per row "
          f"(all {len(items)} items answered together per call, not one call per item)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_config = _run_config(args, item_labels)
    if args.overwrite:
        args.out.unlink(missing_ok=True)
        _meta_path(args.out).unlink(missing_ok=True)
        completed_ids, n_written = set(), 0
    else:
        completed_ids, n_written = _check_resumable(args.out, run_config)
    _meta_path(args.out).write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    t0 = time.time()
    client = make_native_client(args.backend, model_name=args.model_name)
    print(f"backend: {args.backend}; model loaded in {time.time() - t0:.1f}s")

    header_needed = not (args.out.exists() and args.out.stat().st_size > 0)
    for i, (latent_profile_id, profile_rows) in enumerate(roster.groupby("latent_profile_id"), start=1):
        if latent_profile_id in completed_ids:
            continue
        t1 = time.time()
        profile_out_rows = simulate_profile(latent_profile_id, profile_rows, items, client)
        pd.DataFrame(profile_out_rows).to_csv(args.out, mode="a", header=header_needed, index=False)
        header_needed = False
        n_written += len(profile_out_rows)
        print(f"[{i}/{n_profiles}] {latent_profile_id}: {time.time() - t1:.1f}s (checkpointed, {n_written} rows written so far)")

    print(f"\nwrote {n_written} total rows to {args.out}")
    print(f"total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
