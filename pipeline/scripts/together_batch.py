#!/usr/bin/env python3
"""Prepare, submit, poll, retrieve, and parse TogetherAI batch jobs.

Preparation is offline/dry-run by default when `--dry-run` is supplied. Paid
network actions require explicit subcommands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from inference.together_batch import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    download_file,
    is_tiny_smoke_jsonl,
    json_safe,
    parse_batch_results,
    prepare_batch,
    record_tiny_smoke_submission,
    retrieve_batch,
    submit_batch,
    tiny_smoke_safety_guard,
)
from inference.f_reliability_r1_replacement_guard import (  # noqa: E402
    PartitionSequenceError,
    assert_partition_ready_to_submit,
)
from inference.scientific_bakeoff_guard import (  # noqa: E402
    SCIENTIFIC_BAKEOFF_ROOT,
    record_scientific_bakeoff_submission,
    scientific_bakeoff_safety_guard,
)
from inference.target_production_guard import (  # noqa: E402
    TARGET_PRODUCTION_STUDY_ID,
    is_target_production_jsonl,
    record_target_production_submission,
    target_production_safety_guard,
)
from inference.orchinik_domain_confirmation_guard import (  # noqa: E402
    ORCHINIK_V2_ROOT,
    OrchinikDomainConfirmationNotAuthorized,
    orchinik_domain_confirmation_safety_guard,
    record_orchinik_domain_confirmation_submission,
)
from inference.target_g_completion_guard import (  # noqa: E402
    COMPLETION_ROOT as TARGET_G_COMPLETION_ROOT,
    TargetGCompletionNotAuthorized,
    record_target_g_completion_submission,
    target_g_completion_safety_guard,
)
from inference.consensus_exact_guard import (  # noqa: E402
    ConsensusExactNotAuthorized,
    consensus_exact_safety_guard,
    is_consensus_exact_jsonl,
    record_consensus_exact_submission,
)
from inference.together_batch import _under_path  # noqa: E402


def is_orchinik_domain_confirmation_v2_jsonl(jsonl_path: Path) -> bool:
    return _under_path(Path(jsonl_path), ORCHINIK_V2_ROOT)


def is_target_g_completion_jsonl(jsonl_path: Path) -> bool:
    return _under_path(Path(jsonl_path), TARGET_G_COMPLETION_ROOT)


def is_scientific_bakeoff_jsonl(jsonl_path: Path) -> bool:
    return _under_path(Path(jsonl_path), SCIENTIFIC_BAKEOFF_ROOT)


def _manifest_contains_target_rows(manifest_path: Path | None) -> bool:
    """Best-effort detection used only to REFUSE an unguarded submission
    that looks like target-production content -- never used to grant
    access. A JSONL with no --manifest supplied cannot be checked this way
    (it is, by construction, not one of this pipeline's own generated
    scientific manifests) and is left to the generic path unchanged."""
    if manifest_path is None or not Path(manifest_path).exists():
        return False
    import csv

    with open(manifest_path, encoding="utf-8") as f:
        return any(row.get("study_id") == TARGET_PRODUCTION_STUDY_ID for row in csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="write request manifest and JSONL batch input")
    prepare.add_argument("--role", choices=["G", "F"], required=True)
    prepare.add_argument("--requested-model", required=True)
    prepare.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    prepare.add_argument("--max-requests", type=int)
    prepare.add_argument("--successful-results", type=Path, action="append", default=[])
    prepare.add_argument(
        "--consensus-stage-a-success",
        type=Path,
        help="parsed_success.csv from the Consensus Stage A wave; when supplied, prepare emits verified Stage B Consensus requests",
    )
    prepare.add_argument("--dry-run", action="store_true", help="required acknowledgement that no API call should be made")

    submit = sub.add_parser("submit", help="upload JSONL and create Together batch job")
    submit.add_argument("--jsonl", type=Path, required=True)
    submit.add_argument("--metadata-out", type=Path)
    submit.add_argument("--manifest", type=Path, help="request_manifest.csv (required for scientific-bakeoff jsonl paths)")
    submit.add_argument("--phase", help="declared scientific-bakeoff phase name (required for scientific-bakeoff jsonl paths)")
    submit.add_argument(
        "--partition",
        choices=["part1", "part2", "part3", "part4"],
        help="required for phase=f_reliability_r1_replacement -- refuses unless every earlier partition already has a recorded serving-validity PASS",
    )

    validate_smoke = sub.add_parser("validate-smoke-guard", help="validate tiny smoke safety guard without submitting")
    validate_smoke.add_argument("--jsonl", type=Path, required=True)

    status = sub.add_parser("status", help="retrieve Together batch status")
    status.add_argument("--batch-id", required=True)

    retrieve = sub.add_parser("retrieve", help="download Together output/error files by file id")
    retrieve.add_argument("--file-id", required=True)
    retrieve.add_argument("--out", type=Path, required=True)

    parse = sub.add_parser("parse-results", help="parse downloaded batch output JSONL")
    parse.add_argument("--manifest", type=Path, required=True)
    parse.add_argument("--results-jsonl", type=Path, required=True)
    parse.add_argument("--out-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        if not args.dry_run:
            print("prepare only writes local artifacts; pass --dry-run to acknowledge no API call is intended", file=sys.stderr)
            return 2
        result = prepare_batch(
            role=args.role,
            requested_model=args.requested_model,
            output_dir=args.out_dir,
            max_requests=args.max_requests,
            successful_result_paths=args.successful_results,
            consensus_stage_a_success_path=args.consensus_stage_a_success,
        )
    elif args.command == "submit":
        smoke_guard = None
        bakeoff_guard = None
        target_guard = None
        orchinik_guard = None
        target_g_completion_guard = None
        consensus_exact_guard = None
        if is_tiny_smoke_jsonl(args.jsonl):
            try:
                smoke_guard = tiny_smoke_safety_guard(args.jsonl, for_submit=True)
            except RuntimeError as exc:
                print(json.dumps({"smoke_safety_guard": {"submission_allowed": False, "error": str(exc)}}, indent=2))
                return 2
            print(json.dumps({"smoke_safety_guard": smoke_guard}, indent=2, default=str))
        elif is_scientific_bakeoff_jsonl(args.jsonl):
            if not args.manifest or not args.phase:
                print(
                    json.dumps({"scientific_bakeoff_guard": {"submission_allowed": False, "error": "--manifest and --phase are required for scientific-bakeoff jsonl paths"}}, indent=2)
                )
                return 2
            if args.phase == "f_reliability_r1_replacement":
                if not args.partition:
                    print(json.dumps({"scientific_bakeoff_guard": {"submission_allowed": False, "error": "--partition is required for phase=f_reliability_r1_replacement"}}, indent=2))
                    return 2
                try:
                    assert_partition_ready_to_submit(args.partition)
                except PartitionSequenceError as exc:
                    print(json.dumps({"scientific_bakeoff_guard": {"submission_allowed": False, "error": str(exc)}}, indent=2))
                    return 2
            try:
                bakeoff_guard = scientific_bakeoff_safety_guard(args.jsonl, args.manifest, phase=args.phase, for_submit=True)
            except RuntimeError as exc:
                print(json.dumps({"scientific_bakeoff_guard": {"submission_allowed": False, "error": str(exc)}}, indent=2))
                return 2
            print(json.dumps({"scientific_bakeoff_guard": bakeoff_guard}, indent=2, default=str))
        elif is_target_g_completion_jsonl(args.jsonl):
            # Checked BEFORE is_target_production_jsonl: wave1_g_completion/
            # is nested under outputs/target_production/, so the more
            # general check would otherwise wrongly claim it first and
            # require a target_production-declared phase that was never
            # (and must never be) declared for these custom_ids.
            if not args.phase:
                print(json.dumps({"target_g_completion_guard": {"submission_allowed": False, "error": "--phase is required for target G completion jsonl paths"}}, indent=2))
                return 2
            try:
                target_g_completion_guard = target_g_completion_safety_guard(args.jsonl, phase=args.phase, for_submit=True)
            except TargetGCompletionNotAuthorized as exc:
                print(json.dumps({"target_g_completion_guard": {"submission_allowed": False, "error": str(exc)}}, indent=2))
                return 2
            print(json.dumps({"target_g_completion_guard": target_g_completion_guard}, indent=2, default=str))
        elif is_consensus_exact_jsonl(args.jsonl):
            # Checked BEFORE is_target_production_jsonl: consensus_exact/ is
            # nested under outputs/target_production/, same reasoning as
            # is_target_g_completion_jsonl above.
            if not args.phase:
                print(json.dumps({"consensus_exact_guard": {"submission_allowed": False, "error": "--phase is required for Consensus-exact jsonl paths"}}, indent=2))
                return 2
            try:
                consensus_exact_guard = consensus_exact_safety_guard(args.jsonl, phase=args.phase, for_submit=True)
            except ConsensusExactNotAuthorized as exc:
                print(json.dumps({"consensus_exact_guard": {"submission_allowed": False, "error": str(exc)}}, indent=2))
                return 2
            print(json.dumps({"consensus_exact_guard": consensus_exact_guard}, indent=2, default=str))
        elif is_target_production_jsonl(args.jsonl):
            if not args.manifest or not args.phase:
                print(
                    json.dumps({"target_production_guard": {"submission_allowed": False, "error": "--manifest and --phase are required for target-production jsonl paths"}}, indent=2)
                )
                return 2
            try:
                target_guard = target_production_safety_guard(args.jsonl, args.manifest, phase=args.phase, for_submit=True)
            except RuntimeError as exc:
                print(json.dumps({"target_production_guard": {"submission_allowed": False, "error": str(exc)}}, indent=2))
                return 2
            print(json.dumps({"target_production_guard": target_guard}, indent=2, default=str))
        elif is_orchinik_domain_confirmation_v2_jsonl(args.jsonl):
            if not args.phase:
                print(json.dumps({"orchinik_domain_confirmation_guard": {"submission_allowed": False, "error": "--phase is required for Orchinik domain-confirmation v2 jsonl paths"}}, indent=2))
                return 2
            try:
                orchinik_guard = orchinik_domain_confirmation_safety_guard(args.jsonl, phase=args.phase, for_submit=True)
            except OrchinikDomainConfirmationNotAuthorized as exc:
                print(json.dumps({"orchinik_domain_confirmation_guard": {"submission_allowed": False, "error": str(exc)}}, indent=2))
                return 2
            print(json.dumps({"orchinik_domain_confirmation_guard": orchinik_guard}, indent=2, default=str))
        elif _manifest_contains_target_rows(args.manifest):
            # Not under any recognized guarded root, but its own manifest
            # shows target-labeled requests -- refuse rather than fall
            # through to the unguarded generic path. Move it under
            # outputs/target_production/ and resubmit through that guard.
            print(
                json.dumps(
                    {
                        "target_production_guard": {
                            "submission_allowed": False,
                            "error": (
                                f"--manifest {args.manifest} contains study_id='{TARGET_PRODUCTION_STUDY_ID}' rows but "
                                "--jsonl is outside outputs/target_production/; target-production requests must be "
                                "submitted from a target-production-rooted jsonl through target_production_safety_guard"
                            ),
                        }
                    },
                    indent=2,
                )
            )
            return 2
        result = submit_batch(args.jsonl)
        if smoke_guard is not None:
            record_tiny_smoke_submission(smoke_guard, result)
        if bakeoff_guard is not None:
            record_scientific_bakeoff_submission(bakeoff_guard, result)
        if target_guard is not None:
            record_target_production_submission(target_guard, result)
        if orchinik_guard is not None:
            record_orchinik_domain_confirmation_submission(orchinik_guard, result)
        if target_g_completion_guard is not None:
            record_target_g_completion_submission(target_g_completion_guard, result)
        if consensus_exact_guard is not None:
            record_consensus_exact_submission(consensus_exact_guard, result)
        if args.metadata_out:
            args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
            args.metadata_out.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    elif args.command == "validate-smoke-guard":
        try:
            result = tiny_smoke_safety_guard(args.jsonl, for_submit=True)
        except RuntimeError as exc:
            result = {"submission_allowed": False, "error": str(exc)}
            print(json.dumps(result, indent=2))
            return 2
    elif args.command == "status":
        result = retrieve_batch(args.batch_id)
    elif args.command == "retrieve":
        path = download_file(args.file_id, args.out)
        result = {"downloaded": str(path)}
    elif args.command == "parse-results":
        result = parse_batch_results(manifest_path=args.manifest, results_jsonl=args.results_jsonl, output_dir=args.out_dir)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
