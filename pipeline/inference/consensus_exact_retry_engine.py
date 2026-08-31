"""Bounded 3-attempt-per-stage retry ledger for the benchmark-exact
Consensus pipeline (inference/consensus_benchmark_exact.py).

Deliberately INDEPENDENT of inference/target_g_retry_engine.py (which
governs standard target-G Wave-1 retries and must remain completely
untouched by this Consensus-only correction) -- this module reimplements
the same small, generic core (attempt list, first-valid-wins, bounded at 3,
engineering-status-only membership) rather than importing or modifying
that file, so standard target-G production is provably unaffected.

A donor's FOUR logical Consensus stages (step1, step2, step3, outcomes) are
tracked as four SEPARATE per-stage ledgers over four SEPARATE (and
successively smaller) universes: step1's universe is all 1,000 Consensus
donors; step2's universe is only donors with a resolved (schema-valid)
step1 response; step3's only donors with a resolved step2 response;
outcomes' only donors with a resolved step3 response. build_stage_ledger is
the same generic function for all four -- it never knows which logical
stage it is building a ledger for.
"""

from __future__ import annotations

from typing import Any

MAX_PRODUCTION_ATTEMPTS_PER_STAGE = 3
SCHEMA_VALID = "SCHEMA_VALID"
SCHEMA_INVALID = "SCHEMA_INVALID"
PROVIDER_ERROR = "PROVIDER_ERROR"
NOT_ATTEMPTED = "NOT_ATTEMPTED"
TERMINAL_VALID_STATUS = SCHEMA_VALID
STEP_NAMES = ("step1", "step2", "step3", "outcomes")


class ConsensusExactLedgerError(RuntimeError):
    """Fail-closed refusal: malformed attempt order, provenance ambiguity,
    or an attempt count exceeding the bound. Never silently resolved."""


def build_stage_ledger(universe_donor_keys: set[str] | list[str], attempt_rounds: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """attempt_rounds: ordered list of REAL rounds, each shaped as
    {"attempt_number": N, "donor_status": {donor_key: status, ...}}.
    Supplied by the caller once that round has been submitted, retrieved,
    and classified -- never invented here. A donor absent from every round
    has attempt_count=0, next_attempt_number=1 (never attempted yet)."""
    universe = set(universe_donor_keys)
    attempts_by_donor: dict[str, list[dict[str, Any]]] = {donor: [] for donor in universe}
    for round_data in attempt_rounds or []:
        n = round_data["attempt_number"]
        for donor, status in round_data["donor_status"].items():
            if donor not in attempts_by_donor:
                raise ConsensusExactLedgerError(f"attempt round {n} references a donor not in this stage's universe: {donor}")
            attempts_by_donor[donor].append({"attempt_number": n, "status": status})

    ledger: dict[str, dict[str, Any]] = {}
    for donor, attempts in attempts_by_donor.items():
        attempts = sorted(attempts, key=lambda a: a["attempt_number"])
        attempt_numbers = [a["attempt_number"] for a in attempts]
        if attempt_numbers != list(range(1, len(attempts) + 1)):
            raise ConsensusExactLedgerError(f"malformed attempt order for donor {donor}: {attempt_numbers} (must be contiguous starting at 1)")
        if len(attempts) > MAX_PRODUCTION_ATTEMPTS_PER_STAGE:
            raise ConsensusExactLedgerError(f"donor {donor} has {len(attempts)} attempts, exceeds MAX_PRODUCTION_ATTEMPTS_PER_STAGE={MAX_PRODUCTION_ATTEMPTS_PER_STAGE}")
        valid_attempts = [a["attempt_number"] for a in attempts if a["status"] == TERMINAL_VALID_STATUS]
        if len(valid_attempts) > 1:
            raise ConsensusExactLedgerError(f"donor {donor} has more than one schema-valid response ({valid_attempts}) -- provenance ambiguity")
        resolved = bool(valid_attempts)
        resolved_attempt = valid_attempts[0] if valid_attempts else None
        attempt_count = len(attempts)
        next_attempt_number = None if resolved or attempt_count >= MAX_PRODUCTION_ATTEMPTS_PER_STAGE else attempt_count + 1
        ledger[donor] = {"attempts": attempts, "resolved": resolved, "resolved_attempt": resolved_attempt, "attempt_count": attempt_count, "next_attempt_number": next_attempt_number}
    return ledger


def pending_donors(ledger: dict[str, dict[str, Any]]) -> list[str]:
    """Donors needing their next attempt right now for THIS stage -- not
    resolved, attempt_count < MAX. Never reads a scientific response
    value; only status."""
    return sorted(donor for donor, entry in ledger.items() if not entry["resolved"] and entry["attempt_count"] < MAX_PRODUCTION_ATTEMPTS_PER_STAGE)


def resolved_donors(ledger: dict[str, dict[str, Any]]) -> set[str]:
    return {donor for donor, entry in ledger.items() if entry["resolved"]}


def final_row_eligible_donors(step1_ledger: dict, step2_ledger: dict, step3_ledger: dict, outcomes_ledger: dict) -> set[str]:
    """A donor's Consensus row is eligible for the final assembled dataset
    ONLY if all four sequential stages have a resolved (first-valid)
    response. Purely a set intersection over engineering resolution
    status -- no scientific value is read here."""
    return resolved_donors(step1_ledger) & resolved_donors(step2_ledger) & resolved_donors(step3_ledger) & resolved_donors(outcomes_ledger)
