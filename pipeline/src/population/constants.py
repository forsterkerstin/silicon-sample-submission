"""src/population/constants.py

Canonical category orderings, RNG-stream management, and the benchmark-schema
loader shared across the population-construction pipeline. Every module reads
category label strings through `load_benchmark_schema()` rather than
hardcoding them, so config/benchmark_schema.yaml is the single source of
truth for label text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

#: Canonical age-band order used for deterministic sorting (assignment of
#: latent_profile_id, controlled-rounding tie-breaking).
AGE_BAND_ORDER: list[str] = ["18-29", "30-44", "45-59", "60+"]

#: Canonical race order used for deterministic sorting.
RACE_ORDER: list[str] = [
    "White / Caucasian",
    "Black / African American",
    "Hispanic / Latino",
    "Asian / Asian American",
    "Other",
]

#: The primary ACS panel only ever produces Male/Female (see
#: other_gender_mode: none in config/population.yaml, section 20).
GENDER_ORDER: list[str] = ["Male", "Female"]

#: Named, independent RNG streams. Every stream is spawned from one master
#: SeedSequence (see spawn_rngs) and never touches the global np.random state.
RNG_STREAM_NAMES: tuple[str, ...] = (
    "pums_selection",
    "party_sampling",
    "roster_ids",
    "ces_diagnostic_split",
)

_REQUIRED_SCHEMA_KEYS = {"moderators", "conditions", "schema_source", "schema_snapshot_date", "schema_status"}
_REQUIRED_MODERATOR_KEYS = {"gender", "age_band", "race", "education", "income", "party"}


def load_benchmark_schema(path: Path | str) -> dict[str, Any]:
    """Load config/benchmark_schema.yaml: the moderator level lists, the 17
    condition labels, and the schema snapshot's provenance metadata.

    Raises ValueError if the file is missing any of the top-level keys the
    rest of the pipeline depends on.
    """
    with open(path, encoding="utf-8") as f:
        schema = yaml.safe_load(f)
    missing = _REQUIRED_SCHEMA_KEYS - schema.keys()
    if missing:
        raise ValueError(f"{path} is missing required top-level keys: {sorted(missing)}")
    missing_moderators = _REQUIRED_MODERATOR_KEYS - schema["moderators"].keys()
    if missing_moderators:
        raise ValueError(f"{path} moderators block is missing: {sorted(missing_moderators)}")
    return schema


def validate_schema_against(schema: dict[str, Any], other: dict[str, Any]) -> None:
    """Optional consistency check (see config/benchmark_schema.yaml header):
    compare this schema snapshot's levels against another parsed source (e.g.
    a locally added codebook or submission_spec.R-derived schema dict) and
    raise ValueError on the first discrepancy found. `other` uses the same
    shape as the loaded schema (a "moderators" dict and/or a "conditions"
    list); keys/moderators absent from `other` are skipped, not flagged.
    """
    for key, levels in schema["moderators"].items():
        other_levels = other.get("moderators", {}).get(key)
        if other_levels is not None and list(other_levels) != list(levels):
            raise ValueError(
                f"moderator '{key}' mismatch between schema snapshot and other source: "
                f"snapshot={levels!r} other={list(other_levels)!r}"
            )
    other_conditions = other.get("conditions")
    if other_conditions is not None and list(other_conditions) != list(schema["conditions"]):
        raise ValueError(
            f"conditions mismatch between schema snapshot and other source: "
            f"snapshot={schema['conditions']!r} other={list(other_conditions)!r}"
        )


def spawn_rngs(
    master_seed: int, names: tuple[str, ...] = RNG_STREAM_NAMES
) -> tuple[dict[str, np.random.Generator], dict[str, list[int]]]:
    """Spawn one independent, named RNG stream per entry in `names` from a
    single master SeedSequence -- numpy's documented pattern for reproducible,
    mutually-independent streams that never touch global np.random state.

    Returns (generators, spawn_keys): `generators[name]` is a ready-to-use
    np.random.Generator; `spawn_keys[name]` is that stream's SeedSequence
    spawn_key (as a plain list of ints), recorded so build_metadata.json can
    document exactly which streams produced a given build.
    """
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(len(names))
    generators = {name: np.random.default_rng(child) for name, child in zip(names, children)}
    spawn_keys = {name: list(child.spawn_key) for name, child in zip(names, children)}
    return generators, spawn_keys
