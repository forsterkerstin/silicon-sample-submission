"""Validate G/F persona panels and build pre-inference skeleton artifacts.

This script is deliberately limited to profile/population validation. It does
not call any LLM provider and does not fill outcome values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import deque
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_ROOT.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
if str(PIPELINE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT / "src"))

import survey_content as sc  # noqa: E402
from calibration.study_population import largest_remainder_allocations  # noqa: E402

PROFILE_PATH = PIPELINE_ROOT / "data" / "processed" / "population" / "profiles_core_1000.csv"
ROSTER_PATH = PIPELINE_ROOT / "data" / "processed" / "population" / "simulation_roster_17000.csv"
SCHEMA_PATH = PIPELINE_ROOT / "config" / "benchmark_schema.yaml"
QUOTA_GENDER_AGE_1000 = PIPELINE_ROOT / "config" / "quota_gender_age_1000.csv"
QUOTA_GENDER_RACE_1000 = PIPELINE_ROOT / "config" / "quota_gender_race_1000.csv"
QUOTA_GENDER_AGE_500 = PIPELINE_ROOT / "config" / "quota_gender_age_500.csv"
QUOTA_GENDER_RACE_500 = PIPELINE_ROOT / "config" / "quota_gender_race_500.csv"
GENERATED_DIR = PIPELINE_ROOT / "data" / "generated"
OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "persona_validation"

G_MASTER_PATH = GENERATED_DIR / "g_personas_master.csv"
F_PANEL_PATH = GENERATED_DIR / "f_target_panel.csv"
DESIGN_SKELETON_PATH = GENERATED_DIR / "tier1_design_skeleton.csv"
SUBMISSION_SKELETON_PATH = GENERATED_DIR / "tier1_submission_skeleton.csv"

REQUIRED_G_FIELDS = [
    "donor_key",
    "age",
    "age_band",
    "gender",
    "race",
    "education",
    "income",
    "party",
    "state",
]


class ValidationFailure(RuntimeError):
    """Raised when a persona validation hard check fails."""


class Dinic:
    def __init__(self, n_nodes: int) -> None:
        self.graph: list[list[list[int]]] = [[] for _ in range(n_nodes)]

    def add_edge(self, src: int, dst: int, cap: int) -> None:
        forward = [dst, cap, len(self.graph[dst])]
        reverse = [src, 0, len(self.graph[src])]
        self.graph[src].append(forward)
        self.graph[dst].append(reverse)

    def max_flow(self, source: int, sink: int) -> int:
        flow = 0
        n_nodes = len(self.graph)
        while True:
            level = [-1] * n_nodes
            level[source] = 0
            queue: deque[int] = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge[1] > 0 and level[edge[0]] < 0:
                        level[edge[0]] = level[node] + 1
                        queue.append(edge[0])
            if level[sink] < 0:
                return flow
            it = [0] * n_nodes

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while it[node] < len(self.graph[node]):
                    edge = self.graph[node][it[node]]
                    if edge[1] > 0 and level[node] + 1 == level[edge[0]]:
                        pushed = send(edge[0], min(amount, edge[1]))
                        if pushed:
                            edge[1] -= pushed
                            self.graph[edge[0]][edge[2]][1] += pushed
                            return pushed
                    it[node] += 1
                return 0

            while True:
                pushed = send(source, 10**9)
                if not pushed:
                    break
                flow += pushed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=PROFILE_PATH)
    parser.add_argument("--roster", type=Path, default=ROSTER_PATH)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    return parser.parse_args()


def load_schema(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fail_if(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        failures.append(message)


def deterministic_hash(value: str, seed: str = "f-panel-v1") -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def official_submission_columns(schema: dict[str, Any]) -> list[str]:
    trust_items = list(sc.OUTCOME_COMPOSITES["trust_multidimensional"][1])
    outcomes = list(sc.OUTCOME_COMPOSITES.keys())
    non_trust_outcomes = [outcome for outcome in outcomes if outcome != "trust_multidimensional"]
    return [
        "profile_id",
        "condition",
        *schema["moderators"].keys(),
        "trust_multidimensional",
        *trust_items,
        *non_trust_outcomes,
    ]


def raw_item_columns() -> list[str]:
    return [item["target_label"] for item in sc.load_items()]


def outcome_columns() -> list[str]:
    return list(sc.OUTCOME_COMPOSITES.keys())


def build_g_master(core: pd.DataFrame, *, source_dataset: str = "ACS PUMS 2024 + CES-derived party assignment") -> pd.DataFrame:
    abbr_to_state = {abbr: name for name, abbr in sc.STATE_NAME_TO_ABBR.items()}
    master = pd.DataFrame(
        {
            "donor_key": core["latent_profile_id"],
            "source_dataset": source_dataset,
            "source_row_id": core.get("donor_id", pd.Series([pd.NA] * len(core))),
            "source_weight": core.get("pums_person_weight", pd.Series([pd.NA] * len(core))),
            "age": core["age"],
            "age_band": core["age_band"],
            "gender": core["gender"],
            "race": core["race"],
            "education": core["education"],
            "income": core["income"],
            "party": core["party"],
            "state": core["state_abbr"].map(abbr_to_state),
            "state_abbr": core["state_abbr"],
            "state_fips": core.get("state_fips", pd.Series([pd.NA] * len(core))),
            "donor_id": core.get("donor_id", pd.Series([pd.NA] * len(core))),
            "persona_generation_version": "g1000_population_yaml_2026-08-24",
            "population_design": "G final: 1000 latent U.S.-adult donors; one control and 16 interventions",
        }
    )
    optional = [
        "political_ideology",
        "religion",
        "party_prob_republican",
        "party_prob_democrat",
        "party_prob_independent",
        "party_prob_other",
    ]
    for col in optional:
        if col in core.columns:
            master[col] = core[col]
    return master


def solve_gender_flow(
    donors: pd.DataFrame,
    age_targets: dict[str, int],
    race_targets: dict[str, int],
    age_levels: list[str],
    race_levels: list[str],
) -> dict[tuple[str, str], int]:
    source = 0
    age_offset = 1
    race_offset = age_offset + len(age_levels)
    sink = race_offset + len(race_levels)
    flow = Dinic(sink + 1)
    edge_refs: dict[tuple[str, str], list[int]] = {}

    for i, age in enumerate(age_levels):
        flow.add_edge(source, age_offset + i, int(age_targets.get(age, 0)))
    for i, age in enumerate(age_levels):
        for j, race in enumerate(race_levels):
            cap = int(((donors["age_band"] == age) & (donors["race"] == race)).sum())
            edge_refs[(age, race)] = [age_offset + i, len(flow.graph[age_offset + i]), cap]
            flow.add_edge(age_offset + i, race_offset + j, cap)
    for j, race in enumerate(race_levels):
        flow.add_edge(race_offset + j, sink, int(race_targets.get(race, 0)))

    required = sum(age_targets.values())
    achieved = flow.max_flow(source, sink)
    if achieved != required:
        raise ValidationFailure(
            f"cannot construct exact F target panel for one gender: required {required}, achieved {achieved}"
        )

    selected_counts: dict[tuple[str, str], int] = {}
    for cell, (node, edge_index, cap) in edge_refs.items():
        residual = flow.graph[node][edge_index][1]
        selected_counts[cell] = cap - residual
    return selected_counts


def build_f_panel(g_master: pd.DataFrame, schema: dict[str, Any], *, quota_age: pd.DataFrame | None = None, quota_race: pd.DataFrame | None = None) -> pd.DataFrame:
    """quota_age/quota_race default to the frozen 500-total benchmark quota
    CSVs (byte-identical for every existing caller, which omits them). A
    caller MAY pass its own gender x age_band / gender x race target tables
    -- e.g. a donor-source amendment that rescales the published totals to
    reserve some slots for a gender level the benchmark quota never
    constrains -- as long as they cover the same age/race levels; the
    max-flow solve itself (solve_gender_flow) is otherwise unchanged."""
    if quota_age is None:
        quota_age = pd.read_csv(QUOTA_GENDER_AGE_500)
    if quota_race is None:
        quota_race = pd.read_csv(QUOTA_GENDER_RACE_500)
    age_levels = list(schema["moderators"]["age_band"])
    race_levels = list(schema["moderators"]["race"])

    selected_parts = []
    for gender in sorted(set(quota_age["gender"]) | set(quota_race["gender"])):
        donors = g_master[g_master["gender"] == gender].copy()
        age_targets = dict(zip(quota_age.loc[quota_age["gender"] == gender, "age_band"], quota_age.loc[quota_age["gender"] == gender, "target_n"]))
        race_targets = dict(zip(quota_race.loc[quota_race["gender"] == gender, "race"], quota_race.loc[quota_race["gender"] == gender, "target_n"]))
        counts = solve_gender_flow(donors, age_targets, race_targets, age_levels, race_levels)
        donors["_hash"] = donors["donor_key"].map(lambda x: deterministic_hash(str(x)))
        donors = donors.sort_values(["_hash", "donor_key"])
        for (age, race), n_cell in counts.items():
            if n_cell:
                selected_parts.append(donors[(donors["age_band"] == age) & (donors["race"] == race)].head(n_cell))

    panel = pd.concat(selected_parts, ignore_index=True).drop(columns=["_hash"])
    panel = panel.sort_values("donor_key").reset_index(drop=True)
    panel.insert(0, "f_profile_id", [f"F{i:04d}" for i in range(1, len(panel) + 1)])
    panel["f_selection_method"] = "deterministic subset of G donors matching benchmark F 500 cross quotas"
    panel["target_population"] = "same U.S.-adult target population as G benchmark panel"
    return panel


def rescaled_quota_reserving_other(quota_age_path: Path, quota_race_path: Path, *, n_other: int, tie_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rescale the published gender x age_band / gender x race quota margins
    to reserve exactly `n_other` slots for a gender level the quota tables
    themselves never define (e.g. CES's donor-frame Other representation --
    NOT a benchmark quota), preserving relative Male/Female proportions as
    closely as possible via the repository's own largest_remainder_allocations
    (reused unmodified, applied in two stages: gender totals, then each
    gender's age/race weights to its new total -- same convention already
    used to freeze the CES-1000/F-500 target population).

    n_other=0 is an EXACT no-op (verified: largest_remainder_allocations at
    the published total reproduces the published per-cell values exactly),
    so this is a strict generalization of the historical PUMS path, not a
    new rule for it -- the published quota_gender_age_1000.csv/
    quota_gender_race_1000.csv (and their 500-row counterparts) remain the
    sole source of the Male/Female targets in both cases.
    """
    quota_age = pd.read_csv(quota_age_path)
    quota_race = pd.read_csv(quota_race_path)
    gender_totals = quota_age.groupby("gender")["target_n"].sum()
    new_total = int(gender_totals.sum()) - n_other
    new_gender_totals = largest_remainder_allocations(gender_totals, n_f=new_total, tie_key=f"{tie_prefix}_gender_totals")
    age_rows, race_rows = [], []
    for gender in ("Male", "Female"):
        g_total = int(new_gender_totals[gender])
        age_w = quota_age.loc[quota_age["gender"] == gender].set_index("age_band")["target_n"]
        race_w = quota_race.loc[quota_race["gender"] == gender].set_index("race")["target_n"]
        new_age = largest_remainder_allocations(age_w, n_f=g_total, tie_key=f"{tie_prefix}_age_{gender}")
        new_race = largest_remainder_allocations(race_w, n_f=g_total, tie_key=f"{tie_prefix}_race_{gender}")
        for age_band, n in new_age.items():
            age_rows.append({"gender": gender, "age_band": age_band, "target_n": int(n)})
        for race, n in new_race.items():
            race_rows.append({"gender": gender, "race": race, "target_n": int(n)})
    return pd.DataFrame(age_rows), pd.DataFrame(race_rows)


def build_f_panel_with_other(g_master: pd.DataFrame, schema: dict[str, Any], quota_age_mf: pd.DataFrame, quota_race_mf: pd.DataFrame, *, n_other_f: int) -> pd.DataFrame:
    """Male/Female subset via the unmodified max-flow build_f_panel against
    the rescaled Male/Female-only targets, plus (if n_other_f>0) the SAME
    deterministic_hash sort-and-take convention build_f_panel already uses
    for every (age, race) cell, applied to the G master's own Other-gender
    donor pool -- not a fresh draw, not an invented quota for a level the
    F(500) quota tables never define. n_other_f=0 makes this identical to
    calling build_f_panel directly."""
    male_female = build_f_panel(g_master, schema, quota_age=quota_age_mf, quota_race=quota_race_mf).drop(columns=["f_profile_id"])
    if n_other_f == 0:
        panel = male_female
    else:
        other_pool = g_master[g_master["gender"] == "Other"].copy()
        other_pool["_hash"] = other_pool["donor_key"].map(lambda x: deterministic_hash(str(x)))
        other_pool = other_pool.sort_values(["_hash", "donor_key"]).head(n_other_f).drop(columns=["_hash"])
        other_pool["f_selection_method"] = "deterministic subset of G Other-gender donors (same deterministic_hash convention as every Male/Female cell); no age/race quota exists or is enforced for this level"
        other_pool["target_population"] = "same U.S.-adult target population as G benchmark panel"
        panel = pd.concat([male_female, other_pool], ignore_index=True, sort=False)
    if panel["donor_key"].duplicated().any():
        raise ValidationFailure("duplicate donor_key across Male/Female + Other F-panel selection")
    panel = panel.sort_values("donor_key").reset_index(drop=True)
    panel.insert(0, "f_profile_id", [f"F{i:04d}" for i in range(1, len(panel) + 1)])
    return panel


def expand_skeleton(master: pd.DataFrame, schema: dict[str, Any], *, internal: bool) -> pd.DataFrame:
    rows = []
    conditions = list(schema["conditions"])
    official_cols = official_submission_columns(schema)
    raw_cols = raw_item_columns()
    outcomes = outcome_columns()
    for condition in conditions:
        block = master.copy()
        block["condition"] = condition
        block["condition_replicate"] = 1
        block["profile_id"] = block["donor_key"].astype(str) + "__" + condition + "__R1"
        rows.append(block)
    expanded = pd.concat(rows, ignore_index=True)

    response_cols = list(dict.fromkeys([*raw_cols, *outcomes]))
    for col in response_cols:
        expanded[col] = pd.NA

    if internal:
        front = [
            "profile_id",
            "donor_key",
            "condition",
            "condition_replicate",
            "age",
            "age_band",
            "gender",
            "race",
            "education",
            "income",
            "party",
            "state",
            "state_abbr",
            "source_dataset",
            "source_row_id",
            "source_weight",
            "persona_generation_version",
        ]
        return expanded[[col for col in front if col in expanded.columns] + response_cols]
    return expanded[official_cols]


def category_mapping_audit(master: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for variable, allowed in schema["moderators"].items():
        for value, n in master[variable].value_counts(dropna=False).sort_index().items():
            rows.append(
                {
                    "variable": variable,
                    "raw_source_value": value,
                    "benchmark_value": value,
                    "n": int(n),
                    "mapping_status": "exact" if pd.notna(value) and value in allowed else "invalid_or_missing",
                    "source": "profiles_core_1000.csv recoded category",
                }
            )
    for age, rows_for_age in master.groupby("age"):
        age_band = rows_for_age["age_band"].iloc[0]
        rows.append(
            {
                "variable": "age_to_age_band",
                "raw_source_value": int(age),
                "benchmark_value": age_band,
                "n": int(len(rows_for_age)),
                "mapping_status": "exact",
                "source": "age recode in profiles_core_1000.csv",
            }
        )
    for state, n in master["state_abbr"].value_counts(dropna=False).sort_index().items():
        rows.append(
            {
                "variable": "state_abbr",
                "raw_source_value": state,
                "benchmark_value": state,
                "n": int(n),
                "mapping_status": "exact" if state in set(sc.STATE_NAME_TO_ABBR.values()) else "invalid_or_missing",
                "source": "ACS PUMS state FIPS to USPS abbreviation",
            }
        )
    return pd.DataFrame(rows)


def quota_diagnostics(panel: pd.DataFrame, quota: pd.DataFrame, group_cols: list[str], target_total_name: str) -> pd.DataFrame:
    actual = panel.groupby(group_cols).size().reset_index(name="actual_count")
    merged = quota.merge(actual, on=group_cols, how="left").fillna({"actual_count": 0})
    target_total = float(merged["target_n"].sum())
    actual_total = float(len(panel))
    merged["target_count_18000"] = merged["target_n"] / target_total * 18_000
    merged[f"target_count_{target_total_name}"] = merged["target_n"]
    merged["actual_count"] = merged["actual_count"].astype(int)
    merged["actual_proportion"] = merged["actual_count"] / actual_total
    merged["target_proportion"] = merged["target_n"] / target_total
    merged["difference_pp"] = 100 * (merged["actual_proportion"] - merged["target_proportion"])
    merged["abs_difference_pp"] = merged["difference_pp"].abs()
    merged["exact_count_match"] = merged["actual_count"] == merged["target_n"]
    return merged.sort_values("abs_difference_pp", ascending=False).reset_index(drop=True)


def condition_balance_audit(skeleton: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    rows = []
    moderators = list(schema["moderators"].keys()) + ["state_abbr"]
    control = skeleton[skeleton["condition"] == "control"]
    n_control = len(control)
    for variable in moderators:
        control_counts = control[variable].value_counts(dropna=False).to_dict()
        for condition, part in skeleton.groupby("condition", sort=False):
            counts = part[variable].value_counts(dropna=False).to_dict()
            levels = sorted(set(control_counts) | set(counts), key=lambda x: str(x))
            for level in levels:
                count = int(counts.get(level, 0))
                control_count = int(control_counts.get(level, 0))
                rows.append(
                    {
                        "condition": condition,
                        "variable": variable,
                        "level": level,
                        "actual_count": count,
                        "control_count": control_count,
                        "actual_proportion": count / len(part),
                        "control_proportion": control_count / n_control,
                        "count_difference_vs_control": count - control_count,
                        "proportion_difference_vs_control": count / len(part) - control_count / n_control,
                    }
                )
    return pd.DataFrame(rows)


def nonquota_distribution_audit(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    variables = ["education", "income", "party"]
    for variable in variables:
        weighted = master.groupby(variable)["source_weight"].sum() if "source_weight" in master else pd.Series(dtype=float)
        total_weight = float(weighted.sum()) if len(weighted) else 0.0
        counts = master[variable].value_counts(dropna=False)
        for level, count in counts.sort_index().items():
            rows.append(
                {
                    "variable": variable,
                    "level": level,
                    "selected_count": int(count),
                    "selected_proportion": count / len(master),
                    "weighted_selected_proportion": float(weighted.get(level, 0.0) / total_weight) if total_weight else pd.NA,
                    "source_weight_available": "source_weight" in master,
                    "source_proportion": pd.NA,
                    "difference_pp": pd.NA,
                    "comparison_status": "full source universe not materialized in lightweight validation",
                }
            )
    return pd.DataFrame(rows)


def joint_distribution_audit(master: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("gender", "education"),
        ("age_band", "education"),
        ("race", "education"),
        ("party", "education"),
        ("gender", "income"),
        ("age_band", "income"),
        ("race", "income"),
        ("party", "income"),
        ("state_abbr", "party"),
    ]
    rows = []
    for left, right in pairs:
        counts = master.groupby([left, right]).size().reset_index(name="selected_count")
        for row in counts.to_dict("records"):
            rows.append(
                {
                    "joint_variables": f"{left} x {right}",
                    "level_1": row[left],
                    "level_2": row[right],
                    "selected_count": int(row["selected_count"]),
                    "selected_proportion": row["selected_count"] / len(master),
                    "source_proportion": pd.NA,
                    "difference_pp": pd.NA,
                    "comparison_status": "full source universe not materialized in lightweight validation",
                }
            )
    return pd.DataFrame(rows)


def state_audit(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, count in master["state_abbr"].value_counts().sort_index().items():
        stimulus = sc.get_condition_stimulus("Extreme weather predictions", state)
        rows.append(
            {
                "state_abbr": state,
                "state": master.loc[master["state_abbr"] == state, "state"].iloc[0],
                "selected_count": int(count),
                "selected_proportion": count / len(master),
                "state_source": "ACS PUMS state_fips recoded to state_abbr",
                "extreme_weather_stimulus_available": bool(str(stimulus).strip()),
                "source_proportion": pd.NA,
                "comparison_status": "full source universe not materialized in lightweight validation",
            }
        )
    return pd.DataFrame(rows)


def duplication_audit(master: pd.DataFrame, roster: pd.DataFrame, design: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"object": "g_personas_master", "key": "donor_key", "rows": len(master), "unique": master["donor_key"].nunique(), "duplicates": int(master["donor_key"].duplicated().sum())},
            {"object": "g_personas_master", "key": "source_row_id", "rows": len(master), "unique": master["source_row_id"].nunique(), "duplicates": int(master["source_row_id"].duplicated().sum())},
            {"object": "simulation_roster_17000", "key": "profile_id", "rows": len(roster), "unique": roster["profile_id"].nunique(), "duplicates": int(roster["profile_id"].duplicated().sum())},
            {"object": "tier1_design_skeleton", "key": "profile_id", "rows": len(design), "unique": design["profile_id"].nunique(), "duplicates": int(design["profile_id"].duplicated().sum())},
            {"object": "tier1_submission_skeleton", "key": "profile_id", "rows": len(submission), "unique": submission["profile_id"].nunique(), "duplicates": int(submission["profile_id"].duplicated().sum())},
        ]
    )


def existing_persona_files() -> pd.DataFrame:
    rows = []
    roots = [PIPELINE_ROOT / "data", PIPELINE_ROOT / "outputs", PIPELINE_ROOT / "config", REPO_ROOT / "predictions"]
    terms = ("profile", "persona", "roster", "quota", "skeleton", "submission", "audit")
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not any(term in path.name.lower() for term in terms):
                continue
            rel = path.relative_to(REPO_ROOT)
            status = "reference_or_output"
            if path == PROFILE_PATH:
                status = "active_g_source"
            elif path == ROSTER_PATH:
                status = "active_g_roster"
            elif path.name in {"profiles_core_500.csv", "simulation_roster_18000.csv"}:
                status = "stale_or_removed_design"
            elif path in {G_MASTER_PATH, F_PANEL_PATH, DESIGN_SKELETON_PATH, SUBMISSION_SKELETON_PATH}:
                status = "generated_by_persona_validator"
            row_count = pd.NA
            columns = pd.NA
            if path.suffix == ".csv":
                try:
                    row_count = max(sum(1 for _ in open(path, encoding="utf-8")) - 1, 0)
                    columns = "|".join(pd.read_csv(path, nrows=0).columns)
                except Exception as exc:  # noqa: BLE001
                    columns = f"unreadable: {exc}"
            rows.append(
                {
                    "path": str(rel),
                    "status": status,
                    "suffix": path.suffix,
                    "row_count": row_count,
                    "columns": columns,
                }
            )
    return pd.DataFrame(rows)


def write_plots(output_dir: Path, master: pd.DataFrame, g_age: pd.DataFrame, g_race: pd.DataFrame, f_age: pd.DataFrame, f_race: pd.DataFrame, condition_balance: pd.DataFrame, joint: pd.DataFrame) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib.pyplot as plt

    plot_paths: list[str] = []

    def save_current(name: str) -> None:
        path = output_dir / name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        try:
            plot_paths.append(str(path.relative_to(REPO_ROOT)))
        except ValueError:
            # output_dir redirected outside REPO_ROOT (e.g. a test tmp_path):
            # this list is only ever used for the human-readable report text,
            # so fall back to the absolute path rather than crash.
            plot_paths.append(str(path))

    for filename, df, title in [
        ("target_vs_actual_gender_age.png", g_age, "G gender x age"),
        ("target_vs_actual_gender_race.png", g_race, "G gender x race"),
        ("f_target_vs_actual_gender_age.png", f_age, "F gender x age"),
        ("f_target_vs_actual_gender_race.png", f_race, "F gender x race"),
    ]:
        labels = df.iloc[:, 0].astype(str) + " / " + df.iloc[:, 1].astype(str)
        x = range(len(df))
        plt.figure(figsize=(10, 4))
        plt.bar([i - 0.2 for i in x], df["actual_count"], width=0.4, label="actual")
        plt.bar([i + 0.2 for i in x], df["target_n"], width=0.4, label="target")
        plt.xticks(list(x), labels, rotation=45, ha="right")
        plt.title(title)
        plt.legend()
        save_current(filename)

    for filename, df, title in [
        ("error_gender_age.png", g_age, "G gender x age error"),
        ("error_gender_race.png", g_race, "G gender x race error"),
    ]:
        labels = df.iloc[:, 0].astype(str) + " / " + df.iloc[:, 1].astype(str)
        plt.figure(figsize=(10, 4))
        plt.bar(labels, df["difference_pp"])
        plt.xticks(rotation=45, ha="right")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.title(title)
        save_current(filename)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, variable in zip(axes, ["education", "income", "party"]):
        counts = master[variable].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(variable)
        ax.tick_params(axis="x", labelrotation=45)
    save_current("moderator_marginals.png")

    counts = master["state_abbr"].value_counts().sort_index()
    plt.figure(figsize=(14, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("State distribution")
    save_current("state_distribution.png")

    subset = condition_balance.groupby("condition")["count_difference_vs_control"].apply(lambda x: x.abs().max())
    plt.figure(figsize=(12, 4))
    plt.bar(subset.index.astype(str), subset.values)
    plt.xticks(rotation=70, ha="right")
    plt.title("Maximum condition imbalance vs control")
    save_current("condition_balance.png")

    top_joint = joint.sort_values("selected_count", ascending=False).head(25)
    plt.figure(figsize=(12, 4))
    labels = top_joint["joint_variables"] + ": " + top_joint["level_1"].astype(str) + "/" + top_joint["level_2"].astype(str)
    plt.bar(labels, top_joint["selected_count"])
    plt.xticks(rotation=70, ha="right")
    plt.title("Largest selected joint cells")
    save_current("joint_distribution_checks.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, variable in zip(axes, ["education", "income", "party"]):
        weighted = master.groupby(variable)["source_weight"].sum()
        weighted = weighted / weighted.sum()
        unweighted = master[variable].value_counts(normalize=True).sort_index()
        labels = list(unweighted.index.astype(str))
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], [unweighted.get(label, 0) for label in labels], width=0.4, label="selected")
        ax.bar([i + 0.2 for i in x], [weighted.get(label, 0) for label in labels], width=0.4, label="selected weighted")
        ax.set_title(variable)
        ax.set_xticks(list(x), labels, rotation=45, ha="right")
    axes[0].legend()
    save_current("source_vs_selected_nonquota_demographics.png")

    return plot_paths


def write_report(output_dir: Path, summary: dict[str, Any], warnings: list[str], failures: list[str], plot_paths: list[str]) -> None:
    lines = [
        "# Persona Validation Report",
        "",
        "## Status",
        "",
        f"- Hard failures: {len(failures)}",
        f"- Warnings: {len(warnings)}",
        f"- G donors: {summary['n_g_donors']}",
        f"- G design rows: {summary['n_g_design_rows']}",
        f"- F target profiles: {summary['n_f_profiles']}",
        "",
        "## Authoritative population design",
        "",
        "- G: 1000 unique latent U.S.-adult donors, reused across control plus 16 interventions for 17000 rows.",
        "- F: 500 unique forecasting profiles drawn from the same U.S.-adult target population as a deterministic subset of G matching the F cross-quota files.",
        "- Quotas used: gender x age and gender x race; no full gender x age x race quota cube was found or used.",
        "- State comes from ACS PUMS state FIPS recoded to USPS abbreviation and full state name.",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] or ["- None"])
    lines.extend(["", "## Failures", ""])
    lines.extend([f"- {failure}" for failure in failures] or ["- None"])
    lines.extend(
        [
            "",
            "## Generated artifacts",
            "",
            f"- `pipeline/data/generated/g_personas_master.csv`",
            f"- `pipeline/data/generated/f_target_panel.csv`",
            f"- `pipeline/data/generated/tier1_design_skeleton.csv`",
            f"- `pipeline/data/generated/tier1_submission_skeleton.csv`",
            f"- `pipeline/outputs/persona_validation/category_mapping_audit.csv`",
            f"- `pipeline/outputs/persona_validation/quota_diagnostics_gender_age.csv`",
            f"- `pipeline/outputs/persona_validation/quota_diagnostics_gender_race.csv`",
            f"- `pipeline/outputs/persona_validation/f_quota_diagnostics_gender_age.csv`",
            f"- `pipeline/outputs/persona_validation/f_quota_diagnostics_gender_race.csv`",
            f"- `pipeline/outputs/persona_validation/source_nonquota_distribution_audit.csv`",
            f"- `pipeline/outputs/persona_validation/joint_distribution_audit.csv`",
            f"- `pipeline/outputs/persona_validation/state_audit.csv`",
            f"- `pipeline/outputs/persona_validation/duplication_audit.csv`",
            f"- `pipeline/outputs/persona_validation/condition_balance_audit.csv`",
            f"- `pipeline/outputs/persona_validation/existing_persona_files.csv`",
            "",
            "## Plots",
            "",
        ]
    )
    lines.extend([f"- `{path}`" for path in plot_paths] or ["- Plot generation unavailable"])
    lines.extend(
        [
            "",
            "## Five-minute pre-inference checklist",
            "",
            "- Confirm `g_personas_master.csv` has 1000 unique donor_key values and no outcome columns.",
            "- Confirm `f_target_panel.csv` has 500 unique donor_key values and uses the same U.S.-adult target population.",
            "- Confirm the two G quota diagnostics have zero absolute percentage-point error.",
            "- Confirm condition balance has zero demographic/state drift across all 17 conditions.",
            "- Confirm the skeleton outcome/raw-item fields are blank, not zeros or imputed values.",
            "- Confirm no stale 500-profile G or 18000-row roster file is selected by any active script/config.",
            "",
            "## Self-validation command",
            "",
            "```bash",
            "python pipeline/scripts/validate_personas.py",
            "```",
            "",
        ]
    )
    (output_dir / "persona_validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    generated_dir = args.generated_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    warnings: list[str] = []

    schema = load_schema(args.schema)
    core = pd.read_csv(args.profiles)
    roster = pd.read_csv(args.roster)

    # The CES donor-source amendment reserves 8/1000 (G) and 4/500 (F) slots
    # for genuine Other-gender respondents -- a donor-frame representation
    # constraint, NOT a benchmark quota (the quota CSVs never define an
    # Other row). n_other_g is read directly from core, never assumed, so
    # this generalizes cleanly: n_other_g=0 (the historical PUMS core)
    # makes rescaled_quota_reserving_other an exact no-op, reproducing the
    # published quota_gender_age_1000.csv/quota_gender_race_1000.csv values
    # unchanged -- verified, not just asserted (tests/scripts/test_validate_personas_ces_aware.py).
    n_other_g = int((core["gender"] == "Other").sum())
    source_dataset = "CES 2024 Common Content" if n_other_g > 0 else "ACS PUMS 2024 + CES-derived party assignment"
    quota_age_g, quota_race_g = rescaled_quota_reserving_other(QUOTA_GENDER_AGE_1000, QUOTA_GENDER_RACE_1000, n_other=n_other_g, tie_prefix="validate_personas_g")
    n_other_f = round(n_other_g / 1000 * 500) if n_other_g else 0
    quota_age_f, quota_race_f = rescaled_quota_reserving_other(QUOTA_GENDER_AGE_500, QUOTA_GENDER_RACE_500, n_other=n_other_f, tie_prefix="validate_personas_f")

    if not (REPO_ROOT / "data" / "quotas_18000.csv").exists():
        warnings.append("data/quotas_18000.csv not found; using pipeline/config 1000 and 500 cross-quota CSVs as the operative benchmark quota sources.")

    old_profile = PIPELINE_ROOT / "data" / "processed" / "population" / "profiles_core_500.csv"
    old_roster = PIPELINE_ROOT / "data" / "processed" / "population" / "simulation_roster_18000.csv"
    fail_if(old_profile.exists(), "stale profiles_core_500.csv exists in active processed population directory", failures)
    fail_if(old_roster.exists(), "stale simulation_roster_18000.csv exists in active processed population directory", failures)

    g_master = build_g_master(core, source_dataset=source_dataset)
    g_master.to_csv(generated_dir / G_MASTER_PATH.name, index=False)
    f_panel = build_f_panel_with_other(g_master, schema, quota_age_f, quota_race_f, n_other_f=n_other_f)
    f_panel.to_csv(generated_dir / F_PANEL_PATH.name, index=False)
    design = expand_skeleton(g_master, schema, internal=True)
    design.to_csv(generated_dir / DESIGN_SKELETON_PATH.name, index=False)
    submission = expand_skeleton(g_master, schema, internal=False)
    submission.to_csv(generated_dir / SUBMISSION_SKELETON_PATH.name, index=False)

    fail_if(len(g_master) != 1000, f"G master has {len(g_master)} rows, expected 1000", failures)
    fail_if(g_master["donor_key"].nunique() != 1000, "G donor_key is not unique", failures)
    fail_if(g_master[REQUIRED_G_FIELDS].isna().any().any(), "G master has missing required donor fields", failures)
    fail_if((g_master["age"] < 18).any(), "G master contains under-18 donors", failures)
    fail_if(set(g_master["state_abbr"]) - set(sc.STATE_NAME_TO_ABBR.values()), "G master contains invalid state_abbr values", failures)
    for variable, allowed in schema["moderators"].items():
        bad = sorted(set(g_master[variable].dropna().astype(str)) - set(allowed))
        fail_if(bool(bad), f"G master {variable} contains invalid levels: {bad}", failures)
    master_outcome_like = sorted((set(raw_item_columns()) | set(outcome_columns())) & set(g_master.columns))
    fail_if(bool(master_outcome_like), f"G master contains outcome/raw item columns: {master_outcome_like}", failures)

    fail_if(len(f_panel) != 500, f"F target panel has {len(f_panel)} rows, expected 500", failures)
    fail_if(f_panel["donor_key"].nunique() != 500, "F target panel donor_key is not unique", failures)
    fail_if(not set(f_panel["donor_key"]).issubset(set(g_master["donor_key"])), "F target panel includes donors outside G master", failures)
    fail_if(int((f_panel["gender"] == "Other").sum()) != n_other_f, f"F target panel Other count is {int((f_panel['gender'] == 'Other').sum())}, expected {n_other_f}", failures)

    expected_rows = len(schema["conditions"]) * 1000
    fail_if(len(design) != expected_rows, f"internal design skeleton has {len(design)} rows, expected {expected_rows}", failures)
    fail_if(len(submission) != expected_rows, f"submission skeleton has {len(submission)} rows, expected {expected_rows}", failures)
    fail_if(set(design["condition"]) != set(schema["conditions"]), "internal design skeleton conditions differ from schema", failures)
    fail_if(set(submission["condition"]) != set(schema["conditions"]), "submission skeleton conditions differ from schema", failures)
    fail_if("donor_key" in submission.columns or "state" in submission.columns or "state_abbr" in submission.columns, "submission skeleton exposes donor_key/state columns", failures)
    response_cols = list(dict.fromkeys([*raw_item_columns(), *outcome_columns()]))
    fail_if(not design[response_cols].isna().all().all(), "internal design skeleton response columns are not all missing", failures)
    submission_response_cols = [col for col in official_submission_columns(schema) if col not in {"profile_id", "condition", *schema["moderators"].keys()}]
    fail_if(not submission[submission_response_cols].isna().all().all(), "submission skeleton outcome columns are not all missing", failures)
    invariant = design.groupby("donor_key")[["age", "age_band", "gender", "race", "education", "income", "party", "state", "state_abbr"]].nunique()
    fail_if(not (invariant == 1).all().all(), "donor demographics/state vary across design-skeleton conditions", failures)

    for state in g_master["state_abbr"].unique():
        fail_if(not sc.get_condition_stimulus("Extreme weather predictions", state).strip(), f"empty extreme-weather stimulus for {state}", failures)

    mapping = category_mapping_audit(g_master, schema)
    g_total_label = str(1000 - n_other_g)
    f_total_label = str(500 - n_other_f)
    g_master_mf = g_master[g_master["gender"] != "Other"]
    f_panel_mf = f_panel[f_panel["gender"] != "Other"]
    g_age = quota_diagnostics(g_master_mf, quota_age_g, ["gender", "age_band"], g_total_label)
    g_race = quota_diagnostics(g_master_mf, quota_race_g, ["gender", "race"], g_total_label)
    f_age = quota_diagnostics(f_panel_mf, quota_age_f, ["gender", "age_band"], f_total_label)
    f_race = quota_diagnostics(f_panel_mf, quota_race_f, ["gender", "race"], f_total_label)
    condition_balance = condition_balance_audit(design, schema)
    nonquota = nonquota_distribution_audit(g_master)
    joint = joint_distribution_audit(g_master)
    states = state_audit(g_master)
    duplicates = duplication_audit(g_master, roster, design, submission)
    inventory = existing_persona_files()

    fail_if((mapping["mapping_status"] != "exact").any(), "category mapping audit contains invalid_or_missing rows", failures)
    fail_if(not g_age["exact_count_match"].all(), "G gender x age quota counts do not exactly match quota_gender_age_1000.csv", failures)
    fail_if(not g_race["exact_count_match"].all(), "G gender x race quota counts do not exactly match quota_gender_race_1000.csv", failures)
    fail_if(not f_age["exact_count_match"].all(), "F gender x age quota counts do not exactly match quota_gender_age_500.csv", failures)
    fail_if(not f_race["exact_count_match"].all(), "F gender x race quota counts do not exactly match quota_gender_race_500.csv", failures)
    fail_if((condition_balance["count_difference_vs_control"] != 0).any(), "condition balance audit found demographic/state drift vs control", failures)
    fail_if((duplicates.loc[duplicates["key"].isin(["donor_key", "profile_id"]), "duplicates"] != 0).any(), "duplicate donor_key/profile_id found", failures)
    fail_if(not states["extreme_weather_stimulus_available"].all(), "one or more state-specific extreme-weather stimuli are unavailable", failures)

    mapping.to_csv(output_dir / "category_mapping_audit.csv", index=False)
    g_age.to_csv(output_dir / "quota_diagnostics_gender_age.csv", index=False)
    g_race.to_csv(output_dir / "quota_diagnostics_gender_race.csv", index=False)
    f_age.to_csv(output_dir / "f_quota_diagnostics_gender_age.csv", index=False)
    f_race.to_csv(output_dir / "f_quota_diagnostics_gender_race.csv", index=False)
    condition_balance.to_csv(output_dir / "condition_balance_audit.csv", index=False)
    nonquota.to_csv(output_dir / "source_nonquota_distribution_audit.csv", index=False)
    joint.to_csv(output_dir / "joint_distribution_audit.csv", index=False)
    states.to_csv(output_dir / "state_audit.csv", index=False)
    duplicates.to_csv(output_dir / "duplication_audit.csv", index=False)
    inventory.to_csv(output_dir / "existing_persona_files.csv", index=False)

    plot_paths = write_plots(output_dir, g_master, g_age, g_race, f_age, f_race, condition_balance, joint)

    summary = {
        "status": "FAIL" if failures else "PASS",
        "n_g_donors": int(len(g_master)),
        "n_g_design_rows": int(len(design)),
        "n_f_profiles": int(len(f_panel)),
        "n_conditions": int(len(schema["conditions"])),
        "g_gender_age_max_abs_difference_pp": float(g_age["abs_difference_pp"].max()),
        "g_gender_race_max_abs_difference_pp": float(g_race["abs_difference_pp"].max()),
        "f_gender_age_max_abs_difference_pp": float(f_age["abs_difference_pp"].max()),
        "f_gender_race_max_abs_difference_pp": float(f_race["abs_difference_pp"].max()),
        "condition_balance_max_count_difference": int(condition_balance["count_difference_vs_control"].abs().max()),
        "warnings": warnings,
        "failures": failures,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(output_dir, summary, warnings, failures, plot_paths)

    print(json.dumps(summary, indent=2))
    if failures:
        raise ValidationFailure("persona validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
