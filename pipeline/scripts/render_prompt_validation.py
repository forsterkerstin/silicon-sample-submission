#!/usr/bin/env python3
"""Render and audit G/F prompt protocols offline.

No API calls are made. This writes prompt examples, prompt hashes, leakage
audit rows, and paired-F prompt checks under outputs/prompt_validation/.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
import difflib

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PIPELINE_ROOT.parent
sys.path.insert(0, str(PIPELINE_ROOT))
sys.path.insert(0, str(PIPELINE_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import survey_content as sc  # noqa: E402
from inference.prompts import (  # noqa: E402
    F_PROMPT_PROTOCOL,
    F_INTRO_VARIANTS,
    F_PROFILE_FORMAT_VARIANTS,
    F_PROFILE_LABEL_VARIANTS,
    F_SURVEY_FORMAT_VARIANTS,
    F_SYSTEM_PROMPT,
    F_VARIANT_ASSIGNMENT_VERSION,
    CONSENSUS_INTERACTION_PROTOCOL_ID,
    G_PROMPT_PROTOCOL,
    G_QUESTIONNAIRE_VERSION,
    G_SYSTEM_PROMPT,
    PROMPT_COMPILER_VERSION,
    build_f_consensus_stage_a_prompt_render,
    build_f_consensus_stage_b_prompt_render,
    build_f_prompt_render,
    build_f_prompt_render_from_items,
    build_g_consensus_stage_a_prompt_render,
    build_g_consensus_stage_b_prompt_render,
    build_g_prompt_render,
    consensus_stage_a_record,
    f_variant_assignment,
    normalize_prompt_without_stimulus,
    schema_hash,
    target_f_control_variant,
    text_hash,
    validate_compiler_no_leakage,
)
from calibration.study_population import archive_profile_to_prompt_profile  # noqa: E402

OUTPUT_DIR = PIPELINE_ROOT / "outputs" / "prompt_validation"
G_EXAMPLE_DIR = OUTPUT_DIR / "g_prompt_examples"
F_EXAMPLE_DIR = OUTPUT_DIR / "f_prompt_examples"
F_PAIR_DIR = OUTPUT_DIR / "f_pair_examples"
PROFILE_PATH = PIPELINE_ROOT / "data" / "generated" / "g_personas_master.csv"
F_PANEL_PATH = PIPELINE_ROOT / "data" / "generated" / "f_target_panel.csv"
EXTERNAL_F_PANEL_PATH = PIPELINE_ROOT / "data" / "generated" / "external_primary_f_panels.csv"
ATE_ARCHIVE_PATH = PIPELINE_ROOT / "data" / "ate_archive.csv"
ARCHIVE_HYPOTHESES_PATH = PIPELINE_ROOT / "data" / "archive_70studies" / "extracted" / "hypotheses.csv"
ARCHIVE_LLM_RDS_PATH = PIPELINE_ROOT / "data" / "archive_70studies" / "llm_responses.RDS"
SCHEMA_PATH = PIPELINE_ROOT / "config" / "benchmark_schema.yaml"
CODEBOOK_PATH = REPO_ROOT / "codebook.csv"
SURVEY_JSON_PATH = REPO_ROOT / "survey" / "survey.json"
CONDITION_CODENAMES_PATH = REPO_ROOT / "survey" / "condition_codenames.csv"

F_FORBIDDEN_MODEL_VISIBLE_PATTERNS = [
    r"\bexternal primary calibration archive\b",
    r"\bhuman[_ -]?ate\b",
    r"\bhuman treatment mean\b",
    r"\bhuman control mean\b",
    r"\btreatment mean\b",
    r"\bcontrol mean\b",
    r"\beffect direction\b",
    r"\bp[- ]?value\b",
    r"\bstatistical significance\b",
    r"\bsignificance level\b",
    r"\bstatistically significant\b",
    r"\bconfidence interval\b",
    r"\b95% ci\b",
    r"\bpaper conclusion\b",
    r"\bcalibration alpha\b",
    r"\bcalibration lambda\b",
    r"\bbenchmark score\b",
    r"\bsynthetic[_ -]?ate\b",
    r"\baverage treatment effect\b",
    r"\btreatment effect\b",
    r"\bdirect ate\b",
    r"\bestimat(?:e|ing) (?:the )?(?:causal )?effect\b",
]

MISSING_PLACEHOLDER_PATTERNS = [
    r"\bNA\b",
    r"\bNaN\b",
    r"\bN/A\b",
    r"\bnull\b",
    r"\bunknown\b",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args(argv)


def configure_output_dir(output_dir: Path) -> None:
    global OUTPUT_DIR, G_EXAMPLE_DIR, F_EXAMPLE_DIR, F_PAIR_DIR
    OUTPUT_DIR = output_dir
    G_EXAMPLE_DIR = OUTPUT_DIR / "g_prompt_examples"
    F_EXAMPLE_DIR = OUTPUT_DIR / "f_prompt_examples"
    F_PAIR_DIR = OUTPUT_DIR / "f_pair_examples"


def file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def profile_dict(row: pd.Series) -> dict[str, Any]:
    out = {
        "age": row.get("age"),
        "gender": row.get("gender"),
        "race": row.get("race"),
        "education": row.get("education"),
        "income": row.get("income"),
        "party": row.get("party"),
        "state": row.get("state"),
        "state_abbr": row.get("state_abbr"),
    }
    for optional in ("political_ideology", "religion"):
        if optional in row and pd.notna(row[optional]) and str(row[optional]).strip():
            out[optional] = row[optional]
    return out


def write_render(path: Path, render) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    history = "\n\n".join(
        f"{message['role'].upper()}\n{'-' * len(message['role'])}\n{message['content']}"
        for message in render.conversation_history
    )
    path.write_text(
        f"protocol_id: {render.protocol_id}\n"
        f"request_key: {render.request_key}\n"
        f"prompt_variant_id: {render.prompt_variant_id}\n"
        f"questionnaire_order: {json.dumps(render.questionnaire_order or {}, sort_keys=True)}\n"
        f"response_key_map: {json.dumps(render.response_key_map or {}, sort_keys=True)}\n"
        f"provenance: {json.dumps(render.provenance or {}, sort_keys=True)}\n"
        f"system_prompt_hash: {text_hash(render.system_prompt)}\n"
        f"user_prompt_hash: {text_hash(render.user_prompt)}\n"
        f"schema_hash: {schema_hash(render.response_schema)}\n\n"
        "SYSTEM\n"
        "------\n"
        f"{render.system_prompt}\n\n"
        + (f"CONVERSATION_HISTORY\n--------------------\n{history}\n\n" if history else "") +
        "USER\n"
        "----\n"
        f"{render.user_prompt}\n\n"
        "JSON_SCHEMA\n"
        "-----------\n"
        f"{json.dumps(render.response_schema, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )


def audit_row(render, *, donor_or_f_profile_id: str, study_id: str, condition_id: str, outcome_id: str) -> dict[str, Any]:
    problems = validate_compiler_no_leakage(render, condition_id=condition_id)
    return {
        "role": render.role,
        "request_key": render.request_key,
        "protocol_id": render.protocol_id,
        "donor_or_f_profile_id": donor_or_f_profile_id,
        "study_id": study_id,
        "condition_id": condition_id,
        "outcome_id": outcome_id,
        "prompt_variant_id": render.prompt_variant_id,
        "system_prompt_hash": text_hash(render.system_prompt),
        "user_prompt_hash": text_hash(render.user_prompt),
        "stimulus_hash": text_hash(render.stimulus_text),
        "questionnaire_hash": file_hash(CODEBOOK_PATH),
        "g_questionnaire_order": json.dumps(render.questionnaire_order or {}, sort_keys=True),
        "validation_status": "PASS" if not problems else "FAIL: " + "; ".join(problems),
    }


def write_ashokkumar_reference_prompt() -> str | None:
    """Write one exact archived Ashokkumar prompt, if the RDS is available."""
    rds = PIPELINE_ROOT / "data" / "archive_70studies" / "llm_responses.RDS"
    out = F_EXAMPLE_DIR / "external_primary_archive_ashokkumar_verbatim_reference.txt"
    if not rds.exists():
        return None
    r_code = f"""
    llm <- readRDS({json.dumps(str(rds))})
    row <- llm[llm$model == 'gpt-4', ][1, ]
    text <- paste0(
      'This file is a verbatim reference prompt from Ashokkumar archive llm_responses.RDS, not the active adapted F compiler.\\n',
      'study: ', row$study, '\\n',
      'condition.name: ', row$condition.name, '\\n',
      'outcome.name: ', row$outcome.name, '\\n',
      'spec_template_group: ', row$spec_template_group, '\\n\\n',
      row$prompt
    )
    writeLines(text, con={json.dumps(str(out))}, useBytes=TRUE)
    """
    result = subprocess.run(["Rscript", "-e", r_code], cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return str(out.relative_to(REPO_ROOT))
    except ValueError:
        # --output-dir redirected outside REPO_ROOT (e.g. a test tmp_path):
        # this string is only used for the human-readable report text, so
        # fall back to the absolute path rather than crash.
        return str(out)


def _archive_outcome_name(effect_id: str) -> tuple[str, str, str]:
    """study, outcome_name, hypothesis -- hypothesis is always the last
    colon-separated segment (e.g. "hyp1"/"hyp2"/"hyp3", never itself
    containing a colon); outcome_name is everything between the first and
    last colon, rejoined, since an outcome name can itself contain a colon
    (e.g. Haaland874's "Affirmative action: Assistance"). Matches the
    already-correct parsing in build_external_calibration_panels.parse_effect_id;
    for every effect_id with exactly one embedded colon (the overwhelming
    majority) this is byte-identical to a naive 3-way split."""
    parts = effect_id.split(":")
    if len(parts) < 3:
        raise ValueError(f"cannot parse archive effect_id {effect_id!r}")
    return parts[0], ":".join(parts[1:-1]), parts[-1]


def _clean_archive_page(text: str) -> str:
    lines = []
    for line in str(text).splitlines():
        line = re.sub(r"^\s*>\s?", "", line)
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def archive_prompt_pages(prompt: str) -> list[str]:
    pattern = re.compile(
        r"The (?:first|next) page of the survey says:\n> (?P<body>.*?)(?=\n\n(?:You choose|Participant X chooses|The next page of the survey says:|\Z))",
        flags=re.S,
    )
    return [_clean_archive_page(match.group("body")) for match in pattern.finditer(str(prompt))]


def _is_archive_demographic_page(page: str) -> bool:
    text = re.sub(r"\s+", " ", page).lower()
    demographic_markers = [
        "are you liberal, moderate or conservative",
        "how old are you",
        "what is your ethnicity",
        "what is your gender",
        "maximum level of education",
        "partisan affiliation",
    ]
    return any(marker in text for marker in demographic_markers)


def archive_material_and_item(row: pd.Series, *, effect_id: str, is_control_arm: bool = False) -> tuple[str, dict[str, Any]]:
    """`is_control_arm` must come from the frozen t_hypothesis==0 designation in
    hypotheses.csv (via _condition_pair_for_effect), never guessed here. Zero
    non-demographic material pages is accepted ONLY for that designated control
    arm -- i.e. only when the archived human transcript for THIS condition
    itself shows nothing between demographics and the outcome question. A
    treatment arm (or any row not marked as the designated control) with zero
    material pages still fails closed: real interventions must have material."""
    pages = archive_prompt_pages(str(row["prompt"]))
    if len(pages) < 2:
        raise ValueError(f"{effect_id}: archived prompt did not parse into survey pages")
    material_pages = [page for page in pages[:-1] if not _is_archive_demographic_page(page)]
    if not material_pages and not is_control_arm:
        raise ValueError(f"{effect_id}: archived prompt has no non-demographic source material before outcome")
    outcome_page = pages[-1]
    scale_min = int(float(row["outcome_scale_min"]))
    scale_max = int(float(row["outcome_scale_max"]))
    item = {
        "qualtrics_label": "response",
        "target_label": "response",
        "question_text": outcome_page,
        "response_options": "",
        "scale": "external_native_integer",
        "scale_min": scale_min,
        "scale_max": scale_max,
        "source_path": str(ARCHIVE_LLM_RDS_PATH.relative_to(REPO_ROOT)),
        "source_variable": "prompt/outcome_scale_min/outcome_scale_max",
    }
    return "\n\n---\n\n".join(material_pages), item


def export_archive_source_rows(selection: pd.DataFrame) -> pd.DataFrame:
    if selection.empty or not ARCHIVE_LLM_RDS_PATH.exists():
        return pd.DataFrame()
    selected_path = OUTPUT_DIR / "f_external_source_selection.csv"
    source_path = OUTPUT_DIR / "f_external_source_rows.csv"
    selection.to_csv(selected_path, index=False)
    r_code = f"""
    selected <- read.csv({json.dumps(str(selected_path))}, stringsAsFactors=FALSE)
    llm <- readRDS({json.dumps(str(ARCHIVE_LLM_RDS_PATH))})
    rows <- list()
    for (i in seq_len(nrow(selected))) {{
      sub <- llm[
        llm$study == selected$study[i] &
        llm$outcome.name == selected$outcome_name[i] &
        llm$condition.name == selected$condition_name[i],
      ]
      if (nrow(sub) == 0) next
      preferred <- sub[sub$model == 'gpt-4', ]
      if (nrow(preferred) == 0) preferred <- sub
      rows[[length(rows) + 1]] <- preferred[1, c('model','study','condition.name','outcome.name','spec_template_group','prompt','outcome_scale_min','outcome_scale_max')]
    }}
    out <- if (length(rows)) do.call(rbind, rows) else data.frame()
    write.csv(out, file={json.dumps(str(source_path))}, row.names=FALSE, na='')
    """
    result = subprocess.run(["Rscript", "-e", r_code], cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Rscript failed while exporting archive source rows")
    return pd.read_csv(source_path)


def _pick_external_examples() -> pd.DataFrame:
    archive = pd.read_csv(ATE_ARCHIVE_PATH)
    primary = archive[archive["included_primary_calibration"].astype(str).str.lower().eq("true")].copy()
    desired = [
        "AnsonBRIEF60:economy_positivity:hyp1",
        "Braman751:scholar_credibility:hyp1",
        "Terman1029:donating to HRAC:hyp1",
    ]
    chosen = primary[primary["effect_id"].isin(desired)].copy()
    if len(chosen) < 3:
        supplemental = primary[~primary["study_id"].isin(set(chosen["study_id"]))].groupby("study_id", as_index=False).head(1)
        chosen = pd.concat([chosen, supplemental], ignore_index=True).drop_duplicates("effect_id").head(3)
    return chosen


def _condition_pair_for_effect(effect_id: str, hypotheses: pd.DataFrame) -> tuple[str, str]:
    study, outcome_name, hypothesis = _archive_outcome_name(effect_id)
    rows = hypotheses[
        hypotheses["study"].eq(study)
        & hypotheses["outcome.name"].eq(outcome_name)
        & hypotheses["hypothesis"].eq(hypothesis)
    ]
    controls = rows[rows["t_hypothesis"].eq(0)]["condition.name"].dropna().astype(str).tolist()
    treatments = rows[rows["t_hypothesis"].eq(1)]["condition.name"].dropna().astype(str).tolist()
    if not controls or not treatments:
        raise ValueError(f"{effect_id}: cannot identify archived control/treatment condition names")
    return controls[0], treatments[0]


def _first_panel_row(effect_id: str, *, require_missing: bool) -> pd.Series:
    panel = pd.read_csv(EXTERNAL_F_PANEL_PATH)
    rows = panel[panel["effect_id"].eq(effect_id)].copy()
    missing = rows["missing_demographic_fields"].fillna("").astype(str).str.len() > 0
    if require_missing and missing.any():
        return rows[missing].iloc[0]
    if not require_missing and (~missing).any():
        return rows[~missing].iloc[0]
    return rows.iloc[0]


def f_model_visible_text(render) -> str:
    return "\n".join(
        [
            *(message["content"] for message in render.messages),
            json.dumps(render.response_schema, sort_keys=True),
        ]
    )


def f_leakage_terms(render) -> list[str]:
    haystack = f_model_visible_text(render).lower()
    hits = []
    for pattern in F_FORBIDDEN_MODEL_VISIBLE_PATTERNS:
        if re.search(pattern, haystack, flags=re.I):
            hits.append(pattern)
    return hits


def prompt_profile_section(render) -> str:
    text = render.user_prompt
    if "STUDY SETTING" not in text:
        return text
    return text.split("STUDY SETTING", 1)[0]


def missing_placeholder_hits(render) -> list[str]:
    section = prompt_profile_section(render)
    return [pattern for pattern in MISSING_PLACEHOLDER_PATTERNS if re.search(pattern, section)]


def _pair_diff(control, treatment) -> tuple[bool, str]:
    control_norm = normalize_prompt_without_stimulus(control)
    treatment_norm = normalize_prompt_without_stimulus(treatment)
    schema_equal = json.dumps(control.response_schema, sort_keys=True) == json.dumps(treatment.response_schema, sort_keys=True)
    text_equal = control_norm == treatment_norm
    if text_equal and schema_equal and control.prompt_variant_id == treatment.prompt_variant_id:
        return True, "PASS: nonstimulus prompt text, schema, and prompt variant are identical after stimulus normalization."
    diff = "\n".join(
        difflib.unified_diff(
            control_norm.splitlines(),
            treatment_norm.splitlines(),
            fromfile="control_nonstimulus",
            tofile="treatment_nonstimulus",
            lineterm="",
        )
    )
    if not schema_equal:
        diff += "\n\nSCHEMA DIFFERS"
    if control.prompt_variant_id != treatment.prompt_variant_id:
        diff += f"\n\nVARIANT DIFFERS: {control.prompt_variant_id} != {treatment.prompt_variant_id}"
    return False, diff


def slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return text[:90] or "item"


def manual_read_section(name: str, control, treatment) -> str:
    def history_block(render) -> str:
        if not render.conversation_history:
            return ""
        parts = []
        for message in render.conversation_history:
            parts.append(f"{message['role'].upper()}\n{'-' * len(message['role'])}\n{message['content']}")
        return "CONVERSATION HISTORY\n--------------------\n" + "\n\n".join(parts) + "\n\n"

    return (
        f"# {name}\n\n"
        "## Control\n\n"
        "SYSTEM\n"
        "------\n"
        f"{control.system_prompt}\n\n"
        f"{history_block(control)}"
        "USER\n"
        "----\n"
        f"{control.user_prompt}\n\n"
        "## Treatment\n\n"
        "SYSTEM\n"
        "------\n"
        f"{treatment.system_prompt}\n\n"
        f"{history_block(treatment)}"
        "USER\n"
        "----\n"
        f"{treatment.user_prompt}\n"
    )


def target_outcome_context_audit_rows() -> list[dict[str, object]]:
    rows = []
    for outcome in sc.OUTCOME_COMPOSITES:
        if outcome == "newsletter_signup":
            required = "Talking Climate newsletter offer page before signup decision"
            source = "survey/survey.json-derived G_NEWSLETTER_OFFER_TEXT; codebook.csv notes the cross-page dependency"
            self_contained = False
        elif outcome == "donation_ams":
            required = "Bonus/allocation instruction page before donation amount"
            source = "survey/survey.json question QID1721185865-derived G_DONATION_INTRO_TEXT"
            self_contained = False
        else:
            required = "none beyond scored item/battery wording and native response scale"
            source = "codebook.csv measured item rows"
            self_contained = True
        rows.append(
            {
                "outcome": outcome,
                "self_contained_question": self_contained,
                "required_participant_context": required,
                "context_source": source,
                "context_included_in_F": bool(self_contained or required != "none beyond scored item/battery wording and native response scale"),
            }
        )
    return rows


def target_control_filler_audit_rows(f_profiles: pd.DataFrame) -> list[dict[str, object]]:
    labels = {1: "neckties", 2: "baseball", 3: "dances"}
    rows = []
    for _, row in f_profiles.iterrows():
        f_profile_id = str(row["f_profile_id"])
        variant = target_f_control_variant(f_profile_id, 1)
        rows.append({"f_profile_id": f_profile_id, "replicate_id": 1, "control_variant": variant, "control_filler": labels[variant]})
    return rows


def deterministic_stage_a_response(render) -> dict[str, int]:
    """Offline-only plausible response used to render Stage B examples."""
    return {key: 40 + (index * 7) % 45 for index, key in enumerate(render.response_schema["required"], start=1)}


def g_consensus_stage_a_record(profile: dict[str, Any], donor_key: str, items: list[dict[str, Any]]):
    stage_a = build_g_consensus_stage_a_prompt_render(profile, donor_key=donor_key, replicate_id=1)
    response = deterministic_stage_a_response(stage_a)
    record = consensus_stage_a_record(stage_a, response, role="G", subject_id=donor_key, replicate_id=1)
    stage_b = build_g_consensus_stage_b_prompt_render(profile, items, record, donor_key=donor_key, replicate_id=1)
    return stage_a, stage_b, record


def f_consensus_stage_a_record(profile: dict[str, Any], f_profile_id: str):
    stage_a = build_f_consensus_stage_a_prompt_render(profile, f_profile_id=f_profile_id, replicate_id=1)
    response = deterministic_stage_a_response(stage_a)
    record = consensus_stage_a_record(stage_a, response, role="F", subject_id=f_profile_id, replicate_id=1)
    return stage_a, record


def build_target_f_render(profile: dict[str, Any], stimulus: str, outcome: str, *, f_profile_id: str, condition_id: str, replicate_id: int = 1):
    if condition_id == "Consensus":
        _, record = f_consensus_stage_a_record(profile, f_profile_id)
        return build_f_consensus_stage_b_prompt_render(
            profile,
            outcome,
            record,
            f_profile_id=f_profile_id,
            replicate_id=replicate_id,
        )
    return build_f_prompt_render(
        profile,
        stimulus,
        outcome,
        study_id="target",
        f_profile_id=f_profile_id,
        condition_id=condition_id,
        replicate_id=replicate_id,
    )


def consensus_flow_audit() -> dict[str, object]:
    source = sc.get_consensus_interaction_source_audit()
    entries = source["blocks"]
    return {
        **source,
        "status": "PASS",
        "current_g_and_f_representation": "source-faithful two-stage exception: Stage A estimate questions only; Stage B retains Stage A user/assistant history, then shows feedback and post-treatment outcome request",
        "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID,
        "stage_a_incremental_requests_at_n_g_1000_n_f_500_r_f_1": {"G": 1000, "F": 500, "F_formula": "500 * R_F"},
        "blocks": entries,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_output_dir(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    G_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    F_EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    F_PAIR_DIR.mkdir(parents=True, exist_ok=True)
    for directory in (G_EXAMPLE_DIR, F_EXAMPLE_DIR, F_PAIR_DIR):
        for stale_render in directory.glob("*.txt"):
            stale_render.unlink()
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    g_profiles = pd.read_csv(PROFILE_PATH)
    f_profiles = pd.read_csv(F_PANEL_PATH)
    items = sc.load_items()
    audit_rows = []
    leakage_rows = []
    source_rows = []
    pairing_rows = []
    pair_diff_sections = []
    manual_sections = []

    conditions = schema["conditions"]
    g_examples = [
        ("control_filler_1", "control", sc.get_condition_stimulus("control", control_variant=1), g_profiles.iloc[0]),
        ("control_filler_2", "control", sc.get_condition_stimulus("control", control_variant=2), g_profiles.iloc[1]),
        ("control_filler_3", "control", sc.get_condition_stimulus("control", control_variant=3), g_profiles.iloc[2]),
        ("intervention_corporate_reliance", "Corporate reliance", sc.get_condition_stimulus("Corporate reliance"), g_profiles.iloc[3]),
        ("intervention_consensus", "Consensus", sc.get_condition_stimulus("Consensus"), g_profiles.iloc[4]),
        ("intervention_high_public_trust", "High public trust", sc.get_condition_stimulus("High public trust"), g_profiles.iloc[5]),
        ("intervention_funding", "Funding", sc.get_condition_stimulus("Funding"), g_profiles.iloc[6]),
    ]
    for state in ["CA", "NY", "TX"]:
        donor = g_profiles[g_profiles["state_abbr"] == state].head(1)
        if not donor.empty:
            g_examples.append(
                (
                    f"extreme_weather_{state}",
                    "Extreme weather predictions",
                    sc.get_condition_stimulus("Extreme weather predictions", state),
                    donor.iloc[0],
                )
            )

    for name, condition_id, stimulus, donor in g_examples:
        if condition_id == "Consensus":
            donor_profile = profile_dict(donor)
            donor_key = str(donor["donor_key"])
            stage_a, stage_b, _ = g_consensus_stage_a_record(donor_profile, donor_key, items)
            write_render(G_EXAMPLE_DIR / f"{name}_stage_a.txt", stage_a)
            write_render(G_EXAMPLE_DIR / f"{name}_stage_b.txt", stage_b)
            audit_rows.append(
                audit_row(
                    stage_a,
                    donor_or_f_profile_id=donor_key,
                    study_id="target",
                    condition_id=condition_id,
                    outcome_id="consensus_stage_a_estimates",
                )
            )
            audit_rows.append(
                audit_row(
                    stage_b,
                    donor_or_f_profile_id=donor_key,
                    study_id="target",
                    condition_id=condition_id,
                    outcome_id="full_questionnaire",
                )
            )
            continue
        render = build_g_prompt_render(
            profile_dict(donor),
            stimulus,
            items,
            donor_key=str(donor["donor_key"]),
            condition_id=condition_id,
        )
        write_render(G_EXAMPLE_DIR / f"{name}.txt", render)
        audit_rows.append(
            audit_row(
                render,
                donor_or_f_profile_id=str(donor["donor_key"]),
                study_id="target",
                condition_id=condition_id,
                outcome_id="full_questionnaire",
            )
        )

    f_example_specs = [
        ("target_control_trust", "target", "control", "trust_multidimensional", sc.get_condition_stimulus("control", control_variant=target_f_control_variant(str(f_profiles.iloc[0]['f_profile_id']), 1)), f_profiles.iloc[0], 1),
        ("target_treatment_trust", "target", "Corporate reliance", "trust_multidimensional", sc.get_condition_stimulus("Corporate reliance"), f_profiles.iloc[0], 1),
        ("target_treatment_donation", "target", "Consensus", "donation_ams", sc.get_condition_stimulus("Consensus"), f_profiles.iloc[1], 1),
        ("target_extreme_weather_belief", "target", "Extreme weather predictions", "belief_post", sc.get_condition_stimulus("Extreme weather predictions", f_profiles.iloc[2]["state_abbr"]), f_profiles.iloc[2], 1),
    ]
    for i in range(3, min(12, len(f_profiles))):
        variant = f_variant_assignment("target", str(f_profiles.iloc[i]["f_profile_id"]), "trust_post", 1)
        f_example_specs.append(
            (
                "target_variant_" + variant["prompt_variant_id"].replace("+", "_"),
                "target",
                "Peer-review",
                "trust_post",
                sc.get_condition_stimulus("Peer-review"),
                f_profiles.iloc[i],
                1,
            )
        )

    for name, study_id, condition_id, outcome, stimulus, profile_row, replicate in f_example_specs:
        profile = profile_dict(profile_row)
        f_profile_id = str(profile_row["f_profile_id"])
        if study_id == "target":
            if condition_id == "Consensus":
                stage_a, _ = f_consensus_stage_a_record(profile, f_profile_id)
                write_render(F_EXAMPLE_DIR / f"{name}_stage_a.txt", stage_a)
            render = build_target_f_render(profile, stimulus, outcome, f_profile_id=f_profile_id, condition_id=condition_id, replicate_id=replicate)
        else:
            render = build_f_prompt_render(
                profile,
                stimulus,
                outcome,
                study_id=study_id,
                f_profile_id=f_profile_id,
                condition_id=condition_id,
                replicate_id=replicate,
            )
        write_render(F_EXAMPLE_DIR / f"{name}.txt", render)
        audit_rows.append(
            audit_row(
                render,
                donor_or_f_profile_id=str(profile_row["f_profile_id"]),
                study_id=study_id,
                condition_id=condition_id,
                outcome_id=outcome,
            )
        )
        leakage_rows.append(
            {
                "example_name": name,
                "role": "F",
                "study_id": study_id,
                "outcome_id": outcome,
                "condition_id": condition_id,
                "leakage_terms": "|".join(f_leakage_terms(render)),
                "missing_profile_placeholder_terms": "|".join(missing_placeholder_hits(render)),
                "status": "PASS" if not f_leakage_terms(render) and not missing_placeholder_hits(render) else "FAIL",
            }
        )

    target_pair_specs = [
        ("target_corporate_reliance_trust_post", f_profiles.iloc[0], "Corporate reliance", "trust_post", sc.get_condition_stimulus("Corporate reliance")),
        ("target_consensus_newsletter_signup", f_profiles.iloc[1], "Consensus", "newsletter_signup", sc.get_condition_stimulus("Consensus")),
        ("target_funding_donation_ams", f_profiles.iloc[2], "Funding", "donation_ams", sc.get_condition_stimulus("Funding")),
    ]
    for name, profile_row, treatment_condition, outcome, treatment_stimulus in target_pair_specs:
        profile = profile_dict(profile_row)
        f_profile_id = str(profile_row["f_profile_id"])
        control = build_f_prompt_render(
            profile,
            sc.get_condition_stimulus("control", control_variant=target_f_control_variant(str(profile_row["f_profile_id"]), 1)),
            outcome,
            study_id="target",
            f_profile_id=f_profile_id,
            condition_id="control",
            replicate_id=1,
        )
        treatment = build_target_f_render(
            profile,
            treatment_stimulus,
            outcome,
            f_profile_id=f_profile_id,
            condition_id=treatment_condition,
            replicate_id=1,
        )
        write_render(F_PAIR_DIR / f"{name}_control.txt", control)
        write_render(F_PAIR_DIR / f"{name}_treatment.txt", treatment)
        ok, diff = _pair_diff(control, treatment)
        pair_diff_sections.append(f"## {name}\n\n{diff}\n")
        pairing_rows.append(
            {
                "study_id": "target",
                "effect_id": "",
                "f_profile_id": profile_row["f_profile_id"],
                "outcome_id": outcome,
                "replicate_id": 1,
                "n_conditions": 2,
                "prompt_variant_id": control.prompt_variant_id if control.prompt_variant_id == treatment.prompt_variant_id else f"{control.prompt_variant_id};{treatment.prompt_variant_id}",
                "nonstimulus_equal": ok,
                "schema_equal": json.dumps(control.response_schema, sort_keys=True) == json.dumps(treatment.response_schema, sort_keys=True),
                "condition_id_excluded_from_variant": control.variant_assignment == treatment.variant_assignment,
                "status": "PASS" if ok and control.variant_assignment == treatment.variant_assignment else "FAIL",
            }
        )
        for arm, condition_id, render in [("control", "control", control), ("treatment", treatment_condition, treatment)]:
            audit_rows.append(audit_row(render, donor_or_f_profile_id=str(profile_row["f_profile_id"]), study_id="target", condition_id=condition_id, outcome_id=outcome))
            leakage_rows.append(
                {
                    "example_name": f"{name}_{arm}",
                    "role": "F",
                    "study_id": "target",
                    "outcome_id": outcome,
                    "condition_id": condition_id,
                    "leakage_terms": "|".join(f_leakage_terms(render)),
                    "missing_profile_placeholder_terms": "|".join(missing_placeholder_hits(render)),
                    "status": "PASS" if not f_leakage_terms(render) and not missing_placeholder_hits(render) else "FAIL",
                }
            )
        if outcome in {"newsletter_signup", "donation_ams"}:
            manual_sections.append(manual_read_section(name, control, treatment))

    external_examples = _pick_external_examples()
    hypotheses = pd.read_csv(ARCHIVE_HYPOTHESES_PATH)
    selection_rows = []
    external_example_specs = []
    for example_index, row in external_examples.reset_index(drop=True).iterrows():
        effect_id = str(row["effect_id"])
        study, outcome_name, hypothesis = _archive_outcome_name(effect_id)
        control_condition, treatment_condition = _condition_pair_for_effect(effect_id, hypotheses)
        require_missing = example_index == 1
        profile_row = _first_panel_row(effect_id, require_missing=require_missing)
        external_example_specs.append((row, outcome_name, control_condition, treatment_condition, profile_row))
        selection_rows.extend(
            [
                {"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": control_condition, "arm": "control"},
                {"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": treatment_condition, "arm": "treatment"},
            ]
        )
    archive_source = export_archive_source_rows(pd.DataFrame(selection_rows))
    source_key = {
        (str(row["study"]), str(row["outcome.name"]), str(row["condition.name"])): row
        for _, row in archive_source.iterrows()
    }
    for archive_row, outcome_name, control_condition, treatment_condition, profile_row in external_example_specs:
        effect_id = str(archive_row["effect_id"])
        study = str(archive_row["study_id"])
        profile = archive_profile_to_prompt_profile(profile_row)
        control_source = source_key[(study, outcome_name, control_condition)]
        treatment_source = source_key[(study, outcome_name, treatment_condition)]
        control_material, control_item = archive_material_and_item(control_source, effect_id=effect_id, is_control_arm=True)
        treatment_material, treatment_item = archive_material_and_item(treatment_source, effect_id=effect_id)
        if control_item["question_text"] != treatment_item["question_text"]:
            raise RuntimeError(f"{effect_id}: control/treatment archive outcome wording differs")
        item = dict(control_item)
        render_kwargs = {
            "study_id": study,
            "f_profile_id": str(profile_row["f_profile_id"]),
            "outcome_id": effect_id,
            "replicate_id": 1,
            "study_setting": "This is an online survey shown to adult respondents.",
        }
        control = build_f_prompt_render_from_items(
            profile,
            control_material,
            [item],
            condition_id="control",
            **render_kwargs,
        )
        treatment = build_f_prompt_render_from_items(
            profile,
            treatment_material,
            [item],
            condition_id="treatment",
            **render_kwargs,
        )
        name = f"external_{slug(effect_id)}"
        write_render(F_PAIR_DIR / f"{name}_control.txt", control)
        write_render(F_PAIR_DIR / f"{name}_treatment.txt", treatment)
        ok, diff = _pair_diff(control, treatment)
        pair_diff_sections.append(f"## {name}\n\n{diff}\n")
        pairing_rows.append(
            {
                "study_id": study,
                "effect_id": effect_id,
                "f_profile_id": profile_row["f_profile_id"],
                "outcome_id": effect_id,
                "replicate_id": 1,
                "n_conditions": 2,
                "prompt_variant_id": control.prompt_variant_id if control.prompt_variant_id == treatment.prompt_variant_id else f"{control.prompt_variant_id};{treatment.prompt_variant_id}",
                "nonstimulus_equal": ok,
                "schema_equal": json.dumps(control.response_schema, sort_keys=True) == json.dumps(treatment.response_schema, sort_keys=True),
                "condition_id_excluded_from_variant": control.variant_assignment == treatment.variant_assignment,
                "status": "PASS" if ok and control.variant_assignment == treatment.variant_assignment else "FAIL",
            }
        )
        for arm, condition_name, source, render in [
            ("control", control_condition, control_source, control),
            ("treatment", treatment_condition, treatment_source, treatment),
        ]:
            audit_rows.append(audit_row(render, donor_or_f_profile_id=str(profile_row["f_profile_id"]), study_id=study, condition_id=arm, outcome_id=effect_id))
            leakage_rows.append(
                {
                    "example_name": f"{name}_{arm}",
                    "role": "F",
                    "study_id": study,
                    "outcome_id": effect_id,
                    "condition_id": arm,
                    "leakage_terms": "|".join(f_leakage_terms(render)),
                    "missing_profile_placeholder_terms": "|".join(missing_placeholder_hits(render)),
                    "status": "PASS" if not f_leakage_terms(render) and not missing_placeholder_hits(render) else "FAIL",
                }
            )
            source_rows.append(
                {
                    "example_name": f"{name}_{arm}",
                    "study_id": study,
                    "effect_id": effect_id,
                    "arm": arm,
                    "archive_condition_name": condition_name,
                    "source_path": str(ARCHIVE_LLM_RDS_PATH.relative_to(REPO_ROOT)),
                    "source_variables": "prompt|outcome_scale_min|outcome_scale_max",
                    "source_model_row": source["model"],
                    "source_spec_template_group": source["spec_template_group"],
                    "stimulus_source_classification": "exact participant-facing archive pages, demographic pages omitted because active F profile supplies demographics",
                    "outcome_source_classification": "exact participant-facing archive outcome page plus native scale bounds from archive columns",
                    "compiler_adaptation": "active project F prompt wrapper, project-serving neutral JSON key 'response'",
                    "outcome_min": item["scale_min"],
                    "outcome_max": item["scale_max"],
                }
            )
        if not any(section.startswith("# external_") for section in manual_sections):
            manual_sections.append(manual_read_section(name, control, treatment))

    reference_path = write_ashokkumar_reference_prompt()

    for _, profile_row in f_profiles.head(8).iterrows():
        for outcome in ["trust_post", "trust_multidimensional", "donation_ams"]:
            renders = []
            for condition_id, stimulus in [
                ("control", sc.get_condition_stimulus("control", control_variant=target_f_control_variant(str(profile_row["f_profile_id"]), 1))),
                ("Corporate reliance", sc.get_condition_stimulus("Corporate reliance")),
                ("Consensus", sc.get_condition_stimulus("Consensus")),
            ]:
                renders.append(
                    build_target_f_render(
                        profile_dict(profile_row),
                        stimulus,
                        outcome,
                        f_profile_id=str(profile_row["f_profile_id"]),
                        condition_id=condition_id,
                        replicate_id=1,
                    )
                )
            normalized = [normalize_prompt_without_stimulus(render) for render in renders]
            schemas = [json.dumps(render.response_schema, sort_keys=True) for render in renders]
            variants = {render.prompt_variant_id for render in renders}
            pairing_rows.append(
                {
                    "study_id": "target",
                    "f_profile_id": profile_row["f_profile_id"],
                    "outcome_id": outcome,
                    "replicate_id": 1,
                    "n_conditions": len(renders),
                    "prompt_variant_id": sorted(variants)[0] if len(variants) == 1 else ";".join(sorted(variants)),
                    "nonstimulus_equal": len(set(normalized)) == 1 and len(set(schemas)) == 1,
                    "status": "PASS" if len(set(normalized)) == 1 and len(set(schemas)) == 1 and len(variants) == 1 else "FAIL",
                }
            )

    prompt_audit = pd.DataFrame(audit_rows)
    prompt_audit.to_csv(OUTPUT_DIR / "prompt_audit.csv", index=False)
    pairing = pd.DataFrame(pairing_rows)
    pairing.to_csv(OUTPUT_DIR / "f_pairing_audit.csv", index=False)
    leakage = pd.DataFrame(leakage_rows)
    leakage.to_csv(OUTPUT_DIR / "f_effect_leakage_audit.csv", index=False)
    source_fidelity = pd.DataFrame(source_rows)
    source_fidelity.to_csv(OUTPUT_DIR / "f_source_fidelity_audit.csv", index=False)
    (OUTPUT_DIR / "f_pair_diff_report.md").write_text("\n".join(pair_diff_sections), encoding="utf-8")
    target_context = pd.DataFrame(target_outcome_context_audit_rows())
    target_context.to_csv(OUTPUT_DIR / "f_target_outcome_context_audit.csv", index=False)
    control_filler = pd.DataFrame(target_control_filler_audit_rows(f_profiles))
    control_filler.to_csv(OUTPUT_DIR / "f_target_control_filler_audit.csv", index=False)
    (OUTPUT_DIR / "f_consensus_flow_audit.json").write_text(json.dumps(consensus_flow_audit(), indent=2) + "\n", encoding="utf-8")
    manual_sections = sorted(manual_sections, key=lambda section: (0 if section.startswith("# external_") else 1, section))
    (OUTPUT_DIR / "f_outcome_context_manual_read.md").write_text("\n\n---\n\n".join(manual_sections), encoding="utf-8")

    variants = sorted({row["prompt_variant_id"] for row in audit_rows if row["role"] == "F"})
    manifest = {
        "g_protocol_id": G_PROMPT_PROTOCOL,
        "g_questionnaire_version": G_QUESTIONNAIRE_VERSION,
        "consensus_interaction_protocol_id": CONSENSUS_INTERACTION_PROTOCOL_ID,
        "g_literature_source": "Krsteski et al. ACL 2026; demographic-conditioned ATP survey-response template",
        "g_source_doi": "10.18653/v1/2026.acl-long.498",
        "g_adaptation_notes": "Demographic-only profile; no persona-guided prior-response synthesis; one full questionnaire request per donor x condition except Consensus, which uses a source-faithful two-stage interactive adaptation; structured JSON output.",
        "g_not_used": "Krsteski persona-guided prior-response synthesis",
        "f_protocol_id": F_PROMPT_PROTOCOL,
        "f_literature_source": "Ashokkumar et al. Nature 2026; experimental forecasting prompt architecture and prompt-ensemble strategy",
        "f_source_doi": "10.1038/s41586-026-10742-x",
        "f_adaptation_notes": "Small fixed adapted four-component variant library; paired prompt-variant assignment excludes condition_id; N_F remains 500. Consensus uses a source-faithful two-stage interactive adaptation; all other conditions use the standard request design.",
        "consensus_expected_incremental_requests_at_n_g_1000_n_f_500": {
            "g_increment": 1000,
            "f_increment_formula": "500 * R_F",
            "f_increment_at_r_f_1": 500,
        },
        "f_variant_components": {
            "intro_variants": {
                "classification": "close adaptation",
                "values": [value for _, value in F_INTRO_VARIANTS],
            },
            "profile_label_variants": {
                "classification": "project serving adaptation",
                "values": [value for _, value in F_PROFILE_LABEL_VARIANTS],
            },
            "profile_format_variants": {
                "classification": "project serving adaptation",
                "values": [value for _, value in F_PROFILE_FORMAT_VARIANTS],
            },
            "survey_format_variants": {
                "classification": "project serving adaptation",
                "values": [value for _, value in F_SURVEY_FORMAT_VARIANTS],
            },
        },
        "f_assignment_key": "study_id|f_profile_id|outcome_id|replicate_id|f_protocol_id",
        "f_assignment_condition_id_excluded": True,
        "f_assignment_performance_selection": False,
        "exact_system_prompt_hashes": {
            "G": text_hash(G_SYSTEM_PROMPT),
            "F": text_hash(F_SYSTEM_PROMPT),
        },
        "compiler_version": PROMPT_COMPILER_VERSION,
        "questionnaire_version_hash": file_hash(CODEBOOK_PATH),
        "stimulus_library_version_hash": {
            "survey_json": file_hash(SURVEY_JSON_PATH),
            "condition_codenames": file_hash(CONDITION_CODENAMES_PATH),
        },
        "profile_format_version": "demographic_only_fixed_order_v1",
        "selected_F_prompt_variants": variants,
        "deterministic_variant_assignment_algorithm_version": F_VARIANT_ASSIGNMENT_VERSION,
        "ashokkumar_verbatim_reference_prompt_path": reference_path,
        "git_commit": git_commit(),
    }
    (OUTPUT_DIR / "prompt_protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    summary = {
        "prompt_audit_rows": len(prompt_audit),
        "prompt_audit_failures": int(prompt_audit["validation_status"].str.startswith("FAIL").sum()),
        "f_pairing_rows": len(pairing),
        "f_pairing_failures": int((pairing["status"] != "PASS").sum()),
        "f_leakage_rows": len(leakage),
        "f_leakage_failures": int((leakage["status"] != "PASS").sum()) if not leakage.empty else 0,
        "f_source_fidelity_rows": len(source_fidelity),
        "f_target_outcome_context_rows": len(target_context),
        "f_target_control_filler_rows": len(control_filler),
        "f_target_control_filler_counts": control_filler["control_filler"].value_counts().sort_index().to_dict(),
        "f_pair_examples": len(list(F_PAIR_DIR.glob("*.txt"))),
        "g_examples": len(list(G_EXAMPLE_DIR.glob("*.txt"))),
        "f_examples": len(list(F_EXAMPLE_DIR.glob("*.txt"))),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["prompt_audit_failures"] or summary["f_pairing_failures"] or summary["f_leakage_failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
