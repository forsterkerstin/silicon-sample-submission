#!/usr/bin/env python3
"""scripts/build_advocacy_stimuli.py

Extracts the real intervention stimulus text for the climate-advocacy
megastudy's 17 treatment arms (see data/climate_advocacy_megastudy/materials/
intervention_docx/*.docx -- these are Qualtrics' own "Word" export of each
survey block, not hand-written excerpts) for use by
scripts/build_advocacy_validation.py's model-vs-human effect comparison.

Honesty check, not a blanket "it worked": each doc is inspected for two real
failure modes found by hand while building this --
  1. Qualtrics piped-field placeholders (${e://Field/...}) -- the doc
     references a per-respondent value that lives in a separate data source
     this repo doesn't have, so the literal displayed text cannot be
     reconstructed. Found in exactly one condition, MispCorrectionRisks.
  2. An embedded video the on-screen text explicitly refers to ("watch the
     video below") -- the docx has no video, so using the surrounding text
     alone would silently drop part of the real stimulus. Found in 6
     conditions.
A condition is `usable` only if neither applies. The other 10 (+ control,
which is always the empty-string stimulus by this pipeline's convention --
see below) get a mechanically cleaned (Qualtrics block/question-number
scaffolding stripped -- NOT hand-curated down to a single "message", so the
extracted text is the respondent's whole real survey block, matching what an
actual participant read start to finish) stimulus text.

Neutral_Control_Condition.docx's own real control was "watch a neutral
video" -- not literally blank -- but this pipeline (and every other real
condition/outcome comparison in it) already treats "control" as the
no-stimulus baseline by convention, and a neutral video is a defensible
approximation of "no persuasive content" for that purpose; it is NOT
included in the stimuli file below since build_advocacy_validation.py
elicits control with condition_stimulus="" like everywhere else.

Writes data/advocacy_intervention_stimuli.json:
  {condName: {usable: bool, exclusion_reason: str|None, stimulus_text: str|None, n_chars: int}}
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path

from lxml import etree

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

MATERIALS_DIR = PIPELINE_ROOT / "data" / "climate_advocacy_megastudy" / "materials" / "intervention_docx"
OUT_PATH = PIPELINE_ROOT / "data" / "advocacy_intervention_stimuli.json"

#: docx filename (without .docx) -> the exact condName it corresponds to in
#: advocacy_data.csv (verified by hand against both the file listing and
#: df["condName"].unique() -- a fixed, small, known correspondence, safer
#: than fuzzy string matching).
DOCX_TO_CONDNAME: dict[str, str] = {
    "Binding_Moral_Foundations": "BindingMorals",
    "Bipartisan_Elite_Cues": "BipartisanEliteCues",
    "Climate_Activist_Perspective_Taking": "ActivistPerspective",
    "Climate_Policy_Literacy": "ClimatePolicyLiteracy",
    "Co-Benefits": "CoBenefits",
    "Collective_Efficacy_and_Emotional_Benefit": "CollEfficacyEmoBenefit",
    "Connecting_to_Ecological_Disruptions": "EcologicalDisruptions",
    "Dynamic_Anger_Norm": "DynamicAngerNorm",
    "Global_Health_Threat": "GlobalHealthThreat",
    "Guilt-Based_Collective_Responsibility": "GuiltCollResponsibility",
    "Hope_and_Anger_Narratives": "HopeAngerNarratives",
    "Letter_to_Future_Generations": "LetterFuture",
    "Linking_Individual_and_Structural_Change": "IndStructuralChange",
    "Misperception_Correction_Risks": "MispCorrectionRisks",
    "Shifting_Focus_from_Individual_to_Collective_Action": "ShiftFocusIndColl",
    "System_Justification": "SystemJustification",
    "Threat-Injustice-and-Efficacy": "ThreatInjustEfficacy",
    # Neutral_Control_Condition deliberately excluded -- see module docstring.
}

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def raw_text(path: Path) -> str:
    """Every <w:t> text run in document.xml, in document order -- NOT
    python-docx's .paragraphs/.tables, which (verified by hand) return
    nothing for these files: the real text lives nested in table cells and
    structured-document-tag containers python-docx's high-level API doesn't
    traverse. Reading document.xml's raw XML directly recovers all of it.
    """
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = etree.fromstring(xml)
    return "".join(t.text for t in root.iter(f"{W_NS}t") if t.text)


def clean(text: str) -> str:
    """Best-effort Qualtrics-scaffolding stripper -- NOT hand-curation.
    Removes block/question markers, timing-question boilerplate, and
    text-entry-box underscore runs; collapses whitespace. Leaves the
    respondent-facing prose (including benign flow instructions like "Page
    Break") intact and in its original order, since the goal is the real
    survey block a participant read, not an edited-down "message".
    """
    t = re.sub(r"Start of Block:.*?(?=Q\d)", "", text)
    t = re.sub(r"End of Block:[^Q]*", " ", t)
    t = re.sub(r"Q\d+\s", " ", t)
    t = re.sub(r"_{10,}", " ", t)
    t = re.sub(r"Page Break", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def classify(text: str) -> tuple[bool, str | None]:
    if "${e://Field" in text:
        return False, "references Qualtrics piped-field values not present in this export"
    if re.search(r"\bvideo\b", text, re.IGNORECASE):
        return False, "stimulus explicitly refers to an embedded video not present in this export"
    return True, None


def main() -> int:
    stimuli = {}
    for docx_name, cond_name in DOCX_TO_CONDNAME.items():
        path = MATERIALS_DIR / f"{docx_name}.docx"
        text = raw_text(path)
        usable, reason = classify(text)
        cleaned = clean(text) if usable else None
        stimuli[cond_name] = {
            "usable": usable,
            "exclusion_reason": reason,
            "stimulus_text": cleaned,
            "n_chars": len(cleaned) if cleaned else 0,
            "source_docx": f"{docx_name}.docx",
        }
        status = "usable" if usable else f"EXCLUDED ({reason})"
        print(f"{cond_name:28s} {status}")

    n_usable = sum(1 for v in stimuli.values() if v["usable"])
    print(f"\n{n_usable}/{len(stimuli)} conditions usable")

    OUT_PATH.write_text(json.dumps(stimuli, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
