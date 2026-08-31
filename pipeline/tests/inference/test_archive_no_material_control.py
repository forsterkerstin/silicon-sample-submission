"""Regression tests for the narrow no-material-control-arm fix:

- archive_material_and_item() (render_prompt_validation.py): accepts zero
  non-demographic material pages ONLY when explicitly told the row is the
  frozen-designated control arm; still fails closed otherwise.
- build_f_prompt_render_from_items() (inference/prompts.py): still rejects an
  empty stimulus by default; accepts it only when explicitly authorized via
  intentional_no_material_control, and computes nonstimulus_text correctly in
  that case (no fabricated narrative text is ever inserted).
"""

from __future__ import annotations

import pandas as pd
import pytest

from inference.prompts import build_f_prompt_render_from_items
from render_prompt_validation import archive_material_and_item

_ITEM = {
    "qualtrics_label": "response",
    "target_label": "response",
    "question_text": "Would you support this policy?",
    "response_options": "",
    "scale": "external_native_integer",
    "scale_min": 1,
    "scale_max": 5,
}


def _row(prompt: str) -> pd.Series:
    return pd.Series({"prompt": prompt, "outcome_scale_min": 1, "outcome_scale_max": 5})


_DEMOGRAPHICS = (
    "The first page of the survey says:\n> Are you liberal, moderate or conservative?\n\nYou choose: 'Conservative'\n\n"
    "The next page of the survey says:\n> How old are you?\n\nYou choose: '30-39'\n\n"
)
_OUTCOME_PAGE = "The next page of the survey says:\n> Please rate this.\n\nYou choose: '"
_NARRATIVE_PAGE = (
    "The next page of the survey says:\n> Here is a short story about a person named Alex.\n\nYou choose: 'continue'\n\n"
)


def test_intentional_no_material_control_is_accepted():
    """A genuinely source-defined no-material control (demographics -> straight
    to the outcome question, nothing else) succeeds when explicitly flagged."""
    prompt = _DEMOGRAPHICS + _OUTCOME_PAGE
    material, item = archive_material_and_item(_row(prompt), effect_id="Study:outcome:hyp1", is_control_arm=True)
    assert material == ""
    assert item["question_text"] == "Please rate this."


def test_accidental_missing_material_still_fails_closed_without_control_flag():
    """The exact same zero-material prompt, WITHOUT is_control_arm=True (i.e. a
    treatment row, or any row not provably the designated control), must still
    raise -- absence of material is never silently accepted by default."""
    prompt = _DEMOGRAPHICS + _OUTCOME_PAGE
    with pytest.raises(ValueError, match="no non-demographic source material"):
        archive_material_and_item(_row(prompt), effect_id="Study:outcome:hyp1")


def test_treatment_arm_with_zero_material_fails_even_if_mislabeled_control_elsewhere():
    """is_control_arm only ever widens acceptance for the exact call it's passed
    to -- a genuine intervention arm that happens to have no extractable
    material (a real bug, not a design feature) must still fail closed when
    called without the flag, which is how every treatment-arm call site invokes it."""
    prompt = _DEMOGRAPHICS + _OUTCOME_PAGE
    with pytest.raises(ValueError):
        archive_material_and_item(_row(prompt), effect_id="Study:outcome:hyp2")


def test_malformed_prompt_still_fails_regardless_of_control_flag():
    """A prompt that doesn't even parse into >=2 pages is a data/parsing error,
    not a no-material-control case -- is_control_arm must not paper over this."""
    with pytest.raises(ValueError, match="did not parse into survey pages"):
        archive_material_and_item(_row("not a real archived prompt"), effect_id="Study:outcome:hyp1", is_control_arm=True)


def test_control_arm_with_real_material_is_unaffected():
    """is_control_arm=True must not change behavior when material genuinely
    exists -- it only ever widens acceptance of the zero-material case."""
    prompt = _DEMOGRAPHICS + _NARRATIVE_PAGE + _OUTCOME_PAGE
    material, _item = archive_material_and_item(_row(prompt), effect_id="Study:outcome:hyp1", is_control_arm=True)
    assert "Alex" in material


def test_build_f_prompt_render_from_items_rejects_empty_stimulus_by_default():
    with pytest.raises(ValueError, match="non-empty stimulus"):
        build_f_prompt_render_from_items(
            {"gender": "Female"},
            "",
            [_ITEM],
            study_id="Study",
            f_profile_id="F1",
            outcome_id="Study:response:hyp1",
            condition_id="control",
        )


def test_build_f_prompt_render_from_items_allows_explicit_intentional_empty_stimulus():
    render = build_f_prompt_render_from_items(
        {"gender": "Female"},
        "",
        [_ITEM],
        study_id="Study",
        f_profile_id="F1",
        outcome_id="Study:response:hyp1",
        condition_id="control",
        intentional_no_material_control=True,
    )
    assert render.stimulus_text == ""
    # No fabricated narrative text (e.g. "no narrative") is ever inserted.
    assert "narrative" not in render.user_prompt.lower()
    assert "Survey material:\n\n\nOUTCOME QUESTIONS" in render.user_prompt
    # nonstimulus_text must place the placeholder in the material slot, not at position 0.
    assert render.nonstimulus_text.startswith("Researchers") or render.nonstimulus_text.split("\n", 1)[0] == render.user_prompt.split("\n", 1)[0]
    assert "<<STIMULUS>>" in render.nonstimulus_text
    assert not render.nonstimulus_text.startswith("<<STIMULUS>>")


def test_mcginty730_real_archived_data_confirms_intentional_no_material_control(monkeypatch, tmp_path):
    """Integration check against the real archived study: McGinty730's "No
    narrative" control condition (t_hypothesis=0 for treatment_support:hyp2)
    has zero non-demographic material pages in its actual archived transcript,
    and is accepted only via the explicit control flag."""
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import render_prompt_validation as rpv

    if not rpv.ARCHIVE_HYPOTHESES_PATH.exists() or not rpv.ARCHIVE_LLM_RDS_PATH.exists():
        pytest.skip("archived source data not present in this environment")

    # export_archive_source_rows() writes intermediate CSVs to rpv.OUTPUT_DIR;
    # redirect to an isolated tmp_path so this direct module-level call never
    # touches outputs/prompt_validation/f_external_source_{rows,selection}.csv.
    monkeypatch.setattr(rpv, "OUTPUT_DIR", tmp_path)

    effect_id = "McGinty730:treatment_support:hyp2"
    hypotheses = pd.read_csv(rpv.ARCHIVE_HYPOTHESES_PATH)
    study, outcome_name, _hyp = rpv._archive_outcome_name(effect_id)
    control_condition, _treatment_condition = rpv._condition_pair_for_effect(effect_id, hypotheses)
    assert control_condition == "No narrative"

    source_rows = rpv.export_archive_source_rows(
        pd.DataFrame([{"effect_id": effect_id, "study": study, "outcome_name": outcome_name, "condition_name": control_condition, "arm": "control"}])
    )
    control_row = source_rows.iloc[0]
    material, _item = archive_material_and_item(control_row, effect_id=effect_id, is_control_arm=True)
    assert material == ""
    with pytest.raises(ValueError):
        archive_material_and_item(control_row, effect_id=effect_id)
