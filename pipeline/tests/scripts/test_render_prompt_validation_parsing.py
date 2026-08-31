"""Regression test for the Haaland874 embedded-colon effect_id parsing fix:

_archive_outcome_name() (render_prompt_validation.py) used to split an
effect_id into exactly 3 parts on ":", which silently mis-parsed any
effect_id whose outcome name itself contains a colon (Haaland874's
"Affirmative action: Assistance" / "Affirmative action: Preference" are the
only two such effect_ids in the whole archive) -- the hypothesis segment
("hyp1") was dropped entirely and the outcome name was truncated, so the
downstream hypotheses.csv condition-name lookup found no matching rows and
_condition_pair_for_effect() raised "cannot identify archived control/
treatment condition names". The fix takes the LAST segment as hypothesis and
rejoins everything between the first and last colon as the outcome name --
already the correct, proven pattern used by
build_external_calibration_panels.parse_effect_id."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PIPELINE_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pytest  # noqa: E402
from render_prompt_validation import _archive_outcome_name  # noqa: E402


def test_embedded_colon_outcome_name_parsed_correctly():
    assert _archive_outcome_name("Haaland874:Affirmative action: Assistance:hyp1") == ("Haaland874", "Affirmative action: Assistance", "hyp1")
    assert _archive_outcome_name("Haaland874:Affirmative action: Preference:hyp1") == ("Haaland874", "Affirmative action: Preference", "hyp1")


def test_ordinary_single_colon_outcome_name_unaffected():
    """Every other effect_id in the archive has exactly one colon in the
    outcome-name segment (i.e. exactly 3 colon-separated parts total) --
    the fix must be byte-identical to the old naive 3-way split for these."""
    assert _archive_outcome_name("Haaland874:Racial Discrimination perception:hyp1") == ("Haaland874", "Racial Discrimination perception", "hyp1")
    assert _archive_outcome_name("Braman751:legitimacy:hyp2") == ("Braman751", "legitimacy", "hyp2")
    assert _archive_outcome_name("Terman1029:perception of domestic human rights conditions:hyp1") == ("Terman1029", "perception of domestic human rights conditions", "hyp1")


def test_multiple_embedded_colons_all_rejoined_into_outcome_name():
    assert _archive_outcome_name("S:a:b:c:hyp3") == ("S", "a:b:c", "hyp3")


def test_too_few_parts_still_raises():
    with pytest.raises(ValueError):
        _archive_outcome_name("only_one_colon:missing_hypothesis")
