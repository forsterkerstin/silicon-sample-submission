"""pipeline/inference/simulate_response.py

The new primary respondent-generation API: one inference call per (profile,
condition), returning validated native integers for every applicable item.
No probabilities, no per-item calls, no treatment comparison requested of
the model. Every respondent's answer comes from the same prompt template
and model call shape; downstream DELTA assembly estimates any demographic
HTE from the paired F panel, not from extra response-generation settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import survey_content as sc
from inference.prompts import build_g_prompt_render

_ITEMS_BY_LABEL: dict[str, dict] | None = None


def _items_by_label() -> dict[str, dict]:
    global _ITEMS_BY_LABEL
    if _ITEMS_BY_LABEL is None:
        _ITEMS_BY_LABEL = {it["target_label"]: it for it in sc.load_items()}
    return _ITEMS_BY_LABEL


@dataclass(frozen=True)
class SurveyResponse:
    """One respondent's native answers, validated against each item's own
    scale bounds at construction time (Step 10: "validate aggressively").
    Raises ValueError immediately on any out-of-range or wrong-type value
    -- a respondent-generation bug should fail loudly here, not silently
    produce a bad row downstream.
    """

    values: dict[str, int]

    def __post_init__(self):
        items_by_label = _items_by_label()
        for label, value in self.values.items():
            item = items_by_label.get(label)
            if item is None:
                continue  # a caller-restricted item subset is fine; unknown labels aren't validated here
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{label}={value!r} is not an int")
            if item["scale"] == sc.SCALE_SLIDER_0_100 and not (0 <= value <= 100):
                raise ValueError(f"{label}={value} out of bounds [0, 100]")
            elif item["scale"] == sc.SCALE_DONATION_0_10 and not (0 <= value <= 10):
                raise ValueError(f"{label}={value} out of bounds [0, 10]")
            elif item["scale"] == sc.SCALE_BINARY_0_1 and value not in (0, 1):
                raise ValueError(f"{label}={value} not in {{0, 1}}")

    def __getitem__(self, label: str) -> int:
        return self.values[label]

    def items(self):
        return self.values.items()


def simulate_response(
    profile: Mapping[str, object],
    condition_stimulus: str,
    items: list[dict[str, str]],
    client,
    *,
    donor_key: str = "",
    condition_id: str = "",
    replicate_id: int = 1,
) -> SurveyResponse:
    """One inference call, all `items` answered together (within-person
    response dependence retained -- a single call sees its own earlier
    answers as part of the same generation, unlike separately elicited
    items). `client` is inference.client's VLLMNativeClient/HFNativeClient,
    or anything exposing the same __call__(messages, schema) -> dict[str,int]
    shape. `condition_stimulus` is the exact intervention text or one of the
    neutral control filler texts resolved by survey_content.get_condition_stimulus().
    """
    render = build_g_prompt_render(
        profile,
        condition_stimulus,
        items,
        donor_key=donor_key,
        condition_id=condition_id,
        replicate_id=replicate_id,
    )
    raw = client(render.messages, render.response_schema)
    key_map = render.response_key_map or {}
    return SurveyResponse(values={target_label: raw[response_key] for response_key, target_label in key_map.items()})
