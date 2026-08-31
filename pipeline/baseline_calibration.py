"""pipeline/baseline_calibration.py

External baseline-REALISM diagnostics: compare a simulated arm's raw,
native-response control-condition values against external human data
closely matched to the target domain. Diagnostic only, by design (per the
primary specification: distribution/"spread" correction is explicitly
deferred, not invented here) -- this module reports how realistic a
control distribution looks, it does not adjust it. Any future correction
must be validated on held-out real survey questions before being added.

Status: a real, closely domain-matched dataset exists -- a real megastudy
of behavioral interventions to catalyze climate advocacy (not literally
"trust in scientists", but genuinely climate-attitude/behavior-domain data;
see data/README.md and scripts/build_baseline_reference.py). It covers 4 of
this benchmark's 13 outcomes directly, on the same scale (belief_post,
policy_general, donation_ams, newsletter_signup); the other 9 have no
corresponding item and are left uncalibrated by load_baseline_references(),
which returns no entry for them -- not a guessed one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm, wasserstein_distance

BASELINE_REFERENCES_PATH = Path(__file__).resolve().parent / "data" / "baseline_references.json"


@dataclass
class BaselineReference:
    """A summary of an external human baseline for one item: mean and
    variance on the item's own scale, and optionally a demographic gradient
    (e.g. {"gender": {"Male": mean, "Female": mean}, ...}) for a future
    demographic-gradient realism check. All fields optional -- supply
    whatever the source study actually reports.
    """

    mean: float | None = None
    variance: float | None = None
    n: int | None = None
    demographic_gradient: Mapping[str, Mapping[str, float]] | None = None
    source: str = ""


def load_baseline_references(path: Path = BASELINE_REFERENCES_PATH) -> dict[str, BaselineReference]:
    """Load the real, closely domain-matched external baseline references
    built by scripts/build_baseline_reference.py from the climate-advocacy
    megastudy (see data/README.md). Returns {} (not an error) if the file
    doesn't exist yet -- callers should treat that the same as "no reference
    for this outcome", not a fatal error.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        outcome: BaselineReference(mean=v["mean"], variance=v["variance"], n=v.get("n"), source=v["source"])
        for outcome, v in raw.items()
    }


def assess_baseline_realism(elicited_control: Sequence[float], reference: BaselineReference | None) -> dict:
    """Compare a simulated arm's raw, native-response control values
    against an external reference, when one is supplied. Returns a report
    dict; `"status": "no_reference_data"` (not a fabricated pass/fail) when
    `reference` is None.

    `ks_vs_normal_approx`/`wasserstein_vs_normal_approx` are real distances
    between the elicited sample's empirical distribution and a Normal
    approximation built from the reference's mean/variance -- labeled
    "_vs_normal_approx" deliberately, since `data/baseline_references.json`
    stores only summary moments (mean/variance/n), not the reference's own
    raw sample, so a true empirical-vs-empirical Wasserstein/KS comparison
    isn't possible from what's available; this is a disclosed approximation,
    not a substitute for one.
    """
    values = np.asarray(elicited_control, dtype=float)
    elicited_mean = float(values.mean())
    elicited_var = float(values.var())

    if reference is None:
        return {
            "status": "no_reference_data",
            "detail": "no external trust-in-scientists (or comparable domain-matched) baseline dataset is available in data/; "
            "elicited moments are reported for information only, not compared against anything",
            "elicited_mean": elicited_mean,
            "elicited_variance": elicited_var,
        }

    report = {"status": "compared", "elicited_mean": elicited_mean, "elicited_variance": elicited_var, "reference_source": reference.source}
    if reference.mean is not None:
        report["mean_gap"] = elicited_mean - reference.mean
    if reference.variance is not None:
        report["variance_ratio"] = elicited_var / reference.variance if reference.variance > 0 else None
    if reference.mean is not None and reference.variance is not None and reference.variance > 0:
        reference_sample = norm.rvs(loc=reference.mean, scale=reference.variance**0.5, size=max(len(values), 2000), random_state=0)
        report["ks_vs_normal_approx"] = float(_ks_statistic(values, reference_sample))
        report["wasserstein_vs_normal_approx"] = float(wasserstein_distance(values, reference_sample))
    return report


def _ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max empirical-CDF gap),
    implemented directly rather than via scipy.stats.ks_2samp so the only
    new dependency this module needs beyond what's already required is
    scipy.stats.norm/wasserstein_distance (both already available)."""
    all_values = np.sort(np.concatenate([sample_a, sample_b]))
    cdf_a = np.searchsorted(np.sort(sample_a), all_values, side="right") / len(sample_a)
    cdf_b = np.searchsorted(np.sort(sample_b), all_values, side="right") / len(sample_b)
    return float(np.max(np.abs(cdf_a - cdf_b)))
