"""Tests for the Orchinik G-v2 serving-only amendment: scientific
equivalence to v1, zero collision (with v1 Orchinik, target G/G-v2/smoke/F/
ATP/everything else), and that the v1 protocol/manifests are never
modified by building or freezing the v2 amendment."""

from __future__ import annotations

import csv
import glob
import hashlib
import sys
from pathlib import Path

import pytest

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
for p in (PIPELINE_ROOT, PIPELINE_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import build_orchinik_g_domain_confirmation_manifest_v2 as v2_mod  # noqa: E402
import freeze_orchinik_g_v2_serving_amendment as amend_mod  # noqa: E402

pytestmark = pytest.mark.skipif(not v2_mod.OUT_ROOT_V2.exists(), reason="Orchinik G-v2 manifests not built in this environment")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ids(path: Path) -> set[str]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return {row["custom_id"] for row in r} if "custom_id" in (r.fieldnames or []) else set()


def test_v2_manifests_have_2545_requests_each():
    for model in ("google_gemma-4-31B-it", "deepseek-ai_DeepSeek-V4-Pro-0813"):
        ids = _load_ids(v2_mod.OUT_ROOT_V2 / model / "request_manifest.csv")
        assert len(ids) == 2545


def test_v1_protocol_and_manifests_untouched_by_v2_build_and_freeze():
    v1_protocol = amend_mod.V1_PROTOCOL_PATH
    v1_gemma_manifest = amend_mod.V1_ROOT / "google_gemma-4-31B-it" / "request_manifest.csv"
    before = {v1_protocol: v1_protocol.read_bytes(), v1_gemma_manifest: v1_gemma_manifest.read_bytes()}
    v2_mod.main()
    amend_mod.main()
    for path, content in before.items():
        assert path.read_bytes() == content


def test_zero_collision_with_v1_and_every_other_manifest():
    gemma_v2 = _load_ids(v2_mod.OUT_ROOT_V2 / "google_gemma-4-31B-it" / "request_manifest.csv")
    deepseek_v2 = _load_ids(v2_mod.OUT_ROOT_V2 / "deepseek-ai_DeepSeek-V4-Pro-0813" / "request_manifest.csv")
    assert gemma_v2 & deepseek_v2 == set()

    prior: set[str] = set()
    for path in glob.glob(str(PIPELINE_ROOT / "outputs" / "**" / "*.csv"), recursive=True):
        if "orchinik_g_domain_confirmation_v2" in path:
            continue
        prior |= _load_ids(Path(path))
    assert len(prior) > 250_000  # threshold lowered after submission-cleanup removed several deprecated development manifests; the collision invariant below is unaffected
    assert gemma_v2 & prior == set()
    assert deepseek_v2 & prior == set()


def test_amendment_records_required_fields_and_self_hash():
    result = amend_mod.main()
    assert result["amendment_type"] == "ORCHINIK_DOMAIN_CONFIRMATION_SERVING_FORMAT_ONLY_V2"
    assert result["v1_protocol_unmodified_by_this_amendment"] is True
    assert result["v1_protocol_unchanged_verified"] is True
    assert result["format_only_change"]["scientific_equivalence_programmatically_verified"] is True
    assert result["total_requests"] == 5090
    sha_file = (amend_mod.V2_ROOT / "serving_amendment.sha256.txt").read_text(encoding="utf-8").strip()
    assert sha_file == result["amendment_sha256"]
    assert _sha256(amend_mod.V2_ROOT / "serving_amendment.json") == sha_file
