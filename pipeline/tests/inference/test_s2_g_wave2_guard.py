"""Tests for the S2-specific target-G Wave-2 prerequisite guard. Proves it
authorizes against the real repo state today (S2 manifest + projector now
frozen) and that its refusal logic works when a prerequisite is genuinely
missing -- and, critically, that it never checks F*/R_F (isolated via
monkeypatch, since real F* IS frozen in this repo and would mask a bug)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference.s2_g_wave2_guard import S2GWave2NotAuthorized, assert_s2_g_wave2_prerequisites_frozen


def test_prerequisites_frozen_given_real_repo_state():
    result = assert_s2_g_wave2_prerequisites_frozen()
    assert result["selected_g_model"] == "google/gemma-4-31B-it"


def test_never_checks_f_star_or_r_f(monkeypatch):
    import inference.s2_g_wave2_guard as guard

    def fake_selected_model(role, **kw):
        if role == "f":
            raise RuntimeError("F* deliberately broken")
        return "google/gemma-4-31B-it"

    # If F* or R_F were checked, breaking "f" here would raise -- they are not.
    monkeypatch.setattr(guard, "selected_model", fake_selected_model)
    result = assert_s2_g_wave2_prerequisites_frozen()
    assert result["selected_g_model"] == "google/gemma-4-31B-it"


def test_refuses_when_g_star_missing(monkeypatch):
    import inference.s2_g_wave2_guard as guard

    monkeypatch.setattr(guard, "selected_model", lambda role, **kw: (_ for _ in ()).throw(RuntimeError("G* not frozen")))
    with pytest.raises(S2GWave2NotAuthorized, match="G\\* is not frozen"):
        assert_s2_g_wave2_prerequisites_frozen()


def test_refuses_when_s2_manifest_missing(monkeypatch):
    import inference.s2_g_wave2_guard as guard

    monkeypatch.setattr(guard, "S2_FINAL_SUBMISSION_MANIFEST_PATH", Path("/nonexistent/manifest.json"))
    with pytest.raises(S2GWave2NotAuthorized, match="S2 final-submission manifest is not frozen"):
        assert_s2_g_wave2_prerequisites_frozen()


def test_refuses_when_projector_missing(monkeypatch):
    import inference.s2_g_wave2_guard as guard

    monkeypatch.setattr(guard, "PROJECTOR_ARTIFACT_PATH", Path("/nonexistent/projector.json"))
    with pytest.raises(S2GWave2NotAuthorized, match="shared common-shift/support-projection method is not frozen"):
        assert_s2_g_wave2_prerequisites_frozen()
