"""Shared fixtures/sys.path setup -- see tests/elicitation/conftest.py for
the original of this pattern. All synthetic/unit-level -- none call a real
model."""

from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PIPELINE_ROOT))
