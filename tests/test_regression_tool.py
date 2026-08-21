"""Tests for the table comparator bundled with the repository skill."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_SCRIPT = (
    Path(__file__).parents[1]
    / ".agents"
    / "skills"
    / "xrd-scientific-maintenance"
    / "scripts"
    / "compare_tables.py"
)
_SPEC = importlib.util.spec_from_file_location("xrd_compare_tables", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_compare_frames_accepts_values_within_tolerance() -> None:
    baseline = pd.DataFrame({"Peak_ID": [0, 1], "Center": [44.0, 45.0]})
    candidate = pd.DataFrame({"Peak_ID": [1, 0], "Center": [45.0 + 1e-9, 44.0]})

    problems = _MODULE.compare_frames(
        baseline,
        candidate,
        keys=["Peak_ID"],
        rtol=1e-8,
        atol=1e-10,
    )

    assert problems == []


def test_compare_frames_reports_numeric_differences() -> None:
    baseline = pd.DataFrame({"Peak_ID": [0], "Center": [44.0]})
    candidate = pd.DataFrame({"Peak_ID": [0], "Center": [44.1]})

    problems = _MODULE.compare_frames(
        baseline,
        candidate,
        keys=["Peak_ID"],
        rtol=1e-8,
        atol=1e-10,
    )

    assert len(problems) == 1
    assert problems[0].startswith("Center: 1 numeric mismatches")
