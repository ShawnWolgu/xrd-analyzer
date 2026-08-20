"""Regression checks for the tracked historical sample scan."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xrd_analyzer import DataLoader


def test_tracked_sample_scan_is_readable_and_structurally_stable() -> None:
    sample = Path(__file__).parents[1] / "230610_sgPZT(30_70)_004.TXT"

    x_data, y_data = DataLoader.load_txt(sample)

    assert len(x_data) == len(y_data) == 1001
    assert x_data[0] == 90.0
    assert x_data[-1] == 110.0
    assert np.all(np.diff(x_data) > 0.0)
    assert np.isfinite(y_data).all()
