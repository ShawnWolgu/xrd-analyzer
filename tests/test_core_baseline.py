"""Versioned scientific and regression baselines for the backend core."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xrd_analyzer.backend import (
    BraggGeometry,
    Fitter,
    Reporter,
)


BASELINE_DIRECTORY = Path(__file__).parent / "baselines"


def _load_baseline(name: str) -> dict:
    return json.loads((BASELINE_DIRECTORY / name).read_text(encoding="utf-8"))


def _normalized_pseudo_voigt(
    x_data: np.ndarray,
    *,
    area: float,
    center: float,
    sigma: float,
    fraction: float,
) -> np.ndarray:
    """Independent analytical implementation of lmfit's normalized model contract."""
    offset = x_data - center
    gaussian = (
        np.sqrt(np.log(2.0) / np.pi)
        / sigma
        * np.exp(-np.log(2.0) * (offset / sigma) ** 2)
    )
    lorentzian = sigma / (np.pi * (offset**2 + sigma**2))
    return area * ((1.0 - fraction) * gaussian + fraction * lorentzian)


@pytest.mark.scientific
@pytest.mark.baseline
def test_synthetic_reference_recovers_known_peak_parameters() -> None:
    baseline = _load_baseline("core_synthetic_v1.json")
    scan_spec = baseline["scan"]
    x_data = np.linspace(
        scan_spec["start"],
        scan_spec["stop"],
        scan_spec["point_count"],
    )
    y_data = np.full_like(x_data, scan_spec["constant_background"], dtype=float)
    for peak_spec in baseline["peaks"]:
        y_data += _normalized_pseudo_voigt(
            x_data,
            area=peak_spec["area"],
            center=peak_spec["center"],
            sigma=peak_spec["sigma"],
            fraction=peak_spec["fraction"],
        )

    fitter = Fitter(x_data, y_data)
    for peak_spec in baseline["peaks"]:
        peak = fitter.add_peak(
            peak_spec["center_guess"],
            tuple(peak_spec["bounds"]),
            peak_spec["type"],
            peak_spec["name"],
        )
        peak.area_guess = peak_spec["area"] * 0.9
        peak.sigma_guess = peak_spec["sigma"] * 1.1
        peak.fraction_guess = min(1.0, peak_spec["fraction"] + 0.05)

    fit_spec = baseline["fit"]
    fitter.build_model(
        min_peak_separation=fit_spec["minimum_peak_separation"],
        fixed_background=scan_spec["constant_background"],
    )
    fitter.execute_fitting(
        method=fit_spec["method"],
        objective_mode=fit_spec["objective_mode"],
    )
    reporter = Reporter(
        fitter,
        wavelength_angstrom=baseline["wavelength_angstrom"],
    )
    metrics = reporter.calculate_metrics()
    tolerances = baseline["tolerances"]

    assert fitter.result.success is True
    assert metrics["R_squared_fit"] >= tolerances["minimum_r_squared_fit"]
    for peak, expected in zip(fitter.peaks, baseline["peaks"]):
        assert peak.center == pytest.approx(
            expected["center"],
            abs=tolerances["center_absolute_degree"],
        )
        assert peak.fwhm == pytest.approx(
            2.0 * expected["sigma"],
            abs=tolerances["fwhm_absolute_degree"],
        )
        assert peak.area == pytest.approx(
            expected["area"],
            rel=tolerances["area_relative"],
        )
        assert peak.eta == pytest.approx(
            expected["fraction"],
            abs=tolerances["fraction_absolute"],
        )
        characteristic_length = BraggGeometry.d_from_two_theta(
            peak.center,
            baseline["wavelength_angstrom"],
        )
        assert characteristic_length == pytest.approx(
            expected["characteristic_length_angstrom"],
            abs=tolerances["characteristic_length_absolute_angstrom"],
        )
