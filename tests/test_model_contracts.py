"""Contracts for peak-model parameters and derived XRD quantities."""

from __future__ import annotations

import lmfit
import numpy as np
import pandas as pd
import pytest

from xrd_analyzer import Fitter, Peak, Reporter


def test_lmfit_pseudo_voigt_fwhm_is_two_sigma() -> None:
    fitter = Fitter(np.linspace(43.0, 45.0, 101), np.ones(101))
    fitter.add_peak(44.0, (43.5, 44.5))
    fitter.build_model(min_peak_separation=0.0)

    fitter.params["p0_sigma"].set(value=0.2)
    fitter.params.update_constraints()

    assert fitter.params["p0_fwhm"].value == pytest.approx(0.4)


def test_equal_fwhm_option_links_film_peak_sigmas() -> None:
    fitter = Fitter(np.linspace(43.0, 46.0, 151), np.ones(151))
    fitter.add_peak(44.0, (43.5, 44.5), peak_type="film")
    fitter.add_peak(45.0, (44.5, 45.5), peak_type="film")

    fitter.build_model(constrain_fwhm=True, min_peak_separation=0.0)

    assert fitter.params["p1_sigma"].expr == "p0_sigma"


@pytest.mark.scientific
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known legacy issue: Peak.set_result applies the Gaussian 2.3548 factor although "
        "lmfit PseudoVoigtModel defines FWHM as 2*sigma."
    ),
)
def test_peak_result_uses_lmfit_pseudo_voigt_fwhm_definition() -> None:
    params = lmfit.Parameters()
    params.add("p0_center", value=44.0)
    params.add("p0_amplitude", value=100.0)
    params.add("p0_sigma", value=0.2)
    params.add("p0_fraction", value=0.5)
    peak = Peak(0, 44.0)

    peak.set_result(params, fitted_height=150.0)

    assert peak.fwhm == pytest.approx(0.4)


@pytest.mark.scientific
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known legacy issue: the minimum separation is a static bound derived from the first "
        "peak guess, not a relation between fitted centers."
    ),
)
def test_minimum_peak_separation_tracks_the_fitted_first_center() -> None:
    fitter = Fitter(np.linspace(43.0, 47.0, 201), np.ones(201))
    fitter.add_peak(44.0, (43.0, 46.0))
    fitter.add_peak(44.5, (43.0, 47.0))
    fitter.build_model(min_peak_separation=0.3)

    fitter.params["p0_center"].set(value=45.0)
    fitter.params["p1_center"].set(value=44.5)
    fitter.params.update_constraints()

    assert fitter.params["p1_center"].value >= fitter.params["p0_center"].value + 0.3


def _two_theta_from_d_spacing(d_spacing: float, wavelength: float = 1.5406) -> float:
    theta = np.arcsin(wavelength / (2.0 * d_spacing))
    return float(np.degrees(2.0 * theta))


@pytest.mark.scientific
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known legacy issue: Reporter exposes lattice/tetragonality terminology instead of "
        "per-peak Bragg characteristic lengths."
    ),
)
def test_reporter_returns_characteristic_length_without_lattice_inference() -> None:
    d_004 = 1.025
    d_111 = 2.300
    fitter = Fitter(np.array([40.0, 50.0]), np.array([1.0, 1.0]))
    peak_004 = fitter.add_peak(
        _two_theta_from_d_spacing(d_004),
        peak_type="film",
        name="004",
    )
    peak_111 = fitter.add_peak(
        _two_theta_from_d_spacing(d_111),
        peak_type="film",
        name="111",
    )
    peak_004.center = peak_004.center_guess
    peak_111.center = peak_111.center_guess

    result = Reporter(fitter).calculate_characteristic_lengths()

    assert set(result) == {"Peak_0", "Peak_1"}
    assert result["Peak_0"]["characteristic_length_angstrom"] == pytest.approx(d_004)
    assert result["Peak_1"]["characteristic_length_angstrom"] == pytest.approx(d_111)
    assert result["Peak_0"]["reflection_label"] == "004"
    assert result["Peak_1"]["reflection_label"] == "111"
    assert "Tetragonality" not in result


@pytest.mark.scientific
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known legacy issue: Excel labels Bragg d as d_spacing and can export inferred "
        "structure analysis instead of the characteristic-length contract."
    ),
)
def test_export_labels_bragg_d_as_characteristic_length(tmp_path) -> None:
    d_expected = 2.05
    fitter = Fitter(np.array([40.0, 41.0]), np.array([10.0, 12.0]))
    peak = fitter.add_peak(
        _two_theta_from_d_spacing(d_expected),
        peak_type="film",
        name="002",
    )
    peak.center = peak.center_guess
    peak.fwhm = 0.2
    peak.height = 100.0
    peak.area = 20.0
    peak.eta = 0.5
    fitter.y_fit = fitter.y_data.copy()
    output = tmp_path / "results.xlsx"

    Reporter(fitter).export_results(output)
    peak_table = pd.read_excel(output, sheet_name="Peak_Parameters")

    assert "Characteristic_Length_d_Angstrom" in peak_table.columns
    assert "d_spacing_Å" not in peak_table.columns
    assert peak_table.loc[0, "Characteristic_Length_d_Angstrom"] == pytest.approx(d_expected)
    assert all("lattice" not in column.lower() for column in peak_table.columns)
