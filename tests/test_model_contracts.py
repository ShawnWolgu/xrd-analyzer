"""Contracts for peak-model parameters and derived XRD quantities."""

from __future__ import annotations

from types import SimpleNamespace

import lmfit
import numpy as np
import pandas as pd
import pytest

from plot_from_excel import XRDPlotterFromExcel
from xrd_analyzer import (
    DEFAULT_WAVELENGTH_ANGSTROM,
    BraggGeometry,
    Fitter,
    FitterHistory,
    Peak,
    PROJECT_WORKBOOK_SCHEMA_VERSION,
    ProjectWorkbook,
    Reporter,
)
def test_fitter_history_round_trips_fitted_curves_parameters_and_mask() -> None:
    x_data = np.linspace(43.0, 45.0, 21)
    fitter = Fitter(x_data, np.ones_like(x_data))
    peak = fitter.add_peak(44.0, name="004")
    peak.center = 44.1
    peak.area = 12.0
    peak.height = 5.0
    peak.fwhm = 0.2
    peak.eta = 0.4
    params = lmfit.Parameters()
    params.add("c", value=1.0)
    params.add("p0_center", value=44.1)
    params.add("p0_amplitude", value=12.0)
    params.add("p0_sigma", value=0.1)
    params.add("p0_fraction", value=0.4)
    fitter.result = SimpleNamespace(
        params=params,
        chisqr=2.0,
        success=True,
        message="complete",
        nfev=4,
    )
    fitter.y_fit = np.linspace(1.0, 2.0, x_data.size)
    fitter.fit_mask = np.arange(x_data.size) % 2 == 0
    fitter.fit_config = {"objective_mode": "linear"}
    fitter.fit_diagnostics = {"success": True, "nfev": 4}
    fitter.restored_peak_curves = {0: np.linspace(0.0, 1.0, x_data.size)}

    history = FitterHistory(limit=5)
    assert history.record(fitter)
    assert not history.can_undo

    fitter.y_fit[:] = -1.0
    fitter.peaks[0].center = 44.5
    fitter.result.params["p0_center"].set(value=44.5)
    assert history.record(fitter)

    snapshot = history.undo()
    assert snapshot is not None
    restored = snapshot.restore_into(Fitter(x_data, np.ones_like(x_data)))

    assert restored.peaks[0].center == pytest.approx(44.1)
    assert restored.y_fit == pytest.approx(np.linspace(1.0, 2.0, x_data.size))
    assert restored.fit_mask.tolist() == (np.arange(x_data.size) % 2 == 0).tolist()
    assert restored.result.params["p0_center"].value == pytest.approx(44.1)
    assert restored.get_individual_peaks()[0] == pytest.approx(
        np.linspace(0.0, 1.0, x_data.size)
    )


def test_fitter_history_ignores_states_without_completed_fit() -> None:
    fitter = Fitter(np.array([43.0, 44.0]), np.array([1.0, 2.0]))
    fitter.add_peak(43.5)
    history = FitterHistory(limit=5)

    assert not history.record(fitter)
    assert not history.can_undo
    assert not history.can_redo


def test_first_order_bragg_inverse_is_exact_and_round_trips() -> None:
    assert BraggGeometry.two_theta_from_d(
        DEFAULT_WAVELENGTH_ANGSTROM,
        DEFAULT_WAVELENGTH_ANGSTROM,
    ) == pytest.approx(60.0, abs=1e-12)

    for two_theta in (10.0, 44.0, 90.0, 150.0):
        spacing = BraggGeometry.d_from_two_theta(
            two_theta,
            DEFAULT_WAVELENGTH_ANGSTROM,
        )
        restored_two_theta = BraggGeometry.two_theta_from_d(
            spacing,
            DEFAULT_WAVELENGTH_ANGSTROM,
        )
        assert restored_two_theta == pytest.approx(two_theta, abs=1e-12)


@pytest.mark.parametrize("spacing", [0.0, -1.0, 0.5])
def test_bragg_inverse_rejects_nonphysical_spacing(spacing: float) -> None:
    with pytest.raises(ValueError):
        BraggGeometry.two_theta_from_d(spacing, 1.5406)


def test_reporter_uses_explicit_project_wavelength_for_derived_values(tmp_path) -> None:
    fitter = Fitter(np.array([59.0, 61.0]), np.array([1.0, 1.0]))
    peak = fitter.add_peak(60.0, name="reference")
    peak.center = 60.0
    peak.area = 1.0
    peak.height = 1.0
    peak.fwhm = 0.2
    peak.eta = 0.5
    fitter.y_fit = fitter.y_data.copy()
    reporter = Reporter(fitter, wavelength_angstrom=2.0)

    result = reporter.calculate_characteristic_lengths()["Peak_0"]

    assert result["characteristic_length_angstrom"] == pytest.approx(2.0)
    assert result["wavelength_angstrom"] == pytest.approx(2.0)
    output = tmp_path / "wavelength.xlsx"
    reporter.export_results(output)
    peak_table = pd.read_excel(output, sheet_name="Peak_Parameters")
    assert peak_table.loc[0, "Wavelength_Angstrom"] == pytest.approx(2.0)
    restored = ProjectWorkbook.load(output)
    assert restored["project_state"]["wavelength_angstrom"] == pytest.approx(2.0)
    assert restored["project_state"]["radiation_label"] == "自定义波长"


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


def test_peak_area_is_the_fitted_lmfit_amplitude() -> None:
    params = lmfit.Parameters()
    params.add("p0_center", value=44.0)
    params.add("p0_amplitude", value=123.456)
    params.add("p0_sigma", value=0.2)
    params.add("p0_fraction", value=0.5)
    peak = Peak(0, 44.0)

    peak.set_result(params, fitted_height=150.0)

    assert peak.area == pytest.approx(123.456)


def test_fwhm_bounds_are_exactly_twice_the_sigma_bounds() -> None:
    assert Fitter.fwhm_bounds("film") == pytest.approx((0.02, 3.0))
    assert Fitter.fwhm_bounds("substrate") == pytest.approx((0.02, 2.0))


def test_shift_peaks_moves_effective_centers_and_bounds_atomically() -> None:
    fitter = Fitter(np.linspace(40.0, 50.0, 101), np.ones(101))
    first = fitter.add_peak(43.0, (42.5, 43.5), name="first")
    second = fitter.add_peak(47.0, (46.75, 47.25), name="second")
    first.center = 43.2
    fitter.result = SimpleNamespace(params=lmfit.Parameters())
    fitter.y_fit = np.ones(101)

    assert fitter.shift_peaks(0.1) == 2
    assert [peak.center_guess for peak in fitter.peaks] == pytest.approx([43.3, 47.1])
    assert fitter.peaks[0].bounds == pytest.approx((42.8, 43.8))
    assert fitter.peaks[1].bounds == pytest.approx((46.85, 47.35))
    assert fitter.result is None
    assert fitter.y_fit is None

    previous = [(peak.center_guess, peak.bounds) for peak in fitter.peaks]
    with pytest.raises(ValueError, match="数据范围"):
        fitter.shift_peaks(3.0)
    assert [(peak.center_guess, peak.bounds) for peak in fitter.peaks] == previous


def test_accept_result_copies_all_independent_peak_shape_parameters() -> None:
    fitter = Fitter(np.linspace(43.0, 45.0, 101), np.ones(101))
    peak = fitter.add_peak(44.0, (43.5, 44.5))
    params = lmfit.Parameters()
    params.add("p0_center", value=44.1)
    params.add("p0_amplitude", value=123.4)
    params.add("p0_sigma", value=0.2)
    params.add("p0_fraction", value=0.35)
    fitter.result = SimpleNamespace(params=params)

    fitter.accept_current_result()

    assert peak.center_guess == pytest.approx(44.1)
    assert peak.area_guess == pytest.approx(123.4)
    assert peak.sigma_guess == pytest.approx(0.2)
    assert peak.fraction_guess == pytest.approx(0.35)


def test_frozen_peak_shape_fixes_all_independent_pseudo_voigt_parameters() -> None:
    fitter = Fitter(np.linspace(43.0, 45.0, 101), np.ones(101))
    peak = fitter.add_peak(44.0, (43.5, 44.5))
    peak.center_guess = 44.1
    peak.area_guess = 123.4
    peak.sigma_guess = 0.2
    peak.fraction_guess = 0.35
    peak.fit_state = "frozen"

    fitter.build_model(min_peak_separation=0.0)

    for name, expected in {
        "center": 44.1,
        "amplitude": 123.4,
        "sigma": 0.2,
        "fraction": 0.35,
    }.items():
        parameter = fitter.params[f"p0_{name}"]
        assert parameter.value == pytest.approx(expected)
        assert parameter.vary is False


def test_disabled_peak_remains_in_model_as_a_zero_component() -> None:
    fitter = Fitter(np.linspace(43.0, 45.0, 101), np.ones(101))
    peak = fitter.add_peak(44.0, (43.5, 44.5))
    peak.fit_state = "disabled"

    fitter.build_model(min_peak_separation=0.0)

    assert fitter.params["p0_amplitude"].value == pytest.approx(0.0)
    assert fitter.params["p0_amplitude"].vary is False


def test_accepting_a_stage_does_not_overwrite_disabled_peak_guesses() -> None:
    fitter = Fitter(np.linspace(43.0, 46.0, 151), np.ones(151))
    active = fitter.add_peak(44.0, (43.5, 44.5), name="active")
    disabled = fitter.add_peak(45.0, (44.5, 45.5), name="disabled")
    disabled.fit_state = "disabled"
    disabled.area_guess = 50.0
    disabled.sigma_guess = 0.3
    disabled.fraction_guess = 0.25
    params = lmfit.Parameters()
    for index, center in enumerate((44.1, 45.1)):
        params.add(f"p{index}_center", value=center)
        params.add(f"p{index}_amplitude", value=100.0 + index)
        params.add(f"p{index}_sigma", value=0.2 + index * 0.1)
        params.add(f"p{index}_fraction", value=0.4 + index * 0.1)
    fitter.result = SimpleNamespace(params=params)

    fitter.accept_current_result()

    assert active.area_guess == pytest.approx(100.0)
    assert disabled.area_guess == pytest.approx(50.0)
    assert disabled.sigma_guess == pytest.approx(0.3)
    assert disabled.fraction_guess == pytest.approx(0.25)


def test_log_residual_uses_visible_intensity_floor_without_six_decade_zero_penalty() -> None:
    residual = Fitter.log_residual(
        np.array([0.0, 9.0]),
        np.array([1.0, 9.0]),
        intensity_floor=1.0,
    )

    assert residual == pytest.approx(np.array([-np.log10(2.0), 0.0]))


def test_log_residual_rejects_negative_processed_intensity() -> None:
    with pytest.raises(ValueError, match="负强度"):
        Fitter.log_residual(
            np.array([-0.1, 1.0]),
            np.array([0.1, 1.0]),
            intensity_floor=1.0,
        )


def test_fit_mask_combines_manual_include_and_exclude_ranges() -> None:
    x_data = np.arange(0.0, 11.0)

    mask = Fitter.build_fit_mask(
        x_data,
        include_ranges=[(2.0, 8.0)],
        exclude_ranges=[(4.0, 5.0)],
    )

    assert x_data[mask].tolist() == [2.0, 3.0, 6.0, 7.0, 8.0]


def test_fit_mask_excludes_missing_observations() -> None:
    x_data = np.array([42.0, 43.0, 44.0, 45.0])
    y_data = np.array([10.0, np.nan, 30.0, 40.0])

    mask = Fitter.build_fit_mask(x_data, y_data=y_data)

    assert mask.tolist() == [True, False, True, True]


def test_candidate_fit_does_not_replace_guesses_until_explicit_acceptance() -> None:
    x_data = np.linspace(43.0, 45.0, 201)
    source_model = lmfit.models.PseudoVoigtModel(prefix="source_")
    source_params = source_model.make_params(
        amplitude=100.0,
        center=44.0,
        sigma=0.1,
        fraction=0.3,
    )
    y_data = source_model.eval(source_params, x=x_data)
    fitter = Fitter(x_data, y_data)
    peak = fitter.add_peak(44.0, (43.5, 44.5))
    peak.fixed_center = True
    peak.sigma_guess = 0.3
    fitter.build_model(min_peak_separation=0.0, fixed_background=0.0)

    fitter.execute_fitting(
        method="least_squares",
        objective_mode="linear",
        include_ranges=[(43.5, 44.5)],
        exclude_ranges=[(43.9, 43.92)],
    )

    fitted_sigma = fitter.result.params["p0_sigma"].value
    assert fitted_sigma == pytest.approx(0.1, abs=1e-5)
    assert peak.sigma_guess == pytest.approx(0.3)
    assert fitter.fit_diagnostics["method"] == "least_squares"
    assert fitter.fit_diagnostics["objective_mode"] == "linear"
    assert fitter.fit_diagnostics["fit_point_count"] < len(x_data)

    fitter.accept_current_result()

    assert peak.sigma_guess == pytest.approx(fitted_sigma)
    assert peak.area_guess == pytest.approx(
        fitter.result.params["p0_amplitude"].value
    )


def test_fit_metrics_use_only_fit_mask_and_do_not_report_rmse() -> None:
    fitter = Fitter(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
    fitter.y_fit = np.array([1.0, 2.0, 1000.0])
    fitter.fit_mask = np.array([True, True, False])
    fitter.result = SimpleNamespace(chisqr=0.125)
    fitter.fit_config = {
        "objective_mode": "mixed",
        "intensity_floor": 1.0,
        "fit_point_count": 2,
    }

    metrics = Reporter(fitter).calculate_metrics()

    assert metrics["R_squared_fit"] == pytest.approx(1.0)
    assert metrics["R_squared_fit_scale"] == "linear_intensity"
    assert metrics["Fit_Point_Count"] == 2
    assert metrics["Objective_SSE"] == pytest.approx(0.125)
    assert metrics["Objective_Mode"] == "mixed"
    assert "R_squared" not in metrics
    assert "RMSE" not in metrics
    assert "Log_RMSE" not in metrics
    assert "Reduced_Chi_squared" not in metrics
    assert "AIC" not in metrics
    assert "BIC" not in metrics


def test_remove_peak_reindexes_and_invalidates_completed_fit_state() -> None:
    x_data = np.linspace(43.0, 47.0, 201)
    fitter = Fitter(x_data, np.ones_like(x_data))
    for center, name in ((44.0, "A"), (45.0, "B"), (46.0, "C")):
        peak = fitter.add_peak(center, (center - 0.4, center + 0.4), name=name)
        peak.center = center
        peak.height = 100.0
        peak.fwhm = 0.2
        peak.area = 20.0
        peak.eta = 0.5

    fitter.build_model(min_peak_separation=0.0)
    fitter.result = object()
    fitter.y_fit = np.ones_like(x_data)
    fitter.background = np.zeros_like(x_data)

    fitter.remove_peak(1)

    assert [(peak.peak_id, peak.name) for peak in fitter.peaks] == [(0, "A"), (1, "C")]
    assert fitter.model is None
    assert fitter.params is None
    assert fitter.result is None
    assert fitter.y_fit is None
    assert fitter.background is None
    for peak in fitter.peaks:
        assert peak.center is None
        assert peak.height is None
        assert peak.fwhm is None
        assert peak.area is None
        assert peak.eta is None


@pytest.mark.scientific
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
def test_minimum_peak_separation_tracks_the_fitted_first_center() -> None:
    fitter = Fitter(np.linspace(43.0, 47.0, 201), np.ones(201))
    fitter.add_peak(44.0, (43.0, 46.0))
    fitter.add_peak(44.5, (43.0, 47.0))
    fitter.build_model(min_peak_separation=0.3)

    fitter.params["p0_center"].set(value=45.0)
    fitter.params.update_constraints()

    assert fitter.params["p1_center"].expr == "p0_center + p1_center_gap"
    assert fitter.params["p1_center"].value >= fitter.params["p0_center"].value + 0.3


@pytest.mark.scientific
def test_minimum_peak_separation_keeps_each_center_inside_its_own_bounds() -> None:
    fitter = Fitter(np.linspace(43.0, 45.0, 201), np.ones(201))
    fitter.add_peak(43.8, (43.0, 44.2))
    fitter.add_peak(44.3, (44.0, 44.6))

    fitter.build_model(min_peak_separation=0.3)
    fitter.params["p0_center"].set(value=44.2)
    fitter.params["p1_center_gap"].set(value=1.6)
    fitter.params.update_constraints()

    assert fitter.params["p1_center"].value <= 44.6


@pytest.mark.scientific
def test_minimum_peak_separation_does_not_override_a_fixed_center() -> None:
    fitter = Fitter(np.linspace(43.0, 45.0, 201), np.ones(201))
    fitter.add_peak(43.8, (43.0, 44.2))
    fixed_peak = fitter.add_peak(44.3, (44.0, 44.6))
    fixed_peak.fixed_center = True

    fitter.build_model(min_peak_separation=0.3)

    assert fitter.params["p1_center"].expr is None
    assert fitter.params["p1_center"].vary is False
    assert fitter.params["p1_center"].value == pytest.approx(44.3)
    assert fitter.params["p0_center"].max <= 44.0 + 1e-12


def _two_theta_from_d_spacing(d_spacing: float, wavelength: float = 1.5406) -> float:
    theta = np.arcsin(wavelength / (2.0 * d_spacing))
    return float(np.degrees(2.0 * theta))


@pytest.mark.scientific
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
    fitter.fit_config = {
        "method": "least_squares",
        "objective_mode": "log",
        "intensity_floor": 3.0,
        "include_ranges": [(43.0, 45.0)],
        "exclude_ranges": [],
    }
    fitter.fit_diagnostics = {"success": True, "boundary_hits": []}
    output = tmp_path / "results.xlsx"

    Reporter(fitter).export_results(output)
    peak_table = pd.read_excel(output, sheet_name="Peak_Parameters")
    sheet_names = pd.ExcelFile(output).sheet_names
    fit_configuration = pd.read_excel(output, sheet_name="Fit_Configuration")

    assert "Characteristic_Length_d_Angstrom" in peak_table.columns
    assert "d_spacing_Å" not in peak_table.columns
    assert peak_table.loc[0, "Characteristic_Length_d_Angstrom"] == pytest.approx(d_expected)
    assert all("lattice" not in column.lower() for column in peak_table.columns)
    assert "Structure_Analysis" not in sheet_names
    assert "Lattice_Parameters" not in sheet_names
    assert fit_configuration.loc[0, "objective_mode"] == "log"
    assert fit_configuration.loc[0, "intensity_floor"] == pytest.approx(3.0)


def test_excel_project_round_trip_preserves_data_peak_state_and_gui_state(tmp_path) -> None:
    x_data = np.linspace(42.0, 45.0, 31)
    raw_data = np.linspace(5.0, 20.0, 31)
    processed_data = raw_data - 1.0
    fitter = Fitter(x_data, processed_data)
    peak = fitter.add_peak(43.2, (43.0, 43.4), "substrate", "MgO")
    peak.center = 43.21
    peak.center_guess = 43.2
    peak.area = 123.4
    peak.area_guess = 120.0
    peak.height = 500.0
    peak.height_guess = 480.0
    peak.fwhm = 0.08
    peak.sigma_guess = 0.04
    peak.eta = 0.35
    peak.fraction_guess = 0.3
    peak.fixed_center = True
    peak.fixed_fwhm = True
    peak.fit_state = "frozen"
    fitter.y_fit = processed_data.copy()
    fitter.fit_config = {"objective_mode": "log", "intensity_floor": 3.0}
    fitter.fit_diagnostics = {"success": True, "boundary_hits": []}
    output = tmp_path / "project.xlsx"
    project_state = {
        "schema_version": PROJECT_WORKBOOK_SCHEMA_VERSION,
        "range_min": 42.0,
        "range_max": 45.0,
        "filter_type": "Savitzky-Golay",
        "objective_mode": "log",
        "yscale": "log",
        "wavelength_angstrom": DEFAULT_WAVELENGTH_ANGSTROM,
        "radiation_label": "Cu Kα1",
    }

    source_x = np.array([42.0, 42.5, 43.0])
    source_y = np.array([100.0, 110.0, 120.0])
    Reporter(
        fitter,
        x_original=x_data,
        y_original=raw_data,
    ).export_results(
        output,
        project_state=project_state,
        source_datasets=[("scan-001.txt", source_x, source_y)],
    )
    restored = ProjectWorkbook.load(output)

    assert restored["x_data"] == pytest.approx(x_data)
    assert restored["processed_intensity"] == pytest.approx(processed_data)
    assert restored["raw_intensity"] == pytest.approx(raw_data)
    assert restored["project_state"] == project_state
    source_path, restored_source_x, restored_source_y = restored["source_datasets"][0]
    assert source_path == "scan-001.txt"
    assert restored_source_x == pytest.approx(source_x)
    assert restored_source_y == pytest.approx(source_y)
    restored_peak = restored["peaks"][0]
    assert restored_peak["Name"] == "MgO"
    assert restored_peak["Bounds_Min_2theta"] == pytest.approx(43.0)
    assert restored_peak["Bounds_Max_2theta"] == pytest.approx(43.4)
    assert restored_peak["Area_Guess"] == pytest.approx(120.0)
    assert restored_peak["Fixed_Center"] is True
    assert restored_peak["Fixed_FWHM"] is True
    assert restored_peak["Fit_State"] == "frozen"


def test_export_marks_uncovered_original_data_as_missing(tmp_path) -> None:
    fitter = Fitter(
        np.array([42.0, 43.0, 44.0]),
        np.array([10.0, 20.0, 30.0]),
    )
    fitter.y_fit = fitter.y_data.copy()
    output = tmp_path / "missing-original.xlsx"

    Reporter(
        fitter,
        x_original=np.array([43.0, 44.0]),
        y_original=np.array([20.0, 30.0]),
    ).export_results(output)

    full_data = pd.read_excel(output, sheet_name="Full_Data")
    assert np.isnan(full_data.loc[0, "Original_Intensity"])
    assert full_data.loc[1:, "Original_Intensity"].tolist() == pytest.approx([20.0, 30.0])


def test_excel_plotter_accepts_new_and_historical_characteristic_length_columns() -> None:
    assert (
        XRDPlotterFromExcel._characteristic_length_column(
            ["Peak_ID", "Characteristic_Length_d_Angstrom"]
        )
        == "Characteristic_Length_d_Angstrom"
    )
    assert (
        XRDPlotterFromExcel._characteristic_length_column(["Peak_ID", "d_spacing_Å"])
        == "d_spacing_Å"
    )
    assert XRDPlotterFromExcel._characteristic_length_column(["Peak_ID", "FWHM"]) is None


def test_excel_plotter_extracts_only_peak_lengths_from_legacy_sheet(tmp_path) -> None:
    workbook = tmp_path / "legacy-results.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"2theta": [40.0], "Fitted_Intensity": [1.0]}).to_excel(
            writer,
            sheet_name="Full_Data",
            index=False,
        )
        pd.DataFrame(
            {
                "Peak_ID": [0],
                "Type": ["film"],
                "Center_2theta": [44.0],
                "FWHM": [0.2],
                "Height": [100.0],
                "Area": [20.0],
            }
        ).to_excel(writer, sheet_name="Peak_Parameters", index=False)
        pd.DataFrame({"R_squared": [0.99]}).to_excel(
            writer,
            sheet_name="Fit_Metrics",
            index=False,
        )
        pd.DataFrame(
            {
                "Peak_0_d_spacing": [2.05],
                "Tetragonality_c/a_ratio": [1.03],
            }
        ).to_excel(writer, sheet_name="Lattice_Parameters", index=False)

    plotter = XRDPlotterFromExcel(workbook)
    plotter.load_data()

    assert plotter.peaks.loc[0, "Characteristic_Length_d_Angstrom"] == pytest.approx(2.05)
    assert all("Tetragonality" not in column for column in plotter.peaks.columns)
