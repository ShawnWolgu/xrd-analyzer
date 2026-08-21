"""Focused GUI checks for scientific terminology."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMessageBox
import pytest

from xrd_analyzer import (
    DEFAULT_WAVELENGTH_ANGSTROM,
    Fitter,
    PROJECT_WORKBOOK_SCHEMA_VERSION,
    Reporter,
)
from xrd_gui import FittingThread, XRDAnalyzerGUI
from xrd_session import AnalysisSession, PreprocessingStep, ScanData


def _window_with_two_peaks() -> tuple[QApplication, XRDAnalyzerGUI]:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()
    window.x_data = np.linspace(40.0, 50.0, 101)
    window.y_data = np.linspace(10.0, 20.0, 101)
    window.plot_canvas.set_data(window.x_data, window.y_data)
    window.fitter = Fitter(window.x_data, window.y_data)
    window.add_peak_to_fitter(44.0, (43.5, 44.5), "film", "004")
    window.add_peak_to_fitter(46.0, (45.5, 46.5), "film", "200")
    return app, window


def test_theoretical_d_mode_adds_peak_at_inverse_bragg_position() -> None:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()
    window.x_data = np.linspace(40.0, 80.0, 401)
    window.y_data = np.ones_like(window.x_data)
    window.plot_canvas.set_data(window.x_data, window.y_data)
    window.fitter = Fitter(window.x_data, window.y_data)
    window.peak_input_mode.setCurrentIndex(
        window.peak_input_mode.findData("d_spacing")
    )
    window.wavelength_spin.setValue(DEFAULT_WAVELENGTH_ANGSTROM)
    window.peak_value_input.setValue(DEFAULT_WAVELENGTH_ANGSTROM)
    window.peak_name_input.setText("reference")

    window.quick_add_peak()

    assert len(window.fitter.peaks) == 1
    assert window.fitter.peaks[0].center_guess == pytest.approx(60.0, abs=1e-10)
    assert window.fitter.peaks[0].name == "reference"
    assert "d = 1.540600 Å" in window.statusBar().currentMessage()
    assert "2θ = 60.000000°" in window.statusBar().currentMessage()

    window.close()
    app.processEvents()


def test_theoretical_d_mode_rejects_peak_outside_loaded_range(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()
    window.x_data = np.linspace(40.0, 50.0, 101)
    window.y_data = np.ones_like(window.x_data)
    window.fitter = Fitter(window.x_data, window.y_data)
    window.peak_input_mode.setCurrentIndex(
        window.peak_input_mode.findData("d_spacing")
    )
    window.wavelength_spin.setValue(DEFAULT_WAVELENGTH_ANGSTROM)
    window.peak_value_input.setValue(DEFAULT_WAVELENGTH_ANGSTROM)
    warnings = []
    monkeypatch.setattr(
        "xrd_gui.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    window.quick_add_peak()

    assert len(window.fitter.peaks) == 0
    assert warnings

    window.close()
    app.processEvents()


def test_physics_panel_reports_characteristic_length_without_lattice_inference() -> None:
    app = QApplication.instance() or QApplication([])
    fitter = Fitter(np.array([40.0, 41.0]), np.array([10.0, 12.0]))
    peak = fitter.add_peak(44.0, peak_type="film", name="004")
    peak.center = 44.0
    peak.fwhm = 0.2
    peak.height = 100.0
    peak.area = 20.0
    peak.eta = 0.5

    window = XRDAnalyzerGUI()
    window.fitter = fitter
    window.reporter = Reporter(fitter)
    window.display_physics_parameters()
    text = window.physics_text.toPlainText()

    assert "【反射峰特征长度 d】" in text
    assert "Peak 0 (004)" in text
    assert "程序不根据002、004、111、200等峰名自动推断晶格倍数" in text
    assert "c轴晶格常数" not in text
    assert "a轴晶格常数" not in text
    assert "四方度" not in text

    window.close()
    app.processEvents()


def test_fit_plot_callout_shows_only_fit_mask_r_squared() -> None:
    app = QApplication.instance() or QApplication([])
    fitter = Fitter(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 100.0]))
    fitter.y_fit = np.array([1.0, 2.0, 0.0])
    fitter.fit_mask = np.array([True, True, False])
    fitter.result = SimpleNamespace(chisqr=0.0)
    fitter.fit_config = {"objective_mode": "mixed", "fit_point_count": 2}
    fitter.get_individual_peaks = lambda: {}
    window = XRDAnalyzerGUI()
    window.fitter = fitter
    window.reporter = Reporter(fitter)
    window.reporter.calculate_metrics()

    window.plot_fitted_results()

    callout_text = "\n".join(text.get_text() for text in window.plot_canvas.axes.texts)
    assert "R²_fit = 1.000000" in callout_text
    assert "RMSE" not in callout_text
    assert "R² =" not in callout_text

    window.close()
    app.processEvents()


def test_editing_project_wavelength_updates_displayed_derived_values() -> None:
    app = QApplication.instance() or QApplication([])
    fitter = Fitter(np.array([59.0, 61.0]), np.array([1.0, 1.0]))
    peak = fitter.add_peak(60.0, name="reference")
    peak.center = 60.0
    peak.fwhm = 0.2
    peak.height = 1.0
    peak.area = 1.0
    peak.eta = 0.5
    window = XRDAnalyzerGUI()
    window.fitter = fitter
    window.reporter = Reporter(fitter)

    window.wavelength_spin.setValue(2.0)

    assert window.reporter.wavelength_angstrom == pytest.approx(2.0)
    text = window.physics_text.toPlainText()
    assert "λ = 2.000000 Å, 自定义波长" in text
    assert "2.000000 Å" in text
    assert "表观Scherrer相干畴尺寸估算" in text

    window.close()
    app.processEvents()


def test_peak_table_displays_fitted_area_between_position_and_height() -> None:
    app = QApplication.instance() or QApplication([])
    fitter = Fitter(np.array([43.0, 45.0]), np.array([10.0, 12.0]))
    peak = fitter.add_peak(44.0, peak_type="film", name="004")
    peak.center = 44.123
    peak.area = 123.456
    peak.height = 150.0
    peak.fwhm = 0.2345

    window = XRDAnalyzerGUI()
    window.fitter = fitter
    window.update_peak_table()

    headers = [
        window.peak_table.horizontalHeaderItem(column).text()
        for column in range(window.peak_table.columnCount())
    ]
    area_item = window.peak_table.item(0, 3)

    assert headers[2:5] == ["位置 (Pos)", "面积", "高度 (Height)"]
    assert area_item.text() == "123.46"
    assert not area_item.flags() & Qt.ItemIsEditable
    assert not area_item.flags() & Qt.ItemIsUserCheckable
    assert window.peak_table.item(0, 4).text() == "150.00"
    assert window.peak_table.item(0, 5).text() == "0.2345"
    assert headers[6:9] == ["η", "峰状态", "类型"]
    assert window.peak_table.item(0, 6).text() == "NaN"
    assert window.peak_table.item(0, 8).text() == "film"

    window.peak_table.item(0, 4).setCheckState(Qt.Checked)
    window.peak_table.item(0, 5).setCheckState(Qt.Checked)
    assert peak.fixed_height is True
    assert peak.fixed_fwhm is True

    peak.fixed_height = False
    peak.fixed_fwhm = False
    window.sync_table_to_peaks()
    assert peak.fixed_height is True
    assert peak.fixed_fwhm is True

    window.close()
    app.processEvents()


def test_fitting_thread_preserves_selected_method_objective_floor_and_ranges() -> None:
    fitter = Fitter(np.linspace(43.0, 45.0, 101), np.ones(101))
    thread = FittingThread(
        fitter,
        constrain_fwhm=False,
        min_separation=0.2,
        method="least_squares",
        objective_mode="log",
        log_weight=1.0,
        intensity_floor=3.0,
        include_ranges=[(42.5, 43.5)],
        exclude_ranges=[(42.9, 43.0)],
    )

    assert thread.method == "least_squares"
    assert thread.objective_mode == "log"
    assert thread.intensity_floor == pytest.approx(3.0)
    assert thread.include_ranges == [(42.5, 43.5)]
    assert thread.exclude_ranges == [(42.9, 43.0)]


def test_fitting_thread_reserves_native_stack_for_covariance_inversion() -> None:
    fitter = Fitter(np.linspace(43.0, 45.0, 101), np.ones(101))

    thread = FittingThread(fitter, constrain_fwhm=False, min_separation=0.2)

    assert thread.stackSize() >= 8 * 1024 * 1024


def test_fitting_thread_limits_blas_parallelism(monkeypatch) -> None:
    calls = []

    @contextmanager
    def fake_threadpool_limits(*, limits, user_api):
        calls.append((limits, user_api))
        yield

    class StubFitter:
        def __init__(self):
            self.built = False
            self.executed = False

        def build_model(self, **kwargs):
            self.built = True

        def execute_fitting(self, **kwargs):
            self.executed = True
            return object()

    monkeypatch.setattr("xrd_gui.threadpool_limits", fake_threadpool_limits)
    fitter = StubFitter()
    thread = FittingThread(fitter, constrain_fwhm=False, min_separation=0.2)

    thread.run()

    assert calls == [(1, "blas")]
    assert fitter.built is True
    assert fitter.executed is True


def test_background_least_squares_repeatedly_completes_covariance_step() -> None:
    app = QApplication.instance() or QApplication([])
    x_data = np.linspace(42.0, 47.0, 301)
    y_data = (
        5.0
        + 800.0 * np.exp(-((x_data - 43.0) / 0.06) ** 2)
        + 120.0 * np.exp(-((x_data - 44.2) / 0.25) ** 2)
        + 80.0 * np.exp(-((x_data - 45.5) / 0.18) ** 2)
    )

    for _ in range(3):
        fitter = Fitter(x_data, y_data)
        frozen = fitter.add_peak(43.0, (42.8, 43.2), "substrate", "fixed")
        frozen.area_guess = 85.0
        frozen.sigma_guess = 0.04
        frozen.fraction_guess = 0.5
        frozen.fit_state = "frozen"
        fitter.add_peak(44.2, (43.8, 44.6), "film", "free-1")
        fitter.add_peak(45.5, (45.1, 45.9), "film", "free-2")
        errors = []
        thread = FittingThread(
            fitter,
            constrain_fwhm=False,
            min_separation=0.0,
            fixed_background=5.0,
            method="least_squares",
            objective_mode="mixed",
            log_weight=0.0,
        )
        thread.error.connect(errors.append)

        thread.start()
        assert thread.wait(15_000)
        app.processEvents()

        assert errors == []
        assert fitter.result is not None
        assert fitter.result.success is True


def test_peak_state_combo_freezes_only_an_accepted_complete_shape(monkeypatch) -> None:
    app, window = _window_with_two_peaks()
    peak = window.fitter.peaks[0]
    warnings = []
    monkeypatch.setattr(
        "xrd_gui.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    window.set_peak_fit_state(peak.peak_id, "frozen")
    assert peak.fit_state == "optimize"
    assert warnings

    peak.center_guess = 44.1
    peak.area_guess = 123.4
    peak.sigma_guess = 0.2
    peak.fraction_guess = 0.35
    window.set_peak_fit_state(peak.peak_id, "frozen")

    assert peak.fit_state == "frozen"
    assert window.peak_table.item(0, 3).text() == "123.40"
    assert window.peak_table.item(0, 5).text() == "0.4000"
    assert window.peak_table.item(0, 6).text() == "0.3500"
    window.fitter.build_model(min_peak_separation=0.0)
    assert all(
        window.fitter.params[f"p0_{name}"].vary is False
        for name in ("center", "amplitude", "sigma", "fraction")
    )

    window.close()
    app.processEvents()


def test_fwhm_edit_and_lock_use_exact_pseudo_voigt_definition() -> None:
    app, window = _window_with_two_peaks()
    peak = window.fitter.peaks[0]
    fwhm_item = window.peak_table.item(0, 5)

    fwhm_item.setText("0.4000")
    window.fitter.build_model(min_peak_separation=0.0)

    assert peak.sigma_guess == pytest.approx(0.2)
    assert peak.fixed_fwhm is False
    assert window.fitter.params["p0_sigma"].value == pytest.approx(0.2)
    assert window.fitter.params["p0_sigma"].vary is True

    fwhm_item = window.peak_table.item(0, 5)
    fwhm_item.setCheckState(Qt.Checked)
    window.fitter.build_model(min_peak_separation=0.0)

    assert peak.fixed_fwhm is True
    assert window.fitter.params["p0_sigma"].value == pytest.approx(0.2)
    assert window.fitter.params["p0_sigma"].vary is False
    assert window.fitter.params["p0_fwhm"].value == pytest.approx(0.4)

    window.close()
    app.processEvents()


def test_editing_a_fitted_fwhm_invalidates_old_model_before_refit() -> None:
    app, window = _window_with_two_peaks()
    peak = window.fitter.peaks[0]
    window.fitter.build_model(min_peak_separation=0.0)
    window.fitter.result = object()
    peak.fwhm = 0.3
    window.update_peak_table()

    window.peak_table.item(0, 5).setText("0.5000")

    assert peak.sigma_guess == pytest.approx(0.25)
    assert window.fitter.model is None
    assert window.fitter.params is None
    assert window.fitter.result is None

    window.close()
    app.processEvents()


def test_fwhm_edit_outside_peak_type_bounds_is_rejected(monkeypatch) -> None:
    app, window = _window_with_two_peaks()
    peak = window.fitter.peaks[0]
    warnings = []
    monkeypatch.setattr(
        "xrd_gui.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    window.peak_table.item(0, 5).setText("3.1000")

    assert peak.sigma_guess is None
    assert window.peak_table.item(0, 5).text() == "NaN"
    assert warnings

    window.close()
    app.processEvents()


def test_reset_app_restores_a_clean_analysis_session(monkeypatch) -> None:
    app, window = _window_with_two_peaks()
    window.reporter = object()
    window.results_text.setPlainText("stale fit")
    window.physics_text.setPlainText("stale physics")
    window.filter_combo.setCurrentIndex(2)
    window.sg_window.setValue(11)
    window.bg_combo.setCurrentIndex(1)
    window.poly_degree.setValue(4)
    window.constrain_fwhm_cb.setChecked(True)
    window.min_separation.setValue(0.7)
    window.fit_method_combo.setCurrentIndex(1)
    window.log_spin.setValue(0)
    window.bg_spin.setValue(1.5)
    window.bg_fix_cb.setChecked(True)
    window.log_radio.setChecked(True)
    window.peak_threshold.setValue(250.0)
    window.peak_input_mode.setCurrentIndex(
        window.peak_input_mode.findData("d_spacing")
    )
    window.peak_value_input.setValue(2.0)
    window.peak_type_combo.setCurrentIndex(1)
    window.peak_name_input.setText("002")
    window.wavelength_spin.setValue(2.0)
    monkeypatch.setattr(
        "xrd_gui.QMessageBox.question",
        lambda *args: QMessageBox.Yes,
    )

    window.reset_app()

    assert window.fitter is None
    assert window.reporter is None
    assert window.peak_table.rowCount() == 0
    assert window.results_text.toPlainText() == ""
    assert window.physics_text.toPlainText() == ""
    assert window.filter_combo.currentText() == "无"
    assert window.sg_window.value() == 7
    assert window.bg_combo.currentText() == "无"
    assert window.poly_degree.value() == 2
    assert window.constrain_fwhm_cb.isChecked() is False
    assert window.min_separation.value() == pytest.approx(0.2)
    assert window.fit_method_combo.currentText() == "leastsq"
    assert window.objective_combo.currentData() == "mixed"
    assert window.log_spin.value() == 50
    assert window.log_floor_spin.value() == pytest.approx(1.0)
    assert window.include_ranges_input.text() == ""
    assert window.exclude_ranges_input.text() == ""
    assert window.bg_spin.value() == pytest.approx(0.0)
    assert window.bg_fix_cb.isChecked() is False
    assert window.linear_radio.isChecked() is True
    assert window.peak_threshold.value() == pytest.approx(100.0)
    assert window.peak_input_mode.currentData() == "two_theta"
    assert window.peak_value_input.value() == pytest.approx(44.0)
    assert window.peak_type_combo.currentData() == "film"
    assert window.peak_name_input.text() == ""
    assert window.wavelength_spin.value() == pytest.approx(
        DEFAULT_WAVELENGTH_ANGSTROM
    )
    assert window.range_min.value() == pytest.approx(20.0)
    assert window.range_max.value() == pytest.approx(120.0)

    window.close()
    app.processEvents()


def test_manual_fit_range_parser_accepts_multiple_ranges_and_rejects_bad_text() -> None:
    assert XRDAnalyzerGUI.parse_fit_ranges("42.5-43.5; 54:56") == [
        (42.5, 43.5),
        (54.0, 56.0),
    ]
    assert XRDAnalyzerGUI.parse_fit_ranges("") == []
    with pytest.raises(ValueError, match="无法识别"):
        XRDAnalyzerGUI.parse_fit_ranges("MgO")


def test_data_loading_range_commits_project_wide_filter_and_removes_outside_peaks(
    tmp_path,
) -> None:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()
    x_data = np.arange(0.0, 11.0)
    y_data = x_data + 10.0
    window.loaded_files_data = [("scan.txt", x_data.copy(), y_data.copy())]
    window.file_list_widget.addItem("scan.txt")
    window.x_data_raw = x_data.copy()
    window.y_data_raw = y_data.copy()
    window.x_data_original = x_data.copy()
    window.y_data_original = y_data.copy()
    window.x_data = x_data.copy()
    window.y_data = y_data.copy()
    window.fitter = Fitter(window.x_data, window.y_data)
    window.add_peak_to_fitter(2.0, (1.5, 2.5), "film", "outside")
    window.add_peak_to_fitter(5.0, (4.5, 5.5), "film", "inside")
    window.range_min.setValue(3.0)
    window.range_max.setValue(7.0)

    window.apply_range()

    for values in (
        window.x_data,
        window.x_data_raw,
        window.x_data_original,
        window.loaded_files_data[0][1],
        window.plot_canvas.x_data,
    ):
        assert values.tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert window.active_data_range == pytest.approx((3.0, 7.0))
    assert [(peak.peak_id, peak.name) for peak in window.fitter.peaks] == [
        (0, "inside")
    ]
    assert window.peak_table.rowCount() == 1

    window.fitter.build_model(min_peak_separation=0.0, fixed_background=0.0)
    window.fitter.execute_fitting(objective_mode="linear")
    reporter = Reporter(
        window.fitter,
        x_original=window.x_data_original,
        y_original=window.y_data_original,
    )
    reporter.calculate_metrics()
    output = tmp_path / "filtered.xlsx"
    reporter.export_results(output)
    exported = __import__("pandas").read_excel(output, sheet_name="Full_Data")
    assert exported["2theta"].min() == pytest.approx(3.0)
    assert exported["2theta"].max() == pytest.approx(7.0)

    figure = reporter.plot_results()
    for axis in figure.axes:
        x_min, x_max = axis.get_xlim()
        assert x_min == pytest.approx(3.0)
        assert x_max == pytest.approx(7.0)
    figure.clear()

    window.close()
    app.processEvents()


def test_excel_project_load_restores_data_peaks_and_gui_controls(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    x_data = np.linspace(43.0, 45.0, 101)
    y_data = 10.0 + 100.0 * np.exp(-((x_data - 44.0) / 0.12) ** 2)
    fitter = Fitter(x_data, y_data)
    peak = fitter.add_peak(44.0, (43.5, 44.5), "film", "002")
    peak.fixed_center = True
    fitter.build_model(min_peak_separation=0.0)
    fitter.execute_fitting(objective_mode="linear")
    fitter.accept_current_result()
    peak.fit_state = "frozen"
    reporter = Reporter(
        fitter,
        x_original=x_data,
        y_original=y_data,
        wavelength_angstrom=1.2345,
        radiation_label="自定义波长",
    )
    reporter.calculate_metrics()
    project_path = tmp_path / "restorable-project.xlsx"
    reporter.export_results(
        project_path,
        project_state={
            "schema_version": PROJECT_WORKBOOK_SCHEMA_VERSION,
            "range_min": 43.0,
            "range_max": 45.0,
            "filter_type": "高斯滤波",
            "sg_window": 9,
            "background_preprocess": "无",
            "poly_degree": 2,
            "constrain_fwhm": True,
            "min_separation": 0.35,
            "fit_method": "least_squares",
            "objective_mode": "linear",
            "log_weight": 20,
            "log_floor": 2.0,
            "background_value": 10.0,
            "background_fixed": False,
            "yscale": "log",
            "include_ranges_text": "43.5-44.5",
            "exclude_ranges_text": "",
            "peak_threshold": 80.0,
            "wavelength_angstrom": 1.2345,
            "radiation_label": "自定义波长",
            "peak_input_mode": "d_spacing",
            "preprocessing_steps": [
                {"operation": "gaussian", "parameters": {"sigma": 1.5}}
            ],
            "session_fit_configuration": {
                "method": "least_squares",
                "objective_mode": "linear",
                "log_weight": 0.2,
                "intensity_floor": 2.0,
                "constrain_fwhm": True,
                "min_peak_separation": 0.35,
                "fixed_background": None,
                "include_ranges": [[43.5, 44.5]],
                "exclude_ranges": [],
            },
        },
        source_datasets=[("original-scan.txt", x_data, y_data)],
    )

    window = XRDAnalyzerGUI()
    window.load_excel_project(project_path)

    assert window.x_data == pytest.approx(x_data)
    assert window.y_data == pytest.approx(y_data)
    assert window.active_data_range == pytest.approx((43.0, 45.0))
    assert [source[0] for source in window.loaded_files_data] == ["original-scan.txt"]
    assert window.peak_table.rowCount() == 1
    restored_peak = window.fitter.peaks[0]
    assert restored_peak.name == "002"
    assert restored_peak.fit_state == "frozen"
    assert restored_peak.fixed_center is True
    assert window.filter_combo.currentText() == "高斯滤波"
    assert window.constrain_fwhm_cb.isChecked() is True
    assert window.min_separation.value() == pytest.approx(0.35)
    assert window.fit_method_combo.currentText() == "least_squares"
    assert window.objective_combo.currentData() == "linear"
    assert window.log_floor_spin.value() == pytest.approx(2.0)
    assert window.log_radio.isChecked() is True
    assert window.include_ranges_input.text() == "43.5-44.5"
    assert window.wavelength_spin.value() == pytest.approx(1.2345)
    assert window.peak_input_mode.currentData() == "d_spacing"
    assert window.reporter.wavelength_angstrom == pytest.approx(1.2345)
    assert window.fitter.result is not None
    assert window.reporter is not None
    assert [step.operation for step in window.session.preprocessing] == ["gaussian"]
    assert window.session.fit_configuration.objective_mode == "linear"
    assert window.session.fit_configuration.include_ranges == ((43.5, 44.5),)

    window.close()
    app.processEvents()


def test_reapplying_same_preprocessing_is_idempotent_and_invalidates_old_fit() -> None:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()
    x_data = np.linspace(40.0, 48.0, 81)
    y_data = 5.0 + 100.0 * np.exp(-((x_data - 44.0) / 0.2) ** 2)
    window.x_data = x_data.copy()
    window.y_data = y_data.copy()
    window.x_data_raw = x_data.copy()
    window.y_data_raw = y_data.copy()
    window.x_data_original = x_data.copy()
    window.y_data_original = y_data.copy()
    window.fitter = Fitter(window.x_data, window.y_data)
    window.fitter.add_peak(44.0, (43.5, 44.5), "film", "002")
    window.fitter.result = object()
    window.reporter = object()
    window.filter_combo.setCurrentText("高斯滤波")
    window.bg_combo.setCurrentText("无")

    window.apply_preprocessing()
    first_result = window.y_data.copy()
    window.apply_preprocessing()

    np.testing.assert_array_equal(window.y_data, first_result)
    np.testing.assert_array_equal(window.fitter.y_data, window.y_data)
    assert len(window.fitter.peaks) == 1
    assert window.fitter.result is None
    assert window.reporter is None

    window.close()
    app.processEvents()


def test_project_state_records_session_provenance() -> None:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()
    raw_scan = ScanData(
        np.array([42.0, 43.0, 44.0]),
        np.array([10.0, np.nan, 30.0]),
        source_id="scan.txt",
    )
    window.session = AnalysisSession.from_raw(raw_scan).with_preprocessing(
        (PreprocessingStep.gaussian(sigma=1.5),)
    )
    window._sync_legacy_data_views()

    state = window.collect_project_state()

    assert state["schema_version"] == PROJECT_WORKBOOK_SCHEMA_VERSION
    assert state["preprocessing_steps"] == [
        {"operation": "gaussian", "parameters": {"sigma": 1.5}}
    ]
    assert state["raw_scan_sha256"] == window.session.raw_scan.content_sha256
    assert state["processed_scan_sha256"] == window.session.processed_scan.content_sha256
    assert state["raw_point_count"] == 3
    assert state["raw_valid_point_count"] == 2

    window.close()
    app.processEvents()


def test_delete_peak_after_full_plot_redraw_is_safe_and_clears_stale_results() -> None:
    app, window = _window_with_two_peaks()
    window.fitter.build_model(min_peak_separation=0.0)
    window.fitter.result = object()
    window.fitter.y_fit = window.y_data.copy()
    window.reporter = object()
    window.results_text.setPlainText("stale fit")
    window.physics_text.setPlainText("stale physics")

    # Fitted-result plotting performs this full redraw and detaches the old marker artists.
    window.plot_canvas.axes.clear()
    window.peak_table.selectRow(0)

    window.delete_selected_peak()

    assert [(peak.peak_id, peak.name) for peak in window.fitter.peaks] == [(0, "200")]
    assert window.peak_table.rowCount() == 1
    assert window.fitter.result is None
    assert window.reporter is None
    assert window.results_text.toPlainText() == ""
    assert window.physics_text.toPlainText() == ""
    assert window.plot_canvas.peak_markers
    assert all(marker.axes is window.plot_canvas.axes for marker in window.plot_canvas.peak_markers)

    window.close()
    app.processEvents()


def test_delete_peak_is_blocked_while_fitting(monkeypatch) -> None:
    app, window = _window_with_two_peaks()

    class RunningFitThread:
        @staticmethod
        def isRunning() -> bool:
            return True

    warnings = []
    window.fit_thread = RunningFitThread()
    window.peak_table.selectRow(0)
    monkeypatch.setattr(
        "xrd_gui.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    window.delete_selected_peak()

    assert [(peak.peak_id, peak.name) for peak in window.fitter.peaks] == [
        (0, "004"),
        (1, "200"),
    ]
    assert warnings

    window.close()
    app.processEvents()
