"""Focused GUI checks for scientific terminology."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import re
from types import SimpleNamespace

import lmfit
import numpy as np
import matplotlib
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QWidget,
)
import pytest

from xrd_analyzer import (
    DEFAULT_WAVELENGTH_ANGSTROM,
    Fitter,
    PROJECT_WORKBOOK_SCHEMA_VERSION,
    Reporter,
)
from xrd_analyzer.gui import FittingThread, PeakConfigDialog, XRDAnalyzerGUI
from xrd_analyzer.session import AnalysisSession, PreprocessingStep, ScanData
from main import startup_banner_text
from xrd_analyzer.i18n import translate


def test_product_identity_is_general_xrd_analysis() -> None:
    banner = startup_banner_text()
    assert "XRD Analyzer v1.1.0" in banner
    assert "通用 X 射线衍射分析工具" in banner
    assert "PZT" not in banner
    assert "薄膜专用" not in banner

    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()
    assert window.windowTitle() == "XRD Analyzer v1.1.0 - 通用 X 射线衍射分析工具"
    assert window.peak_type_combo.currentText() == "样品峰"
    assert window.peak_type_combo.currentData() == "film"
    assert window.constrain_fwhm_cb.text() == "强制样品峰FWHM相等"
    assert "PZT" not in window.peak_name_input.placeholderText()
    window.close()
    app.processEvents()


def test_language_switch_updates_main_ui_immediately() -> None:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()

    window.set_ui_language("ja")
    assert window.windowTitle() == "XRD Analyzer v1.1.0 - 汎用 X 線回折解析ツール"
    assert window.file_group.title() == "1. データの読み込みと結合"
    assert window.execute_fit_btn.text() == "フィッティングを実行"
    assert window.analysis_tabs.tabText(0) == "データとフィッティング"
    assert window.peak_table.horizontalHeaderItem(3).text() == "面積"
    assert window.filter_combo.currentData() == "none"
    message_box = QMessageBox()
    message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    assert message_box.button(QMessageBox.Yes).text().startswith("はい")

    window.set_ui_language("en")
    assert window.windowTitle() == "XRD Analyzer v1.1.0 - General-purpose X-ray diffraction analysis tool"
    assert window.file_group.title() == "1. Data Loading and Merging"
    assert window.execute_fit_btn.text() == "Run Fit"
    assert window.analysis_tabs.tabText(0) == "Data and Fitting"
    assert window.peak_table.horizontalHeaderItem(3).text() == "Area"
    assert window.filter_combo.currentData() == "none"
    assert translate("没有数据", "en") == "No Data"
    message_box = QMessageBox()
    message_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    assert "Yes" in message_box.button(QMessageBox.Yes).text()

    window.close()
    app.processEvents()


def _visible_widget_strings(root: QWidget) -> list[str]:
    strings: list[str] = []
    language_combo = getattr(root, "language_combo", None)
    for widget in [root, *root.findChildren(QWidget)]:
        if isinstance(widget, QGroupBox):
            strings.append(widget.title())
        if isinstance(widget, (QLabel, QAbstractButton)):
            strings.append(widget.text())
        if isinstance(widget, QLineEdit):
            strings.append(widget.placeholderText())
        strings.append(widget.toolTip())
        if isinstance(widget, QComboBox) and widget is not language_combo:
            strings.extend(
                widget.itemText(index) for index in range(widget.count())
            )
        if isinstance(widget, QTableWidget):
            strings.extend(
                widget.horizontalHeaderItem(index).text()
                for index in range(widget.columnCount())
                if widget.horizontalHeaderItem(index) is not None
            )
    return [text for text in strings if text]


def test_english_ui_contains_no_untranslated_chinese_widget_text() -> None:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()
    window.set_ui_language("ja")
    window.set_ui_language("en")
    dialog = PeakConfigDialog(language="en")

    untranslated = [
        text
        for root in (window, dialog)
        for text in _visible_widget_strings(root)
        if re.search(r"[\u3400-\u9fff]", text)
    ]

    assert untranslated == []
    window.close()
    dialog.close()
    app.processEvents()


def test_plot_style_uses_arial_as_primary_font() -> None:
    assert matplotlib.rcParams["font.family"][0] == "Arial"


def test_peak_setup_stays_left_and_management_table_is_below_plot() -> None:
    app = QApplication.instance() or QApplication([])
    window = XRDAnalyzerGUI()

    assert window.peak_setup_group is window.peak_group
    assert window.peak_table_group.isAncestorOf(window.peak_table)
    assert not window.peak_setup_group.isAncestorOf(window.peak_table)
    assert window.right_splitter.indexOf(window.analysis_tabs) == 0
    assert window.right_splitter.indexOf(window.peak_table_group) == 1
    assert window.analysis_tabs.count() == 1
    assert window.analysis_tabs.tabText(0) == "数据与拟合"
    assert not any(
        button.text() == "自动寻峰"
        for button in window.peak_setup_group.findChildren(QPushButton)
    )

    window.close()
    app.processEvents()


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
    window.set_ui_language("en")
    warnings = []
    monkeypatch.setattr(
        "xrd_analyzer.gui.QMessageBox.warning",
        lambda *args: warnings.append(args),
    )

    window.quick_add_peak()

    assert len(window.fitter.peaks) == 0
    assert warnings
    assert warnings[0][1] == "Peak Outside Data Range"
    assert not re.search(r"[\u3400-\u9fff]", warnings[0][2])

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


def test_editing_project_wavelength_updates_reporter_provenance() -> None:
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
    assert window.reporter.radiation_label == "自定义波长"

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
    assert window.peak_table.item(0, 8).text() == "样品峰"

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


def test_non_fitting_peak_changes_do_not_create_history_snapshots() -> None:
    app, window = _window_with_two_peaks()
    window.fit_history.clear()
    window.update_fit_history_buttons()

    window.set_peak_fit_state(0, "disabled")
    assert not window.fit_history.can_undo

    window.peak_table.selectRow(1)
    window.delete_selected_peak()

    assert [peak.name for peak in window.fitter.peaks] == ["004"]
    assert not window.fit_history.can_undo

    window.add_peak_to_fitter(48.0, (47.5, 48.5), "film", "new")
    assert [peak.name for peak in window.fitter.peaks] == ["004", "new"]
    assert not window.fit_history.can_undo

    window.close()
    app.processEvents()


def test_export_fitted_peak_positions_uses_project_database_directory(
    monkeypatch,
    tmp_path,
) -> None:
    app, window = _window_with_two_peaks()
    window.fitter.result = SimpleNamespace(params=lmfit.Parameters())
    window.fitter.peaks[0].center = 44.1234564
    window.fitter.peaks[1].center = 46.6543214
    window.fitter.peaks[1].fit_state = "disabled"
    selected_path = tmp_path / "my-fitted-peaks"
    dialog_defaults = []

    def choose_path(_parent, _title, default_path, _filters):
        dialog_defaults.append(Path(default_path))
        return str(selected_path), "Text Files (*.txt)"

    monkeypatch.setattr(
        "xrd_analyzer.gui.QFileDialog.getSaveFileName", choose_path
    )

    window.export_fitted_peaks_to_file()

    expected_default_dir = Path(__file__).parents[1] / "database"
    assert dialog_defaults[0].parent == expected_default_dir
    output_path = selected_path.with_suffix(".txt")
    assert output_path.exists()
    assert "44.123456 - film - 004" in output_path.read_text(encoding="utf-8")
    assert "200" not in output_path.read_text(encoding="utf-8")

    window.fitter.clear_peaks()
    monkeypatch.setattr(
        "xrd_analyzer.gui.QFileDialog.getOpenFileName",
        lambda *_args: (str(output_path), "Text Files (*.txt)"),
    )
    window.import_peaks_from_file()
    assert len(window.fitter.peaks) == 1
    assert window.fitter.peaks[0].center_guess == pytest.approx(44.123456)
    assert window.fitter.peaks[0].name == "004"

    window.close()
    app.processEvents()


def test_peak_shift_buttons_and_slider_move_all_peaks_without_fit_history() -> None:
    app, window = _window_with_two_peaks()
    window.fit_history.clear()
    window.peak_shift_step.setValue(0.02)

    window.shift_right_btn.click()
    assert [peak.center_guess for peak in window.fitter.peaks] == pytest.approx(
        [44.02, 46.02]
    )

    window.peak_shift_slider.setValue(-2)
    window.apply_peak_shift_from_slider()
    assert [peak.center_guess for peak in window.fitter.peaks] == pytest.approx(
        [43.98, 45.98]
    )
    assert window.peak_shift_slider.value() == 0
    assert not window.fit_history.can_undo

    window.close()
    app.processEvents()


def _install_displayed_fit(
    window: XRDAnalyzerGUI,
    *,
    centers: tuple[float, float],
    total_curve: np.ndarray,
    component_curves: tuple[np.ndarray, np.ndarray],
) -> None:
    params = lmfit.Parameters()
    params.add("c", value=1.0)
    for peak, center, component in zip(
        window.fitter.peaks,
        centers,
        component_curves,
    ):
        peak.center = center
        peak.area = float(np.trapz(component, window.fitter.x_data))
        peak.height = float(np.max(component))
        peak.fwhm = 0.2
        peak.eta = 0.5
        prefix = f"p{peak.peak_id}_"
        params.add(f"{prefix}center", value=center)
        params.add(f"{prefix}amplitude", value=peak.area)
        params.add(f"{prefix}sigma", value=0.1)
        params.add(f"{prefix}fraction", value=0.5)

    window.fitter.result = SimpleNamespace(
        params=params,
        chisqr=1.0,
        success=True,
        message="synthetic",
        nfev=1,
        covar=np.eye(1),
    )
    window.fitter.y_fit = np.asarray(total_curve, dtype=float)
    window.fitter.fit_mask = np.ones_like(total_curve, dtype=bool)
    window.fitter.fit_config = {"objective_mode": "linear", "exclude_ranges": []}
    window.fitter.fit_diagnostics = {"success": True}
    curves = {
        peak.peak_id: np.asarray(curve, dtype=float)
        for peak, curve in zip(window.fitter.peaks, component_curves)
    }
    window.fitter.get_individual_peaks = lambda: curves
    window.reporter = Reporter(window.fitter)
    window.reporter.calculate_metrics()
    window.update_peak_table()
    window.plot_fitted_results()


def test_live_fit_legend_uses_total_and_peak_names_only() -> None:
    app, window = _window_with_two_peaks()
    x_data = window.fitter.x_data
    first = np.exp(-((x_data - 44.0) / 0.2) ** 2)
    second = 0.5 * np.exp(-((x_data - 46.0) / 0.3) ** 2)
    _install_displayed_fit(
        window,
        centers=(44.0, 46.0),
        total_curve=1.0 + first + second,
        component_curves=(first, second),
    )

    labels = [text.get_text() for text in window.plot_canvas.axes.get_legend().texts]

    assert "Total" in labels
    assert "004" in labels
    assert "200" in labels
    assert not any("Peak 0" in label or "Peak 1" in label for label in labels)
    assert "Fit" not in labels

    window.fitter.fit_mask[0] = False
    window.fitter.fit_config["exclude_ranges"] = [(40.0, 40.1)]
    window.plot_fitted_results()
    labels = [text.get_text() for text in window.plot_canvas.axes.get_legend().texts]
    assert "Fit data" in labels
    assert "Excluded range" in labels
    assert not any(re.search(r"[\u3400-\u9fff]", label) for label in labels)

    window.close()
    app.processEvents()


def test_undo_redo_restores_table_and_complete_fitted_plot() -> None:
    app, window = _window_with_two_peaks()
    x_data = window.fitter.x_data
    first_a = np.exp(-((x_data - 44.0) / 0.2) ** 2)
    second_a = 0.5 * np.exp(-((x_data - 46.0) / 0.3) ** 2)
    total_a = 1.0 + first_a + second_a
    _install_displayed_fit(
        window,
        centers=(44.0, 46.0),
        total_curve=total_a,
        component_curves=(first_a, second_a),
    )
    window.fit_history.clear()
    window.record_completed_fit()

    first_b = 1.2 * np.exp(-((x_data - 44.2) / 0.25) ** 2)
    second_b = 0.7 * np.exp(-((x_data - 46.2) / 0.35) ** 2)
    total_b = 1.0 + first_b + second_b
    _install_displayed_fit(
        window,
        centers=(44.2, 46.2),
        total_curve=total_b,
        component_curves=(first_b, second_b),
    )
    window.record_completed_fit()

    window.undo_fit_result()
    assert window.peak_table.item(0, 2).text() == "44.000"
    assert window.reporter is not None
    total_line = next(
        line for line in window.plot_canvas.axes.lines if line.get_label() == "Total"
    )
    assert total_line.get_ydata() == pytest.approx(total_a)

    window.redo_fit_result()
    assert window.peak_table.item(0, 2).text() == "44.200"
    total_line = next(
        line for line in window.plot_canvas.axes.lines if line.get_label() == "Total"
    )
    assert total_line.get_ydata() == pytest.approx(total_b)

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

    monkeypatch.setattr("xrd_analyzer.gui.threadpool_limits", fake_threadpool_limits)
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
        "xrd_analyzer.gui.QMessageBox.warning",
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
        "xrd_analyzer.gui.QMessageBox.warning",
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
    window.peak_input_mode.setCurrentIndex(
        window.peak_input_mode.findData("d_spacing")
    )
    window.peak_value_input.setValue(2.0)
    window.peak_type_combo.setCurrentIndex(1)
    window.peak_name_input.setText("002")
    window.wavelength_spin.setValue(2.0)
    window.peak_shift_step.setValue(0.2)
    monkeypatch.setattr(
        "xrd_analyzer.gui.QMessageBox.question",
        lambda *args: QMessageBox.Yes,
    )

    window.reset_app()

    assert window.fitter is None
    assert window.reporter is None
    assert window.peak_table.rowCount() == 0
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
    assert window.peak_input_mode.currentData() == "two_theta"
    assert window.peak_value_input.value() == pytest.approx(44.0)
    assert window.peak_type_combo.currentData() == "film"
    assert window.peak_name_input.text() == ""
    assert window.wavelength_spin.value() == pytest.approx(
        DEFAULT_WAVELENGTH_ANGSTROM
    )
    assert window.peak_shift_step.value() == pytest.approx(0.01)
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
            "wavelength_angstrom": 1.2345,
            "radiation_label": "自定义波长",
            "peak_input_mode": "d_spacing",
            "peak_shift_step_2theta_deg": 0.025,
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
    assert window.peak_shift_step.value() == pytest.approx(0.025)
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
    window.set_ui_language("ja")

    state = window.collect_project_state()

    assert state["schema_version"] == PROJECT_WORKBOOK_SCHEMA_VERSION
    assert state["preprocessing_steps"] == [
        {"operation": "gaussian", "parameters": {"sigma": 1.5}}
    ]
    assert state["raw_scan_sha256"] == window.session.raw_scan.content_sha256
    assert state["processed_scan_sha256"] == window.session.processed_scan.content_sha256
    assert state["raw_point_count"] == 3
    assert state["raw_valid_point_count"] == 2
    assert state["ui_language"] == "ja"

    window.close()
    app.processEvents()


def test_delete_peak_after_full_plot_redraw_is_safe_and_clears_stale_results() -> None:
    app, window = _window_with_two_peaks()
    window.fitter.build_model(min_peak_separation=0.0)
    window.fitter.result = object()
    window.fitter.y_fit = window.y_data.copy()
    window.reporter = object()

    # Fitted-result plotting performs this full redraw and detaches the old marker artists.
    window.plot_canvas.axes.clear()
    window.peak_table.selectRow(0)

    window.delete_selected_peak()

    assert [(peak.peak_id, peak.name) for peak in window.fitter.peaks] == [(0, "200")]
    assert window.peak_table.rowCount() == 1
    assert window.fitter.result is None
    assert window.reporter is None
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
        "xrd_analyzer.gui.QMessageBox.warning",
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
