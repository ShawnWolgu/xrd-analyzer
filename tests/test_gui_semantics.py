"""Focused GUI checks for scientific terminology."""

from __future__ import annotations

import numpy as np
from PyQt5.QtWidgets import QApplication

from xrd_analyzer import Fitter, Reporter
from xrd_gui import XRDAnalyzerGUI


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
