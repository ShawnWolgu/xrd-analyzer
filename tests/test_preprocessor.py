"""Tests for deterministic preprocessing behavior."""

from __future__ import annotations

import numpy as np

from xrd_analyzer import Preprocessor


def test_polynomial_background_recovers_exact_quadratic() -> None:
    x_data = np.linspace(-2.0, 2.0, 41)
    background_expected = 3.0 * x_data**2 - 2.0 * x_data + 7.0

    corrected, background = Preprocessor.subtract_background_polynomial(
        x_data,
        background_expected,
        degree=2,
    )

    np.testing.assert_allclose(background, background_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(corrected, 0.0, rtol=0.0, atol=1e-12)


def test_filters_preserve_shape_and_finite_values() -> None:
    x_data = np.linspace(0.0, 2.0 * np.pi, 101)
    y_data = np.sin(x_data) + 0.1 * np.sin(20.0 * x_data)

    filtered = [
        Preprocessor.apply_savgol_filter(y_data, window_length=11, polyorder=3),
        Preprocessor.apply_gaussian_filter(y_data, sigma=1.5),
        Preprocessor.apply_fft_filter(y_data, cutoff_freq=0.1),
    ]

    for result in filtered:
        assert result.shape == y_data.shape
        assert np.isfinite(result).all()
