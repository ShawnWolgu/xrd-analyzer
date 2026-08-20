"""Tests for deterministic preprocessing behavior."""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "filter_function, kwargs",
    [
        (Preprocessor.apply_savgol_filter, {"window_length": 5, "polyorder": 2}),
        (Preprocessor.apply_gaussian_filter, {"sigma": 1.0}),
        (Preprocessor.apply_fft_filter, {"cutoff_freq": 0.25}),
    ],
)
def test_filters_preserve_scan_gaps_without_contaminating_valid_segments(
    filter_function,
    kwargs,
) -> None:
    y_data = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0])

    result = filter_function(y_data, **kwargs)

    assert result.shape == y_data.shape
    assert np.isnan(result[4:6]).all()
    assert np.isfinite(result[:4]).all()
    assert np.isfinite(result[6:]).all()


def test_polynomial_background_ignores_missing_scan_ranges() -> None:
    x_data = np.arange(7.0)
    y_data = 2.0 * x_data + 3.0
    y_data[3:5] = np.nan

    corrected, background = Preprocessor.subtract_background_polynomial(
        x_data,
        y_data,
        degree=1,
    )

    assert np.isnan(corrected[3:5]).all()
    assert np.isnan(background[3:5]).all()
    np.testing.assert_allclose(corrected[np.isfinite(y_data)], 0.0, atol=1e-12)


def test_snip_background_preserves_missing_scan_ranges() -> None:
    y_data = np.array([2.0, 3.0, 2.0, np.nan, np.nan, 4.0, 6.0, 4.0])

    corrected, background = Preprocessor.subtract_background_snip(y_data, iterations=2)

    assert np.isnan(corrected[3:5]).all()
    assert np.isnan(background[3:5]).all()
    assert np.isfinite(corrected[:3]).all()
    assert np.isfinite(corrected[5:]).all()
