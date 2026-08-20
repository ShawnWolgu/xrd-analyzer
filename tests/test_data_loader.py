"""Characterization and scientific-contract tests for scan loading."""

from __future__ import annotations

import numpy as np
import pytest

from xrd_analyzer import DataLoader


def test_load_txt_reads_valid_two_column_rows(tmp_path) -> None:
    source = tmp_path / "valid-scan.txt"
    source.write_text("44.00 100\n44.02 120\n", encoding="utf-8")

    x_data, y_data = DataLoader.load_txt(source)

    np.testing.assert_array_equal(x_data, [44.00, 44.02])
    np.testing.assert_array_equal(y_data, [100.0, 120.0])


def test_load_txt_skips_headers_and_partly_numeric_rows_atomically(tmp_path) -> None:
    source = tmp_path / "scan.txt"
    source.write_text(
        "2theta intensity\n"
        "44.00 100\n"
        "ignored row with extra columns\n"
        "44.02 120\n"
        "44.04 not-a-number\n",
        encoding="utf-8",
    )

    x_data, y_data = DataLoader.load_txt(source)

    np.testing.assert_array_equal(x_data, [44.00, 44.02])
    np.testing.assert_array_equal(y_data, [100.0, 120.0])


def test_trim_range_is_inclusive_and_does_not_mutate_input() -> None:
    x_data = np.array([43.0, 44.0, 45.0, 46.0])
    y_data = np.array([1.0, 2.0, 3.0, 4.0])
    x_before = x_data.copy()
    y_before = y_data.copy()

    x_trimmed, y_trimmed = DataLoader.trim_range(x_data, y_data, 44.0, 45.0)

    np.testing.assert_array_equal(x_trimmed, [44.0, 45.0])
    np.testing.assert_array_equal(y_trimmed, [2.0, 3.0])
    np.testing.assert_array_equal(x_data, x_before)
    np.testing.assert_array_equal(y_data, y_before)


def test_stitch_datasets_averages_a_simple_overlap() -> None:
    first = (np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
    second = (np.array([2.0, 3.0, 4.0]), np.array([30.0, 40.0, 50.0]))

    x_data, y_data = DataLoader.stitch_datasets([first, second])

    np.testing.assert_allclose(x_data, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(y_data, [10.0, 25.0, 35.0, 50.0])


@pytest.mark.scientific
def test_stitch_datasets_preserves_uncovered_gaps() -> None:
    first = (np.array([0.0, 1.0]), np.array([10.0, 11.0]))
    second = (np.array([4.0, 5.0]), np.array([14.0, 15.0]))

    x_data, y_data = DataLoader.stitch_datasets([first, second])
    gap = (x_data > 1.0) & (x_data < 4.0)

    assert gap.any()
    assert np.isnan(y_data[gap]).all()
