"""Core contracts for reproducible XRD analysis sessions."""

from __future__ import annotations

import numpy as np
import pytest

from xrd_session import AnalysisSession, FitConfiguration, PreprocessingStep, ScanData


def test_scan_data_owns_read_only_array_copies() -> None:
    two_theta = np.array([42.0, 43.0, 44.0])
    intensity = np.array([10.0, 20.0, 30.0])

    scan = ScanData(two_theta, intensity, source_id="scan.txt")
    two_theta[0] = 0.0
    intensity[0] = 0.0

    assert scan.two_theta.tolist() == [42.0, 43.0, 44.0]
    assert scan.intensity.tolist() == [10.0, 20.0, 30.0]
    with pytest.raises(ValueError):
        scan.intensity[0] = 99.0


def test_scan_content_hash_is_stable_and_changes_with_numeric_content() -> None:
    first = ScanData(np.array([42.0, 43.0]), np.array([10.0, 20.0]))
    equivalent = ScanData(np.array([42.0, 43.0]), np.array([10.0, 20.0]))
    changed = ScanData(np.array([42.0, 43.0]), np.array([10.0, 21.0]))

    assert first.content_sha256 == equivalent.content_sha256
    assert first.content_sha256 != changed.content_sha256


def test_session_preprocessing_is_recomputed_from_raw_scan() -> None:
    raw = ScanData(
        np.arange(9.0),
        np.array([0.0, 0.0, 0.0, 10.0, 20.0, 10.0, 0.0, 0.0, 0.0]),
    )
    session = AnalysisSession.from_raw(raw)
    steps = (PreprocessingStep.gaussian(sigma=1.5),)

    first = session.with_preprocessing(steps)
    second = first.with_preprocessing(steps)

    np.testing.assert_array_equal(first.processed_scan.intensity, second.processed_scan.intensity)
    np.testing.assert_array_equal(second.raw_scan.intensity, raw.intensity)
    assert second.preprocessing == steps


def test_session_crop_preserves_raw_and_processed_alignment() -> None:
    raw = ScanData(np.arange(10.0), np.arange(10.0) * 2.0)
    session = AnalysisSession.from_raw(raw).with_preprocessing(
        (PreprocessingStep.gaussian(sigma=1.0),)
    )

    cropped = session.crop(3.0, 7.0)

    assert cropped.active_range == pytest.approx((3.0, 7.0))
    assert cropped.raw_scan.two_theta.tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert cropped.processed_scan.two_theta.tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]


def test_session_crop_applies_to_each_source_scan() -> None:
    first = ScanData(np.arange(0.0, 6.0), np.arange(0.0, 6.0), source_id="a.txt")
    second = ScanData(np.arange(4.0, 10.0), np.arange(4.0, 10.0), source_id="b.txt")
    merged = ScanData(np.arange(0.0, 10.0), np.arange(0.0, 10.0))
    session = AnalysisSession.from_raw(merged, source_scans=(first, second))

    cropped = session.crop(3.0, 7.0)

    assert [scan.source_id for scan in cropped.source_scans] == ["a.txt", "b.txt"]
    assert cropped.source_scans[0].two_theta.tolist() == [3.0, 4.0, 5.0]
    assert cropped.source_scans[1].two_theta.tolist() == [4.0, 5.0, 6.0, 7.0]


def test_restored_session_keeps_distinct_raw_and_processed_scans() -> None:
    x_data = np.array([42.0, 43.0, 44.0])
    raw = np.array([10.0, 20.0, 30.0])
    processed = np.array([11.0, 19.0, 31.0])

    session = AnalysisSession.restore(
        x_data,
        raw,
        processed,
        source_id="project.xlsx",
        active_range=(42.0, 44.0),
    )

    np.testing.assert_array_equal(session.raw_scan.intensity, raw)
    np.testing.assert_array_equal(session.processed_scan.intensity, processed)
    assert session.preprocessing == ()


def test_fit_configuration_is_validated_and_serializable() -> None:
    configuration = FitConfiguration(
        method="least_squares",
        objective_mode="mixed",
        log_weight=0.25,
        intensity_floor=1e-6,
        constrain_fwhm=True,
        min_peak_separation=0.3,
        fixed_background=2.0,
        include_ranges=((45.0, 43.0),),
        exclude_ranges=((44.2, 44.0),),
    )

    restored = FitConfiguration.from_record(configuration.to_record())

    assert restored == configuration
    assert restored.include_ranges == ((43.0, 45.0),)
    assert restored.exclude_ranges == ((44.0, 44.2),)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"objective_mode": "unknown"},
        {"log_weight": -0.1},
        {"log_weight": 1.1},
        {"intensity_floor": 0.0},
        {"min_peak_separation": -0.1},
    ],
)
def test_fit_configuration_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        FitConfiguration(**kwargs)
