"""Application-service tests at the frontend/backend boundary."""

from __future__ import annotations

import numpy as np

from xrd_analyzer.backend import Peak, PreprocessingStep, XRDApplicationService


def test_backend_service_owns_merge_preprocess_crop_and_fitter_creation() -> None:
    service = XRDApplicationService()
    sources = [
        ("a.txt", np.array([40.0, 41.0, 42.0]), np.array([1.0, 3.0, 1.0])),
        ("b.txt", np.array([42.0, 43.0, 44.0]), np.array([2.0, 4.0, 2.0])),
    ]

    merged = service.merge_sources(sources, project_id="merged")
    first = service.apply_preprocessing((PreprocessingStep.gaussian(1.0),))
    first_intensity = first.processed_scan.intensity.copy()
    second = service.apply_preprocessing((PreprocessingStep.gaussian(1.0),))
    cropped = service.crop(41.0, 43.0)
    peak = Peak(0, 42.0, (41.5, 42.5), "film", "reference")
    fitter = service.create_fitter((peak,))

    assert merged.project_id == "merged"
    assert [scan.source_id for scan in merged.source_scans] == ["a.txt", "b.txt"]
    np.testing.assert_array_equal(second.processed_scan.intensity, first_intensity)
    assert cropped.raw_scan.two_theta.tolist() == [41.0, 42.0, 43.0]
    np.testing.assert_array_equal(fitter.x_data, cropped.processed_scan.two_theta)
    np.testing.assert_array_equal(fitter.y_data, cropped.processed_scan.intensity)
    assert fitter.peaks == [peak]
