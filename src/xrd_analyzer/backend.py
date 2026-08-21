"""XRD desktop application backend public API.

The PyQt frontend imports scientific and application services only from this module.
No module in this dependency closure imports PyQt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .engine import (
    Fitter,
    FitterHistory,
    PROJECT_WORKBOOK_SCHEMA_VERSION,
    Reporter,
    pseudo_voigt,
)
from .crystallography import (
    BraggGeometry,
    DEFAULT_RADIATION_LABEL,
    DEFAULT_WAVELENGTH_ANGSTROM,
)
from .io import DataLoader
from .peaks import Peak, PSEUDO_VOIGT_FWHM_FACTOR
from .preprocessing import Preprocessor
from .project import ProjectWorkbook, RestoredFitResult
from .session import (
    AnalysisSession,
    FitConfiguration,
    PreprocessingStep,
    ScanData,
)


@dataclass
class XRDApplicationService:
    """协调扫描状态和核心对象，不包含任何界面行为。"""

    session: AnalysisSession = AnalysisSession.empty()

    def clear(self) -> AnalysisSession:
        self.session = AnalysisSession.empty()
        return self.session

    def set_session(self, session: AnalysisSession) -> AnalysisSession:
        self.session = session
        return self.session

    def restore_session(
        self,
        two_theta: np.ndarray,
        raw_intensity: np.ndarray,
        processed_intensity: np.ndarray,
        *,
        source_id: str = "",
        active_range: Optional[Tuple[float, float]] = None,
        project_id: str = "",
        preprocessing: Iterable[PreprocessingStep] = (),
        source_scans: Iterable[ScanData] = (),
        fit_configuration: Optional[FitConfiguration] = None,
    ) -> AnalysisSession:
        session = AnalysisSession.restore(
            two_theta,
            raw_intensity,
            processed_intensity,
            source_id=source_id,
            active_range=active_range,
            project_id=project_id,
            preprocessing=preprocessing,
            source_scans=source_scans,
        )
        if fit_configuration is not None:
            session = session.with_fit_configuration(fit_configuration)
        return self.set_session(session)

    def merge_sources(
        self,
        sources: Sequence[Tuple[str, np.ndarray, np.ndarray]],
        *,
        active_range: Optional[Tuple[float, float]] = None,
        project_id: str = "",
    ) -> AnalysisSession:
        if not sources:
            return self.clear()
        x_data, y_data = DataLoader.stitch_datasets(
            [(source_x, source_y) for _, source_x, source_y in sources]
        )
        if active_range is not None:
            x_data, y_data = DataLoader.trim_range(
                x_data,
                y_data,
                active_range[0],
                active_range[1],
            )
        source_scans = tuple(
            ScanData(source_x, source_y, source_id=str(path))
            for path, source_x, source_y in sources
        )
        raw_scan = ScanData(x_data, y_data, source_id=project_id)
        return self.set_session(
            AnalysisSession.from_raw(
                raw_scan,
                active_range=active_range,
                project_id=project_id,
                source_scans=source_scans,
            )
        )

    @staticmethod
    def load_source(
        input_path: str,
        active_range: Optional[Tuple[float, float]] = None,
    ) -> ScanData:
        two_theta, intensity = DataLoader.load_txt(input_path)
        if active_range is not None:
            two_theta, intensity = DataLoader.trim_range(
                two_theta,
                intensity,
                active_range[0],
                active_range[1],
            )
        if two_theta.size == 0:
            raise ValueError("所选2θ范围内没有数据")
        return ScanData(two_theta, intensity, source_id=str(input_path))

    def apply_preprocessing(
        self,
        steps: Iterable[PreprocessingStep],
    ) -> AnalysisSession:
        return self.set_session(self.session.with_preprocessing(steps))

    def reset_preprocessing(self) -> AnalysisSession:
        return self.set_session(self.session.reset_preprocessing())

    def crop(self, lower: float, upper: float) -> AnalysisSession:
        return self.set_session(self.session.crop(lower, upper))

    def set_fit_configuration(
        self,
        configuration: FitConfiguration,
    ) -> AnalysisSession:
        return self.set_session(self.session.with_fit_configuration(configuration))

    def create_fitter(self, peaks: Iterable[Peak] = ()) -> Optional[Fitter]:
        if not self.session.has_data:
            return None
        fitter = Fitter(
            self.session.processed_scan.two_theta,
            self.session.processed_scan.intensity,
        )
        fitter.peaks = list(peaks)
        for peak_id, peak in enumerate(fitter.peaks):
            peak.peak_id = peak_id
        return fitter

    @staticmethod
    def load_project(input_path: str) -> dict:
        return ProjectWorkbook.load(input_path)


__all__ = [
    "AnalysisSession",
    "BraggGeometry",
    "DEFAULT_RADIATION_LABEL",
    "DEFAULT_WAVELENGTH_ANGSTROM",
    "DataLoader",
    "FitConfiguration",
    "Fitter",
    "FitterHistory",
    "Peak",
    "PreprocessingStep",
    "Preprocessor",
    "PROJECT_WORKBOOK_SCHEMA_VERSION",
    "PSEUDO_VOIGT_FWHM_FACTOR",
    "ProjectWorkbook",
    "Reporter",
    "RestoredFitResult",
    "ScanData",
    "XRDApplicationService",
    "pseudo_voigt",
]
