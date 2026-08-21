"""XRD 峰配置、猜测值与拟合结果语义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


PSEUDO_VOIGT_FWHM_FACTOR = 2.0


class Peak:
    """单个 Pseudo-Voigt 峰的配置、猜测和候选结果。"""

    def __init__(
        self,
        peak_id: int,
        center_guess: float,
        bounds: Optional[Tuple[float, float]] = None,
        peak_type: str = "film",
        name: str = "",
    ):
        self.peak_id = peak_id
        self.center_guess = center_guess
        self.bounds = bounds if bounds else (center_guess - 0.5, center_guess + 0.5)
        self.peak_type = peak_type
        self.name = name

        self.sigma_guess = None
        self.height_guess = None
        self.area_guess = None
        self.fraction_guess = None
        self.fit_state = "optimize"

        self.center = None
        self.height = None
        self.fwhm = None
        self.area = None
        self.eta = None

        self.fixed_center = False
        self.fixed_height = False
        self.fixed_fwhm = False

    def clear_result(self) -> None:
        """清除失效结果，同时保留下一轮需要的初始猜测。"""
        self.center = None
        self.height = None
        self.fwhm = None
        self.area = None
        self.eta = None

    def set_result(self, params: Dict, fitted_height: float = None) -> None:
        """从 lmfit 参数读取峰结果。"""
        prefix = f"p{self.peak_id}_"
        self.center = params[f"{prefix}center"].value
        amplitude = params[f"{prefix}amplitude"].value
        sigma = params[f"{prefix}sigma"].value

        fwhm_parameter = params.get(f"{prefix}fwhm")
        if fwhm_parameter is not None:
            self.fwhm = fwhm_parameter.value
        else:
            self.fwhm = sigma * PSEUDO_VOIGT_FWHM_FACTOR

        fraction_parameter = params.get(f"{prefix}fraction")
        if fraction_parameter is not None:
            self.eta = fraction_parameter.value
        else:
            self.eta = 0.5

        if fitted_height is not None:
            self.height = fitted_height
        else:
            term1 = (1 - self.eta) / np.sqrt(2 * np.pi)
            term2 = self.eta / np.pi
            self.height = (amplitude / sigma) * (term1 + term2)
        self.area = amplitude

    def calculate_area(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """在峰中心正负三倍 FWHM 范围内数值积分。"""
        if self.fwhm is None:
            return 0
        mask = np.abs(x_data - self.center) <= 3 * self.fwhm
        area = np.trapz(y_data[mask], x_data[mask])
        self.area = area
        return area


@dataclass(frozen=True)
class PeakSnapshot:
    """Immutable copy of all user-visible configuration and result fields of one peak."""

    peak_id: int
    center_guess: float
    bounds: Tuple[float, float]
    peak_type: str
    name: str
    sigma_guess: Optional[float]
    height_guess: Optional[float]
    area_guess: Optional[float]
    fraction_guess: Optional[float]
    fit_state: str
    center: Optional[float]
    height: Optional[float]
    fwhm: Optional[float]
    area: Optional[float]
    eta: Optional[float]
    fixed_center: bool
    fixed_height: bool
    fixed_fwhm: bool

    @classmethod
    def from_peak(cls, peak: Peak) -> "PeakSnapshot":
        return cls(
            peak_id=peak.peak_id,
            center_guess=peak.center_guess,
            bounds=tuple(peak.bounds),
            peak_type=peak.peak_type,
            name=peak.name,
            sigma_guess=peak.sigma_guess,
            height_guess=peak.height_guess,
            area_guess=peak.area_guess,
            fraction_guess=peak.fraction_guess,
            fit_state=peak.fit_state,
            center=peak.center,
            height=peak.height,
            fwhm=peak.fwhm,
            area=peak.area,
            eta=peak.eta,
            fixed_center=peak.fixed_center,
            fixed_height=peak.fixed_height,
            fixed_fwhm=peak.fixed_fwhm,
        )

    def to_peak(self) -> Peak:
        peak = Peak(
            self.peak_id,
            self.center_guess,
            self.bounds,
            self.peak_type,
            self.name,
        )
        peak.sigma_guess = self.sigma_guess
        peak.height_guess = self.height_guess
        peak.area_guess = self.area_guess
        peak.fraction_guess = self.fraction_guess
        peak.fit_state = self.fit_state
        peak.center = self.center
        peak.height = self.height
        peak.fwhm = self.fwhm
        peak.area = self.area
        peak.eta = self.eta
        peak.fixed_center = self.fixed_center
        peak.fixed_height = self.fixed_height
        peak.fixed_fwhm = self.fixed_fwhm
        return peak
