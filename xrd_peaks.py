"""XRD 峰配置、猜测值与拟合结果语义。"""

from __future__ import annotations

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
