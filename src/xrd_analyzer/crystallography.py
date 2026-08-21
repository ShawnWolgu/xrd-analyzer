"""XRD 衍射几何和带显式物理语义的派生量。"""

from __future__ import annotations

import numpy as np


DEFAULT_WAVELENGTH_ANGSTROM = 1.5406
DEFAULT_RADIATION_LABEL = "Cu Kα1"


class BraggGeometry:
    """一级布拉格条件下 2θ 与晶面间距 d 的换算。"""

    @staticmethod
    def _validated_wavelength(wavelength_angstrom: float) -> float:
        wavelength = float(wavelength_angstrom)
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("X射线波长必须是大于0的有限值")
        return wavelength

    @classmethod
    def d_from_two_theta(
        cls,
        two_theta_deg: float,
        wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    ) -> float:
        """由 2θ（度）计算一级反射的晶面间距 d（Å）。"""
        wavelength = cls._validated_wavelength(wavelength_angstrom)
        two_theta = float(two_theta_deg)
        if not np.isfinite(two_theta) or not 0 < two_theta < 180:
            raise ValueError("2θ必须位于0°到180°之间")
        theta_rad = np.radians(two_theta / 2.0)
        return float(wavelength / (2.0 * np.sin(theta_rad)))

    @classmethod
    def two_theta_from_d(
        cls,
        d_angstrom: float,
        wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
    ) -> float:
        """由晶面间距 d（Å）反算一级反射的 2θ（度）。"""
        wavelength = cls._validated_wavelength(wavelength_angstrom)
        spacing = float(d_angstrom)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("理论晶面间距d必须是大于0的有限值")
        sine_theta = wavelength / (2.0 * spacing)
        if sine_theta > 1.0:
            raise ValueError("理论晶面间距d小于λ/2，无法满足一级布拉格条件")
        return float(2.0 * np.degrees(np.arcsin(sine_theta)))

    @classmethod
    def apparent_scherrer_size_nm(
        cls,
        two_theta_deg: float,
        fwhm_deg: float,
        wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
        shape_factor: float = 0.9,
    ) -> float:
        """计算未作仪器展宽修正的表观 Scherrer 相干畴尺寸（nm）。"""
        wavelength = cls._validated_wavelength(wavelength_angstrom)
        two_theta = float(two_theta_deg)
        fwhm = float(fwhm_deg)
        factor = float(shape_factor)
        if not np.isfinite(two_theta) or not 0 < two_theta < 180:
            raise ValueError("2θ必须位于0°到180°之间")
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError("FWHM必须是大于0的有限值")
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Scherrer形状因子必须是大于0的有限值")
        beta_rad = np.radians(fwhm)
        theta_rad = np.radians(two_theta / 2.0)
        return float(factor * wavelength / (beta_rad * np.cos(theta_rad)) / 10.0)
