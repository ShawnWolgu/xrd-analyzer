"""无 Qt 依赖的 XRD 扫描与可复现分析会话状态。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Iterable, Mapping, Optional, Tuple

import numpy as np

from xrd_io import DataLoader
from xrd_preprocessing import Preprocessor


def _read_only_float_array(values: np.ndarray) -> np.ndarray:
    """复制一维浮点数组并禁止原地修改。"""
    array = np.array(values, dtype=float, copy=True)
    if array.ndim != 1:
        raise ValueError("XRD扫描数据必须是一维数组")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ScanData:
    """一份带坐标语义和来源标识的不可变 XRD 扫描。"""

    two_theta: np.ndarray
    intensity: np.ndarray
    source_id: str = ""
    coordinate_unit: str = "degree_2theta"
    intensity_unit: str = "a.u."

    def __post_init__(self) -> None:
        two_theta = _read_only_float_array(self.two_theta)
        intensity = _read_only_float_array(self.intensity)
        if two_theta.shape != intensity.shape:
            raise ValueError("XRD扫描的2θ与强度数组长度不一致")
        if two_theta.size == 0:
            raise ValueError("XRD扫描不能为空")
        if not np.isfinite(two_theta).all():
            raise ValueError("XRD扫描的2θ坐标必须全部为有限值")
        object.__setattr__(self, "two_theta", two_theta)
        object.__setattr__(self, "intensity", intensity)

    @property
    def valid_mask(self) -> np.ndarray:
        """返回真实有限观测点掩码。"""
        return np.isfinite(self.two_theta) & np.isfinite(self.intensity)

    @property
    def content_sha256(self) -> str:
        """返回坐标、强度和单位语义的稳定 SHA-256 标识。"""
        digest = hashlib.sha256()
        digest.update(self.coordinate_unit.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.intensity_unit.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(self.two_theta, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(self.intensity, dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def with_intensity(self, intensity: np.ndarray) -> "ScanData":
        """保留坐标和来源，返回具有新强度值的扫描。"""
        return ScanData(
            self.two_theta,
            intensity,
            source_id=self.source_id,
            coordinate_unit=self.coordinate_unit,
            intensity_unit=self.intensity_unit,
        )

    def crop(self, lower: float, upper: float) -> "ScanData":
        """按永久数据范围裁剪扫描，范围单位为 2θ 度。"""
        x_data, y_data = DataLoader.trim_range(
            self.two_theta,
            self.intensity,
            lower,
            upper,
        )
        if x_data.size == 0:
            raise ValueError("所选2θ范围内没有数据")
        return ScanData(
            x_data,
            y_data,
            source_id=self.source_id,
            coordinate_unit=self.coordinate_unit,
            intensity_unit=self.intensity_unit,
        )


@dataclass(frozen=True)
class PreprocessingStep:
    """一个有序、可序列化的预处理操作及其参数。"""

    operation: str
    parameters: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted((str(key), float(value)) for key, value in self.parameters)
        )
        object.__setattr__(self, "parameters", normalized)

    @property
    def parameter_dict(self) -> dict[str, float]:
        return dict(self.parameters)

    def to_record(self) -> dict[str, object]:
        """转换为可写入项目文件的普通字典。"""
        return {
            "operation": self.operation,
            "parameters": self.parameter_dict,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "PreprocessingStep":
        """从项目文件中的字典恢复预处理步骤。"""
        operation = str(record.get("operation", ""))
        raw_parameters = record.get("parameters", {})
        if not isinstance(raw_parameters, Mapping):
            raise ValueError("预处理步骤参数必须是键值映射")
        parameters = tuple(
            sorted((str(key), float(value)) for key, value in raw_parameters.items())
        )
        return cls(operation=operation, parameters=parameters)

    @classmethod
    def savgol(cls, window_length: int, polyorder: int = 3) -> "PreprocessingStep":
        return cls(
            "savgol",
            (("window_length", float(window_length)), ("polyorder", float(polyorder))),
        )

    @classmethod
    def gaussian(cls, sigma: float = 1.5) -> "PreprocessingStep":
        return cls("gaussian", (("sigma", float(sigma)),))

    @classmethod
    def fft(cls, cutoff_freq: float = 0.1) -> "PreprocessingStep":
        return cls("fft", (("cutoff_freq", float(cutoff_freq)),))

    @classmethod
    def polynomial_background(cls, degree: int = 2) -> "PreprocessingStep":
        return cls("polynomial_background", (("degree", float(degree)),))

    @classmethod
    def snip_background(cls, iterations: int = 40) -> "PreprocessingStep":
        return cls("snip_background", (("iterations", float(iterations)),))


def _normalized_ranges(
    ranges: Iterable[Tuple[float, float]],
) -> Tuple[Tuple[float, float], ...]:
    return tuple(
        tuple(sorted((float(lower), float(upper))))
        for lower, upper in ranges
    )


@dataclass(frozen=True)
class FitConfiguration:
    """与 GUI 无关、可恢复的完整拟合配置。"""

    method: str = "leastsq"
    objective_mode: str = "mixed"
    log_weight: float = 0.5
    intensity_floor: float = 1.0
    constrain_fwhm: bool = False
    min_peak_separation: float = 0.2
    fixed_background: Optional[float] = None
    include_ranges: Tuple[Tuple[float, float], ...] = ()
    exclude_ranges: Tuple[Tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.method:
            raise ValueError("拟合方法不能为空")
        if self.objective_mode not in {"linear", "log", "mixed"}:
            raise ValueError("拟合目标必须是linear、log或mixed")
        if not 0.0 <= float(self.log_weight) <= 1.0:
            raise ValueError("Log权重必须位于0到1之间")
        if not np.isfinite(self.intensity_floor) or self.intensity_floor <= 0:
            raise ValueError("Log强度底值必须为有限正数")
        if (
            not np.isfinite(self.min_peak_separation)
            or self.min_peak_separation < 0
        ):
            raise ValueError("最小峰间距必须是非负有限值")
        if self.fixed_background is not None and not np.isfinite(self.fixed_background):
            raise ValueError("固定背景必须是有限值")
        object.__setattr__(self, "log_weight", float(self.log_weight))
        object.__setattr__(self, "intensity_floor", float(self.intensity_floor))
        object.__setattr__(
            self,
            "min_peak_separation",
            float(self.min_peak_separation),
        )
        object.__setattr__(self, "include_ranges", _normalized_ranges(self.include_ranges))
        object.__setattr__(self, "exclude_ranges", _normalized_ranges(self.exclude_ranges))

    def to_record(self) -> dict[str, object]:
        return {
            "method": self.method,
            "objective_mode": self.objective_mode,
            "log_weight": self.log_weight,
            "intensity_floor": self.intensity_floor,
            "constrain_fwhm": self.constrain_fwhm,
            "min_peak_separation": self.min_peak_separation,
            "fixed_background": self.fixed_background,
            "include_ranges": [list(bounds) for bounds in self.include_ranges],
            "exclude_ranges": [list(bounds) for bounds in self.exclude_ranges],
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "FitConfiguration":
        return cls(
            method=str(record.get("method", "leastsq")),
            objective_mode=str(record.get("objective_mode", "mixed")),
            log_weight=float(record.get("log_weight", 0.5)),
            intensity_floor=float(record.get("intensity_floor", 1.0)),
            constrain_fwhm=bool(record.get("constrain_fwhm", False)),
            min_peak_separation=float(record.get("min_peak_separation", 0.2)),
            fixed_background=(
                None
                if record.get("fixed_background") is None
                else float(record["fixed_background"])
            ),
            include_ranges=tuple(record.get("include_ranges", ())),
            exclude_ranges=tuple(record.get("exclude_ranges", ())),
        )


def apply_preprocessing_steps(
    raw_scan: ScanData,
    steps: Iterable[PreprocessingStep],
) -> ScanData:
    """始终从同一份原始扫描执行完整的有序预处理流水线。"""
    intensity = np.array(raw_scan.intensity, dtype=float, copy=True)
    for step in steps:
        parameters = step.parameter_dict
        if step.operation == "savgol":
            intensity = Preprocessor.apply_savgol_filter(
                intensity,
                window_length=int(parameters["window_length"]),
                polyorder=int(parameters["polyorder"]),
            )
        elif step.operation == "gaussian":
            intensity = Preprocessor.apply_gaussian_filter(
                intensity,
                sigma=parameters["sigma"],
            )
        elif step.operation == "fft":
            intensity = Preprocessor.apply_fft_filter(
                intensity,
                cutoff_freq=parameters["cutoff_freq"],
            )
        elif step.operation == "polynomial_background":
            intensity, _ = Preprocessor.subtract_background_polynomial(
                raw_scan.two_theta,
                intensity,
                degree=int(parameters["degree"]),
            )
        elif step.operation == "snip_background":
            intensity, _ = Preprocessor.subtract_background_snip(
                intensity,
                iterations=int(parameters["iterations"]),
            )
        else:
            raise ValueError(f"未知预处理操作: {step.operation}")
    return raw_scan.with_intensity(intensity)


@dataclass(frozen=True)
class AnalysisSession:
    """原始扫描、处理结果及其可复现配置的单一状态源。"""

    raw_scan: Optional[ScanData] = None
    processed_scan: Optional[ScanData] = None
    source_scans: Tuple[ScanData, ...] = ()
    preprocessing: Tuple[PreprocessingStep, ...] = ()
    active_range: Optional[Tuple[float, float]] = None
    project_id: str = ""
    fit_configuration: FitConfiguration = FitConfiguration()

    @classmethod
    def empty(cls) -> "AnalysisSession":
        return cls()

    @classmethod
    def from_raw(
        cls,
        raw_scan: ScanData,
        *,
        active_range: Optional[Tuple[float, float]] = None,
        project_id: str = "",
        source_scans: Iterable[ScanData] = (),
    ) -> "AnalysisSession":
        return cls(
            raw_scan=raw_scan,
            processed_scan=raw_scan.with_intensity(raw_scan.intensity),
            source_scans=tuple(source_scans),
            active_range=(
                tuple(map(float, active_range))
                if active_range is not None
                else None
            ),
            project_id=str(project_id or raw_scan.source_id),
        )

    @classmethod
    def restore(
        cls,
        two_theta: np.ndarray,
        raw_intensity: np.ndarray,
        processed_intensity: np.ndarray,
        *,
        source_id: str = "",
        active_range: Optional[Tuple[float, float]] = None,
        project_id: str = "",
        preprocessing: Iterable[PreprocessingStep] = (),
        source_scans: Iterable[ScanData] = (),
    ) -> "AnalysisSession":
        raw_scan = ScanData(two_theta, raw_intensity, source_id=source_id)
        processed_scan = ScanData(two_theta, processed_intensity, source_id=source_id)
        if active_range is None:
            active_range = (
                float(np.min(two_theta)),
                float(np.max(two_theta)),
            )
        return cls(
            raw_scan=raw_scan,
            processed_scan=processed_scan,
            source_scans=tuple(source_scans),
            preprocessing=tuple(preprocessing),
            active_range=tuple(map(float, active_range)),
            project_id=str(project_id or source_id),
        )

    @property
    def has_data(self) -> bool:
        return self.raw_scan is not None and self.processed_scan is not None

    def with_preprocessing(
        self,
        steps: Iterable[PreprocessingStep],
    ) -> "AnalysisSession":
        if self.raw_scan is None:
            raise ValueError("尚未加载可预处理的XRD扫描")
        ordered_steps = tuple(steps)
        processed = apply_preprocessing_steps(self.raw_scan, ordered_steps)
        return replace(
            self,
            processed_scan=processed,
            preprocessing=ordered_steps,
        )

    def reset_preprocessing(self) -> "AnalysisSession":
        if self.raw_scan is None:
            return self
        return replace(
            self,
            processed_scan=self.raw_scan.with_intensity(self.raw_scan.intensity),
            preprocessing=(),
        )

    def with_fit_configuration(
        self,
        configuration: FitConfiguration,
    ) -> "AnalysisSession":
        return replace(self, fit_configuration=configuration)

    def crop(self, lower: float, upper: float) -> "AnalysisSession":
        if not self.has_data:
            raise ValueError("尚未加载可裁剪的XRD扫描")
        lower_value, upper_value = sorted((float(lower), float(upper)))
        raw_scan = self.raw_scan.crop(lower_value, upper_value)
        processed_scan = self.processed_scan.crop(lower_value, upper_value)
        cropped_sources = []
        for source_scan in self.source_scans:
            try:
                cropped_sources.append(source_scan.crop(lower_value, upper_value))
            except ValueError:
                continue
        return replace(
            self,
            raw_scan=raw_scan,
            processed_scan=processed_scan,
            source_scans=tuple(cropped_sources),
            active_range=(lower_value, upper_value),
        )
