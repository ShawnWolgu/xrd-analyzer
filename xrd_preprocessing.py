"""无 GUI 依赖的 XRD 预处理变换。"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy import fft, ndimage
from scipy.signal import savgol_filter


class Preprocessor:
    """对强度数组执行纯预处理变换，不修改输入数组。"""

    @staticmethod
    def _finite_segments(y_data: np.ndarray) -> List[slice]:
        """返回连续有限强度区间，避免滤波跨越未测量的扫描空隙。"""
        finite_indices = np.flatnonzero(np.isfinite(y_data))
        if finite_indices.size == 0:
            return []

        split_points = np.flatnonzero(np.diff(finite_indices) > 1) + 1
        index_groups = np.split(finite_indices, split_points)
        return [slice(group[0], group[-1] + 1) for group in index_groups]

    @classmethod
    def _filter_finite_segments(
        cls,
        y_data: np.ndarray,
        operation: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """仅对连续有效数据段执行滤波，并原样保留缺失区间。"""
        values = np.asarray(y_data, dtype=float)
        result = np.full_like(values, np.nan, dtype=float)
        for segment in cls._finite_segments(values):
            result[segment] = operation(values[segment])
        return result

    @staticmethod
    def apply_savgol_filter(
        y_data: np.ndarray,
        window_length: int = 11,
        polyorder: int = 3,
    ) -> np.ndarray:
        """应用 Savitzky-Golay 滤波。"""
        if window_length % 2 == 0:
            window_length += 1

        def filter_segment(segment: np.ndarray) -> np.ndarray:
            if segment.size <= polyorder:
                return segment.copy()
            local_window = min(window_length, segment.size)
            if local_window % 2 == 0:
                local_window -= 1
            if local_window <= polyorder:
                return segment.copy()
            return savgol_filter(segment, local_window, polyorder)

        return Preprocessor._filter_finite_segments(y_data, filter_segment)

    @staticmethod
    def apply_gaussian_filter(
        y_data: np.ndarray,
        sigma: float = 1.0,
    ) -> np.ndarray:
        """应用一维高斯滤波。"""
        return Preprocessor._filter_finite_segments(
            y_data,
            lambda segment: ndimage.gaussian_filter1d(segment, sigma),
        )

    @staticmethod
    def apply_fft_filter(
        y_data: np.ndarray,
        cutoff_freq: float = 0.1,
    ) -> np.ndarray:
        """应用 FFT 低通滤波。"""

        def filter_segment(segment: np.ndarray) -> np.ndarray:
            fft_vals = fft.fft(segment)
            freq = fft.fftfreq(len(segment))
            fft_vals[np.abs(freq) > cutoff_freq] = 0
            return np.real(fft.ifft(fft_vals))

        return Preprocessor._filter_finite_segments(y_data, filter_segment)

    @staticmethod
    def subtract_background_polynomial(
        x_data: np.ndarray,
        y_data: np.ndarray,
        degree: int = 2,
        anchor_points: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """拟合并扣除多项式背景。"""
        x_values = np.asarray(x_data, dtype=float)
        y_values = np.asarray(y_data, dtype=float)
        valid_data = np.isfinite(x_values) & np.isfinite(y_values)
        if anchor_points:
            x_anchors = np.array([point[0] for point in anchor_points])
            y_anchors = np.array([point[1] for point in anchor_points])
            valid_anchors = np.isfinite(x_anchors) & np.isfinite(y_anchors)
            x_anchors = x_anchors[valid_anchors]
            y_anchors = y_anchors[valid_anchors]
            if x_anchors.size <= degree:
                raise ValueError("有效背景锚点数量不足以完成多项式拟合")
            coeffs = np.polyfit(x_anchors, y_anchors, degree)
        else:
            if np.count_nonzero(valid_data) <= degree:
                raise ValueError("有效数据点数量不足以完成多项式背景拟合")
            coeffs = np.polyfit(x_values[valid_data], y_values[valid_data], degree)

        background = np.full_like(y_values, np.nan, dtype=float)
        background[valid_data] = np.polyval(coeffs, x_values[valid_data])
        return y_values - background, background

    @staticmethod
    def subtract_background_snip(
        y_data: np.ndarray,
        iterations: int = 40,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用 SNIP 峰裁剪背景估计。"""
        values = np.asarray(y_data, dtype=float)

        def snip_segment(segment: np.ndarray) -> np.ndarray:
            data = segment.copy()
            for iteration in range(iterations):
                window = 2**iteration
                for index in range(window, len(data) - window):
                    data[index] = min(
                        data[index],
                        (data[index - window] + data[index + window]) / 2,
                    )
            return data

        background = Preprocessor._filter_finite_segments(values, snip_segment)
        return values - background, background
