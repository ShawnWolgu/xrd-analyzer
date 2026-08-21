"""XRD 文本扫描加载、范围裁剪和多扫描合并。"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d


class DataLoader:
    """无 GUI 依赖的 XRD 输入与组合操作。"""

    @staticmethod
    def load_txt(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """读取包含两列数值的 2θ 与强度文本数据。"""
        x_data, y_data = [], []

        with open(filepath, "r", encoding="utf-8", errors="ignore") as input_file:
            for line in input_file:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        two_theta = float(parts[0])
                        intensity = float(parts[1])
                    except ValueError:
                        continue
                    x_data.append(two_theta)
                    y_data.append(intensity)

        return np.array(x_data), np.array(y_data)

    @staticmethod
    def trim_range(
        x_data: np.ndarray,
        y_data: np.ndarray,
        x_min: float,
        x_max: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按闭区间裁剪 2θ 数据。"""
        mask = (x_data >= x_min) & (x_data <= x_max)
        return x_data[mask], y_data[mask]

    @staticmethod
    def stitch_datasets(
        datasets: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """在平均步长网格上合并扫描，重叠处平均，空隙保留为 NaN。"""
        if not datasets:
            return np.array([]), np.array([])

        if len(datasets) == 1:
            return datasets[0]

        all_x = []
        steps = []
        for x_data, _ in datasets:
            all_x.append(x_data)
            if len(x_data) > 1:
                steps.append(np.mean(np.diff(x_data)))

        x_concat = np.concatenate(all_x)
        x_min, x_max = x_concat.min(), x_concat.max()
        average_step = np.mean(steps) if steps else 0.02
        if average_step <= 0 or np.isnan(average_step):
            average_step = 0.02

        point_count = int((x_max - x_min) / average_step) + 1
        x_uniform = np.linspace(x_min, x_max, point_count)
        y_accumulated = np.zeros_like(x_uniform)
        weights = np.zeros_like(x_uniform)

        for source_x, source_y in datasets:
            interpolate = interp1d(
                source_x,
                source_y,
                kind="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            interpolated = interpolate(x_uniform)
            valid_mask = ~np.isnan(interpolated)
            y_accumulated[valid_mask] += interpolated[valid_mask]
            weights[valid_mask] += 1

        measured_mask = weights > 0
        merged = np.full_like(x_uniform, np.nan, dtype=float)
        merged[measured_mask] = y_accumulated[measured_mask] / weights[measured_mask]
        return x_uniform, merged
