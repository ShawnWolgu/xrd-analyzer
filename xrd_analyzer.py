# xrd_analyzer.py - 核心处理引擎

import json
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import lmfit
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Iterable, List, Optional, Tuple
import warnings

from plot_style import apply_plot_style

from xrd_preprocessing import Preprocessor
from xrd_io import DataLoader
from xrd_crystallography import (
    BraggGeometry,
    DEFAULT_RADIATION_LABEL,
    DEFAULT_WAVELENGTH_ANGSTROM,
)
from xrd_peaks import Peak, PeakSnapshot, PSEUDO_VOIGT_FWHM_FACTOR
from xrd_project import ProjectWorkbook, RestoredFitResult


apply_plot_style()


PROJECT_WORKBOOK_SCHEMA_VERSION = 3


class Fitter:
    """拟合引擎核心类"""

    SIGMA_BOUNDS = {
        'film': (0.01, 1.5),
        'substrate': (0.01, 1.0),
    }
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = x_data
        self.y_data = y_data
        self.peaks: List[Peak] = []
        self.model = None
        self.params = None
        self.result = None
        self.y_fit = None
        self.background = None
        self.fit_mask = np.ones_like(x_data, dtype=bool)
        self.fit_config = {}
        self.fit_diagnostics = {}
        self.fit_warnings = []
        self.result_accepted = False
        self.restored_peak_curves = None

    def invalidate_fit_state(self) -> None:
        """峰配置改变后清除不再对应当前模型的拟合状态。"""
        self.model = None
        self.params = None
        self.result = None
        self.y_fit = None
        self.background = None
        self.fit_diagnostics = {}
        self.fit_warnings = []
        self.restored_peak_curves = None
        for peak in self.peaks:
            peak.clear_result()

    @classmethod
    def sigma_bounds(cls, peak_type: str) -> Tuple[float, float]:
        """返回指定峰类型的Pseudo-Voigt sigma搜索范围，单位为度。"""
        normalized_type = 'substrate' if peak_type == 'substrate' else 'film'
        return cls.SIGMA_BOUNDS[normalized_type]

    @classmethod
    def fwhm_bounds(cls, peak_type: str) -> Tuple[float, float]:
        """返回指定峰类型的FWHM搜索范围，单位为2theta度。"""
        sigma_min, sigma_max = cls.sigma_bounds(peak_type)
        return (
            PSEUDO_VOIGT_FWHM_FACTOR * sigma_min,
            PSEUDO_VOIGT_FWHM_FACTOR * sigma_max,
        )

    @staticmethod
    def log_residual(
        observed: np.ndarray,
        calculated: np.ndarray,
        intensity_floor: float,
    ) -> np.ndarray:
        """计算带显式强度底值的对数残差。"""
        if intensity_floor <= 0 or not np.isfinite(intensity_floor):
            raise ValueError("Log强度底值必须为有限正数")
        if np.any(observed < 0):
            raise ValueError("当前拟合数据包含负强度，不能使用Log目标")
        if np.any(calculated < 0):
            raise ValueError("当前模型产生负强度，不能使用Log目标")
        return np.log10(observed + intensity_floor) - np.log10(
            calculated + intensity_floor
        )

    @staticmethod
    def build_fit_mask(
        x_data: np.ndarray,
        include_ranges: Optional[List[Tuple[float, float]]] = None,
        exclude_ranges: Optional[List[Tuple[float, float]]] = None,
        y_data: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """从人工区间和有效观测构建拟合点掩码。"""
        if include_ranges:
            mask = np.zeros_like(x_data, dtype=bool)
            for lower, upper in include_ranges:
                lo, hi = sorted((lower, upper))
                mask |= (x_data >= lo) & (x_data <= hi)
        else:
            mask = np.ones_like(x_data, dtype=bool)

        for lower, upper in exclude_ranges or []:
            lo, hi = sorted((lower, upper))
            mask &= ~((x_data >= lo) & (x_data <= hi))

        mask &= np.isfinite(x_data)
        if y_data is not None:
            y_values = np.asarray(y_data)
            if y_values.shape != np.asarray(x_data).shape:
                raise ValueError("拟合数据的2θ与强度长度不一致")
            mask &= np.isfinite(y_values)

        if not np.any(mask):
            raise ValueError("人工拟合区间没有包含任何数据点")
        return mask

    def accept_current_result(self) -> None:
        """将当前候选结果保存为下一轮初值。"""
        if self.result is None:
            raise ValueError("当前没有可接受的拟合结果")

        for peak in self.peaks:
            if peak.fit_state != 'optimize':
                continue
            prefix = f'p{peak.peak_id}_'
            peak.center_guess = self.result.params[f'{prefix}center'].value
            peak.area_guess = self.result.params[f'{prefix}amplitude'].value
            peak.sigma_guess = self.result.params[f'{prefix}sigma'].value
            peak.fraction_guess = self.result.params[f'{prefix}fraction'].value
            height_factor = (
                (1 - peak.fraction_guess) * np.sqrt(np.log(2) / np.pi)
                + peak.fraction_guess / np.pi
            )
            peak.height_guess = (
                peak.area_guess / peak.sigma_guess * height_factor
            )
            if peak.bounds:
                width = peak.bounds[1] - peak.bounds[0]
                peak.bounds = (
                    peak.center_guess - width / 2,
                    peak.center_guess + width / 2,
                )
        self.result_accepted = True
        
    def add_peak(self, center_guess: float, 
                 bounds: Tuple[float, float] = None,
                 peak_type: str = 'film',
                 name: str = '') -> Peak:
        """添加峰"""
        self.invalidate_fit_state()
        peak_id = len(self.peaks)
        peak = Peak(peak_id, center_guess, bounds, peak_type, name)
        self.peaks.append(peak)
        return peak
    
    def remove_peaks(self, peak_ids: Iterable[int]) -> int:
        """原子删除多个峰、重新编号，并使旧拟合结果失效。"""
        ids_to_remove = set(peak_ids)
        remaining_peaks = [p for p in self.peaks if p.peak_id not in ids_to_remove]
        removed_count = len(self.peaks) - len(remaining_peaks)
        if removed_count == 0:
            return 0

        self.peaks = remaining_peaks
        for i, peak in enumerate(self.peaks):
            peak.peak_id = i

        self.invalidate_fit_state()
        return removed_count

    def remove_peak(self, peak_id: int) -> bool:
        """删除一个峰；成功删除时返回True。"""
        return self.remove_peaks([peak_id]) == 1

    def clear_peaks(self) -> int:
        """清除全部峰并使旧拟合结果失效。"""
        removed_count = len(self.peaks)
        if removed_count == 0:
            return 0
        self.peaks.clear()
        self.invalidate_fit_state()
        return removed_count

    def shift_peaks(self, delta_2theta_deg: float) -> int:
        """将全部峰中心和搜索边界整体平移指定的2theta角度。"""
        delta = float(delta_2theta_deg)
        if not np.isfinite(delta):
            raise ValueError("峰位平移量必须是有限的2θ角度")
        if not self.peaks or delta == 0.0:
            return 0

        finite_x = np.asarray(self.x_data)[np.isfinite(self.x_data)]
        if finite_x.size == 0:
            raise ValueError("当前XRD数据不包含有效的2θ坐标")
        data_min = float(np.min(finite_x))
        data_max = float(np.max(finite_x))

        shifted = []
        for peak in self.peaks:
            effective_center = (
                float(peak.center)
                if peak.center is not None
                else float(peak.center_guess)
            )
            new_center = effective_center + delta
            if not data_min <= new_center <= data_max:
                raise ValueError(
                    f"平移后的峰位2θ={new_center:.6f}°超出当前数据范围"
                    f"{data_min:.6f}–{data_max:.6f}°"
                )
            bounds_width = float(peak.bounds[1] - peak.bounds[0])
            shifted.append(
                (peak, new_center, (new_center - bounds_width / 2, new_center + bounds_width / 2))
            )

        for peak, new_center, new_bounds in shifted:
            peak.center_guess = new_center
            peak.bounds = new_bounds
        self.invalidate_fit_state()
        return len(shifted)
    
    def update_guesses_from_result(self):
        """兼容旧调用；明确接受当前完整峰形作为下一轮初值。"""
        if self.result is not None:
            self.accept_current_result()
    
    def build_model(self, constrain_fwhm: bool = False,
                   min_peak_separation: float = 0.2,
                   fixed_background: Optional[float] = None):
        """构建lmfit模型"""
        self.model = lmfit.models.ConstantModel()
        self.params = self.model.make_params()
        
        # 约束背景
        if 'c' in self.params:
            if fixed_background is not None:
                self.params['c'].set(value=fixed_background, vary=False)
            else:
                self.params['c'].set(min=0, vary=True)
        
        for peak in self.peaks:
            prefix = f'p{peak.peak_id}_'
            is_frozen = peak.fit_state == 'frozen'
            is_disabled = peak.fit_state == 'disabled'

            if is_frozen and any(
                value is None
                for value in (
                    peak.area_guess,
                    peak.sigma_guess,
                    peak.fraction_guess,
                )
            ):
                raise ValueError(f"Peak {peak.peak_id}尚无完整已接受峰形，不能冻结")
            
            # 使用Pseudo-Voigt模型
            peak_model = lmfit.models.PseudoVoigtModel(prefix=prefix)
            self.model += peak_model
            
            # 初始参数
            pars = peak_model.make_params()
            pars[f'{prefix}center'].set(value=peak.center_guess, 
                                       min=peak.bounds[0], 
                                       max=peak.bounds[1],
                                       vary=(
                                           not peak.fixed_center
                                           and not is_frozen
                                           and not is_disabled
                                       ))
            
            # 估算初始高度
            if peak.height_guess is not None:
                initial_height = peak.height_guess
            else:
                idx = np.argmin(np.abs(self.x_data - peak.center_guess))
                initial_height = self.y_data[idx]
            
            # FWHM约束 - 关键修改：确保vary=True，除非被锁定
            is_sigma_vary = (
                not peak.fixed_fwhm and not is_frozen and not is_disabled
            )
            
            if peak.sigma_guess is not None:
                # 如果有之前的拟合结果，优先使用
                initial_sigma = peak.sigma_guess
            elif peak.peak_type == 'substrate':
                # 基底峰通常很窄
                initial_sigma = 0.05
            else:
                # 样品峰使用较宽的通用初值；内部类型名保留为film以兼容旧项目。
                initial_sigma = 0.15
            
            # 设置Sigma参数；FWHM范围由精确关系FWHM=2*sigma确定。
            sigma_min, sigma_max = self.sigma_bounds(peak.peak_type)
            if not sigma_min <= initial_sigma <= sigma_max:
                fwhm_min, fwhm_max = self.fwhm_bounds(peak.peak_type)
                raise ValueError(
                    f"Peak {peak.peak_id} FWHM超出允许范围 "
                    f"[{fwhm_min:.4f}, {fwhm_max:.4f}]°"
                )
            pars[f'{prefix}sigma'].set(
                value=initial_sigma,
                min=sigma_min,
                max=sigma_max,
                vary=is_sigma_vary,
            )

            # Pseudo-Voigt混合参数
            initial_fraction = (
                peak.fraction_guess if peak.fraction_guess is not None else 0.5
            )
            pars[f'{prefix}fraction'].set(
                value=initial_fraction,
                min=0,
                max=1,
                vary=not is_frozen and not is_disabled,
            )
            
            # 设置Amplitude (Area) 参数
            # 如果锁定高度，我们需要约束Amplitude，使其随Sigma和Eta变化以保持Height恒定
            if is_disabled:
                pars[f'{prefix}amplitude'].set(value=0.0, min=0, vary=False)
            elif is_frozen:
                pars[f'{prefix}amplitude'].set(
                    value=peak.area_guess,
                    min=0,
                    vary=False,
                )
            elif peak.fixed_height:
                # Formula (LMFIT Pseudo-Voigt):
                # H = (A / sigma) * [ (1-eta)*sqrt(ln(2)/pi) + eta/pi ]
                # Therefore: A = H * sigma / [ (1-eta)*sqrt(ln(2)/pi) + eta/pi ]
                # Constants: sqrt(ln(2)/pi) ≈ 0.4697186, 1/pi ≈ 0.3183099
                
                # Use the target height (initial_height) as the fixed value
                h_val = initial_height
                if h_val <= 0: h_val = 1e-6 # Avoid zero
                
                expr = f'{h_val:.6f} * {prefix}sigma / ((1-{prefix}fraction)*0.4697186 + {prefix}fraction*0.3183099)'
                
                # Set initial value for amplitude based on current sigma/fraction to start cleanly
                term1 = (1 - initial_fraction) * 0.4697186
                term2 = initial_fraction * 0.3183099
                initial_amp = h_val * initial_sigma / (term1 + term2)
                
                pars[f'{prefix}amplitude'].set(value=initial_amp, min=0, expr=expr)
            else:
                # Normal estimation: Area approx Height * Sigma * 2.5 (rough guess)
                # Better: calculate based on assumed fraction=0.5
                term1 = (1 - initial_fraction) * 0.4697186
                term2 = initial_fraction * 0.3183099
                initial_amp = (
                    peak.area_guess
                    if peak.area_guess is not None
                    else initial_height * initial_sigma / (term1 + term2)
                )
                
                pars[f'{prefix}amplitude'].set(value=initial_amp, min=0, vary=True)
            
            self.params.update(pars)
        
        # 添加FWHM相等约束 - 只在勾选时生效
        if constrain_fwhm and len(self.peaks) >= 2:
            film_peaks = [
                p
                for p in self.peaks
                if p.peak_type == 'film' and p.fit_state == 'optimize'
            ]
            if len(film_peaks) >= 2:
                reference_peak = film_peaks[0]
                for peak in film_peaks[1:]:
                    # 如果该峰单独被锁定FWHM，则跳过全局约束
                    if not peak.fixed_fwhm:
                        self.params[f'p{peak.peak_id}_sigma'].set(
                            expr=f'p{reference_peak.peak_id}_sigma'
                        )
        
        # 添加峰间距约束。使用独立的gap参数表达两个
        # 中心之间的关系，避免把后一个峰绑定到前一个峰的初始猜测。
        if min_peak_separation > 0 and len(self.peaks) >= 2:
            sorted_peaks = sorted(
                (p for p in self.peaks if p.fit_state == 'optimize'),
                key=lambda p: p.center_guess,
            )
            for i in range(len(sorted_peaks) - 1):
                p1 = sorted_peaks[i]
                p2 = sorted_peaks[i + 1]
                center1 = self.params[f'p{p1.peak_id}_center']
                center2 = self.params[f'p{p2.peak_id}_center']

                if p1.fixed_center and p2.fixed_center:
                    if center2.value - center1.value < min_peak_separation:
                        raise ValueError(
                            f"固定的Peak {p1.peak_id}与Peak {p2.peak_id}"
                            "不满足最小峰间距"
                        )
                    continue

                if p1.fixed_center:
                    new_minimum = max(center2.min, center1.value + min_peak_separation)
                    if new_minimum > center2.max:
                        raise ValueError(
                            f"Peak {p2.peak_id}的中心边界无法满足最小峰间距"
                        )
                    center2.set(min=new_minimum)
                    continue

                if p2.fixed_center:
                    new_maximum = min(center1.max, center2.value - min_peak_separation)
                    if new_maximum < center1.min:
                        raise ValueError(
                            f"Peak {p1.peak_id}的中心边界无法满足最小峰间距"
                        )
                    center1.set(max=new_maximum)
                    continue

                gap_name = f'p{p2.peak_id}_center_gap'
                initial_gap = max(
                    p2.center_guess - p1.center_guess,
                    min_peak_separation,
                )
                maximum_gap = p2.bounds[1] - p1.bounds[0]
                if maximum_gap < min_peak_separation:
                    raise ValueError(
                        f"Peak {p1.peak_id}与Peak {p2.peak_id}的中心边界"
                        "无法满足最小峰间距"
                    )
                self.params.add(
                    gap_name,
                    value=min(initial_gap, maximum_gap),
                    min=min_peak_separation,
                    max=maximum_gap,
                    vary=True,
                )
                center2.set(
                    expr=f'p{p1.peak_id}_center + {gap_name}'
                )
        
    def execute_fitting(
        self,
        method: str = 'leastsq',
        use_log_scale: bool = True,
        log_weight: float = 0.5,
        objective_mode: Optional[str] = None,
        intensity_floor: float = 1.0,
        include_ranges: Optional[List[Tuple[float, float]]] = None,
        exclude_ranges: Optional[List[Tuple[float, float]]] = None,
    ) -> lmfit.model.ModelResult:
        """
        执行拟合
        
        Args:
            method: 拟合方法
            use_log_scale: 是否使用对数尺度 (True: 混合对数/线性, False: 纯线性)
            log_weight: Mixed目标中对数残差的比例 (0.0 - 1.0)
            objective_mode: linear / log / mixed；None时兼容旧use_log_scale参数
            intensity_floor: Log目标中的显式强度底值I0
            include_ranges: 本轮参与拟合的2theta区间
            exclude_ranges: 本轮排除的2theta区间
        """
        if self.model is None:
            self.build_model()

        if objective_mode is None:
            objective_mode = 'mixed' if use_log_scale else 'linear'
        objective_mode = objective_mode.lower()
        if objective_mode not in {'linear', 'log', 'mixed'}:
            raise ValueError(f"未知拟合目标: {objective_mode}")
        if not 0.0 <= log_weight <= 1.0:
            raise ValueError("Log权重必须在0和1之间")

        self.fit_mask = self.build_fit_mask(
            self.x_data,
            include_ranges=include_ranges,
            exclude_ranges=exclude_ranges,
            y_data=self.y_data,
        )
        x_fit = self.x_data[self.fit_mask]
        y_fit_observed = self.y_data[self.fit_mask]
        if objective_mode in {'log', 'mixed'} and np.any(y_fit_observed < 0):
            raise ValueError("当前拟合数据包含负强度，不能使用Log或Mixed目标")

        self.fit_config = {
            'method': method,
            'objective_mode': objective_mode,
            'log_weight': log_weight,
            'intensity_floor': intensity_floor,
            'include_ranges': list(include_ranges or []),
            'exclude_ranges': list(exclude_ranges or []),
            'fit_point_count': int(np.sum(self.fit_mask)),
        }

        # 调试：打印初始参数
        print("Initial parameters:")
        for peak in self.peaks:
            prefix = f'p{peak.peak_id}_'
            print(f"  Peak {peak.peak_id} sigma: {self.params[f'{prefix}sigma'].value}, vary={self.params[f'{prefix}sigma'].vary}")
        
        def configured_residual(params, model, x, data):
            model_vals = model.eval(params, x=x)
            linear_scale = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1.0
            res_linear = (data - model_vals) / linear_scale

            if objective_mode == 'linear':
                return res_linear

            res_log = self.log_residual(data, model_vals, intensity_floor)
            if objective_mode == 'log':
                return res_log

            residual_parts = []
            if log_weight < 1.0:
                residual_parts.append(np.sqrt(1.0 - log_weight) * res_linear)
            if log_weight > 0.0:
                residual_parts.append(np.sqrt(log_weight) * res_log)
            return np.concatenate(residual_parts)

        print(
            f"Executing fitting with {objective_mode.upper()} objective "
            f"(Log weight={log_weight:.2%}, I0={intensity_floor:g}, method={method})..."
        )
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            self.result = lmfit.minimize(
                configured_residual,
                self.params,
                args=(self.model, x_fit, y_fit_observed),
                method=method,
            )
        self.fit_warnings = [str(item.message) for item in caught_warnings]
        self.y_fit = self.model.eval(self.result.params, x=self.x_data)
        
        print("\nFitted parameters:")
        for peak in self.peaks:
            prefix = f'p{peak.peak_id}_'
            print(f"  Peak {peak.peak_id} sigma: {self.result.params[f'{prefix}sigma'].value}")
        
        # 确保 y_fit 是最新的
        if self.y_fit is None:
             self.y_fit = self.model.eval(self.result.params, x=self.x_data)

        # 获取各组件的独立曲线以便精确计算峰高
        if hasattr(self.result, 'eval_components'):
             components = self.result.eval_components(x=self.x_data)
        else:
             components = self.model.eval_components(params=self.result.params, x=self.x_data)

        # 更新各峰的参数
        for peak in self.peaks:
            # 从组件曲线中获取峰中心处的高度
            prefix = f'p{peak.peak_id}_'
            if prefix in components:
                # 找到最接近中心的数据点索引
                center_val = self.result.params[f'{prefix}center'].value
                # 或者更精确地，评估该组件在center处的值
                # 这里我们简单取最接近点的组件值作为高度近似，
                # 或者为了更精确，我们可以重新evaluate component at single point x=center
                
                # 方法2: 精确计算
                peak_model = lmfit.models.PseudoVoigtModel(prefix=prefix)
                # 构造只包含该峰参数的字典
                peak_params = peak_model.make_params()
                for p_name in peak_params:
                    if p_name in self.result.params:
                        peak_params[p_name].value = self.result.params[p_name].value
                
                fitted_height = peak_model.eval(peak_params, x=np.array([center_val]))[0]
            else:
                fitted_height = None

            peak.set_result(self.result.params, fitted_height)

            # 移除 calculate_area 调用，因为它计算的是 gross area (包含背景)
            # 我们现在只使用 set_result 中设置的 amplitude (net area)
            # peak.calculate_area(self.x_data, self.y_fit)
        
        self.fit_diagnostics = self._collect_fit_diagnostics()
        return self.result

    def _collect_fit_diagnostics(self) -> Dict:
        """汇总求解状态、协方差和独立参数边界命中。"""
        if self.result is None:
            return {}

        boundary_hits = []
        for name, parameter in self.result.params.items():
            if parameter.expr is not None or not parameter.vary:
                continue
            scale = max(abs(parameter.value), 1.0)
            tolerance = 1e-6 * scale
            if np.isfinite(parameter.min) and abs(parameter.value - parameter.min) <= tolerance:
                boundary_hits.append(f'{name}:min')
            if np.isfinite(parameter.max) and abs(parameter.value - parameter.max) <= tolerance:
                boundary_hits.append(f'{name}:max')

        return {
            'success': bool(getattr(self.result, 'success', False)),
            'message': str(getattr(self.result, 'message', '')),
            'nfev': int(getattr(self.result, 'nfev', 0)),
            'covariance_available': getattr(self.result, 'covar', None) is not None,
            'boundary_hits': boundary_hits,
            'warnings': self.fit_warnings,
            **self.fit_config,
        }
    
    def get_individual_peaks(self) -> Dict[int, np.ndarray]:
        """获取各个峰的独立曲线"""
        if self.result is None:
            return {}

        if self.restored_peak_curves is not None:
            return {
                peak_id: curve.copy()
                for peak_id, curve in self.restored_peak_curves.items()
            }
        
        peak_curves = {}
        
        # 兼容ModelResult和MinimizerResult
        if hasattr(self.result, 'eval_components'):
             components = self.result.eval_components(x=self.x_data)
        else:
             components = self.model.eval_components(params=self.result.params, x=self.x_data)

        for peak in self.peaks:
            prefix = f'p{peak.peak_id}_'
            peak_curves[peak.peak_id] = components[prefix]
        
        return peak_curves

    def get_fit_report(self) -> str:
        """获取拟合报告"""
        if self.result is None:
            return "No fit result available."
        if getattr(self.result, 'restored', False):
            return (
                "Result restored from an XRD project workbook.\n"
                "Solver diagnostics and parameters are preserved in the workbook; "
                "run a new fit to generate a fresh lmfit report."
            )
        return lmfit.fit_report(self.result)


def _snapshot_array(values) -> Optional[np.ndarray]:
    if values is None:
        return None
    copied = np.array(values, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class FitterSnapshot:
    """Complete peak-table and fitted-plot state for undo/redo."""

    peaks: Tuple[PeakSnapshot, ...]
    has_result: bool
    y_fit: Optional[np.ndarray]
    background: Optional[np.ndarray]
    fit_mask: Optional[np.ndarray]
    fit_config: Dict
    fit_diagnostics: Dict
    fit_warnings: Tuple[str, ...]
    result_accepted: bool
    result_chisqr: float
    result_success: bool
    result_message: str
    result_nfev: int
    parameter_values: Tuple[Tuple[str, float], ...]
    peak_curves: Tuple[Tuple[int, np.ndarray], ...]

    @classmethod
    def capture(cls, fitter: Optional[Fitter]) -> "FitterSnapshot":
        if fitter is None:
            return cls(
                peaks=(),
                has_result=False,
                y_fit=None,
                background=None,
                fit_mask=None,
                fit_config={},
                fit_diagnostics={},
                fit_warnings=(),
                result_accepted=False,
                result_chisqr=np.nan,
                result_success=False,
                result_message="",
                result_nfev=0,
                parameter_values=(),
                peak_curves=(),
            )

        result = fitter.result
        has_result = (
            result is not None
            and fitter.y_fit is not None
            and hasattr(result, 'params')
        )
        parameter_values = ()
        peak_curves = ()
        if has_result:
            parameter_values = tuple(
                (name, float(parameter.value))
                for name, parameter in result.params.items()
            )
            peak_curves = tuple(
                (peak_id, _snapshot_array(curve))
                for peak_id, curve in fitter.get_individual_peaks().items()
            )

        return cls(
            peaks=tuple(PeakSnapshot.from_peak(peak) for peak in fitter.peaks),
            has_result=has_result,
            y_fit=_snapshot_array(fitter.y_fit),
            background=_snapshot_array(fitter.background),
            fit_mask=_snapshot_array(fitter.fit_mask),
            fit_config=deepcopy(fitter.fit_config),
            fit_diagnostics=deepcopy(fitter.fit_diagnostics),
            fit_warnings=tuple(fitter.fit_warnings),
            result_accepted=bool(fitter.result_accepted),
            result_chisqr=float(getattr(result, 'chisqr', np.nan)),
            result_success=bool(getattr(result, 'success', False)),
            result_message=str(getattr(result, 'message', '')),
            result_nfev=int(getattr(result, 'nfev', 0)),
            parameter_values=parameter_values,
            peak_curves=peak_curves,
        )

    def restore_into(self, fitter: Fitter) -> Fitter:
        fitter.peaks = [snapshot.to_peak() for snapshot in self.peaks]
        fitter.fit_config = deepcopy(self.fit_config)
        fitter.fit_diagnostics = deepcopy(self.fit_diagnostics)
        fitter.fit_warnings = list(self.fit_warnings)
        fitter.result_accepted = self.result_accepted
        fitter.fit_mask = (
            self.fit_mask.copy()
            if self.fit_mask is not None
            else np.ones_like(fitter.x_data, dtype=bool)
        )
        fitter.background = (
            self.background.copy() if self.background is not None else None
        )
        fitter.y_fit = self.y_fit.copy() if self.y_fit is not None else None
        fitter.restored_peak_curves = (
            {peak_id: curve.copy() for peak_id, curve in self.peak_curves}
            if self.has_result
            else None
        )

        if self.has_result:
            params = lmfit.Parameters()
            for name, value in self.parameter_values:
                params.add(name, value=value)
            fitter.params = params
            fitter.result = RestoredFitResult(
                params=params,
                success=self.result_success,
                message=self.result_message or 'Restored from peak history',
                nfev=self.result_nfev,
                covar=None,
                chisqr=self.result_chisqr,
            )
        return fitter


class FitterHistory:
    """Bounded timeline containing completed fitting results only."""

    def __init__(self, limit: int = 5):
        if limit < 1:
            raise ValueError("拟合历史步数必须至少为1")
        self.limit = limit
        self._states: List[FitterSnapshot] = []
        self._index = -1

    @property
    def can_undo(self) -> bool:
        return self._index > 0

    @property
    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._states) - 1

    def clear(self) -> None:
        self._states.clear()
        self._index = -1

    def record(self, fitter: Optional[Fitter]) -> bool:
        """Append one completed fit and discard any forward branch."""
        snapshot = FitterSnapshot.capture(fitter)
        if not snapshot.has_result:
            return False
        if self._index < len(self._states) - 1:
            del self._states[self._index + 1:]
        self._states.append(snapshot)
        if len(self._states) > self.limit:
            del self._states[:len(self._states) - self.limit]
        self._index = len(self._states) - 1
        return True

    def undo(self) -> Optional[FitterSnapshot]:
        if not self.can_undo:
            return None
        self._index -= 1
        return self._states[self._index]

    def redo(self) -> Optional[FitterSnapshot]:
        if not self.can_redo:
            return None
        self._index += 1
        return self._states[self._index]


class Reporter:
    """结果报告类"""
    def __init__(
        self,
        fitter: Fitter,
        x_original=None,
        y_original=None,
        wavelength_angstrom: float = DEFAULT_WAVELENGTH_ANGSTROM,
        radiation_label: Optional[str] = None,
    ):
        self.fitter = fitter
        self.metrics = {}
        self.x_original = x_original  # 新增
        self.y_original = y_original  # 新增
        self.wavelength_angstrom = BraggGeometry._validated_wavelength(
            wavelength_angstrom
        )
        if radiation_label is None:
            radiation_label = (
                DEFAULT_RADIATION_LABEL
                if np.isclose(
                    self.wavelength_angstrom,
                    DEFAULT_WAVELENGTH_ANGSTROM,
                    rtol=0.0,
                    atol=5e-7,
                )
                else '自定义波长'
            )
        self.radiation_label = str(radiation_label)
        
    def calculate_metrics(self) -> Dict:
        """仅在本轮拟合掩码内计算线性强度空间的R²诊断。"""
        if self.fitter.result is None:
            return {}

        fit_mask = np.asarray(self.fitter.fit_mask, dtype=bool)
        if fit_mask.shape != self.fitter.y_data.shape:
            raise ValueError("拟合掩码与数据长度不一致，无法计算R²_fit")
        y_obs = self.fitter.y_data[fit_mask]
        y_calc = self.fitter.y_fit[fit_mask]
        residuals = y_obs - y_calc

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

        self.metrics = {
            'R_squared_fit': r_squared,
            'R_squared_fit_scale': 'linear_intensity',
            'Fit_Point_Count': int(np.sum(fit_mask)),
            'Objective_SSE': float(getattr(self.fitter.result, 'chisqr', np.nan)),
            'Objective_Mode': self.fitter.fit_config.get('objective_mode', 'unknown'),
        }
        
        return self.metrics
    
    def calculate_characteristic_lengths(
        self,
        wavelength: Optional[float] = None,
    ) -> Dict:
        """按Bragg定律计算各反射峰对应的特征长度d。"""
        wavelength = (
            self.wavelength_angstrom
            if wavelength is None
            else BraggGeometry._validated_wavelength(wavelength)
        )
        characteristic_lengths = {}

        for peak in self.fitter.peaks:
            if peak.center is None:
                continue

            characteristic_length = BraggGeometry.d_from_two_theta(
                peak.center,
                wavelength,
            )

            characteristic_lengths[f'Peak_{peak.peak_id}'] = {
                'peak_id': peak.peak_id,
                'reflection_label': peak.name,
                '2theta_deg': peak.center,
                'characteristic_length_angstrom': characteristic_length,
                'wavelength_angstrom': wavelength,
                'radiation_label': self.radiation_label,
            }

        return characteristic_lengths
    
    def export_results(
        self,
        output_path: str,
        project_state: Optional[Dict] = None,
        source_files: Optional[List[str]] = None,
        source_datasets: Optional[
            Iterable[Tuple[str, np.ndarray, np.ndarray]]
        ] = None,
    ):
        """导出可恢复项目工作簿及兼容的结果工作表。"""
        characteristic_lengths = self.calculate_characteristic_lengths()
        
        results_list = []
        
        for peak in self.fitter.peaks:
            characteristic_length = None
            peak_key = f'Peak_{peak.peak_id}'
            if peak_key in characteristic_lengths:
                characteristic_length = characteristic_lengths[peak_key].get(
                    'characteristic_length_angstrom'
                )

            results_list.append({
                'Peak_ID': peak.peak_id,
                'Name': peak.name,          # 新增: 峰名称
                'Type': peak.peak_type,
                'Center_2theta': peak.center,
                'Center_Guess_2theta': peak.center_guess,
                'Bounds_Min_2theta': peak.bounds[0],
                'Bounds_Max_2theta': peak.bounds[1],
                'Characteristic_Length_d_Angstrom': characteristic_length,
                'Wavelength_Angstrom': self.wavelength_angstrom,
                'Radiation_Label': self.radiation_label,
                'FWHM': peak.fwhm,
                'Sigma_Guess': peak.sigma_guess,
                'Height': peak.height,
                'Height_Guess': peak.height_guess,
                'Area': peak.area,
                'Area_Guess': peak.area_guess,
                'Eta_PseudoVoigt': peak.eta,
                'Fraction_Guess': peak.fraction_guess,
                'Fixed_Center': peak.fixed_center,
                'Fixed_Height': peak.fixed_height,
                'Fixed_FWHM': peak.fixed_fwhm,
                'Fit_State': peak.fit_state,
            })
        
        df_peaks = pd.DataFrame(results_list)
        
        # 评估指标
        metrics_df = pd.DataFrame([self.metrics])

        configuration = {
            **self.fitter.fit_config,
            **self.fitter.fit_diagnostics,
        }
        configuration = {
            key: repr(value) if isinstance(value, (list, tuple, dict)) else value
            for key, value in configuration.items()
        }
        configuration_df = pd.DataFrame([configuration])

        stored_project_state = {
            **(project_state or {}),
            'wavelength_angstrom': self.wavelength_angstrom,
            'radiation_label': self.radiation_label,
            'schema_version': PROJECT_WORKBOOK_SCHEMA_VERSION,
        }
        project_state_df = pd.DataFrame(
            [
                {
                    'Key': key,
                    'Value_JSON': json.dumps(value, ensure_ascii=False),
                }
                for key, value in stored_project_state.items()
            ]
        )
        source_dataframes = []
        source_rows = []
        for index, (path, source_x, source_y) in enumerate(source_datasets or []):
            source_x = np.asarray(source_x, dtype=float)
            source_y = np.asarray(source_y, dtype=float)
            if len(source_x) != len(source_y):
                raise ValueError(f"源数据长度不一致: {path}")
            sheet_name = f'Source_Data_{index + 1:03d}'
            source_rows.append({'Path': str(path), 'Sheet_Name': sheet_name})
            source_dataframes.append(
                (
                    sheet_name,
                    pd.DataFrame({'2theta': source_x, 'Intensity': source_y}),
                )
            )
        recorded_paths = {row['Path'] for row in source_rows}
        for path in source_files or []:
            if str(path) not in recorded_paths:
                source_rows.append({'Path': str(path), 'Sheet_Name': None})
        source_files_df = pd.DataFrame(
            source_rows,
            columns=['Path', 'Sheet_Name'],
        )
        
        # 完整数据表
        data_dict = {
            '2theta': self.fitter.x_data,
            'Processed_Intensity': self.fitter.y_data,  # 处理后的数据
            'Fitted_Intensity': self.fitter.y_fit,
            'Residuals': self.fitter.y_data - self.fitter.y_fit
        }
        
        # 如果有原始数据，插值到相同的x点
        if self.x_original is not None and self.y_original is not None:
            # 插值原始数据到拟合数据的x点
            from scipy.interpolate import interp1d
            f_interp = interp1d(self.x_original, self.y_original, 
                               kind='linear', bounds_error=False, fill_value=np.nan)
            y_original_interp = f_interp(self.fitter.x_data)
            data_dict['Original_Intensity'] = y_original_interp
        
        # 添加各峰的分量
        peak_curves = self.fitter.get_individual_peaks()
        for peak_id in sorted(peak_curves.keys()):
            curve = peak_curves[peak_id]
            data_dict[f'Peak_{peak_id}_Component'] = curve
        
        # 添加背景（constant）
        if self.fitter.result is not None:
            constant = self.fitter.result.params.get('c')
            if constant is not None:
                data_dict['Background'] = np.full_like(self.fitter.x_data, constant.value)
        
        df_data = pd.DataFrame(data_dict)
        
        # 写入Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_peaks.to_excel(writer, sheet_name='Peak_Parameters', index=False)
            metrics_df.to_excel(writer, sheet_name='Fit_Metrics', index=False)
            configuration_df.to_excel(
                writer,
                sheet_name='Fit_Configuration',
                index=False,
            )
            project_state_df.to_excel(writer, sheet_name='Project_State', index=False)
            source_files_df.to_excel(writer, sheet_name='Source_Files', index=False)
            for sheet_name, source_dataframe in source_dataframes:
                source_dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 完整数据
            df_data.to_excel(writer, sheet_name='Full_Data', index=False)
        
        print(f"Results exported to {output_path}")
        
    def plot_results(self, save_path: str = None, show_components: bool = True):
        """绘制拟合结果 (包含线性、对数和残差图)"""
        # 创建3个子图: 线性, 对数, 残差
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), 
                                       gridspec_kw={'height_ratios': [2, 2, 1]},
                                       sharex=True)
        
        # --- 1. 线性坐标图 (Linear) ---
        ax1.scatter(self.fitter.x_data, self.fitter.y_data, 
                   s=10, alpha=0.6, label='Raw Data', color='gray')
        ax1.plot(self.fitter.x_data, self.fitter.y_fit, 
                'r-', linewidth=2, label='Fit')
        
        if show_components:
            peak_curves = self.fitter.get_individual_peaks()
            colors = plt.cm.tab10(np.linspace(0, 1, len(peak_curves)))
            
            for i, (peak_id, curve) in enumerate(peak_curves.items()):
                peak = self.fitter.peaks[peak_id]
                label = f'Peak {peak_id} ({peak.peak_type})'
                if peak.name:
                    label += f': {peak.name}'
                ax1.plot(self.fitter.x_data, curve, 
                        '--', color=colors[i], linewidth=1.5, 
                        alpha=0.7, label=label)

        # 添加峰名注记 (Linear)
        for peak in self.fitter.peaks:
            label = peak.name if peak.name else f'P{peak.peak_id}'
            # 使用拟合后的位置，如果还没拟合完则使用猜测值
            center = peak.center if peak.center is not None else peak.center_guess
            height = peak.height if peak.height is not None else 0
            ax1.annotate(label, xy=(center, height), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9, color='darkblue', fontweight='bold')

        ax1.set_ylabel('Intensity (Linear)', fontsize=12, fontweight='bold')
        ax1.set_title('XRD Pattern Fitting Results', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        # Legend outside right
        ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

        # --- 2. 对数坐标图 (Log) ---
        ax2.scatter(self.fitter.x_data, self.fitter.y_data, 
                   s=10, alpha=0.6, color='gray') # No label to avoid duplicate legend
        ax2.plot(self.fitter.x_data, self.fitter.y_fit, 
                'r-', linewidth=2)
        
        if show_components:
            for i, (peak_id, curve) in enumerate(peak_curves.items()):
                ax2.plot(self.fitter.x_data, curve, 
                        '--', color=colors[i], linewidth=1.5, alpha=0.7)
        
        # 添加峰名注记 (Log)
        for peak in self.fitter.peaks:
            label = peak.name if peak.name else f'P{peak.peak_id}'
            center = peak.center if peak.center is not None else peak.center_guess
            height = peak.height if peak.height is not None else 0
            # 只有当高度大于0时才标注，避免对数坐标报错
            if height > 0:
                ax2.annotate(label, xy=(center, height), 
                            xytext=(0, 10), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9, color='darkblue', fontweight='bold')

        ax2.set_yscale('log')
        ax2.set_ylabel('Intensity (Log)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 强制设置Log图的Y轴范围（基于原始数据）
        y_pos = self.fitter.y_data[self.fitter.y_data > 0]
        if len(y_pos) > 0:
            y_min = np.min(y_pos)
            y_max = np.max(self.fitter.y_data)
            ax2.set_ylim(y_min * 0.5, y_max * 2.0)

        # 对数图不重复显示图例

        # --- 3. 残差图 (Residuals) ---
        residuals = self.fitter.y_data - self.fitter.y_fit
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax3.scatter(self.fitter.x_data, residuals, 
                   s=5, alpha=0.5, color='blue')
        ax3.set_xlabel('2θ (degree)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Residuals', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 添加评估指标文本 (to the first plot)
        metrics_text = f"R²_fit = {self.metrics.get('R_squared_fit', np.nan):.6f}"
        ax1.text(0.02, 0.98, metrics_text, 
                transform=ax1.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        # 强制设置X轴范围为数据范围
        if self.fitter.x_data is not None:
            x_min, x_max = self.fitter.x_data.min(), self.fitter.x_data.max()
            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)
            ax3.set_xlim(x_min, x_max)
            
        plt.tight_layout()
        
        if save_path:
            # 使用 bbox_inches='tight' 确保外置图例被包含
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig


# 辅助函数
def pseudo_voigt(x, amplitude, center, sigma, fraction):
    """Pseudo-Voigt函数"""
    gaussian = np.exp(-((x - center) / sigma)**2 / 2)
    lorentzian = 1 / (1 + ((x - center) / sigma)**2)
    return amplitude * (fraction * gaussian + (1 - fraction) * lorentzian)
