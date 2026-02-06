# xrd_analyzer.py - 核心处理引擎

import numpy as np
import pandas as pd
from scipy import signal, ndimage, optimize, fft
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
import lmfit
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class Peak:
    """单个峰的数据结构"""
    def __init__(self, peak_id: int, center_guess: float, 
                 bounds: Tuple[float, float] = None,
                 peak_type: str = 'film',
                 name: str = ''):
        self.peak_id = peak_id
        self.center_guess = center_guess
        self.bounds = bounds if bounds else (center_guess - 0.5, center_guess + 0.5)
        self.peak_type = peak_type  # 'film' or 'substrate'
        self.name = name
        
        # 拟合结果
        self.center = None
        self.height = None
        self.fwhm = None
        self.area = None
        self.eta = None  # Pseudo-Voigt混合参数
        
    def set_result(self, params: Dict):
        """从lmfit结果设置参数"""
        prefix = f'p{self.peak_id}_'
        self.center = params[f'{prefix}center'].value
        self.height = params[f'{prefix}amplitude'].value
        self.fwhm = params[f'{prefix}sigma'].value * 2.355  # 转换为FWHM
            # 修改这一行：
        fraction_param = params.get(f'{prefix}fraction')
        if fraction_param is not None:
            self.eta = fraction_param.value
        else:
            self.eta = 0.5  # 默认值
        
    def calculate_area(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
        """计算峰面积"""
        # 在峰中心±3*FWHM范围内积分
        if self.fwhm is None:
            return 0
        mask = np.abs(x_data - self.center) <= 3 * self.fwhm
        area = np.trapz(y_data[mask], x_data[mask])
        self.area = area
        return area


class DataLoader:
    """数据加载类"""
    
    @staticmethod
    def load_txt(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载XRD数据"""
        x_data, y_data = [], []
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    try:
                        x_data.append(float(parts[0]))
                        y_data.append(float(parts[1]))
                    except ValueError:
                        continue
        
        return np.array(x_data), np.array(y_data)
    
    @staticmethod
    def trim_range(x_data: np.ndarray, y_data: np.ndarray,
                   x_min: float, x_max: float) -> Tuple[np.ndarray, np.ndarray]:
        """裁剪数据范围"""
        mask = (x_data >= x_min) & (x_data <= x_max)
        return x_data[mask], y_data[mask]

    @staticmethod
    def stitch_datasets(datasets: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        拼接多个数据集 (Stitch multiple datasets)
        
        策略:
        1. 确定整体的X范围
        2. 生成均匀网格
        3. 将每个数据集插值到网格上，未覆盖区域设为基线值(1e-5)
        4. 重叠区域取平均
        """
        if not datasets:
            return np.array([]), np.array([])
            
        if len(datasets) == 1:
            return datasets[0]
            
        # 1. 确定全范围和步长
        all_x = []
        steps = []
        
        for x, y in datasets:
            all_x.append(x)
            if len(x) > 1:
                # 取每个文件中间段的平均步长
                steps.append(np.mean(np.diff(x)))
                
        x_concat = np.concatenate(all_x)
        x_min, x_max = x_concat.min(), x_concat.max()
        
        # 确定平均步长
        avg_step = np.mean(steps) if steps else 0.02
        if avg_step <= 0 or np.isnan(avg_step):
            avg_step = 0.02
            
        # 2. 生成均匀网格
        num_points = int((x_max - x_min) / avg_step) + 1
        x_uniform = np.linspace(x_min, x_max, num_points)
        
        # 3. 累积数据
        y_accum = np.zeros_like(x_uniform)
        weights = np.zeros_like(x_uniform)
        
        for x_src, y_src in datasets:
            # 创建该数据集的插值函数
            # bounds_error=False, fill_value=np.nan 让未覆盖区域为NaN
            f = interp1d(x_src, y_src, kind='linear', bounds_error=False, fill_value=np.nan)
            y_interp = f(x_uniform)
            
            # 找到有效值（非NaN）的掩码
            valid_mask = ~np.isnan(y_interp)
            
            # 累积有效值
            y_accum[valid_mask] += y_interp[valid_mask]
            weights[valid_mask] += 1
            
        # 4. 计算最终结果
        # 有数据的地方取平均
        mask_data = weights > 0
        y_final = np.ones_like(x_uniform) * 1e-5  # 默认填充基线值
        
        y_final[mask_data] = y_accum[mask_data] / weights[mask_data]
        
        return x_uniform, y_final


class Preprocessor:
    """数据预处理类"""
    
    @staticmethod
    def apply_savgol_filter(y_data: np.ndarray, 
                           window_length: int = 11, 
                           polyorder: int = 3) -> np.ndarray:
        """Savitzky-Golay滤波"""
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(y_data, window_length, polyorder)
    
    @staticmethod
    def apply_gaussian_filter(y_data: np.ndarray, 
                             sigma: float = 1.0) -> np.ndarray:
        """高斯滤波"""
        return ndimage.gaussian_filter1d(y_data, sigma)
    
    @staticmethod
    def apply_fft_filter(y_data: np.ndarray, 
                        cutoff_freq: float = 0.1) -> np.ndarray:
        """FFT低通滤波"""
        fft_vals = fft.fft(y_data)
        freq = fft.fftfreq(len(y_data))
        fft_vals[np.abs(freq) > cutoff_freq] = 0
        return np.real(fft.ifft(fft_vals))
    
    @staticmethod
    def subtract_background_polynomial(x_data: np.ndarray, 
                                      y_data: np.ndarray,
                                      degree: int = 2,
                                      anchor_points: List[Tuple[float, float]] = None) -> np.ndarray:
        """多项式背景扣除"""
        if anchor_points:
            # 使用锚点进行多项式拟合
            x_anchors = np.array([p[0] for p in anchor_points])
            y_anchors = np.array([p[1] for p in anchor_points])
            coeffs = np.polyfit(x_anchors, y_anchors, degree)
        else:
            # 使用全部数据拟合
            coeffs = np.polyfit(x_data, y_data, degree)
        
        background = np.polyval(coeffs, x_data)
        return y_data - background, background
    
    @staticmethod
    def subtract_background_snip(y_data: np.ndarray, 
                                iterations: int = 40) -> Tuple[np.ndarray, np.ndarray]:
        """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) 算法"""
        data = y_data.copy()
        background = np.zeros_like(data)
        
        for i in range(iterations):
            window = 2**i
            for j in range(window, len(data) - window):
                data[j] = min(data[j], 
                             (data[j - window] + data[j + window]) / 2)
        
        background = data
        return y_data - background, background


class Fitter:
    """拟合引擎核心类"""
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = x_data
        self.y_data = y_data
        self.peaks: List[Peak] = []
        self.model = None
        self.params = None
        self.result = None
        self.y_fit = None
        self.background = None
        
    def add_peak(self, center_guess: float, 
                 bounds: Tuple[float, float] = None,
                 peak_type: str = 'film',
                 name: str = '') -> Peak:
        """添加峰"""
        peak_id = len(self.peaks)
        peak = Peak(peak_id, center_guess, bounds, peak_type, name)
        self.peaks.append(peak)
        return peak
    
    def remove_peak(self, peak_id: int):
        """删除峰"""
        self.peaks = [p for p in self.peaks if p.peak_id != peak_id]
        # 重新分配ID
        for i, peak in enumerate(self.peaks):
            peak.peak_id = i
    
    def auto_find_peaks(self, height_threshold: float = None,
                       distance: int = 10) -> List[float]:
        """自动寻峰"""
        if height_threshold is None:
            height_threshold = np.max(self.y_data) * 0.1
        
        peaks_idx, properties = find_peaks(self.y_data, 
                                          height=height_threshold,
                                          distance=distance)
        
        peak_positions = self.x_data[peaks_idx]
        return peak_positions.tolist()
    
    def build_model(self, constrain_fwhm: bool = False,
                   min_peak_separation: float = 0.2):
        """构建lmfit模型"""
        self.model = lmfit.models.ConstantModel()
        self.params = self.model.make_params()
        
        for peak in self.peaks:
            prefix = f'p{peak.peak_id}_'
            
            # 使用Pseudo-Voigt模型
            peak_model = lmfit.models.PseudoVoigtModel(prefix=prefix)
            self.model += peak_model
            
            # 初始参数
            pars = peak_model.make_params()
            pars[f'{prefix}center'].set(value=peak.center_guess, 
                                       min=peak.bounds[0], 
                                       max=peak.bounds[1])
            
            # 估算初始高度
            idx = np.argmin(np.abs(self.x_data - peak.center_guess))
            initial_height = self.y_data[idx]
            pars[f'{prefix}amplitude'].set(value=initial_height, min=0)
            
            # FWHM约束 - 关键修改：确保vary=True
            if peak.peak_type == 'substrate':
                # 基底峰通常很窄
                pars[f'{prefix}sigma'].set(value=0.05, min=0.01, max=1.0, vary=True)  # 添加vary=True
            else:
                # 薄膜峰较宽
                pars[f'{prefix}sigma'].set(value=0.15, min=0.01, max=1.5, vary=True)  # 添加vary=True
            
            # Pseudo-Voigt混合参数
            pars[f'{prefix}fraction'].set(value=0.5, min=0, max=1, vary=True)  # 添加vary=True
            
            self.params.update(pars)
        
        # 添加FWHM相等约束 - 只在勾选时生效
        if constrain_fwhm and len(self.peaks) >= 2:
            film_peaks = [p for p in self.peaks if p.peak_type == 'film']
            if len(film_peaks) >= 2:
                reference_peak = film_peaks[0]
                for peak in film_peaks[1:]:
                    self.params[f'p{peak.peak_id}_sigma'].set(
                        expr=f'p{reference_peak.peak_id}_sigma'
                    )
        
        # 添加峰间距约束
        if min_peak_separation > 0 and len(self.peaks) >= 2:
            sorted_peaks = sorted(self.peaks, key=lambda p: p.center_guess)
            for i in range(len(sorted_peaks) - 1):
                p1 = sorted_peaks[i]
                p2 = sorted_peaks[i + 1]
                # 确保p2的中心至少比p1大min_peak_separation
                current_min = p2.bounds[0]
                new_min = max(current_min, p1.center_guess + min_peak_separation)
                self.params[f'p{p2.peak_id}_center'].set(min=new_min)
        
    def execute_fitting(self, method: str = 'leastsq', use_log_scale: bool = True) -> lmfit.model.ModelResult:
        """
        执行拟合
        
        Args:
            method: 拟合方法
            use_log_scale: 是否使用对数尺度 (True: 混合对数/线性, False: 纯线性)
        """
        if self.model is None:
            self.build_model()

        # 调试：打印初始参数
        print("Initial parameters:")
        for peak in self.peaks:
            prefix = f'p{peak.peak_id}_'
            print(f"  Peak {peak.peak_id} sigma: {self.params[f'{prefix}sigma'].value}, vary={self.params[f'{prefix}sigma'].vary}")
        
        if use_log_scale:
            # 混合损失函数: 80% Log + 20% Linear
            def mixed_residual(params, model, x, data):
                model_vals = model.eval(params, x=x)
                eps = 1e-6
                
                # Log residual (Log10 difference)
                res_log = np.log10(np.maximum(data, eps)) - np.log10(np.maximum(model_vals, eps))
                res_log = np.nan_to_num(res_log)
                
                # Linear residual (Normalized by max intensity)
                max_val = np.max(data) if np.max(data) > 0 else 1.0
                res_lin = (data - model_vals) / max_val
                
                # Weights: 80% Log, 20% Linear (applied as sqrt since leastsq minimizes sum of squares)
                w_log = np.sqrt(0.8)
                w_lin = np.sqrt(0.2)
                
                # Concatenate weighted residuals
                return np.concatenate([w_log * res_log, w_lin * res_lin])

            print(f"Executing fitting with MIXED optimization (80% Log + 20% Linear, method={method})...")
            self.result = lmfit.minimize(mixed_residual, 
                                       self.params, 
                                       args=(self.model, self.x_data, self.y_data),
                                       method=method)
            
            # 手动计算拟合曲线
            self.y_fit = self.model.eval(self.result.params, x=self.x_data)
            
        else:
            print(f"Executing fitting with linear scale optimization (method={method})...")
            self.result = self.model.fit(self.y_data, 
                                         self.params, 
                                         x=self.x_data,
                                         method=method)
            self.y_fit = self.result.best_fit
        
        print("\nFitted parameters:")
        for peak in self.peaks:
            prefix = f'p{peak.peak_id}_'
            print(f"  Peak {peak.peak_id} sigma: {self.result.params[f'{prefix}sigma'].value}")
        
        # 确保 y_fit 是最新的
        if self.y_fit is None:
             self.y_fit = self.model.eval(self.result.params, x=self.x_data)

        # 更新各峰的参数
        for peak in self.peaks:
            peak.set_result(self.result.params)
            peak.calculate_area(self.x_data, self.y_fit)
        
        return self.result
    
    def get_individual_peaks(self) -> Dict[int, np.ndarray]:
        """获取各个峰的独立曲线"""
        if self.result is None:
            return {}
        
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
        return lmfit.fit_report(self.result)


class Reporter:
    """结果报告类"""
    def __init__(self, fitter: Fitter, x_original=None, y_original=None):
        self.fitter = fitter
        self.metrics = {}
        self.x_original = x_original  # 新增
        self.y_original = y_original  # 新增
        
    def calculate_metrics(self) -> Dict:
        """计算评估指标"""
        if self.fitter.result is None:
            return {}
        
        y_obs = self.fitter.y_data
        y_calc = self.fitter.y_fit
        residuals = y_obs - y_calc
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Reduced Chi-squared
        n_data = len(y_obs)
        n_params = len(self.fitter.result.params)
        chi_squared = self.fitter.result.chisqr
        reduced_chi_squared = chi_squared / (n_data - n_params)
        
        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        
        self.metrics = {
            'R_squared': r_squared,
            'Chi_squared': chi_squared,
            'Reduced_Chi_squared': reduced_chi_squared,
            'RMSE': rmse,
            'AIC': self.fitter.result.aic,
            'BIC': self.fitter.result.bic
        }
        
        return self.metrics
    
    def calculate_lattice_parameters(self, wavelength: float = 1.5406) -> Dict:
        """计算晶格常数 (λ=1.5406 Å for Cu Kα)"""
        lattice_params = {}
        
        for peak in self.fitter.peaks:
            if peak.center is None:
                continue
            
            # Bragg's Law: λ = 2d*sinθ
            theta_rad = np.radians(peak.center / 2)
            d_spacing = wavelength / (2 * np.sin(theta_rad))
            
            lattice_params[f'Peak_{peak.peak_id}'] = {
                '2theta': peak.center,
                'd_spacing': d_spacing,
                'FWHM': peak.fwhm,
                'Area': peak.area,
                'Height': peak.height
            }
        
        # 计算Tetragonality (假设002和200峰)
        film_peaks = [p for p in self.fitter.peaks if p.peak_type == 'film']
        if len(film_peaks) >= 2:
            # 按角度排序
            sorted_peaks = sorted(film_peaks, key=lambda p: p.center)
            
            theta1 = np.radians(sorted_peaks[0].center / 2)
            theta2 = np.radians(sorted_peaks[1].center / 2)
            
            d1 = wavelength / (2 * np.sin(theta1))
            d2 = wavelength / (2 * np.sin(theta2))
            
            # 假设较小角度是c轴，较大角度是a轴
            c_axis = d1
            a_axis = d2
            tetragonality = c_axis / a_axis
            
            lattice_params['Tetragonality'] = {
                'c_axis': c_axis,
                'a_axis': a_axis,
                'c/a_ratio': tetragonality
            }
        
        return lattice_params
    
    def export_results(self, output_path: str):
        """导出结果到Excel（包含原始数据和处理后数据）"""
        # 计算晶格参数以便获取d-spacing和四方度
        lattice_params = self.calculate_lattice_parameters()
        
        results_list = []
        
        for peak in self.fitter.peaks:
            # 从lattice_params中获取d_spacing
            d_spacing = None
            peak_key = f'Peak_{peak.peak_id}'
            if peak_key in lattice_params:
                d_spacing = lattice_params[peak_key].get('d_spacing')

            results_list.append({
                'Peak_ID': peak.peak_id,
                'Name': peak.name,          # 新增: 峰名称
                'Type': peak.peak_type,
                'Center_2theta': peak.center,
                'd_spacing_Å': d_spacing,   # 新增: d间距
                'FWHM': peak.fwhm,
                'Height': peak.height,
                'Area': peak.area,
                'Eta_PseudoVoigt': peak.eta
            })
        
        df_peaks = pd.DataFrame(results_list)
        
        # 评估指标
        metrics_df = pd.DataFrame([self.metrics])
        
        # 结构分析参数 (如 Tetragonality) - 单独放入一张表，避免压扁在一行
        structure_list = []
        if 'Tetragonality' in lattice_params:
            tet = lattice_params['Tetragonality']
            structure_list.append({
                'Parameter': 'Tetragonality',
                'c_axis': tet.get('c_axis'),
                'a_axis': tet.get('a_axis'),
                'c/a_ratio': tet.get('c/a_ratio')
            })
        
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
                               kind='linear', bounds_error=False, fill_value=0)
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
            
            # 结构分析参数
            if structure_list:
                df_structure = pd.DataFrame(structure_list)
                df_structure.to_excel(writer, sheet_name='Structure_Analysis', index=False)
            
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
        metrics_text = f"R² = {self.metrics.get('R_squared', 0):.6f}\n"
        metrics_text += f"χ² = {self.metrics.get('Reduced_Chi_squared', 0):.4f}"
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

