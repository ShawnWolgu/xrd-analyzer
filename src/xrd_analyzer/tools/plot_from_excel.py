# 从 Excel 项目重新绘制拟合图

"""
从XRD分析结果Excel文件重新生成高质量拟合图

使用方法:
    xrd-plot-results results.xlsx
# 基本用法
xrd-plot-results results.xlsx

# 指定输出文件
xrd-plot-results results.xlsx -o my_plot.png

# 生成所有图表
xrd-plot-results results.xlsx --all

# 不显示残差和原始数据
xrd-plot-results results.xlsx --no-residuals --no-original

# 只生成对比图
xrd-plot-results results.xlsx --comparison

# 打印摘要报告
xrd-plot-results results.xlsx --summary

或者在脚本中直接指定文件路径
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from ..plotting import apply_plot_style


apply_plot_style()
import argparse


class XRDPlotterFromExcel:
    """从Excel绘制XRD拟合图"""
    
    def __init__(self, excel_path: str):
        self.excel_path = Path(excel_path)
        self.data = None
        self.peaks = None
        self.metrics = None

    @staticmethod
    def _characteristic_length_column(columns) -> Optional[str]:
        """查找新旧工作簿中的反射峰特征长度列。"""
        candidates = (
            'Characteristic_Length_d_Angstrom',
            'd_spacing_Å',
            'd_spacing_Angstrom',
        )
        return next((column for column in candidates if column in columns), None)
        
    def load_data(self):
        """加载Excel数据"""
        print(f"加载文件: {self.excel_path}")
        
        # 读取各个sheet
        try:
            self.data = pd.read_excel(self.excel_path, sheet_name='Full_Data')
            self.peaks = pd.read_excel(self.excel_path, sheet_name='Peak_Parameters')
            self.metrics = pd.read_excel(self.excel_path, sheet_name='Fit_Metrics')

            if self._characteristic_length_column(self.peaks.columns) is None:
                try:
                    legacy_lengths = pd.read_excel(
                        self.excel_path,
                        sheet_name='Lattice_Parameters',
                    )
                except ValueError:
                    legacy_lengths = None

                if legacy_lengths is not None and not legacy_lengths.empty:
                    for row_index, peak_id in self.peaks['Peak_ID'].items():
                        legacy_column = f'Peak_{int(peak_id)}_d_spacing'
                        if legacy_column in legacy_lengths.columns:
                            self.peaks.loc[
                                row_index,
                                'Characteristic_Length_d_Angstrom',
                            ] = legacy_lengths.loc[0, legacy_column]

            print(f"  成功加载 {len(self.data)} 个数据点")
            print(f"  包含 {len(self.peaks)} 个峰")
            
        except Exception as e:
            print(f"加载失败: {e}")
            raise
    
    def plot_fitting(self, 
                    output_path: Optional[str] = None,
                    show_original: bool = True,
                    show_components: bool = True,
                    show_residuals: bool = True,
                    dpi: int = 300,
                    figsize: tuple = (12, 10)):
        """
        绘制拟合图
        
        Parameters:
        -----------
        output_path : str, optional
            输出文件路径，如果为None则显示而不保存
        show_original : bool
            是否显示原始数据（预处理前）
        show_components : bool
            是否显示各峰分量
        show_residuals : bool
            是否显示残差图
        dpi : int
            图片分辨率
        figsize : tuple
            图片尺寸 (width, height)
        """
        
        if self.data is None:
            self.load_data()
        
        # 创建子图
        if show_residuals:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                          gridspec_kw={'height_ratios': [3, 1],
                                                      'hspace': 0.05})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]*0.75))
        
        # 提取数据
        x_data = self.data['2theta'].values
        y_processed = self.data['Processed_Intensity'].values
        y_fit = self.data['Fitted_Intensity'].values
        
        # 主图 - 数据点
        if show_original and 'Original_Intensity' in self.data.columns:
            y_original = self.data['Original_Intensity'].values
            ax1.scatter(x_data, y_original, s=8, alpha=0.3, 
                       label='Original Data', color='lightgray', zorder=1)
        
        ax1.scatter(x_data, y_processed, s=15, alpha=0.6, 
                   label='Processed Data', color='gray', zorder=2)
        
        # 拟合曲线
        ax1.plot(x_data, y_fit, 'r-', linewidth=2.5, 
                label='Fit', zorder=4)
        
        # 各峰分量
        if show_components:
            # 找出所有峰分量列
            peak_columns = [col for col in self.data.columns 
                          if col.startswith('Peak_') and col.endswith('_Component')]
            
            # 使用不同颜色
            colors = plt.cm.tab10(np.linspace(0, 1, len(peak_columns)))
            
            for i, col in enumerate(sorted(peak_columns)):
                peak_id = int(col.split('_')[1])
                y_component = self.data[col].values
                
                # 从峰参数表获取信息
                peak_info = self.peaks[self.peaks['Peak_ID'] == peak_id].iloc[0]
                peak_type = peak_info['Type']
                center = peak_info['Center_2theta']
                
                label = f"Peak {peak_id} ({peak_type})\n2θ={center:.3f}°"
                ax1.plot(x_data, y_component, '--', 
                        color=colors[i], linewidth=2, 
                        alpha=0.7, label=label, zorder=3)
        
        # 背景线（如果有）
        if 'Background' in self.data.columns:
            bg = self.data['Background'].values
            ax1.plot(x_data, bg, 'k:', linewidth=1.5, 
                    label='Background', alpha=0.5, zorder=3)
        
        # 设置主图标签和样式
        ax1.set_ylabel('Intensity (a.u.)', fontsize=13, fontweight='bold')
        ax1.set_title('XRD Pattern Fitting', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 新项目只显示拟合掩码内的R²；旧工作簿保留原标签以免误称。
        if 'R_squared_fit' in self.metrics.columns:
            r2 = self.metrics['R_squared_fit'].values[0]
            text = f'R²_fit = {r2:.6f}'
        else:
            r2 = self.metrics['R_squared'].values[0]
            text = f'R² (legacy) = {r2:.6f}'
        
        ax1.text(0.02, 0.98, text, 
                transform=ax1.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=11, fontweight='bold')
        
        # 残差图
        if show_residuals:
            residuals = self.data['Residuals'].values
            
            ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=1)
            ax2.scatter(x_data, residuals, s=8, alpha=0.6, color='blue', zorder=2)
            
            ax2.set_xlabel('2θ (degree)', fontsize=13, fontweight='bold')
            ax2.set_ylabel('Residuals', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax2.set_xlim(ax1.get_xlim())  # 保持x轴一致
            
            # 移除上图的x轴标签
            ax1.set_xticklabels([])
        else:
            ax1.set_xlabel('2θ (degree)', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存或显示
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"\n图片已保存至: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_comparison(self, output_path: Optional[str] = None, dpi: int = 300):
        """
        绘制对比图：原始数据 vs 处理后数据 vs 拟合结果
        """
        if self.data is None:
            self.load_data()
        
        if 'Original_Intensity' not in self.data.columns:
            print("警告: 数据中没有原始信号，无法生成对比图")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        x_data = self.data['2theta'].values
        
        # 子图1: 原始数据
        axes[0].plot(x_data, self.data['Original_Intensity'].values, 
                    'b-', linewidth=1, label='Original Data')
        axes[0].set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
        axes[0].set_title('Original Data', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')
        
        # 子图2: 处理后数据
        axes[1].plot(x_data, self.data['Processed_Intensity'].values, 
                    'g-', linewidth=1, label='Processed Data')
        axes[1].set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
        axes[1].set_title('Processed Data (After Filtering & Background Subtraction)', 
                         fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')
        
        # 子图3: 拟合结果
        axes[2].scatter(x_data, self.data['Processed_Intensity'].values, 
                       s=10, alpha=0.5, label='Data', color='gray')
        axes[2].plot(x_data, self.data['Fitted_Intensity'].values, 
                    'r-', linewidth=2, label='Fit')
        axes[2].set_xlabel('2θ (degree)', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
        axes[2].set_title('Fitting Result', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='best')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"\n对比图已保存至: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_peak_components(self, output_path: Optional[str] = None, dpi: int = 300):
        """
        单独绘制各个峰的分量
        """
        if self.data is None:
            self.load_data()
        
        peak_columns = [col for col in self.data.columns 
                       if col.startswith('Peak_') and col.endswith('_Component')]
        
        n_peaks = len(peak_columns)
        
        if n_peaks == 0:
            print("没有找到峰分量数据")
            return
        
        # 计算子图布局
        ncols = min(2, n_peaks)
        nrows = (n_peaks + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
        if n_peaks == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        x_data = self.data['2theta'].values
        
        for i, col in enumerate(sorted(peak_columns)):
            peak_id = int(col.split('_')[1])
            y_component = self.data[col].values
            
            # 获取峰信息
            peak_info = self.peaks[self.peaks['Peak_ID'] == peak_id].iloc[0]
            
            ax = axes[i]
            ax.plot(x_data, y_component, 'b-', linewidth=2)
            ax.fill_between(x_data, 0, y_component, alpha=0.3)
            
            # 标题包含详细信息
            title = f"Peak {peak_id} ({peak_info['Type']})\n"
            title += f"Center: {peak_info['Center_2theta']:.4f}°, "
            title += f"FWHM: {peak_info['FWHM']:.4f}°\n"
            title += f"Area: {peak_info['Area']:.2f}"
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('2θ (degree)', fontsize=10)
            ax.set_ylabel('Intensity (a.u.)', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_peaks, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"\n峰分量图已保存至: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_summary_report(self):
        """
        打印摘要报告
        """
        if self.data is None:
            self.load_data()
        
        print("\n" + "="*60)
        print("XRD拟合结果摘要")
        print("="*60)
        
        # 拟合质量
        print("\n【拟合质量】")
        print("-"*40)
        for col in self.metrics.columns:
            value = self.metrics[col].values[0]
            print(f"  {col:25s}: {value:.6e}")
        
        # 峰参数
        print("\n【峰参数】")
        print("-"*40)
        characteristic_length_column = self._characteristic_length_column(self.peaks.columns)
        for _, peak in self.peaks.iterrows():
            print(f"\nPeak {int(peak['Peak_ID'])} ({peak['Type']}):")
            print(f"  中心位置: {peak['Center_2theta']:.4f}°")
            if characteristic_length_column is not None:
                characteristic_length = peak[characteristic_length_column]
                if pd.notna(characteristic_length):
                    print(f"  特征长度 d: {characteristic_length:.6f} Å")
            print(f"  FWHM:     {peak['FWHM']:.4f}°")
            print(f"  峰高:     {peak['Height']:.2f}")
            print(f"  峰面积:   {peak['Area']:.2f}")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='从Excel文件重新绘制XRD拟合图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  xrd-plot-results results.xlsx
  
  # 指定输出文件名
  xrd-plot-results results.xlsx -o output.png
  
  # 生成所有类型的图
  xrd-plot-results results.xlsx --all
  
  # 不显示原始数据和残差
  xrd-plot-results results.xlsx --no-original --no-residuals
        """
    )
    
    parser.add_argument('excel_file', help='输入的Excel文件路径')
    parser.add_argument('-o', '--output', help='输出图片路径（默认：自动生成）')
    parser.add_argument('--no-original', action='store_true', 
                       help='不显示原始数据')
    parser.add_argument('--no-components', action='store_true',
                       help='不显示峰分量')
    parser.add_argument('--no-residuals', action='store_true',
                       help='不显示残差图')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图片分辨率（默认：300）')
    parser.add_argument('--all', action='store_true',
                       help='生成所有类型的图（主图、对比图、峰分量图）')
    parser.add_argument('--comparison', action='store_true',
                       help='生成对比图')
    parser.add_argument('--peaks', action='store_true',
                       help='生成峰分量图')
    parser.add_argument('--summary', action='store_true',
                       help='打印摘要报告')
    
    args = parser.parse_args()
    
    # 创建绘图器
    plotter = XRDPlotterFromExcel(args.excel_file)
    plotter.load_data()
    
    # 生成输出文件名
    base_name = Path(args.excel_file).stem
    output_dir = Path(args.excel_file).parent
    
    # 主拟合图
    if args.output:
        output_path = args.output
    else:
        output_path = output_dir / f"{base_name}_fitting.png"
    
    print("\n生成主拟合图...")
    plotter.plot_fitting(
        output_path=str(output_path),
        show_original=not args.no_original,
        show_components=not args.no_components,
        show_residuals=not args.no_residuals,
        dpi=args.dpi
    )
    
    # 对比图
    if args.all or args.comparison:
        comparison_path = output_dir / f"{base_name}_comparison.png"
        print("\n生成对比图...")
        plotter.plot_comparison(str(comparison_path), dpi=args.dpi)
    
    # 峰分量图
    if args.all or args.peaks:
        peaks_path = output_dir / f"{base_name}_peak_components.png"
        print("\n生成峰分量图...")
        plotter.plot_peak_components(str(peaks_path), dpi=args.dpi)
    
    # 摘要报告
    if args.summary or args.all:
        plotter.export_summary_report()
    
    print("\n完成！")


if __name__ == '__main__':
    # 如果没有命令行参数，使用交互模式
    if len(sys.argv) == 1:
        print("="*60)
        print("XRD拟合结果可视化工具")
        print("="*60)
        
        excel_file = input("\n请输入Excel文件路径: ").strip().strip('"').strip("'")
        
        if not Path(excel_file).exists():
            print(f"错误：文件不存在 - {excel_file}")
            sys.exit(1)
        
        plotter = XRDPlotterFromExcel(excel_file)
        plotter.load_data()
        
        print("\n请选择要生成的图表:")
        print("1. 主拟合图")
        print("2. 数据对比图")
        print("3. 峰分量图")
        print("4. 全部")
        print("5. 仅打印摘要")
        
        choice = input("\n选择 (1-5): ").strip()
        
        base_name = Path(excel_file).stem
        output_dir = Path(excel_file).parent
        
        if choice == '1':
            output_path = output_dir / f"{base_name}_fitting.png"
            plotter.plot_fitting(str(output_path))
        elif choice == '2':
            output_path = output_dir / f"{base_name}_comparison.png"
            plotter.plot_comparison(str(output_path))
        elif choice == '3':
            output_path = output_dir / f"{base_name}_peak_components.png"
            plotter.plot_peak_components(str(output_path))
        elif choice == '4':
            plotter.plot_fitting(str(output_dir / f"{base_name}_fitting.png"))
            plotter.plot_comparison(str(output_dir / f"{base_name}_comparison.png"))
            plotter.plot_peak_components(str(output_dir / f"{base_name}_peak_components.png"))
            plotter.export_summary_report()
        elif choice == '5':
            plotter.export_summary_report()
        else:
            print("无效选择")
            sys.exit(1)
        
        print("\n完成！")
    else:
        main()
