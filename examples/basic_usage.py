# XRD Analyzer 使用示例

"""
XRD分析系统使用示例

这个脚本展示了如何在不使用GUI的情况下，
通过编程方式使用核心分析引擎。
"""

from pathlib import Path

from xrd_analyzer import (
    DataLoader, Preprocessor, Fitter, Reporter
)


def example_full_analysis(file_path):
    """完整分析示例"""
    
    print("=" * 60)
    print("XRD数据分析示例")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    x_data, y_data = DataLoader.load_txt(file_path)
    print(f"  加载了 {len(x_data)} 个数据点")
    
    # 2. 数据裁剪（可选）
    print("\n[2/5] 裁剪数据范围...")
    x_data, y_data = DataLoader.trim_range(x_data, y_data, 43, 47)
    print(f"  裁剪后: {len(x_data)} 个数据点")
    
    # 3. 预处理
    print("\n[3/5] 数据预处理...")
    # Savitzky-Golay滤波
    y_data = Preprocessor.apply_savgol_filter(y_data, window_length=11, polyorder=3)
    # 背景扣除
    y_data, background = Preprocessor.subtract_background_polynomial(
        x_data, y_data, degree=2
    )
    print("  完成滤波和背景扣除")
    
    # 4. 峰拟合
    print("\n[4/5] 执行峰拟合...")
    fitter = Fitter(x_data, y_data)
    
    # 手动添加两个示例反射峰；实际名称和位置应由使用者给出
    fitter.add_peak(
        center_guess=44.3,
        bounds=(44.0, 44.6),
        peak_type='film'
    )
    
    fitter.add_peak(
        center_guess=44.8,
        bounds=(44.6, 45.1),
        peak_type='film'
    )
    
    # 可选：添加基底峰
    # fitter.add_peak(
    #     center_guess=46.5,
    #     bounds=(46.0, 47.0),
    #     peak_type='substrate'
    # )
    
    # 构建模型并拟合
    fitter.build_model(
        constrain_fwhm=True,  # 强制FWHM相等
        min_peak_separation=0.3  # 最小峰间距
    )
    
    result = fitter.execute_fitting(method='leastsq')
    print(f"  拟合完成！")
    
    # 5. 结果分析
    print("\n[5/5] 生成报告...")
    reporter = Reporter(fitter)
    metrics = reporter.calculate_metrics()
    
    print("\n拟合质量:")
    print(f"  R² = {metrics['R_squared']:.6f}")
    print(f"  χ²ᵣ = {metrics['Reduced_Chi_squared']:.4f}")
    
    print("\n峰参数:")
    for peak in fitter.peaks:
        print(f"\n  Peak {peak.peak_id} ({peak.peak_type}):")
        print(f"    中心: {peak.center:.4f}°")
        print(f"    FWHM: {peak.fwhm:.4f}°")
        print(f"    面积: {peak.area:.2f}")
    
    # 计算各反射峰的特征长度（Bragg d间距）
    characteristic_lengths = reporter.calculate_characteristic_lengths()
    print("\n反射峰特征长度:")
    for value in characteristic_lengths.values():
        label = value['reflection_label'] or '未指定'
        print(
            f"  Peak {value['peak_id']} ({label}): "
            f"d = {value['characteristic_length_angstrom']:.6f} Å"
        )
    
    # 6. 保存结果
    output_dir = Path(file_path).parent
    
    # 保存Excel
    excel_path = output_dir / "analysis_results.xlsx"
    reporter.export_results(str(excel_path))
    print(f"\n结果已保存至: {excel_path}")
    
    # 保存图片
    fig_path = output_dir / "fitting_result.png"
    reporter.plot_results(save_path=str(fig_path), show_components=True)
    print(f"图片已保存至: {fig_path}")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == '__main__':
    # 使用示例
    # 请替换为您的实际文件路径
    
    # 示例1: 完整分析
    # file_path = "path/to/your/xrd_data.txt"
    # example_full_analysis(file_path)
    
    print("请取消注释上面的代码行并指定您的数据文件路径")
