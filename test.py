from xrd_analyzer import DataLoader, Fitter, Reporter

# 加载数据
x, y = DataLoader.load_txt("your_data.txt")

# 创建拟合器并添加峰
fitter = Fitter(x, y)
fitter.add_peak(44.3, (44.0, 44.6), 'film')
fitter.add_peak(44.8, (44.6, 45.1), 'film')

# 执行拟合
fitter.build_model(constrain_fwhm=True, min_peak_separation=0.3)
result = fitter.execute_fitting()

# 生成报告
reporter = Reporter(fitter)
reporter.export_results("results.xlsx")
reporter.plot_results("fitting.png")

