from plot_from_excel import XRDPlotterFromExcel

plotter = XRDPlotterFromExcel("results.xlsx")
plotter.plot_fitting("output.png", dpi=600)
plotter.plot_comparison("comparison.png")

