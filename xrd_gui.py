# xrd_gui.py - PyQt5 GUI界面

import sys
import os
from pathlib import Path
from typing import List, Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableWidget,
    QTableWidgetItem, QGroupBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QTextEdit, QSplitter, QTabWidget, QMessageBox,
    QProgressBar, QListWidget, QListWidgetItem, QRadioButton,
    QButtonGroup, QDialog, QDialogButtonBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QThread
from PyQt5.QtGui import QFont, QColor

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from xrd_analyzer import (
    DataLoader, Preprocessor, Fitter, Reporter, Peak
)


class MplCanvas(FigureCanvas):
    """Matplotlib画布"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


class InteractivePlotCanvas(MplCanvas):
    """交互式绘图画布"""
    
    peak_added = pyqtSignal(float, float)  # center, intensity
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_data = None
        self.y_data = None
        self.peak_markers = []
        self.mode = 'view'  # 'view' or 'add_peak'
        self.yscale = 'linear'  # 'linear' or 'log'
        
        # 连接鼠标事件
        self.mpl_connect('button_press_event', self.on_click)
        
    def set_data(self, x_data, y_data):
        """设置数据"""
        self.x_data = x_data
        self.y_data = y_data
        self.plot_data()
        
    def set_yscale(self, scale):
        """设置Y轴刻度"""
        self.yscale = scale
        if self.axes:
            self.axes.set_yscale(scale)
            self.draw()
        
    def plot_data(self):
        """绘制数据"""
        self.axes.clear()
        if self.x_data is not None and self.y_data is not None:
            self.axes.plot(self.x_data, self.y_data, 'b-', linewidth=1.5, label='Data')
            self.axes.set_xlabel('2θ (degree)', fontsize=11, fontweight='bold')
            self.axes.set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
            self.axes.set_yscale(self.yscale)  # Apply current scale
            self.axes.grid(True, alpha=0.3)
            self.axes.legend()
        self.draw()
    
    def add_peak_marker(self, x_pos, y_pos, peak_id):
        """添加峰标记"""
        marker = self.axes.plot(x_pos, y_pos, 'ro', markersize=8, 
                               label=f'Peak {peak_id}')[0]
        self.peak_markers.append(marker)
        self.axes.legend()
        self.draw()
    
    def clear_peak_markers(self):
        """清除所有峰标记"""
        for marker in self.peak_markers:
            marker.remove()
        self.peak_markers.clear()
        self.draw()
    
    def on_click(self, event):
        """鼠标点击事件"""
        if self.mode == 'add_peak' and event.inaxes == self.axes:
            if event.button == 1:  # 左键
                x_pos = event.xdata
                y_pos = event.ydata
                self.peak_added.emit(x_pos, y_pos)


class PeakConfigDialog(QDialog):
    """峰配置对话框"""
    
    def __init__(self, parent=None, peak_center=None):
        super().__init__(parent)
        self.peak_center = peak_center
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("峰参数配置")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        layout = QFormLayout()
        
        # 峰中心
        self.center_spin = QDoubleSpinBox()
        self.center_spin.setRange(0, 180)
        self.center_spin.setDecimals(3)
        self.center_spin.setSingleStep(0.01)
        if self.peak_center:
            self.center_spin.setValue(self.peak_center)
        layout.addRow("峰中心 (2θ):", self.center_spin)
        
        # 范围下限
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(0, 180)
        self.min_spin.setDecimals(3)
        self.min_spin.setSingleStep(0.01)
        if self.peak_center:
            self.min_spin.setValue(self.peak_center - 0.5)
        layout.addRow("范围下限:", self.min_spin)
        
        # 范围上限
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0, 180)
        self.max_spin.setDecimals(3)
        self.max_spin.setSingleStep(0.01)
        if self.peak_center:
            self.max_spin.setValue(self.peak_center + 0.5)
        layout.addRow("范围上限:", self.max_spin)
        
        # 峰类型
        self.type_combo = QComboBox()
        self.type_combo.addItems(['film', 'substrate'])
        layout.addRow("峰类型:", self.type_combo)
        
        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        
        self.setLayout(layout)
    
    def get_values(self):
        """获取配置值"""
        return {
            'center': self.center_spin.value(),
            'min': self.min_spin.value(),
            'max': self.max_spin.value(),
            'type': self.type_combo.currentText()
        }


class FittingThread(QThread):
    """拟合线程"""
    
    finished = pyqtSignal(object)  # 发送result
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, fitter, constrain_fwhm, min_separation):
        super().__init__()
        self.fitter = fitter
        self.constrain_fwhm = constrain_fwhm
        self.min_separation = min_separation
        
    def run(self):
        try:
            self.progress.emit(20)
            
            # 构建模型
            self.fitter.build_model(
                constrain_fwhm=self.constrain_fwhm,
                min_peak_separation=self.min_separation
            )
            
            self.progress.emit(40)
            
            # 执行拟合
            result = self.fitter.execute_fitting()
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class XRDAnalyzerGUI(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        
        # 数据
        self.x_data = None
        self.y_data = None
        self.x_data_raw = None
        self.y_data_raw = None
        self.x_data_original = None  # 新增：最原始数据（加载时的）
        self.y_data_original = None  # 新增
        self.current_file = None
        
        # 多文件管理
        self.loaded_files_data = []  # List[Tuple[filepath, x, y]]
        
        # 处理对象
        self.fitter = None
        self.reporter = None
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("XRD数据分析系统 - PZT薄膜专用")
        self.setGeometry(100, 100, 1400, 900)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 左侧控制面板
        left_panel = self.create_left_panel()
        
        # 右侧绘图区域
        right_panel = self.create_right_panel()
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        
        main_layout.addWidget(splitter)
        
        # 状态栏
        self.statusBar().showMessage('就绪')
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.setVisible(False)
        
    def create_left_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        # 改为水平布局以容纳两列
        main_layout = QHBoxLayout()
        panel.setLayout(main_layout)
        
        # === 左列：数据加载与预处理 ===
        col1_layout = QVBoxLayout()
        
        # 1. 文件加载组
        file_group = QGroupBox("1. 数据加载与合并")
        file_layout = QVBoxLayout()
        
        # 按钮区
        btn_layout = QHBoxLayout()
        load_btn = QPushButton("添加数据文件")
        load_btn.clicked.connect(self.add_files)
        btn_layout.addWidget(load_btn)
        
        remove_btn = QPushButton("移除选中")
        remove_btn.clicked.connect(self.remove_file)
        btn_layout.addWidget(remove_btn)
        
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_files)
        btn_layout.addWidget(clear_btn)
        
        file_layout.addLayout(btn_layout)
        
        # 文件列表
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMaximumHeight(150)
        file_layout.addWidget(self.file_list_widget)
        
        # 数据范围
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("2θ范围:"))
        self.range_min = QDoubleSpinBox()
        self.range_min.setRange(0, 180)
        self.range_min.setValue(20)
        self.range_max = QDoubleSpinBox()
        self.range_max.setRange(0, 180)
        self.range_max.setValue(120)
        range_layout.addWidget(self.range_min)
        range_layout.addWidget(QLabel("-"))
        range_layout.addWidget(self.range_max)
        file_layout.addLayout(range_layout)
        
        apply_range_btn = QPushButton("应用范围")
        apply_range_btn.clicked.connect(self.apply_range)
        file_layout.addWidget(apply_range_btn)
        
        # Y轴显示模式
        yscale_layout = QHBoxLayout()
        yscale_layout.addWidget(QLabel("Y轴显示:"))
        self.linear_radio = QRadioButton("Linear")
        self.linear_radio.setChecked(True)
        self.log_radio = QRadioButton("Log")
        self.linear_radio.toggled.connect(self.toggle_yscale)
        yscale_layout.addWidget(self.linear_radio)
        yscale_layout.addWidget(self.log_radio)
        file_layout.addLayout(yscale_layout)
        
        file_group.setLayout(file_layout)
        col1_layout.addWidget(file_group)
        
        # 2. 预处理组
        preprocess_group = QGroupBox("2. 数据预处理")
        preprocess_layout = QVBoxLayout()
        
        # 滤波器选择
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("滤波器:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(['无', 'Savitzky-Golay', '高斯滤波', 'FFT滤波'])
        filter_layout.addWidget(self.filter_combo)
        preprocess_layout.addLayout(filter_layout)
        
        # SG参数
        sg_layout = QHBoxLayout()
        sg_layout.addWidget(QLabel("窗口长度:"))
        self.sg_window = QSpinBox()
        self.sg_window.setRange(3, 51)
        self.sg_window.setValue(11)
        self.sg_window.setSingleStep(2)
        sg_layout.addWidget(self.sg_window)
        preprocess_layout.addLayout(sg_layout)
        
        # 背景扣除
        bg_layout = QHBoxLayout()
        bg_layout.addWidget(QLabel("背景扣除:"))
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(['无', '多项式', 'SNIP'])
        bg_layout.addWidget(self.bg_combo)
        preprocess_layout.addLayout(bg_layout)
        
        # 多项式阶数
        poly_layout = QHBoxLayout()
        poly_layout.addWidget(QLabel("多项式阶数:"))
        self.poly_degree = QSpinBox()
        self.poly_degree.setRange(1, 5)
        self.poly_degree.setValue(2)
        poly_layout.addWidget(self.poly_degree)
        preprocess_layout.addLayout(poly_layout)
        
        apply_preprocess_btn = QPushButton("应用预处理")
        apply_preprocess_btn.clicked.connect(self.apply_preprocessing)
        preprocess_layout.addWidget(apply_preprocess_btn)
        
        reset_btn = QPushButton("重置为原始数据")
        reset_btn.clicked.connect(self.reset_to_raw)
        preprocess_layout.addWidget(reset_btn)
        
        preprocess_group.setLayout(preprocess_layout)
        col1_layout.addWidget(preprocess_group)
        
        col1_layout.addStretch()
        main_layout.addLayout(col1_layout)
        
        # === 右列：峰管理与拟合 ===
        col2_layout = QVBoxLayout()
        
        # 3. 峰管理组
        peak_group = QGroupBox("3. 峰识别与管理")
        peak_layout = QVBoxLayout()
        
        # 自动寻峰
        auto_peak_layout = QHBoxLayout()
        auto_peak_btn = QPushButton("自动寻峰")
        auto_peak_btn.clicked.connect(self.auto_find_peaks)
        auto_peak_layout.addWidget(auto_peak_btn)
        
        self.peak_threshold = QDoubleSpinBox()
        self.peak_threshold.setRange(0, 100000)
        self.peak_threshold.setValue(100)
        self.peak_threshold.setPrefix("阈值: ")
        auto_peak_layout.addWidget(self.peak_threshold)
        peak_layout.addLayout(auto_peak_layout)
        
        # 手动添加峰
        manual_peak_btn = QPushButton("手动添加峰 (点击图上)")
        manual_peak_btn.setCheckable(True)
        manual_peak_btn.clicked.connect(self.toggle_add_peak_mode)
        peak_layout.addWidget(manual_peak_btn)
        self.manual_peak_btn = manual_peak_btn
        
        # 峰列表 (移除最大高度限制)
        self.peak_table = QTableWidget()
        self.peak_table.setColumnCount(5)
        self.peak_table.setHorizontalHeaderLabels(
            ['ID', '中心', '最小', '最大', '类型']
        )
        self.peak_table.setMinimumHeight(200)
        peak_layout.addWidget(self.peak_table)
        
        # 删除峰按钮
        delete_peak_btn = QPushButton("删除选中峰")
        delete_peak_btn.clicked.connect(self.delete_selected_peak)
        peak_layout.addWidget(delete_peak_btn)
        
        clear_peaks_btn = QPushButton("清除所有峰")
        clear_peaks_btn.clicked.connect(self.clear_all_peaks)
        peak_layout.addWidget(clear_peaks_btn)
        
        peak_group.setLayout(peak_layout)
        col2_layout.addWidget(peak_group, 1) # 让其占据多余空间
        
        # 4. 拟合配置组
        fit_group = QGroupBox("4. 拟合配置")
        fit_layout = QVBoxLayout()
        
        # 约束选项
        self.constrain_fwhm_cb = QCheckBox("强制薄膜峰FWHM相等")
        fit_layout.addWidget(self.constrain_fwhm_cb)
        
        # 最小峰间距
        separation_layout = QHBoxLayout()
        separation_layout.addWidget(QLabel("最小峰间距:"))
        self.min_separation = QDoubleSpinBox()
        self.min_separation.setRange(0, 5)
        self.min_separation.setValue(0.2)
        self.min_separation.setSingleStep(0.05)
        self.min_separation.setDecimals(3)
        separation_layout.addWidget(self.min_separation)
        fit_layout.addLayout(separation_layout)
        
        # 拟合方法
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("拟合方法:"))
        self.fit_method_combo = QComboBox()
        self.fit_method_combo.addItems(['leastsq', 'least_squares', 'nelder'])
        method_layout.addWidget(self.fit_method_combo)
        fit_layout.addLayout(method_layout)
        
        # 执行拟合按钮
        execute_fit_btn = QPushButton("执行拟合")
        execute_fit_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        execute_fit_btn.clicked.connect(self.execute_fitting)
        fit_layout.addWidget(execute_fit_btn)
        
        fit_group.setLayout(fit_layout)
        col2_layout.addWidget(fit_group)
        
        # 5. 结果导出组
        export_group = QGroupBox("5. 结果导出")
        export_layout = QVBoxLayout()
        
        export_excel_btn = QPushButton("导出Excel报告")
        export_excel_btn.clicked.connect(self.export_excel)
        export_layout.addWidget(export_excel_btn)
        
        export_figure_btn = QPushButton("导出高清图片")
        export_figure_btn.clicked.connect(self.export_figure)
        export_layout.addWidget(export_figure_btn)
        
        export_group.setLayout(export_layout)
        col2_layout.addWidget(export_group)
        
        main_layout.addLayout(col2_layout)
        
        return panel
    
    def create_right_panel(self):
        """创建右侧绘图区域"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # 标签页
        tabs = QTabWidget()
        
        # Tab 1: 数据与拟合
        data_tab = QWidget()
        data_layout = QVBoxLayout()
        
        self.plot_canvas = InteractivePlotCanvas(width=8, height=6)
        self.plot_canvas.peak_added.connect(self.on_peak_added_from_plot)
        toolbar = NavigationToolbar(self.plot_canvas, self)
        
        data_layout.addWidget(toolbar)
        data_layout.addWidget(self.plot_canvas)
        data_tab.setLayout(data_layout)
        
        tabs.addTab(data_tab, "数据与拟合")
        
        # Tab 2: 结果详情
        results_tab = QWidget()
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Courier", 10))
        results_layout.addWidget(self.results_text)
        
        results_tab.setLayout(results_layout)
        tabs.addTab(results_tab, "拟合结果")
        
        # Tab 3: 物理参数
        physics_tab = QWidget()
        physics_layout = QVBoxLayout()
        
        self.physics_text = QTextEdit()
        self.physics_text.setReadOnly(True)
        self.physics_text.setFont(QFont("Courier", 10))
        physics_layout.addWidget(self.physics_text)
        
        physics_tab.setLayout(physics_layout)
        tabs.addTab(physics_tab, "物理参数")
        
        layout.addWidget(tabs)
        
        return panel
    
    # === 事件处理函数 ===
    
    def add_files(self):
        """添加文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择XRD数据文件", "", "Text Files (*.txt *.TXT)"
        )
        
        if not file_paths:
            return
            
        success_count = 0
        for file_path in file_paths:
            try:
                # 检查是否已加载
                if any(f[0] == file_path for f in self.loaded_files_data):
                    continue
                    
                x, y = DataLoader.load_txt(file_path)
                self.loaded_files_data.append((file_path, x, y))
                
                # 添加到列表UI
                item = QListWidgetItem(Path(file_path).name)
                item.setToolTip(file_path)
                self.file_list_widget.addItem(item)
                
                success_count += 1
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        if success_count > 0:
            self.update_merged_data()
            self.statusBar().showMessage(f'成功添加 {success_count} 个文件')
        else:
            QMessageBox.warning(self, "警告", "未添加任何新文件（可能格式错误或已存在）")

    def remove_file(self):
        """移除选中的文件"""
        current_row = self.file_list_widget.currentRow()
        if current_row < 0:
            return
            
        # 从数据中移除
        del self.loaded_files_data[current_row]
        
        # 从UI中移除
        self.file_list_widget.takeItem(current_row)
        
        self.update_merged_data()
        self.statusBar().showMessage('文件已移除')
        
    def clear_files(self):
        """清空文件"""
        self.loaded_files_data.clear()
        self.file_list_widget.clear()
        self.x_data = None
        self.y_data = None
        self.x_data_raw = None
        self.y_data_raw = None
        self.plot_canvas.axes.clear()
        self.plot_canvas.draw()
        self.current_file = None
        self.statusBar().showMessage('列表已清空')

    def update_merged_data(self):
        """更新合并后的数据"""
        if not self.loaded_files_data:
            self.clear_files()
            return
            
        # 提取所有数据进行拼接
        datasets = [(d[1], d[2]) for d in self.loaded_files_data]
        
        try:
            x_merged, y_merged = DataLoader.stitch_datasets(datasets)
            
            self.x_data_raw = x_merged
            self.y_data_raw = y_merged
            self.x_data_original = x_merged.copy()
            self.y_data_original = y_merged.copy()
            
            # 更新当前文件名为第一个文件 + 标识
            first_file = Path(self.loaded_files_data[0][0]).stem
            if len(self.loaded_files_data) > 1:
                self.current_file = f"{first_file}_merged_{len(self.loaded_files_data)}files"
            else:
                self.current_file = self.loaded_files_data[0][0]
                
            # 更新范围显示
            self.range_min.setValue(x_merged.min())
            self.range_max.setValue(x_merged.max())
            
            # 应用当前范围设置（这也将触发绘图更新）
            self.apply_range()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据合并失败:\n{str(e)}")

    def toggle_yscale(self):
        """切换Y轴显示模式"""
        if self.linear_radio.isChecked():
            self.plot_canvas.set_yscale('linear')
        else:
            self.plot_canvas.set_yscale('log')


    
    def apply_range(self):
        """应用数据范围"""
        if self.x_data_raw is None:
            return
        
        x_min = self.range_min.value()
        x_max = self.range_max.value()
        
        self.x_data, self.y_data = DataLoader.trim_range(
            self.x_data_raw, self.y_data_raw, x_min, x_max
        )
        
        self.plot_canvas.set_data(self.x_data, self.y_data)
        self.statusBar().showMessage(f'数据范围: {x_min:.2f} - {x_max:.2f}°')
    
    def apply_preprocessing(self):
        """应用预处理"""
        if self.y_data is None:
            return
        
        y_processed = self.y_data.copy()
        
        # 滤波
        filter_type = self.filter_combo.currentText()
        if filter_type == 'Savitzky-Golay':
            window = self.sg_window.value()
            if window % 2 == 0:
                window += 1
            y_processed = Preprocessor.apply_savgol_filter(y_processed, window, 3)
        elif filter_type == '高斯滤波':
            y_processed = Preprocessor.apply_gaussian_filter(y_processed, sigma=1.5)
        elif filter_type == 'FFT滤波':
            y_processed = Preprocessor.apply_fft_filter(y_processed, cutoff_freq=0.1)
        
        # 背景扣除
        bg_type = self.bg_combo.currentText()
        if bg_type == '多项式':
            degree = self.poly_degree.value()
            y_processed, bg = Preprocessor.subtract_background_polynomial(
                self.x_data, y_processed, degree
            )
        elif bg_type == 'SNIP':
            y_processed, bg = Preprocessor.subtract_background_snip(y_processed, iterations=40)
        
        self.y_data = y_processed
        self.plot_canvas.set_data(self.x_data, self.y_data)
        self.statusBar().showMessage('预处理完成')
    
    def reset_to_raw(self):
        """重置为原始数据"""
        if self.x_data_raw is not None:
            self.x_data = self.x_data_raw.copy()
            self.y_data = self.y_data_raw.copy()
            self.plot_canvas.set_data(self.x_data, self.y_data)
            self.statusBar().showMessage('已重置为原始数据')

    
    def toggle_add_peak_mode(self, checked):
        """切换添加峰模式"""
        if checked:
            self.plot_canvas.mode = 'add_peak'
            self.manual_peak_btn.setText("手动添加峰 (已激活)")
            self.statusBar().showMessage('点击图上添加峰...')
        else:
            self.plot_canvas.mode = 'view'
            self.manual_peak_btn.setText("手动添加峰 (点击图上)")
            self.statusBar().showMessage('就绪')


    def on_peak_added_from_plot(self, x_pos, y_pos):
        """从图上添加峰"""
        dialog = PeakConfigDialog(self, peak_center=x_pos)
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            self.add_peak_to_fitter(
                values['center'],
                (values['min'], values['max']),
                values['type']
            )
            
    def auto_find_peaks(self):
        """自动寻峰"""
        if self.x_data is None or self.y_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return
        
        # 创建临时fitter
        temp_fitter = Fitter(self.x_data, self.y_data)
        threshold = self.peak_threshold.value()
        
        peak_positions = temp_fitter.auto_find_peaks(height_threshold=threshold)
        
        if not peak_positions:
            QMessageBox.information(self, "信息", "未找到峰")
            return
        
        # 清除现有峰
        self.clear_all_peaks()
        
        # 添加找到的峰
        for pos in peak_positions:
            self.add_peak_to_fitter(pos, (pos - 0.5, pos + 0.5), 'film')
        
        self.statusBar().showMessage(f'自动找到 {len(peak_positions)} 个峰')
    
    def add_peak_to_fitter(self, center, bounds, peak_type):
        """添加峰到拟合器"""
        if self.fitter is None:
            self.fitter = Fitter(self.x_data, self.y_data)
        
        peak = self.fitter.add_peak(center, bounds, peak_type)
        
        # 更新表格
        self.update_peak_table()
        
        # 在图上标记
        idx = np.argmin(np.abs(self.x_data - center))
        self.plot_canvas.add_peak_marker(center, self.y_data[idx], peak.peak_id)
        
    def update_peak_table(self):
        """更新峰表格"""
        if self.fitter is None:
            return
        
        self.peak_table.setRowCount(len(self.fitter.peaks))
        
        for i, peak in enumerate(self.fitter.peaks):
            self.peak_table.setItem(i, 0, QTableWidgetItem(str(peak.peak_id)))
            self.peak_table.setItem(i, 1, QTableWidgetItem(f"{peak.center_guess:.3f}"))
            self.peak_table.setItem(i, 2, QTableWidgetItem(f"{peak.bounds[0]:.3f}"))
            self.peak_table.setItem(i, 3, QTableWidgetItem(f"{peak.bounds[1]:.3f}"))
            self.peak_table.setItem(i, 4, QTableWidgetItem(peak.peak_type))
    
    def delete_selected_peak(self):
        """删除选中的峰"""
        if self.fitter is None:
            return
        
        selected_rows = set(item.row() for item in self.peak_table.selectedItems())
        
        if not selected_rows:
            QMessageBox.warning(self, "警告", "请选择要删除的峰")
            return
        
        # 获取要删除的峰ID
        peak_ids_to_delete = []
        for row in selected_rows:
            peak_id = int(self.peak_table.item(row, 0).text())
            peak_ids_to_delete.append(peak_id)
        
        # 删除峰
        for peak_id in sorted(peak_ids_to_delete, reverse=True):
            self.fitter.remove_peak(peak_id)
        
        # 更新显示
        self.update_peak_table()
        self.plot_canvas.clear_peak_markers()
        
        # 重新添加标记
        for peak in self.fitter.peaks:
            idx = np.argmin(np.abs(self.x_data - peak.center_guess))
            self.plot_canvas.add_peak_marker(
                peak.center_guess, self.y_data[idx], peak.peak_id
            )
    
    def clear_all_peaks(self):
        """清除所有峰"""
        if self.fitter is not None:
            self.fitter.peaks.clear()
            self.update_peak_table()
            self.plot_canvas.clear_peak_markers()
    
    def execute_fitting(self):
        """执行拟合"""
        if self.fitter is None or len(self.fitter.peaks) == 0:
            QMessageBox.warning(self, "警告", "请先添加峰")
            return
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建拟合线程
        self.fit_thread = FittingThread(
            self.fitter,
            self.constrain_fwhm_cb.isChecked(),
            self.min_separation.value()
        )
        
        self.fit_thread.progress.connect(self.progress_bar.setValue)
        self.fit_thread.finished.connect(self.on_fitting_finished)
        self.fit_thread.error.connect(self.on_fitting_error)
        
        self.fit_thread.start()
        self.statusBar().showMessage('正在拟合...')
    
    def on_fitting_finished(self, result):
        """拟合完成处理"""
        self.progress_bar.setVisible(False)
        
        # 创建Reporter
        self.reporter = Reporter(
            self.fitter, 
            x_original=self.x_data_original,  # 新增
            y_original=self.y_data_original   # 新增
        )
        self.reporter.calculate_metrics()
        
        # 更新绘图
        self.plot_fitted_results()
        
        # 显示结果
        self.display_results()
        
        # 计算物理参数
        self.display_physics_parameters()
        
        self.statusBar().showMessage('拟合完成！')
        
        QMessageBox.information(
            self, "成功", 
            f"拟合完成！\nR² = {self.reporter.metrics['R_squared']:.6f}"
        )
    
    def on_fitting_error(self, error_msg):
        """拟合错误处理"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "拟合错误", f"拟合过程中出错:\n{error_msg}")
        self.statusBar().showMessage('拟合失败')
    
    def plot_fitted_results(self):
        """绘制拟合结果"""
        if self.fitter.result is None:
            return
        
        # 清除并重绘
        ax = self.plot_canvas.axes
        ax.clear()
        
        # 原始数据
        ax.scatter(self.fitter.x_data, self.fitter.y_data, 
                  s=15, alpha=0.5, label='Data', color='gray', zorder=1)
        
        # 拟合曲线
        ax.plot(self.fitter.x_data, self.fitter.y_fit, 
               'r-', linewidth=2.5, label='Fit', zorder=3)
        
        # 各个峰的分量
        peak_curves = self.fitter.get_individual_peaks()
        colors = plt.cm.tab10(np.linspace(0, 1, len(peak_curves)))
        
        for i, (peak_id, curve) in enumerate(peak_curves.items()):
            peak = self.fitter.peaks[peak_id]
            label = f'Peak {peak_id} ({peak.peak_type})\n2θ={peak.center:.3f}°'
            ax.plot(self.fitter.x_data, curve, 
                   '--', color=colors[i], linewidth=2, 
                   alpha=0.7, label=label, zorder=2)
        
        # 设置标签
        ax.set_xlabel('2θ (degree)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
        ax.set_title('XRD Pattern Fitting', fontsize=14, fontweight='bold')
        ax.set_yscale(self.plot_canvas.yscale)  # Keep consistent scale
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加R²文本
        r2 = self.reporter.metrics['R_squared']
        chi2 = self.reporter.metrics['Reduced_Chi_squared']
        text = f'R² = {r2:.6f}\nχ²ᵣ = {chi2:.4f}'
        ax.text(0.02, 0.98, text, 
               transform=ax.transAxes, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=11, fontweight='bold')
        
        self.plot_canvas.draw()
    
    def display_results(self):
        """显示拟合结果"""
        if self.reporter is None:
            return
        
        text = "=" * 60 + "\n"
        text += "拟合结果详情\n"
        text += "=" * 60 + "\n\n"
        
        # 评估指标
        text += "【拟合质量评估】\n"
        text += "-" * 40 + "\n"
        for key, value in self.reporter.metrics.items():
            text += f"{key:25s}: {value:.6e}\n"
        text += "\n"
        
        # 各峰参数
        text += "【峰参数】\n"
        text += "-" * 40 + "\n"
        for peak in self.fitter.peaks:
            text += f"\nPeak {peak.peak_id} ({peak.peak_type}):\n"
            text += f"  中心位置 (2θ)    : {peak.center:.4f}°\n"
            text += f"  峰高 (Intensity) : {peak.height:.2f}\n"
            text += f"  FWHM             : {peak.fwhm:.4f}°\n"
            text += f"  峰面积           : {peak.area:.2f}\n"
            text += f"  η (P-V混合比)    : {peak.eta:.4f}\n"
        
        # lmfit报告
        text += "\n" + "=" * 60 + "\n"
        text += "【Lmfit拟合报告】\n"
        text += "=" * 60 + "\n"
        text += self.fitter.result.fit_report()
        
        self.results_text.setPlainText(text)
    
    def display_physics_parameters(self):
        """显示物理参数"""
        if self.reporter is None:
            return
        
        text = "=" * 60 + "\n"
        text += "物理参数计算\n"
        text += "=" * 60 + "\n\n"
        
        # 计算晶格参数
        lattice_params = self.reporter.calculate_lattice_parameters()
        
        text += "【晶格参数】(λ = 1.5406 Å, Cu Kα)\n"
        text += "-" * 40 + "\n"
        
        for key, value in lattice_params.items():
            if key == 'Tetragonality':
                text += f"\n{key}:\n"
                text += f"  c轴晶格常数: {value['c_axis']:.6f} Å\n"
                text += f"  a轴晶格常数: {value['a_axis']:.6f} Å\n"
                text += f"  四方度 (c/a): {value['c/a_ratio']:.6f}\n"
                
                # 判断相结构
                if value['c/a_ratio'] > 1.01:
                    text += f"  → 四方相占主导 (Tetragonal)\n"
                elif value['c/a_ratio'] < 0.99:
                    text += f"  → 可能存在压应变\n"
                else:
                    text += f"  → 接近立方相或应变释放\n"
            else:
                text += f"\n{key}:\n"
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            text += f"  {k:20s}: {v:.6f}\n"
        
        # 晶粒尺寸估算 (Scherrer公式)
        text += "\n【晶粒尺寸估算】(Scherrer公式)\n"
        text += "-" * 40 + "\n"
        text += "D = Kλ / (β·cosθ)\n"
        text += "K = 0.9 (形状因子)\n\n"
        
        for peak in self.fitter.peaks:
            if peak.fwhm is not None and peak.center is not None:
                # FWHM转换为弧度
                beta_rad = np.radians(peak.fwhm)
                theta_rad = np.radians(peak.center / 2)
                
                # Scherrer公式
                K = 0.9
                wavelength = 1.5406  # Å
                D_nm = (K * wavelength) / (beta_rad * np.cos(theta_rad)) / 10  # 转换为nm
                
                text += f"Peak {peak.peak_id} ({peak.peak_type}):\n"
                text += f"  晶粒尺寸: {D_nm:.2f} nm\n"
        
        # 应力/应变分析
        text += "\n【应力分析提示】\n"
        text += "-" * 40 + "\n"
        text += "• 如果002和200峰分裂明显 → 存在四方相畸变\n"
        text += "• 峰位偏移 → 存在晶格应变\n"
        text += "• 峰展宽 → 晶粒尺寸减小或微观应变增大\n"
        text += "• 建议结合sin²ψ方法进行残余应力定量分析\n"
        
        self.physics_text.setPlainText(text)
    
    def export_excel(self):
        """导出Excel报告"""
        if self.reporter is None:
            QMessageBox.warning(self, "警告", "请先完成拟合")
            return
        
        if self.current_file is None:
            default_name = "xrd_analysis_results.xlsx"
        else:
            default_name = Path(self.current_file).stem + "_results.xlsx"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存Excel报告", default_name, "Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                self.reporter.export_results(file_path)
                QMessageBox.information(self, "成功", f"结果已导出至:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败:\n{str(e)}")
    
    def export_figure(self):
        """导出高清图片"""
        if self.fitter is None or self.fitter.result is None:
            QMessageBox.warning(self, "警告", "请先完成拟合")
            return
        
        if self.current_file is None:
            default_name = "xrd_fitting.png"
        else:
            default_name = Path(self.current_file).stem + "_fitting.png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default_name, 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if file_path:
            try:
                # 创建完整的报告图
                fig = self.reporter.plot_results(save_path=None, show_components=True)
                
                # 保存
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                QMessageBox.information(self, "成功", f"图片已保存至:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    # 创建主窗口
    window = XRDAnalyzerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

