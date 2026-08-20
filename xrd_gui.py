# xrd_gui.py - PyQt5 GUI界面

import sys
import os
import re
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
from threadpoolctl import threadpool_limits

from xrd_analyzer import (
    BraggGeometry, DEFAULT_RADIATION_LABEL, DEFAULT_WAVELENGTH_ANGSTROM,
    DataLoader, Preprocessor, Fitter, PROJECT_WORKBOOK_SCHEMA_VERSION,
    PSEUDO_VOIGT_FWHM_FACTOR, ProjectWorkbook, Reporter, RestoredFitResult, Peak
)


FITTING_THREAD_STACK_SIZE_BYTES = 16 * 1024 * 1024


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
        self.clear_axes()
        if self.x_data is not None and self.y_data is not None:
            self.axes.plot(self.x_data, self.y_data, 'b-', linewidth=1.5, label='Data')
            self.axes.set_xlabel('2θ (degree)', fontsize=11, fontweight='bold')
            self.axes.set_ylabel('Intensity (a.u.)', fontsize=11, fontweight='bold')
            self.axes.set_yscale(self.yscale)  # Apply current scale
            self.axes.set_xlim(self.x_data.min(), self.x_data.max()) # Enforce x-limits
            self.axes.grid(True, alpha=0.3)
            self.axes.legend()
        self.draw()

    def clear_axes(self) -> None:
        """完整清空坐标轴，并丢弃随之失效的峰标记引用。"""
        self.peak_markers.clear()
        self.axes.clear()
    
    def add_peak_marker(self, x_pos, y_pos, peak_id, name=''):
        """添加峰标记"""
        label_text = name if name else f'Peak {peak_id}'
        marker = self.axes.plot(x_pos, y_pos, 'ro', markersize=8, 
                               label=label_text)[0]
        
        # Add text annotation
        if name:
            annotation = self.axes.annotate(name, (x_pos, y_pos), 
                                          xytext=(0, 10), textcoords='offset points',
                                          ha='center', fontsize=9, color='red')
            self.peak_markers.append(annotation)
            
        self.peak_markers.append(marker)
        self.axes.legend()
        self.draw()
    
    def clear_peak_markers(self):
        """清除所有峰标记"""
        for marker in self.peak_markers:
            if getattr(marker, 'axes', None) is self.axes:
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
        
        # 峰名称
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("可选，例如: PZT(002)")
        layout.addRow("峰名称:", self.name_edit)
        
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
            'type': self.type_combo.currentText(),
            'name': self.name_edit.text().strip()
        }


class FittingThread(QThread):
    """拟合线程"""
    
    finished = pyqtSignal(object)  # 发送result
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(
        self,
        fitter,
        constrain_fwhm,
        min_separation,
        log_weight=0.5,
        fixed_background=None,
        method='leastsq',
        objective_mode='mixed',
        intensity_floor=1.0,
        include_ranges=None,
        exclude_ranges=None,
    ):
        super().__init__()
        # macOS的默认QThread栈不足以承载OpenBLAS在lmfit协方差求逆时的
        # 原生栈分配。显式预留空间，避免SIGBUS直接终止整个进程。
        self.setStackSize(FITTING_THREAD_STACK_SIZE_BYTES)
        self.fitter = fitter
        self.constrain_fwhm = constrain_fwhm
        self.min_separation = min_separation
        self.log_weight = log_weight
        self.fixed_background = fixed_background
        self.method = method
        self.objective_mode = objective_mode
        self.intensity_floor = intensity_floor
        self.include_ranges = list(include_ranges or [])
        self.exclude_ranges = list(exclude_ranges or [])
        
    def run(self):
        try:
            # 科学结果不依赖BLAS线程数。限制为单线程可避免OpenBLAS在
            # Qt后台线程中走并行LU求逆，同时也防止多套BLAS过度订阅。
            with threadpool_limits(limits=1, user_api='blas'):
                self.progress.emit(20)

                self.fitter.build_model(
                    constrain_fwhm=self.constrain_fwhm,
                    min_peak_separation=self.min_separation,
                    fixed_background=self.fixed_background
                )

                self.progress.emit(40)

                result = self.fitter.execute_fitting(
                    method=self.method,
                    log_weight=self.log_weight,
                    objective_mode=self.objective_mode,
                    intensity_floor=self.intensity_floor,
                    include_ranges=self.include_ranges,
                    exclude_ranges=self.exclude_ranges,
                )
            
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
        self.last_data_dir = None    # 新增：记录上次打开的数据文件夹
        
        # 多文件管理
        self.loaded_files_data = []  # List[Tuple[filepath, x, y]]
        self.active_data_range = None  # 数据加载模块已提交的永久2theta范围
        
        # 处理对象
        self.fitter = None
        self.reporter = None
        self.fit_thread = None
        
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
        # 改为垂直布局：上部为两列，下部为宽大的峰管理区
        main_layout = QVBoxLayout()
        panel.setLayout(main_layout)
        
        # === 上部区域 (两列布局) ===
        top_section = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_section.setLayout(top_layout)
        
        # --- 左上列：数据加载与预处理 ---
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
        
        clear_btn = QPushButton("清空文件")
        clear_btn.clicked.connect(self.clear_files)
        btn_layout.addWidget(clear_btn)
        
        # 新增：重置/刷新按钮
        app_reset_btn = QPushButton("重置系统")
        app_reset_btn.setToolTip("清除所有数据、峰和结果，恢复初始状态")
        app_reset_btn.setStyleSheet("color: red; font-weight: bold;")
        app_reset_btn.clicked.connect(self.reset_app)
        btn_layout.addWidget(app_reset_btn)
        
        file_layout.addLayout(btn_layout)

        load_project_btn = QPushButton("加载Excel项目")
        load_project_btn.setToolTip("读取本程序导出的Excel报告并恢复数据、峰和界面配置")
        load_project_btn.clicked.connect(self.open_excel_project)
        file_layout.addWidget(load_project_btn)
        
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
        self.sg_window.setValue(7)
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
        col1_layout.addStretch() # 确保紧凑
        
        top_layout.addLayout(col1_layout)
        
        # --- 右上列：拟合配置与导出 (原Group 4 & 5) ---
        col2_layout = QVBoxLayout()
        
        # 4. 拟合配置组 (原 Group 4)
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

        objective_layout = QHBoxLayout()
        objective_layout.addWidget(QLabel("目标函数:"))
        self.objective_combo = QComboBox()
        self.objective_combo.addItem("Linear", "linear")
        self.objective_combo.addItem("Log", "log")
        self.objective_combo.addItem("Mixed", "mixed")
        self.objective_combo.setCurrentIndex(2)
        objective_layout.addWidget(self.objective_combo)
        fit_layout.addLayout(objective_layout)
        
        # 优化权重 (Log/Linear)
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(QLabel("Log权重:"))
        
        self.log_weight_slider = QSpinBox() # 使用SpinBox简单直观，或者Slider+SpinBox联动
        from PyQt5.QtWidgets import QSlider # 确保引入
        
        self.log_slider = QSlider(Qt.Horizontal)
        self.log_slider.setRange(0, 100)
        self.log_slider.setValue(50)
        self.log_slider.setSingleStep(10)
        
        self.log_spin = QSpinBox()
        self.log_spin.setRange(0, 100)
        self.log_spin.setValue(50)
        self.log_spin.setSuffix("%")
        
        # 联动
        self.log_slider.valueChanged.connect(self.log_spin.setValue)
        self.log_spin.valueChanged.connect(self.log_slider.setValue)
        
        weight_layout.addWidget(self.log_slider)
        weight_layout.addWidget(self.log_spin)
        fit_layout.addLayout(weight_layout)

        floor_layout = QHBoxLayout()
        floor_layout.addWidget(QLabel("Log强度底值 I₀:"))
        self.log_floor_spin = QDoubleSpinBox()
        self.log_floor_spin.setRange(0.000001, 1000000.0)
        self.log_floor_spin.setDecimals(6)
        self.log_floor_spin.setValue(1.0)
        self.log_floor_spin.setToolTip(
            "Log残差使用log10(I + I₀)；I₀应依据噪声底人工设置"
        )
        floor_layout.addWidget(self.log_floor_spin)
        fit_layout.addLayout(floor_layout)

        self.objective_combo.currentIndexChanged.connect(self.update_objective_controls)
        self.update_objective_controls()

        include_layout = QHBoxLayout()
        include_layout.addWidget(QLabel("包含区间:"))
        self.include_ranges_input = QLineEdit()
        self.include_ranges_input.setPlaceholderText("留空=全部；例 42.5-43.5; 54-56")
        include_layout.addWidget(self.include_ranges_input)
        fit_layout.addLayout(include_layout)

        exclude_layout = QHBoxLayout()
        exclude_layout.addWidget(QLabel("排除区间:"))
        self.exclude_ranges_input = QLineEdit()
        self.exclude_ranges_input.setPlaceholderText("例 42.8-43.2")
        exclude_layout.addWidget(self.exclude_ranges_input)
        fit_layout.addLayout(exclude_layout)
        
        # 背景强度设置
        bg_setting_layout = QHBoxLayout()
        bg_setting_layout.addWidget(QLabel("背景强度:"))
        
        self.bg_spin = QDoubleSpinBox()
        self.bg_spin.setRange(0, 1000000)
        self.bg_spin.setDecimals(2)
        self.bg_spin.setValue(0.0)
        self.bg_spin.setSingleStep(1.0)
        self.bg_spin.setToolTip("设置固定的背景强度值或显示拟合后的背景值")
        
        self.bg_fix_cb = QCheckBox("固定")
        self.bg_fix_cb.setToolTip("勾选此项将强制使用设定的背景值，不进行拟合")
        
        bg_setting_layout.addWidget(self.bg_spin)
        bg_setting_layout.addWidget(self.bg_fix_cb)
        fit_layout.addLayout(bg_setting_layout)
        
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
        
        # 优化拟合按钮 (Refine Fit)
        refine_fit_btn = QPushButton("基于当前结果优化拟合 (Refine)")
        refine_fit_btn.setToolTip("使用上一次的拟合结果作为初始值，并应用表格中的锁定设置进行再拟合")
        refine_fit_btn.setStyleSheet("color: blue; font-weight: bold;")
        refine_fit_btn.clicked.connect(self.refine_fitting)
        fit_layout.addWidget(refine_fit_btn)

        accept_fit_btn = QPushButton("接受当前结果作为下一轮初值")
        accept_fit_btn.setToolTip(
            "人工确认当前候选结果，并保存center、area、FWHM和η作为下一轮初值"
        )
        accept_fit_btn.clicked.connect(self.accept_current_fit)
        fit_layout.addWidget(accept_fit_btn)
        self.accept_fit_btn = accept_fit_btn
        
        fit_group.setLayout(fit_layout)
        col2_layout.addWidget(fit_group)
        
        # 5. 结果导出组 (原 Group 5)
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
        col2_layout.addStretch() # 确保紧凑
        
        top_layout.addLayout(col2_layout)
        
        # 添加上部区域到主布局
        main_layout.addWidget(top_section)
        
        # === 下部区域：峰管理 (全宽) ===
        
        # 3. 峰管理组 (原 Group 3)
        peak_group = QGroupBox("3. 峰识别与管理")
        self.peak_group = peak_group
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
        
        # 数值添加：直接2θ或由理论晶面间距d反算
        quick_add_layout = QHBoxLayout()
        self.peak_input_mode = QComboBox()
        self.peak_input_mode.addItem("2θ位置 (°)", "two_theta")
        self.peak_input_mode.addItem("理论晶面间距 d (Å)", "d_spacing")
        quick_add_layout.addWidget(self.peak_input_mode)

        self.peak_value_input = QDoubleSpinBox()
        self.peak_value_input.setDecimals(6)
        self.peak_value_input.setRange(0.000001, 179.999999)
        self.peak_value_input.setValue(44.0)
        self.peak_value_input.setSuffix(" °")
        self.peak_value_input.setKeyboardTracking(False)
        quick_add_layout.addWidget(self.peak_value_input)

        self.peak_type_combo = QComboBox()
        self.peak_type_combo.addItem("薄膜", "film")
        self.peak_type_combo.addItem("基底", "substrate")
        quick_add_layout.addWidget(self.peak_type_combo)

        self.peak_name_input = QLineEdit()
        self.peak_name_input.setPlaceholderText("峰名，例如 PZT(002)")
        quick_add_layout.addWidget(self.peak_name_input)
        
        quick_add_btn = QPushButton("添加")
        quick_add_btn.clicked.connect(self.quick_add_peak)
        quick_add_layout.addWidget(quick_add_btn)
        peak_layout.addLayout(quick_add_layout)

        wavelength_layout = QHBoxLayout()
        wavelength_layout.addWidget(QLabel("X射线波长 λ:"))
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setDecimals(6)
        self.wavelength_spin.setRange(0.000001, 100.0)
        self.wavelength_spin.setValue(DEFAULT_WAVELENGTH_ANGSTROM)
        self.wavelength_spin.setSuffix(" Å")
        self.wavelength_spin.setKeyboardTracking(False)
        self.wavelength_spin.setToolTip(
            "项目统一使用的X射线波长；默认 Cu Kα1 = 1.5406 Å"
        )
        self.wavelength_spin.valueChanged.connect(self.on_wavelength_changed)
        wavelength_layout.addWidget(self.wavelength_spin)
        wavelength_layout.addWidget(QLabel("默认 Cu Kα1"))
        wavelength_layout.addStretch()
        peak_layout.addLayout(wavelength_layout)
        self.peak_input_mode.currentIndexChanged.connect(
            self.update_peak_input_mode
        )
        
        # 导入/导出峰
        io_peak_layout = QHBoxLayout()
        import_peak_btn = QPushButton("导入峰列表 (.txt)")
        import_peak_btn.clicked.connect(self.import_peaks_from_file)
        io_peak_layout.addWidget(import_peak_btn)
        peak_layout.addLayout(io_peak_layout)
        
        # 手动添加峰
        manual_peak_btn = QPushButton("手动添加峰 (点击图上)")
        manual_peak_btn.setCheckable(True)
        manual_peak_btn.clicked.connect(self.toggle_add_peak_mode)
        peak_layout.addWidget(manual_peak_btn)
        self.manual_peak_btn = manual_peak_btn
        
        # 峰列表 (移除最大高度限制)
        self.peak_table = QTableWidget()
        self.peak_table.setColumnCount(9)
        self.peak_table.setHorizontalHeaderLabels(
            [
                'ID', '名称', '位置 (Pos)', '面积', '高度 (Height)',
                'FWHM', 'η', '峰状态', '类型'
            ]
        )
        self.peak_table.horizontalHeaderItem(3).setToolTip(
            "拟合Pseudo-Voigt分布的积分面积（lmfit amplitude）"
        )
        self.peak_table.horizontalHeaderItem(5).setToolTip(
            "Pseudo-Voigt半高全宽（2θ度）。薄膜峰允许0.02–3.00°，"
            "基底峰允许0.02–2.00°；勾选后按输入值精确固定。"
        )
        self.peak_table.horizontalHeaderItem(6).setToolTip(
            "Pseudo-Voigt Lorentzian比例，范围0–1"
        )
        self.peak_table.horizontalHeaderItem(7).setToolTip(
            "优化=参数参与拟合；冻结=固定完整峰形；禁用=本轮峰分量为零"
        )
        self.peak_table.setMinimumHeight(200)
        
        # 设置列宽
        self.peak_table.setColumnWidth(0, 40) # ID
        self.peak_table.setColumnWidth(1, 80) # Name
        self.peak_table.setColumnWidth(2, 100) # Pos
        self.peak_table.setColumnWidth(3, 110) # Area
        self.peak_table.setColumnWidth(4, 100) # Height
        self.peak_table.setColumnWidth(5, 100) # FWHM
        self.peak_table.setColumnWidth(6, 70) # Eta
        self.peak_table.setColumnWidth(7, 90) # State
        self.peak_table.setColumnWidth(8, 80) # Type
        
        # Connect itemChanged signal for live editing
        self.peak_table.itemChanged.connect(self.on_peak_table_changed)
        
        peak_layout.addWidget(self.peak_table)
        
        # 删除峰按钮
        peak_btn_layout = QHBoxLayout()
        delete_peak_btn = QPushButton("删除选中峰")
        delete_peak_btn.clicked.connect(self.delete_selected_peak)
        peak_btn_layout.addWidget(delete_peak_btn)
        
        clear_peaks_btn = QPushButton("清除所有峰")
        clear_peaks_btn.clicked.connect(self.clear_all_peaks)
        peak_btn_layout.addWidget(clear_peaks_btn)
        peak_layout.addLayout(peak_btn_layout)
        
        peak_group.setLayout(peak_layout)
        
        # 添加峰管理组到主布局，并给予伸缩因子1，使其占据剩余空间
        main_layout.addWidget(peak_group, 1)
        
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

    def collect_project_state(self) -> dict:
        """收集恢复当前GUI分析会话所需的显式状态。"""
        active_range = self.active_data_range
        if active_range is None and self.x_data is not None and len(self.x_data):
            active_range = (float(np.min(self.x_data)), float(np.max(self.x_data)))
        return {
            'schema_version': PROJECT_WORKBOOK_SCHEMA_VERSION,
            'range_min': active_range[0] if active_range else self.range_min.value(),
            'range_max': active_range[1] if active_range else self.range_max.value(),
            'filter_type': self.filter_combo.currentText(),
            'sg_window': self.sg_window.value(),
            'background_preprocess': self.bg_combo.currentText(),
            'poly_degree': self.poly_degree.value(),
            'constrain_fwhm': self.constrain_fwhm_cb.isChecked(),
            'min_separation': self.min_separation.value(),
            'fit_method': self.fit_method_combo.currentText(),
            'objective_mode': self.objective_combo.currentData(),
            'log_weight': self.log_spin.value(),
            'log_floor': self.log_floor_spin.value(),
            'background_value': self.bg_spin.value(),
            'background_fixed': self.bg_fix_cb.isChecked(),
            'yscale': self.plot_canvas.yscale,
            'include_ranges_text': self.include_ranges_input.text(),
            'exclude_ranges_text': self.exclude_ranges_input.text(),
            'peak_threshold': self.peak_threshold.value(),
            'wavelength_angstrom': self.wavelength_spin.value(),
            'radiation_label': self.current_radiation_label(),
            'peak_input_mode': self.peak_input_mode.currentData(),
        }

    @staticmethod
    def _set_combo_text(combo: QComboBox, value) -> None:
        if value is None:
            return
        index = combo.findText(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value) -> None:
        if value is None:
            return
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def apply_project_state(self, state: dict) -> None:
        """恢复工作簿中保存的GUI配置，但不重复执行预处理。"""
        self._set_combo_text(self.filter_combo, state.get('filter_type'))
        self.sg_window.setValue(int(state.get('sg_window', 7)))
        self._set_combo_text(
            self.bg_combo,
            state.get('background_preprocess', '无'),
        )
        self.poly_degree.setValue(int(state.get('poly_degree', 2)))
        self.constrain_fwhm_cb.setChecked(bool(state.get('constrain_fwhm', False)))
        self.min_separation.setValue(float(state.get('min_separation', 0.2)))
        self._set_combo_text(self.fit_method_combo, state.get('fit_method', 'leastsq'))
        self._set_combo_data(self.objective_combo, state.get('objective_mode', 'mixed'))
        self.log_spin.setValue(int(state.get('log_weight', 50)))
        self.log_floor_spin.setValue(float(state.get('log_floor', 1.0)))
        self.bg_spin.setValue(float(state.get('background_value', 0.0)))
        self.bg_fix_cb.setChecked(bool(state.get('background_fixed', False)))
        self.include_ranges_input.setText(str(state.get('include_ranges_text', '')))
        self.exclude_ranges_input.setText(str(state.get('exclude_ranges_text', '')))
        self.peak_threshold.setValue(float(state.get('peak_threshold', 100.0)))
        self.wavelength_spin.setValue(
            float(state.get('wavelength_angstrom', DEFAULT_WAVELENGTH_ANGSTROM))
        )
        self._set_combo_data(
            self.peak_input_mode,
            state.get('peak_input_mode', 'two_theta'),
        )
        if state.get('yscale') == 'log':
            self.log_radio.setChecked(True)
            self.plot_canvas.yscale = 'log'
        else:
            self.linear_radio.setChecked(True)
            self.plot_canvas.yscale = 'linear'

    def open_excel_project(self) -> None:
        """选择并加载XRD Excel项目工作簿。"""
        start_dir = self.last_data_dir if self.last_data_dir else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "加载Excel项目",
            start_dir,
            "Excel Projects (*.xlsx *.xls)",
        )
        if not file_path:
            return
        if self.x_data is not None:
            reply = QMessageBox.question(
                self,
                "替换当前项目",
                "加载Excel项目将替换当前数据和峰设置，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        try:
            self.load_excel_project(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "项目加载失败", str(exc))

    def load_excel_project(self, file_path: str) -> None:
        """从Excel报告恢复数据、峰、候选拟合结果和GUI状态。"""
        project = ProjectWorkbook.load(file_path)
        x_data = np.asarray(project['x_data'], dtype=float)
        processed = np.asarray(project['processed_intensity'], dtype=float)
        raw = np.asarray(project['raw_intensity'], dtype=float)
        state = project['project_state']

        self.clear_files()
        self.reporter = None
        self.peak_table.setRowCount(0)
        self.results_text.clear()
        self.physics_text.clear()

        self.x_data = x_data.copy()
        self.y_data = processed.copy()
        self.x_data_raw = x_data.copy()
        self.y_data_raw = raw.copy()
        self.x_data_original = x_data.copy()
        self.y_data_original = raw.copy()
        self.current_file = str(file_path)
        self.last_data_dir = str(Path(file_path).parent)
        range_min = float(state.get('range_min', np.min(x_data)))
        range_max = float(state.get('range_max', np.max(x_data)))
        self.active_data_range = (range_min, range_max)
        self.range_min.setValue(range_min)
        self.range_max.setValue(range_max)
        source_datasets = project.get('source_datasets', [])
        self.loaded_files_data = [
            (str(path), np.asarray(source_x), np.asarray(source_y))
            for path, source_x, source_y in source_datasets
        ]
        source_paths = project.get('source_files', [])
        if self.loaded_files_data:
            for path, _, _ in self.loaded_files_data:
                item = QListWidgetItem(Path(path).name)
                item.setToolTip(path)
                self.file_list_widget.addItem(item)
        else:
            self.loaded_files_data = [(str(file_path), x_data.copy(), raw.copy())]
            project_item = QListWidgetItem(f"项目: {Path(file_path).name}")
            project_item.setToolTip(
                "\n".join(source_paths) if source_paths else str(file_path)
            )
            self.file_list_widget.addItem(project_item)
        self.apply_project_state(state)

        self.fitter = Fitter(self.x_data, self.y_data)
        peak_records = sorted(
            project['peaks'],
            key=lambda record: record.get('Peak_ID', 0),
        )

        def number(record, key, fallback=None):
            value = record.get(key)
            return fallback if value is None else float(value)

        for record in peak_records:
            fitted_center = number(record, 'Center_2theta')
            center_guess = number(record, 'Center_Guess_2theta', fitted_center)
            if center_guess is None:
                continue
            bounds = (
                number(record, 'Bounds_Min_2theta', center_guess - 0.5),
                number(record, 'Bounds_Max_2theta', center_guess + 0.5),
            )
            peak = self.fitter.add_peak(
                center_guess,
                bounds,
                str(record.get('Type') or 'film'),
                str(record.get('Name') or ''),
            )
            peak.center = fitted_center
            peak.area = number(record, 'Area')
            peak.height = number(record, 'Height')
            peak.fwhm = number(record, 'FWHM')
            peak.eta = number(record, 'Eta_PseudoVoigt')
            peak.area_guess = number(record, 'Area_Guess', peak.area)
            peak.height_guess = number(record, 'Height_Guess', peak.height)
            peak.sigma_guess = number(
                record,
                'Sigma_Guess',
                peak.fwhm / PSEUDO_VOIGT_FWHM_FACTOR if peak.fwhm is not None else None,
            )
            peak.fraction_guess = number(record, 'Fraction_Guess', peak.eta)
            peak.fixed_center = bool(record.get('Fixed_Center') or False)
            peak.fixed_height = bool(record.get('Fixed_Height') or False)
            peak.fixed_fwhm = bool(record.get('Fixed_FWHM') or False)
            fit_state = str(record.get('Fit_State') or 'optimize')
            peak.fit_state = (
                fit_state
                if fit_state in {'optimize', 'frozen', 'disabled'}
                else 'optimize'
            )

        fitted_intensity = project.get('fitted_intensity')
        has_complete_fit = fitted_intensity is not None and all(
            peak.center is not None
            and peak.area is not None
            and peak.fwhm is not None
            and peak.eta is not None
            for peak in self.fitter.peaks
        )
        self.fitter.fit_config = dict(project.get('fit_config', {}))
        self.fitter.fit_diagnostics = dict(project.get('fit_config', {}))

        if has_complete_fit and self.fitter.peaks:
            saved_states = [
                (
                    peak.fit_state,
                    peak.fixed_center,
                    peak.fixed_height,
                    peak.fixed_fwhm,
                )
                for peak in self.fitter.peaks
            ]
            for peak in self.fitter.peaks:
                peak.fit_state = 'optimize'
                peak.fixed_center = False
                peak.fixed_height = False
                peak.fixed_fwhm = False
            background = project.get('background')
            background_value = float(background[0]) if background is not None else 0.0
            self.fitter.build_model(
                min_peak_separation=0.0,
                fixed_background=background_value,
            )
            for peak, record in zip(self.fitter.peaks, peak_records):
                prefix = f'p{peak.peak_id}_'
                self.fitter.params[f'{prefix}center'].set(value=peak.center)
                self.fitter.params[f'{prefix}amplitude'].set(value=peak.area)
                self.fitter.params[f'{prefix}sigma'].set(
                    value=peak.fwhm / PSEUDO_VOIGT_FWHM_FACTOR
                )
                self.fitter.params[f'{prefix}fraction'].set(value=peak.eta)
            self.fitter.params.update_constraints()
            for peak, saved in zip(self.fitter.peaks, saved_states):
                (
                    peak.fit_state,
                    peak.fixed_center,
                    peak.fixed_height,
                    peak.fixed_fwhm,
                ) = saved

            diagnostics = self.fitter.fit_diagnostics
            metrics = project.get('metrics', {})
            self.fitter.result = RestoredFitResult(
                params=self.fitter.params,
                success=bool(diagnostics.get('success', True)),
                message=str(diagnostics.get('message', 'Restored from Excel project')),
                nfev=int(diagnostics.get('nfev') or 0),
                covar=None,
                chisqr=float(metrics.get('Objective_SSE') or np.nan),
            )
            self.fitter.y_fit = np.asarray(fitted_intensity, dtype=float)
            try:
                include_ranges = self.parse_fit_ranges(
                    str(state.get('include_ranges_text', ''))
                )
                exclude_ranges = self.parse_fit_ranges(
                    str(state.get('exclude_ranges_text', ''))
                )
                self.fitter.fit_mask = Fitter.build_fit_mask(
                    self.x_data,
                    include_ranges,
                    exclude_ranges,
                )
            except ValueError:
                self.fitter.fit_mask = np.ones_like(self.x_data, dtype=bool)
            self.reporter = Reporter(
                self.fitter,
                x_original=self.x_data_original,
                y_original=self.y_data_original,
                wavelength_angstrom=self.wavelength_spin.value(),
                radiation_label=self.current_radiation_label(),
            )
            self.reporter.calculate_metrics()
            self.plot_fitted_results()
            self.update_peak_table()
            self.display_results()
            self.display_physics_parameters()
        else:
            self.reporter = None
            self.update_peak_table()
            self.redraw_peak_guesses()

        self.statusBar().showMessage(f"Excel项目已恢复: {Path(file_path).name}")
    
    def add_files(self):
        """添加文件"""
        start_dir = self.last_data_dir if self.last_data_dir else ""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择XRD数据文件", start_dir, "Text Files (*.txt *.TXT)"
        )
        
        if not file_paths:
            return
            
        # 更新上次打开的文件夹
        if file_paths:
            self.last_data_dir = str(Path(file_paths[0]).parent)
            
        success_count = 0
        for file_path in file_paths:
            try:
                # 检查是否已加载
                if any(f[0] == file_path for f in self.loaded_files_data):
                    continue
                    
                x, y = DataLoader.load_txt(file_path)
                if self.active_data_range is not None:
                    x, y = DataLoader.trim_range(
                        x,
                        y,
                        self.active_data_range[0],
                        self.active_data_range[1],
                    )
                    if len(x) == 0:
                        continue
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
        self.x_data_original = None
        self.y_data_original = None
        self.plot_canvas.clear_axes()
        self.plot_canvas.draw()
        self.current_file = None
        self.active_data_range = None
        
        # 同时也应该清除fitter，因为数据没了
        self.fitter = None
        
        self.statusBar().showMessage('列表已清空')

    def reset_app(self):
        """重置整个应用状态"""
        if self.is_fitting():
            QMessageBox.warning(self, "拟合进行中", "请等待当前拟合结束后再重置系统")
            return

        reply = QMessageBox.question(
            self, '确认重置', 
            "确定要重置系统吗？\n这将清除所有已加载的数据、峰设置和拟合结果。",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # clear_files会销毁fitter，因此先直接清空依赖fitter的峰表格。
            if self.fitter is not None:
                self.fitter.clear_peaks()
            self.peak_table.setRowCount(0)

            # 清除文件、数据、绘图和模型对象。
            self.clear_files()
            self.fitter = None
            self.reporter = None
            self.fit_thread = None

            # 清除结果显示。
            self.results_text.clear()
            self.physics_text.clear()

            # 恢复所有会改变下一次分析的控件默认值。
            self.filter_combo.setCurrentIndex(0)
            self.sg_window.setValue(7)
            self.bg_combo.setCurrentIndex(0)
            self.poly_degree.setValue(2)
            self.constrain_fwhm_cb.setChecked(False)
            self.min_separation.setValue(0.2)
            self.fit_method_combo.setCurrentIndex(0)
            self.objective_combo.setCurrentIndex(2)
            self.log_spin.setValue(50)
            self.log_floor_spin.setValue(1.0)
            self.include_ranges_input.clear()
            self.exclude_ranges_input.clear()
            self.bg_spin.setValue(0.0)
            self.bg_fix_cb.setChecked(False)
            self.linear_radio.setChecked(True)
            self.plot_canvas.yscale = 'linear'
            self.peak_threshold.setValue(100.0)
            self.peak_input_mode.setCurrentIndex(0)
            self.peak_value_input.setValue(44.0)
            self.peak_type_combo.setCurrentIndex(0)
            self.peak_name_input.clear()
            self.wavelength_spin.setValue(DEFAULT_WAVELENGTH_ANGSTROM)

            self.manual_peak_btn.setChecked(False)
            self.plot_canvas.mode = 'view'
            self.manual_peak_btn.setText("手动添加峰 (点击图上)")
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)

            self.range_min.setValue(20)
            self.range_max.setValue(120)

            self.statusBar().showMessage('系统已重置，就绪')

    def update_merged_data(self):
        """更新合并后的数据"""
        if not self.loaded_files_data:
            self.clear_files()
            return
            
        # 提取所有数据进行拼接
        datasets = [(d[1], d[2]) for d in self.loaded_files_data]
        
        try:
            x_merged, y_merged = DataLoader.stitch_datasets(datasets)
            if self.active_data_range is not None:
                x_merged, y_merged = DataLoader.trim_range(
                    x_merged,
                    y_merged,
                    self.active_data_range[0],
                    self.active_data_range[1],
                )
            
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
                
            self.x_data = x_merged.copy()
            self.y_data = y_merged.copy()

            # 数据源改变后，旧峰和拟合结果不再对应当前数据。
            self.fitter = None
            self.reporter = None
            self.peak_table.setRowCount(0)
            self.results_text.clear()
            self.physics_text.clear()

            if self.active_data_range is None:
                self.range_min.setValue(x_merged.min())
                self.range_max.setValue(x_merged.max())
            else:
                self.range_min.setValue(self.active_data_range[0])
                self.range_max.setValue(self.active_data_range[1])

            self.plot_canvas.set_data(self.x_data, self.y_data)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据合并失败:\n{str(e)}")

    def toggle_yscale(self):
        """切换Y轴显示模式"""
        if self.linear_radio.isChecked():
            self.plot_canvas.set_yscale('linear')
        else:
            self.plot_canvas.set_yscale('log')

    def update_objective_controls(self) -> None:
        """根据目标函数显示真正生效的Log配置。"""
        mode = self.objective_combo.currentData()
        self.log_slider.setEnabled(mode == 'mixed')
        self.log_spin.setEnabled(mode == 'mixed')
        self.log_floor_spin.setEnabled(mode in {'log', 'mixed'})

    @staticmethod
    def parse_fit_ranges(text: str) -> List[tuple[float, float]]:
        """解析用户输入的2theta区间列表。"""
        stripped = text.strip()
        if not stripped:
            return []

        ranges = []
        pattern = re.compile(
            r"^\s*([0-9]+(?:\.[0-9]*)?)\s*[-–—:]\s*"
            r"([0-9]+(?:\.[0-9]*)?)\s*$"
        )
        for token in re.split(r"[;,，；]+", stripped):
            match = pattern.match(token)
            if match is None:
                raise ValueError(f"无法识别拟合区间: {token.strip()}")
            lower, upper = map(float, match.groups())
            ranges.append(tuple(sorted((lower, upper))))
        return ranges


    
    def apply_range(self):
        """提交数据加载范围，并永久裁剪当前项目中的全部数据。"""
        if self.x_data_raw is None:
            return
        
        x_min = self.range_min.value()
        x_max = self.range_max.value()
        if x_min >= x_max:
            QMessageBox.warning(self, "范围错误", "2θ范围下限必须小于上限")
            return

        trimmed_raw_x, trimmed_raw_y = DataLoader.trim_range(
            self.x_data_raw,
            self.y_data_raw,
            x_min,
            x_max,
        )
        if len(trimmed_raw_x) == 0:
            QMessageBox.warning(self, "范围错误", "所选2θ范围内没有数据")
            return

        old_peaks = list(self.fitter.peaks) if self.fitter is not None else []

        self.x_data_raw = trimmed_raw_x.copy()
        self.y_data_raw = trimmed_raw_y.copy()
        self.x_data, self.y_data = DataLoader.trim_range(
            self.x_data,
            self.y_data,
            x_min,
            x_max,
        )
        if self.x_data_original is not None and self.y_data_original is not None:
            self.x_data_original, self.y_data_original = DataLoader.trim_range(
                self.x_data_original,
                self.y_data_original,
                x_min,
                x_max,
            )

        filtered_sources = []
        for path, source_x, source_y in self.loaded_files_data:
            source_x, source_y = DataLoader.trim_range(
                source_x,
                source_y,
                x_min,
                x_max,
            )
            if len(source_x) > 0:
                filtered_sources.append((path, source_x, source_y))
        self.loaded_files_data = filtered_sources
        self.file_list_widget.clear()
        for path, _, _ in self.loaded_files_data:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            self.file_list_widget.addItem(item)

        self.active_data_range = (x_min, x_max)
        self.reporter = None
        self.results_text.clear()
        self.physics_text.clear()

        removed_peak_count = 0
        if old_peaks:
            in_range_peaks = []
            for peak in old_peaks:
                if x_min <= peak.center_guess <= x_max:
                    peak.bounds = (
                        max(peak.bounds[0], x_min),
                        min(peak.bounds[1], x_max),
                    )
                    peak.clear_result()
                    in_range_peaks.append(peak)
                else:
                    removed_peak_count += 1
            self.fitter = Fitter(self.x_data, self.y_data)
            self.fitter.peaks = in_range_peaks
            for peak_id, peak in enumerate(self.fitter.peaks):
                peak.peak_id = peak_id
            self.update_peak_table()
            self.redraw_peak_guesses()
        else:
            self.fitter = None
            self.peak_table.setRowCount(0)
            self.plot_canvas.set_data(self.x_data, self.y_data)

        message = f"项目数据已裁剪至 {x_min:.2f}–{x_max:.2f}°"
        if removed_peak_count:
            message += f"，移除 {removed_peak_count} 个范围外峰"
        self.statusBar().showMessage(message)
    
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
                values['type'],
                values['name']
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
    
    def add_peak_to_fitter(self, center, bounds, peak_type, name=''):
        """添加峰到拟合器"""
        if self.fitter is None:
            self.fitter = Fitter(self.x_data, self.y_data)
        
        self.fitter.add_peak(center, bounds, peak_type, name)
        self.refresh_after_peak_change()

    def is_fitting(self) -> bool:
        """返回后台拟合线程是否仍在运行。"""
        return self.fit_thread is not None and self.fit_thread.isRunning()

    def set_peak_editing_enabled(self, enabled: bool) -> None:
        """拟合期间统一锁定或恢复峰编辑区域。"""
        if not enabled:
            self.manual_peak_btn.setChecked(False)
            self.manual_peak_btn.setText("手动添加峰 (点击图上)")
            self.plot_canvas.mode = 'view'
        self.peak_group.setEnabled(enabled)

    def clear_stale_fit_display(self) -> None:
        """清除不再对应当前峰配置的结果展示。"""
        self.reporter = None
        self.results_text.clear()
        self.physics_text.clear()

    def redraw_peak_guesses(self) -> None:
        """以当前数据和峰初始位置重绘配置视图。"""
        if self.x_data is None or self.y_data is None:
            self.plot_canvas.clear_axes()
            self.plot_canvas.draw()
            return

        self.plot_canvas.set_data(self.x_data, self.y_data)
        if self.fitter is None:
            return

        for peak in self.fitter.peaks:
            idx = np.argmin(np.abs(self.x_data - peak.center_guess))
            self.plot_canvas.add_peak_marker(
                peak.center_guess,
                self.y_data[idx],
                peak.peak_id,
                peak.name,
            )

    def refresh_after_peak_change(self) -> None:
        """同步峰表格、结果面板和绘图，避免显示旧拟合状态。"""
        self.update_peak_table()
        self.clear_stale_fit_display()
        self.redraw_peak_guesses()
        
    def update_peak_table(self):
        """更新峰表格"""
        if self.fitter is None:
            return
        
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(len(self.fitter.peaks))
        boundary_hits = set(self.fitter.fit_diagnostics.get('boundary_hits', []))
        
        for i, peak in enumerate(self.fitter.peaks):
            # 0. ID
            self.peak_table.setItem(i, 0, QTableWidgetItem(str(peak.peak_id)))
            
            # 1. 名称
            self.peak_table.setItem(i, 1, QTableWidgetItem(peak.name))
            
            # 2. 位置 (Pos) - 总是显示猜测值或结果值
            center_val = peak.center if peak.center is not None else peak.center_guess
            item_pos = QTableWidgetItem(f"{center_val:.3f}")
            item_pos.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            item_pos.setCheckState(Qt.Checked if peak.fixed_center else Qt.Unchecked)
            if any(
                f"p{peak.peak_id}_center:{side}" in boundary_hits
                for side in ('min', 'max')
            ):
                item_pos.setBackground(QColor(255, 235, 130))
            self.peak_table.setItem(i, 2, item_pos)
            
            # 3. 面积 (Area) - lmfit PseudoVoigtModel的amplitude，拟合前NaN
            area_value = peak.area if peak.area is not None else peak.area_guess
            if area_value is not None:
                val_str = f"{area_value:.2f}"
            else:
                val_str = "NaN"
            item_area = QTableWidgetItem(val_str)
            item_area.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            if any(
                f"p{peak.peak_id}_amplitude:{side}" in boundary_hits
                for side in ('min', 'max')
            ):
                item_area.setBackground(QColor(255, 235, 130))
            self.peak_table.setItem(i, 3, item_area)

            # 4. 高度 (Height) - 拟合前NaN
            height_value = peak.height if peak.height is not None else peak.height_guess
            if height_value is not None:
                val_str = f"{height_value:.2f}"
            else:
                val_str = "NaN"
            item_h = QTableWidgetItem(val_str)
            item_h.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            item_h.setCheckState(Qt.Checked if peak.fixed_height else Qt.Unchecked)
            self.peak_table.setItem(i, 4, item_h)
            
            # 5. FWHM - 拟合结果优先，否则显示用户设置的初始猜测
            if peak.fwhm is not None:
                val_str = f"{peak.fwhm:.4f}"
            elif peak.sigma_guess is not None:
                val_str = f"{PSEUDO_VOIGT_FWHM_FACTOR * peak.sigma_guess:.4f}"
            else:
                val_str = "NaN"
            item_w = QTableWidgetItem(val_str)
            item_w.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            item_w.setCheckState(Qt.Checked if peak.fixed_fwhm else Qt.Unchecked)
            if any(
                f"p{peak.peak_id}_sigma:{side}" in boundary_hits
                for side in ('min', 'max')
            ):
                item_w.setBackground(QColor(255, 235, 130))
            fwhm_min, fwhm_max = Fitter.fwhm_bounds(peak.peak_type)
            item_w.setToolTip(
                f"允许范围：{fwhm_min:.2f}–{fwhm_max:.2f}°（2θ）。"
                "未勾选时作为拟合初值；勾选时精确固定为该值。"
            )
            self.peak_table.setItem(i, 5, item_w)

            # 6. Pseudo-Voigt Lorentzian比例η
            eta_value = peak.eta if peak.eta is not None else peak.fraction_guess
            item_eta = QTableWidgetItem(
                f"{eta_value:.4f}" if eta_value is not None else "NaN"
            )
            item_eta.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
            )
            item_eta.setToolTip("0=Gaussian，1=Lorentzian；冻结时作为完整峰形的一部分固定")
            if any(
                f"p{peak.peak_id}_fraction:{side}" in boundary_hits
                for side in ('min', 'max')
            ):
                item_eta.setBackground(QColor(255, 235, 130))
            self.peak_table.setItem(i, 6, item_eta)

            # 7. 本轮峰状态
            state_combo = QComboBox()
            state_combo.addItem("优化", "optimize")
            state_combo.addItem("冻结", "frozen")
            state_combo.addItem("禁用", "disabled")
            state_index = state_combo.findData(peak.fit_state)
            state_combo.setCurrentIndex(max(state_index, 0))
            state_combo.currentIndexChanged.connect(
                lambda _index, peak_id=peak.peak_id, combo=state_combo:
                self.set_peak_fit_state(peak_id, combo.currentData())
            )
            self.peak_table.setCellWidget(i, 7, state_combo)

            # 8. 类型
            self.peak_table.setItem(i, 8, QTableWidgetItem(peak.peak_type))
            
        self.peak_table.blockSignals(False)

    def on_peak_table_changed(self, item):
        """表格单元格修改事件"""
        if self.fitter is None:
            return
            
        row = item.row()
        col = item.column()
        
        # Get peak ID from first column
        peak_id_item = self.peak_table.item(row, 0)
        if not peak_id_item: return
        peak_id = int(peak_id_item.text())
        
        peak = next((p for p in self.fitter.peaks if p.peak_id == peak_id), None)
        if not peak: return
        
        # Sync CheckState first (locking)
        if col == 2:
            peak.fixed_center = (item.checkState() == Qt.Checked)
        elif col == 4:
            peak.fixed_height = (item.checkState() == Qt.Checked)
        elif col == 5:
            peak.fixed_fwhm = (item.checkState() == Qt.Checked)
            
        # Parse value for editing
        try:
            text = item.text().strip()
            if text == "NaN" or not text:
                if col in {2, 4, 5}:
                    self.invalidate_after_peak_edit()
                return
            val = float(text)
            
            if col == 2: # Center
                peak.center_guess = val
                # Update bounds
                if peak.bounds:
                    width = peak.bounds[1] - peak.bounds[0]
                    peak.bounds = (val - width/2, val + width/2)
                # Update center result too, so it persists
                peak.center = val
                
            elif col == 4: # Height
                peak.height_guess = val
                peak.height = val
                
            elif col == 5: # FWHM
                fwhm_min, fwhm_max = Fitter.fwhm_bounds(peak.peak_type)
                if not np.isfinite(val) or not fwhm_min <= val <= fwhm_max:
                    previous_fwhm = (
                        peak.fwhm
                        if peak.fwhm is not None
                        else PSEUDO_VOIGT_FWHM_FACTOR * peak.sigma_guess
                        if peak.sigma_guess is not None
                        else None
                    )
                    self.peak_table.blockSignals(True)
                    item.setText(
                        f"{previous_fwhm:.4f}"
                        if previous_fwhm is not None
                        else "NaN"
                    )
                    self.peak_table.blockSignals(False)
                    QMessageBox.warning(
                        self,
                        "FWHM超出范围",
                        f"{peak.peak_type}峰的FWHM必须在 "
                        f"{fwhm_min:.2f}–{fwhm_max:.2f}°（2θ）之间。",
                    )
                    return

                # lmfit PseudoVoigtModel精确定义FWHM = 2 * sigma。
                peak.sigma_guess = val / PSEUDO_VOIGT_FWHM_FACTOR
                peak.fwhm = val

            elif col == 6: # Pseudo-Voigt fraction
                if not np.isfinite(val) or not 0.0 <= val <= 1.0:
                    previous_eta = (
                        peak.eta
                        if peak.eta is not None
                        else peak.fraction_guess
                    )
                    self.peak_table.blockSignals(True)
                    item.setText(
                        f"{previous_eta:.4f}" if previous_eta is not None else "NaN"
                    )
                    self.peak_table.blockSignals(False)
                    QMessageBox.warning(
                        self,
                        "η超出范围",
                        "Pseudo-Voigt Lorentzian比例η必须在0和1之间。",
                    )
                    return
                peak.fraction_guess = val
                peak.eta = val

            if col in {2, 4, 5, 6}:
                self.invalidate_after_peak_edit()
                
        except ValueError:
            pass # Ignore invalid input

    def invalidate_after_peak_edit(self) -> None:
        """峰参数或锁定状态改变后，确保下一次拟合重建模型。"""
        if self.fitter is None:
            return
        self.fitter.invalidate_fit_state()
        self.clear_stale_fit_display()
        self.update_peak_table()
        self.redraw_peak_guesses()
        self.statusBar().showMessage("峰参数已修改，请重新拟合")

    def set_peak_fit_state(self, peak_id: int, state: str) -> None:
        """设置本轮峰状态；冻结要求已有完整且被接受的峰形参数。"""
        if self.fitter is None:
            return
        peak = next((p for p in self.fitter.peaks if p.peak_id == peak_id), None)
        if peak is None or state not in {'optimize', 'frozen', 'disabled'}:
            return

        if state == 'frozen' and any(
            value is None
            for value in (
                peak.area_guess,
                peak.sigma_guess,
                peak.fraction_guess,
            )
        ):
            QMessageBox.warning(
                self,
                "不能冻结峰形",
                "请先拟合并点击“接受当前结果作为下一轮初值”。",
            )
            self.update_peak_table()
            return

        if peak.fit_state == state:
            return
        peak.fit_state = state
        self.fitter.invalidate_fit_state()
        self.refresh_after_peak_change()
    
    def update_peak_input_mode(self) -> None:
        """切换数值添加峰时的输入语义和单位。"""
        if self.peak_input_mode.currentData() == 'd_spacing':
            self.peak_value_input.setRange(0.000001, 1000.0)
            self.peak_value_input.setSuffix(" Å")
            self.peak_value_input.setToolTip(
                "一级布拉格条件下的理论晶面间距d，不是晶格常数"
            )
        else:
            self.peak_value_input.setRange(0.000001, 179.999999)
            self.peak_value_input.setSuffix(" °")
            self.peak_value_input.setToolTip("衍射峰中心位置2θ，单位为度")

    def current_radiation_label(self) -> str:
        """根据项目波长返回用于结果溯源的辐射标签。"""
        if np.isclose(
            self.wavelength_spin.value(),
            DEFAULT_WAVELENGTH_ANGSTROM,
            rtol=0.0,
            atol=5e-7,
        ):
            return DEFAULT_RADIATION_LABEL
        return '自定义波长'

    def on_wavelength_changed(self) -> None:
        """同步项目波长到派生量和后续导出，不重新执行峰拟合。"""
        if self.reporter is None:
            return
        self.reporter.wavelength_angstrom = self.wavelength_spin.value()
        self.reporter.radiation_label = self.current_radiation_label()
        self.display_physics_parameters()

    def quick_add_peak(self):
        """按直接2θ或理论晶面间距d添加峰。"""
        if self.x_data is None or self.y_data is None or len(self.x_data) == 0:
            QMessageBox.warning(self, "没有数据", "请先加载XRD数据")
            return

        try:
            input_value = self.peak_value_input.value()
            wavelength = self.wavelength_spin.value()
            input_mode = self.peak_input_mode.currentData()
            if input_mode == 'd_spacing':
                center = BraggGeometry.two_theta_from_d(
                    input_value,
                    wavelength,
                )
            else:
                center = input_value
                BraggGeometry.d_from_two_theta(center, wavelength)
        except ValueError as exc:
            QMessageBox.warning(self, "输入无效", str(exc))
            return

        data_min = float(np.min(self.x_data))
        data_max = float(np.max(self.x_data))
        if not data_min <= center <= data_max:
            QMessageBox.warning(
                self,
                "峰位超出数据范围",
                f"计算得到的峰位 2θ = {center:.6f}°，"
                f"不在当前数据范围 {data_min:.6f}–{data_max:.6f}° 内。",
            )
            return

        peak_type = self.peak_type_combo.currentData()
        name = self.peak_name_input.text().strip()
        bounds = (max(center - 0.5, data_min), min(center + 0.5, data_max))
        self.add_peak_to_fitter(center, bounds, peak_type, name)
        self.peak_name_input.clear()
        if input_mode == 'd_spacing':
            self.statusBar().showMessage(
                f"已添加峰: d = {input_value:.6f} Å, "
                f"λ = {wavelength:.6f} Å → 2θ = {center:.6f}° {name}"
            )
        else:
            self.statusBar().showMessage(f"已添加峰: 2θ = {center:.6f}° {name}")

    def import_peaks_from_file(self):
        """从文件导入峰列表"""
        # 确定初始目录: 程序所在目录/database
        program_dir = Path(__file__).parent.absolute()
        database_dir = program_dir / "database"
        start_dir = str(database_dir) if database_dir.exists() else str(program_dir)
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入峰列表", start_dir, "Text Files (*.txt);;All Files (*)"
        )
        
        if not file_path:
            return
            
        # 获取当前X轴范围
        x_min, x_max = -np.inf, np.inf
        if self.x_data is not None:
            x_min = self.x_data.min()
            x_max = self.x_data.max()
            
        try:
            count = 0
            skipped = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = [p.strip() for p in line.split('-')]
                    if len(parts) < 1:
                        continue
                        
                    try:
                        center = float(parts[0])
                        
                        # 检查范围
                        if center < x_min or center > x_max:
                            skipped += 1
                            continue
                        
                        peak_type = 'film'
                        if len(parts) >= 2:
                            t = parts[1].lower()
                            if 'sub' in t:
                                peak_type = 'substrate'
                        
                        name = ''
                        if len(parts) >= 3:
                            name = parts[2]
                            
                        self.add_peak_to_fitter(center, (center-0.5, center+0.5), peak_type, name)
                        count += 1
                    except ValueError:
                        continue
                        
            msg = f"成功导入 {count} 个峰"
            if skipped > 0:
                msg += f" (已忽略 {skipped} 个超出范围的峰)"
            self.statusBar().showMessage(msg)
            
        except Exception as e:
            QMessageBox.critical(self, "导入失败", f"错误: {str(e)}")

    def delete_selected_peak(self):
        """删除选中的峰"""
        if self.fitter is None:
            return

        if self.is_fitting():
            QMessageBox.warning(self, "拟合进行中", "请等待当前拟合结束后再删除峰")
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
        
        self.fitter.remove_peaks(peak_ids_to_delete)
        self.refresh_after_peak_change()
        self.statusBar().showMessage("峰已删除，请重新拟合")
    
    def clear_all_peaks(self):
        """清除所有峰"""
        if self.fitter is not None:
            self.fitter.clear_peaks()
            self.refresh_after_peak_change()
    
    def sync_table_to_peaks(self):
        """将表格中的锁定状态同步回Peak对象"""
        if self.fitter is None:
            return
            
        for i in range(self.peak_table.rowCount()):
            peak_id = int(self.peak_table.item(i, 0).text())
            # 找到对应的peak对象 (peak_id可能不等于索引，如果删除过)
            peak = next((p for p in self.fitter.peaks if p.peak_id == peak_id), None)
            if peak:
                # 位置、高度和FWHM的复选框分别位于2、4、5列
                peak.fixed_center = (self.peak_table.item(i, 2).checkState() == Qt.Checked)
                peak.fixed_height = (self.peak_table.item(i, 4).checkState() == Qt.Checked)
                peak.fixed_fwhm = (self.peak_table.item(i, 5).checkState() == Qt.Checked)

    def refine_fitting(self):
        """基于当前人工设置或已接受初值重新拟合。"""
        if self.fitter is None:
            QMessageBox.warning(self, "警告", "请先添加峰")
            return
            
        # 1. 同步锁定状态 (fallback)
        self.sync_table_to_peaks()
        
        # 候选结果不会自动覆盖初值；这里只使用人工输入或已接受的初值。
        
        # 2. 重新执行拟合
        self.statusBar().showMessage('正在优化拟合...')
        self.execute_fitting()

    def accept_current_fit(self) -> None:
        """人工接受当前候选结果，保存完整峰形为下一轮初值。"""
        if self.fitter is None or self.fitter.result is None:
            QMessageBox.warning(self, "没有候选结果", "请先完成一次拟合")
            return

        diagnostics = self.fitter.fit_diagnostics
        if not diagnostics.get('success', False):
            reply = QMessageBox.question(
                self,
                "结果未收敛",
                "求解器没有报告成功。仍然接受这个候选结果作为下一轮初值吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.fitter.accept_current_result()
        self.update_peak_table()
        self.statusBar().showMessage("已接受当前结果，可冻结峰形或继续优化")

    def execute_fitting(self):
        """执行拟合"""
        if self.fitter is None or len(self.fitter.peaks) == 0:
            QMessageBox.warning(self, "警告", "请先添加峰")
            return

        # Excel 项目中的 result 只用于复现显示；继续拟合时必须按当前 GUI 中的
        # 冻结/锁定状态重新建模，不能沿用恢复时为绘图构造的全固定参数模型。
        if self.fitter.result is not None and getattr(
            self.fitter.result, "restored", False
        ):
            self.fitter.invalidate_fit_state()
            
        # 确保锁定状态同步（即使是初次拟合，用户也可能先勾选了锁定）
        self.sync_table_to_peaks()
        
        try:
            include_ranges = self.parse_fit_ranges(self.include_ranges_input.text())
            exclude_ranges = self.parse_fit_ranges(self.exclude_ranges_input.text())
        except ValueError as exc:
            QMessageBox.warning(self, "拟合区间格式错误", str(exc))
            return

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建拟合线程
        log_weight = self.log_spin.value() / 100.0
        
        fixed_bg = self.bg_spin.value() if self.bg_fix_cb.isChecked() else None
        
        self.fit_thread = FittingThread(
            self.fitter,
            self.constrain_fwhm_cb.isChecked(),
            self.min_separation.value(),
            method=self.fit_method_combo.currentText(),
            objective_mode=self.objective_combo.currentData(),
            log_weight=log_weight,
            intensity_floor=self.log_floor_spin.value(),
            include_ranges=include_ranges,
            exclude_ranges=exclude_ranges,
            fixed_background=fixed_bg
        )
        
        self.fit_thread.progress.connect(self.progress_bar.setValue)
        self.fit_thread.finished.connect(self.on_fitting_finished)
        self.fit_thread.error.connect(self.on_fitting_error)
        
        self.set_peak_editing_enabled(False)
        self.fit_thread.start()
        self.statusBar().showMessage('正在拟合...')
    
    def on_fitting_finished(self, result):
        """拟合完成处理"""
        self.progress_bar.setVisible(False)
        self.set_peak_editing_enabled(True)
        
        # 更新背景显示 (如果未固定)
        if not self.bg_fix_cb.isChecked() and self.fitter.result is not None:
            if 'c' in self.fitter.result.params:
                bg_val = self.fitter.result.params['c'].value
                self.bg_spin.setValue(bg_val)
        
        # 创建Reporter
        self.reporter = Reporter(
            self.fitter, 
            x_original=self.x_data_original,  # 新增
            y_original=self.y_data_original,  # 新增
            wavelength_angstrom=self.wavelength_spin.value(),
            radiation_label=self.current_radiation_label(),
        )
        self.reporter.calculate_metrics()
        
        # 更新绘图
        self.plot_fitted_results()
        
        # 更新峰列表表格
        self.update_peak_table()
        
        # 显示结果
        self.display_results()
        
        # 计算物理参数
        self.display_physics_parameters()
        
        diagnostics = self.fitter.fit_diagnostics
        boundary_hits = diagnostics.get('boundary_hits', [])
        success = diagnostics.get('success', False)
        covariance_available = diagnostics.get('covariance_available', False)
        message = diagnostics.get('message', '')
        nfev = diagnostics.get('nfev', 0)

        details = [
            f"求解器消息：{message}",
            f"函数评估次数：{nfev}",
            f"R²_fit：{self.reporter.metrics['R_squared_fit']:.6f}",
            f"协方差：{'可用' if covariance_available else '不可用'}",
        ]
        if boundary_hits:
            details.append("边界命中：" + ", ".join(boundary_hits))
        fit_warnings = diagnostics.get('warnings', [])
        if fit_warnings:
            details.append("数值警告：" + " | ".join(fit_warnings))

        if success and covariance_available and not boundary_hits and not fit_warnings:
            self.statusBar().showMessage('拟合完成，结果尚未接受')
            QMessageBox.information(
                self,
                "候选拟合结果",
                "\n".join(details) + "\n\n请检查后点击“接受当前结果”。",
            )
        else:
            self.statusBar().showMessage('拟合结束，但结果需要检查')
            QMessageBox.warning(
                self,
                "候选结果需要检查",
                "\n".join(details) + "\n\n该结果不会自动成为下一轮初值。",
            )
    
    def on_fitting_error(self, error_msg):
        """拟合错误处理"""
        self.progress_bar.setVisible(False)
        self.set_peak_editing_enabled(True)
        QMessageBox.critical(self, "拟合错误", f"拟合过程中出错:\n{error_msg}")
        self.statusBar().showMessage('拟合失败')
    
    def plot_fitted_results(self):
        """绘制拟合结果"""
        if self.fitter.result is None:
            return
        
        # 清除并重绘
        self.plot_canvas.clear_axes()
        ax = self.plot_canvas.axes
        
        # 原始数据
        ax.scatter(self.fitter.x_data, self.fitter.y_data, 
                  s=15, alpha=0.5, label='Data', color='gray', zorder=1)
        if self.fitter.fit_mask is not None and not np.all(self.fitter.fit_mask):
            ax.scatter(
                self.fitter.x_data[self.fitter.fit_mask],
                self.fitter.y_data[self.fitter.fit_mask],
                s=18,
                alpha=0.75,
                facecolors='none',
                edgecolors='black',
                linewidths=0.5,
                label='本轮拟合点',
                zorder=2,
            )
        
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

        for index, (lower, upper) in enumerate(
            self.fitter.fit_config.get('exclude_ranges', [])
        ):
            ax.axvspan(
                lower,
                upper,
                color='gray',
                alpha=0.18,
                label='本轮排除区间' if index == 0 else None,
            )
        
        # 设置标签
        ax.set_xlabel('2θ (degree)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Intensity (a.u.)', fontsize=12, fontweight='bold')
        ax.set_title('XRD Pattern Fitting', fontsize=14, fontweight='bold')
        ax.set_yscale(self.plot_canvas.yscale)  # Keep consistent scale
        
        # 强制设置X轴范围为当前数据范围
        if self.fitter.x_data is not None:
            ax.set_xlim(self.fitter.x_data.min(), self.fitter.x_data.max())

        # 强制设置Y轴范围（基于原始数据，避免拟合曲线的拖尾影响Log显示）
        if self.plot_canvas.yscale == 'log' and self.fitter.y_data is not None:
            # 过滤出大于0的数据点
            y_pos = self.fitter.y_data[self.fitter.y_data > 0]
            if len(y_pos) > 0:
                y_min = np.min(y_pos)
                y_max = np.max(self.fitter.y_data)
                # 设置Y轴范围: 下限为最小正值的0.5倍，上限为最大值的2倍
                ax.set_ylim(y_min * 0.5, y_max * 2.0)
            
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加峰标记信息
        for peak in self.fitter.peaks:
            label = peak.name if peak.name else f'Peak {peak.peak_id}'
            # 使用拟合后的位置，如果还没拟合完则使用猜测值
            center = peak.center if peak.center is not None else peak.center_guess
            height = peak.height if peak.height is not None else 0
            
            # 在峰顶添加文字
            ax.annotate(label, 
                       xy=(center, height), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=9, color='darkblue', fontweight='bold')
        
        # 只显示本轮实际参与优化的数据点上的线性强度R²。
        r2_fit = self.reporter.metrics['R_squared_fit']
        text = f'R²_fit = {r2_fit:.6f}'
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

        diagnostics = self.fitter.fit_diagnostics
        if diagnostics:
            text += "【求解状态与本轮配置】\n"
            text += "-" * 40 + "\n"
            text += f"Success                  : {diagnostics.get('success')}\n"
            text += f"Message                  : {diagnostics.get('message')}\n"
            text += f"Function evaluations     : {diagnostics.get('nfev')}\n"
            text += (
                "Covariance available      : "
                f"{diagnostics.get('covariance_available')}\n"
            )
            text += f"Method                   : {diagnostics.get('method')}\n"
            text += f"Objective                : {diagnostics.get('objective_mode')}\n"
            text += f"Log weight               : {diagnostics.get('log_weight')}\n"
            text += f"Log intensity floor I0   : {diagnostics.get('intensity_floor')}\n"
            text += f"Fit points               : {diagnostics.get('fit_point_count')}\n"
            hits = diagnostics.get('boundary_hits', [])
            text += f"Boundary hits            : {', '.join(hits) if hits else 'None'}\n\n"
            fit_warnings = diagnostics.get('warnings', [])
            text += f"Warnings                 : {' | '.join(fit_warnings) if fit_warnings else 'None'}\n\n"
        
        # 评估指标
        text += "【拟合质量评估】\n"
        text += "-" * 40 + "\n"
        for key, value in self.reporter.metrics.items():
            if isinstance(value, (int, float, np.number)):
                text += f"{key:25s}: {value:.6e}\n"
            else:
                text += f"{key:25s}: {value}\n"
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
        text += self.fitter.get_fit_report()
        
        self.results_text.setPlainText(text)
    
    def display_physics_parameters(self):
        """显示物理参数"""
        if self.reporter is None:
            return
        
        text = "=" * 60 + "\n"
        text += "物理参数计算\n"
        text += "=" * 60 + "\n\n"
        
        characteristic_lengths = self.reporter.calculate_characteristic_lengths()
        wavelength = self.reporter.wavelength_angstrom
        radiation_label = self.reporter.radiation_label
        
        text += (
            "【反射峰特征长度 d】"
            f"(λ = {wavelength:.6f} Å, {radiation_label})\n"
        )
        text += "-" * 40 + "\n"
        text += "说明：该值是对应反射峰的Bragg d间距，不等同于晶格常数。\n"
        text += "程序不根据002、004、111、200等峰名自动推断晶格倍数。\n"

        for value in characteristic_lengths.values():
            reflection_label = value['reflection_label'] or '未指定'
            text += f"\nPeak {value['peak_id']} ({reflection_label}):\n"
            text += f"  峰位 2θ             : {value['2theta_deg']:.6f}°\n"
            text += (
                "  特征长度 d           : "
                f"{value['characteristic_length_angstrom']:.6f} Å\n"
            )
        
        # 未校正仪器展宽时，只报告表观Scherrer相干畴尺寸。
        text += "\n【表观Scherrer相干畴尺寸估算】\n"
        text += "-" * 40 + "\n"
        text += "D = Kλ / (β·cosθ)\n"
        text += "K = 0.9；β为拟合FWHM，未作仪器展宽修正\n\n"
        
        for peak in self.fitter.peaks:
            if peak.fwhm is not None and peak.center is not None:
                D_nm = BraggGeometry.apparent_scherrer_size_nm(
                    peak.center,
                    peak.fwhm,
                    wavelength,
                    shape_factor=0.9,
                )
                
                text += f"Peak {peak.peak_id} ({peak.peak_type}):\n"
                text += f"  表观相干畴尺寸: {D_nm:.2f} nm\n"
        
        # 应力/应变分析
        text += "\n【应力分析提示】\n"
        text += "-" * 40 + "\n"
        text += "• 特征长度只表示当前反射的d间距，不直接代表晶格常数\n"
        text += "• 晶相或应变判断需要明确的(hkl)指认和独立参考峰位\n"
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
            
        # 使用上次的数据文件夹作为起点
        start_dir = self.last_data_dir if self.last_data_dir else ""
        default_path = os.path.join(start_dir, default_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存Excel报告", default_path, "Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                self.reporter.export_results(
                    file_path,
                    project_state=self.collect_project_state(),
                    source_files=[entry[0] for entry in self.loaded_files_data],
                    source_datasets=self.loaded_files_data,
                )
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
            
        # 使用上次的数据文件夹作为起点
        start_dir = self.last_data_dir if self.last_data_dir else ""
        default_path = os.path.join(start_dir, default_name)
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default_path, 
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
