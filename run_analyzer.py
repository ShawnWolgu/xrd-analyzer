# run_analyzer.py - 启动脚本

"""
XRD数据分析系统 - 启动脚本
专为PZT薄膜002/200峰分裂分析设计

使用方法:
    python run_analyzer.py

依赖库:
    - PyQt5
    - numpy
    - scipy
    - matplotlib
    - pandas
    - lmfit
    - openpyxl
"""

import sys
import os

# 检查依赖
required_packages = {
    'PyQt5': 'PyQt5',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'pandas': 'pandas',
    'lmfit': 'lmfit',
    'openpyxl': 'openpyxl'
}

missing_packages = []
for module_name, package_name in required_packages.items():
    try:
        __import__(module_name)
    except ImportError:
        missing_packages.append(package_name)

if missing_packages:
    print("=" * 60)
    print("缺少必要的Python包！")
    print("=" * 60)
    print("\n请运行以下命令安装：")
    print(f"\npip install {' '.join(missing_packages)}")
    print("\n或者：")
    print(f"\nconda install {' '.join(missing_packages)}")
    print("\n" + "=" * 60)
    sys.exit(1)

# 导入GUI
from xrd_gui import main

if __name__ == '__main__':
    print("=" * 60)
    print("XRD数据分析系统 v1.0")
    print("=" * 60)
    print("\n专为PZT薄膜分析设计")
    print("支持002/200峰分裂拟合")
    print("包含晶格常数与四方度计算")
    print("\n启动GUI...")
    print("=" * 60 + "\n")
    
    main()

