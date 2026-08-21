# run_analyzer.py - 启动脚本

"""
XRD数据分析系统 - 启动脚本
用于PZT薄膜多反射峰拟合与特征长度分析

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

from main import main

if __name__ == '__main__':
    print("=" * 60)
    print("XRD数据分析系统 v1.0")
    print("=" * 60)
    print("\n用于PZT薄膜多反射峰分析")
    print("支持自定义反射峰拟合")
    print("包含反射峰特征长度（Bragg d间距）与峰宽分析")
    print("\n启动GUI...")
    print("=" * 60 + "\n")
    
    raise SystemExit(main())
