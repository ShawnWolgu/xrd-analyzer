# AGENTS.md - Instructions for Agentic Coding Agents

This document provides context, conventions, and instructions for AI agents working on the XRD Analyzer codebase.

## 1. Project Overview
This project is an XRD (X-ray Diffraction) analysis tool built with Python. It includes core analysis logic (`xrd_analyzer.py`), a GUI (`xrd_gui.py`), and utility scripts for data processing and visualization.

## 2. Environment & Dependencies
- **Language**: Python 3.8+
- **Dependencies**: Listed in `requirements.txt`. Key libraries:
  - `numpy`, `scipy`, `pandas`: Data manipulation and numerical operations.
  - `matplotlib`: Plotting and visualization.
  - `lmfit`: Non-linear least-squares minimization and curve fitting.
  - `openpyxl`: Excel file I/O.
  - `PyQt5`: GUI framework.
- **Setup Command**:
  ```bash
  pip install -r requirements.txt
  ```

## 3. Build, Lint, and Test Commands

### Running Tests
There is no formal test suite (like `pytest`) configured yet. Verification is done by running example scripts:
- **Core Logic Verification**:
  ```bash
  python test.py
  ```
- **Plotting Verification**:
  ```bash
  python plot_test.py
  ```
- **GUI Verification**:
  ```bash
  python xrd_gui.py
  ```

*Agent Note*: When adding new features, prefer creating a dedicated test script or adding to `test.py` to verify functionality. If a formal test runner is needed, suggest `pytest`.

### Linting & Formatting
No strict linters are currently enforced, but the following are recommended standards:
- **Linting**: `flake8`
  ```bash
  flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
  ```
- **Formatting**: `black` style is preferred but not enforced.
- **Imports**: `isort` compatible sorting (Standard -> Third-party -> Local).

## 4. Code Style & Conventions

### General
- **Style Guide**: Follow PEP 8.
- **Docstrings**: **MUST use Chinese** for docstrings and comments, as per existing files (e.g., `xrd_analyzer.py`).
- **Type Hinting**: **REQUIRED** for all new function definitions. Use `typing` module (`List`, `Dict`, `Tuple`, `Optional`).
  ```python
  def calculate_area(self, x_data: np.ndarray, y_data: np.ndarray) -> float:
      ...
  ```

### File Structure & Naming
- **File Names**: `snake_case.py` (e.g., `xrd_analyzer.py`, `plot_from_excel.py`).
- **Class Names**: `PascalCase` (e.g., `DataLoader`, `Fitter`, `Peak`).
- **Function/Variable Names**: `snake_case` (e.g., `load_txt`, `center_guess`).
- **Constants**: `UPPER_CASE` (if any).

### Architecture & Patterns
- **Core Logic (`xrd_analyzer.py`)**:
  - Organized into classes: `Peak` (data structure), `DataLoader` (I/O), `Preprocessor` (filtering/background), `Fitter` (engine), `Reporter` (export/stats).
  - Uses static methods for utility classes (`DataLoader`, `Preprocessor`).
- **Fitting**: Uses `lmfit` models. `PseudoVoigtModel` is the standard peak shape.
- **Data Handling**:
  - Use `pathlib.Path` for file paths.
  - Use `numpy.ndarray` for signal data.
  - Use `pandas.DataFrame` for tabular data (results export).

### Error Handling
- Use `try...except` blocks for I/O operations and parsing.
- Fail gracefully with informative error messages.
- Use `warnings.warn` for non-critical issues.

### Plotting
- Use `matplotlib.pyplot` and object-oriented interface (`fig, ax = plt.subplots(...)`).
- Ensure plots are saved with high DPI (e.g., `dpi=300`) when exporting.

## 5. Specific Rules & Constraints
- **GUI**: Any changes to `xrd_gui.py` must maintain compatibility with PyQt5. Ensure signals and slots are correctly connected.
- **Excel Export**: Use `openpyxl` engine for `pandas` to write Excel files. Maintain the multi-sheet structure (`Peak_Parameters`, `Fit_Metrics`, etc.).
- **Math**: Use `numpy` for vector operations. Avoid explicit loops for math calculations where possible.

## 6. Example Code Snippet
```python
import numpy as np
from typing import Tuple

def process_data(data: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, int]:
    """
    处理数据并返回筛选后的结果 (Process data and return filtered results)
    
    Args:
        data: 输入数据数组
        threshold: 阈值
        
    Returns:
        处理后的数组和计数
    """
    mask = data > threshold
    filtered_data = data[mask]
    return filtered_data, len(filtered_data)
```
