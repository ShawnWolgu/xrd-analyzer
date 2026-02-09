# AGENTS.md - Instructions for Agentic Coding Agents

This document provides context, conventions, and instructions for AI agents working on the XRD Analyzer codebase.

## 1. Project Overview
This project is an XRD (X-ray Diffraction) analysis tool built with Python. It includes:
- **Core Logic**: `xrd_analyzer.py` (Data loading, preprocessing, fitting, reporting).
- **GUI**: `xrd_gui.py` (PyQt5-based interface).
- **Utility Scripts**: `plot_from_excel.py`, `convert_csv_to_txt.py`.

The system is designed for analyzing PZT thin films on substrates, handling peak fitting (Pseudo-Voigt), and exporting results to Excel and high-quality plots.

## 2. Environment & Dependencies

### Setup
- **Language**: Python 3.8+
- **Virtual Environment**: Recommended.
- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```

### Key Libraries
- **`numpy`, `scipy`, `pandas`**: Core numerical and data processing.
- **`lmfit`**: Non-linear least-squares minimization.
- **`matplotlib`**: Plotting (Qt5Agg backend for GUI).
- **`PyQt5`**: Graphical User Interface.
- **`openpyxl`**: Excel file I/O.

## 3. Build, Lint, and Test Commands

### Running Tests
There is no formal `pytest` suite. Verification is performed by running specific scripts.
*Note: Ensure you have sample data (e.g., in `database/`) before running tests.*

1.  **Core Logic Verification**:
    Runs the analyzer on a sample file (modify script to point to actual data).
    ```bash
    python test.py
    ```

2.  **GUI Verification**:
    Launches the main application window.
    ```bash
    python xrd_gui.py
    ```

3.  **Plotting Verification**:
    Tests the plotting logic independently.
    ```bash
    python plot_test.py
    ```

*Agent Instruction*: When modifying core logic, run `test.py`. When modifying UI, run `xrd_gui.py` to ensure no crashes on startup.

### Linting & Formatting
Follow these standards to maintain code quality:

1.  **Linting** (Recommended):
    ```bash
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    ```

2.  **Formatting**:
    - Style: PEP 8.
    - Tool: `black` (preferred but not strictly enforced).
    - Import Sorting: `isort` style (Standard Library -> Third Party -> Local Application).

## 4. Code Style & Conventions

### Language & Documentation
- **Docstrings**: **MUST use Chinese** (Mandarin) for all docstrings and comments.
  - Example: `""" 计算峰面积 (Calculate peak area) """`
- **Comments**: Sparse, focusing on *why*, not *what*. Use Chinese.

### Type Hinting
- **REQUIRED** for all new function definitions.
- Use `typing` module (`List`, `Dict`, `Tuple`, `Optional`).
```python
def calculate_metrics(self, data: np.ndarray) -> Dict[str, float]:
    ...
```

### Naming Conventions
- **Files**: `snake_case.py` (e.g., `xrd_analyzer.py`)
- **Classes**: `PascalCase` (e.g., `DataLoader`, `Peak`)
- **Functions/Variables**: `snake_case` (e.g., `load_txt`, `center_guess`)
- **Constants**: `UPPER_CASE` (e.g., `DEFAULT_DPI = 300`)

### Architecture & Patterns
- **Classes**:
  - `Peak`: Data structure for single peak parameters.
  - `DataLoader`: Static methods for file I/O.
  - `Preprocessor`: Static methods for filtering/background subtraction.
  - `Fitter`: Main engine wrapping `lmfit`.
  - `Reporter`: Handles stats calculation and export.
- **Data Structures**:
  - Use `numpy.ndarray` for spectral data.
  - Use `pandas.DataFrame` for organizing results.
- **Path Handling**: Always use `pathlib.Path`.

### Error Handling
- Use `try...except` for file I/O and parsing.
- Raise informative `ValueError` or `RuntimeError` for logical failures.
- Use `warnings.warn` for non-critical data issues (e.g., missing columns).

## 5. Specific Rules & Constraints

### GUI Development (`xrd_gui.py`)
- **Framework**: PyQt5.
- **Threading**: Use `QThread` for long-running tasks (fitting) to avoid freezing the UI.
- **Signals**: Define custom signals (`pyqtSignal`) for component communication.
- **Matplotlib Integration**: Use `FigureCanvasQTAgg`.

### Data Analysis (`xrd_analyzer.py`)
- **Fitting**: Use `lmfit.Model` with `PseudoVoigtModel`.
- **Math**: Vectorize operations with `numpy`. Avoid `for` loops over data arrays.
- **Excel**: Use `openpyxl` engine. Maintain multi-sheet structure (`Peak_Parameters`, `Structure_Analysis`, `Fit_Metrics`).

### File Operations
- **Absolute Paths**: Always resolve paths to absolute before using them.
- **Safety**: Do not overwrite input files. Create new output files with clear suffixes (e.g., `_results.xlsx`).

## 6. Example Code Snippet

```python
import numpy as np
from typing import Tuple, Optional

class DataProcessor:
    """数据处理类 (Data Processor Class)"""

    @staticmethod
    def smooth_signal(
        data: np.ndarray, 
        window_length: int = 7, 
        polyorder: int = 2
    ) -> np.ndarray:
        """
        对信号应用Savitzky-Golay平滑滤波
        
        Args:
            data: 输入信号数组
            window_length: 窗口长度 (必须为奇数)
            polyorder: 多项式阶数
            
        Returns:
            平滑后的信号数组
        """
        if window_length % 2 == 0:
            window_length += 1
            
        try:
            from scipy.signal import savgol_filter
            return savgol_filter(data, window_length, polyorder)
        except Exception as e:
            print(f"平滑处理失败: {e}")
            return data
```

## 7. Cursor/Copilot Rules
- **Proactiveness**: If a user asks for a feature, implement it fully including UI controls and backend logic.
- **Safety**: Always warn before deleting files.
- **Context**: Read `xrd_analyzer.py` before making changes to `xrd_gui.py` to understand the API.
