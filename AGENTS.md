# AGENTS.md - Instructions for Agentic Coding Agents

This document provides context, conventions, and instructions for AI agents working on the XRD Analyzer codebase.

## 1. Project Overview
This project is an **XRD (X-ray Diffraction) Analysis Tool** built with Python.
- **Core Logic**: `xrd_analyzer.py` (Data loading, preprocessing, fitting, reporting).
- **GUI**: `xrd_gui.py` (PyQt5-based interface).
- **Goal**: Analyze PZT thin films, perform Pseudo-Voigt peak fitting, and export results to Excel/Plots.

## 2. Environment & Dependencies
- **Language**: Python 3.8+
- **Install Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
- **Key Libraries**:
  - `numpy`, `pandas`, `scipy` (Data processing)
  - `lmfit` (Non-linear least-squares fitting)
  - `PyQt5` (GUI)
  - `matplotlib` (Plotting)
  - `openpyxl` (Excel I/O)

## 3. Build, Lint, and Test Commands

### Running Tests
The project uses **script-based verification** rather than a formal test runner like `pytest`.
*Agent Instruction: Always verify core logic before GUI changes.*

1.  **Run All Core Logic Tests**:
    Executes the full analysis pipeline on sample data.
    ```bash
    python test.py
    ```

2.  **Run a "Single Test" (Component Verification)**:
    Since there are no individual test units, verify specific components by creating a temporary script or using a one-liner.
    *Example: Testing the Peak Area calculation*
    ```bash
    python -c "from xrd_analyzer import Peak; p=Peak(44.0, 1000, 0.5, 'Test'); print(f'Area: {p.area}')"
    ```
    *Example: Testing Plotting independently*
    ```bash
    python plot_test.py
    ```

3.  **Verify GUI Startup**:
    Launches the main application to ensure no crashes.
    ```bash
    python xrd_gui.py
    ```

### Linting & Formatting
Maintain high code quality using these standard commands:

1.  **Linting**:
    ```bash
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    ```

2.  **Formatting**:
    - **Style**: PEP 8.
    - **Tool**: `black` (preferred).
    ```bash
    black .
    ```

## 4. Code Style Guidelines

### Language & Documentation
- **Docstrings**: **MUST use Chinese (Mandarin)** for all docstrings.
  - Format: Google or NumPy style.
  - Example: `""" 计算峰的积分面积 (Calculate integrated peak area) """`
- **Comments**: Sparse, focusing on *why* (physics/logic), not *what*. Use Chinese.

### Imports
Group imports in the following order:
1.  **Standard Library** (`os`, `sys`, `typing`)
2.  **Third-Party** (`numpy`, `pandas`, `PyQt5`, `lmfit`)
3.  **Local Application** (`xrd_analyzer`)

### Type Hinting
- **MANDATORY** for all new function definitions.
- Use `typing` module generics (`List`, `Dict`, `Tuple`, `Optional`).
```python
def fit_spectrum(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    ...
```

### Naming Conventions
- **Files**: `snake_case.py` (e.g., `xrd_analyzer.py`)
- **Classes**: `PascalCase` (e.g., `PeakFitter`, `MainWindow`)
- **Functions/Variables**: `snake_case` (e.g., `load_data`, `calculate_fwhm`)
- **Constants**: `UPPER_CASE` (e.g., `MAX_ITERATIONS = 100`)
- **Private Members**: Prefix with `_` (e.g., `_update_plot`)

### Error Handling
- **File I/O**: Always wrap in `try...except` blocks.
- **Fitting**: Catch `ValueError` or `RuntimeError` from `lmfit`.
- **GUI**: Use `QMessageBox` to display errors to the user (do not just print to console).
- **Exceptions**: Raise descriptive exceptions with context.

## 5. Architecture & Patterns

### Core Logic (`xrd_analyzer.py`)
- **Vectorization**: Use `numpy` array operations. Avoid explicit loops over data points.
- **Model**: Use `lmfit.models.PseudoVoigtModel` for peak fitting.
- **Data Class**: `Peak` class encapsulates single peak parameters (center, height, fwhm).
- **Calculation Rules**:
  - **Height**: `model.eval(x=center)` (Intensity at peak center).
  - **Area**: `amplitude` parameter from `lmfit` (Net Area). Do not use integration.

### GUI (`xrd_gui.py`)
- **Pattern**: Signal-Slot mechanism.
- **Threading**: Heavy computations (fitting) **MUST** run in a `QThread` to keep the UI responsive.
- **State**: The GUI should track the current `Fitter` instance and `file_path`.

## 6. Rules for Agents (Cursor/Copilot)

### Proactiveness
- If a user asks for a feature, implement both the backend logic and the UI controls.
- If a bug is reported, create a reproduction script (`repro.py`) if possible before fixing.

### Safety & Integrity
- **File Safety**: Never overwrite input raw data files (`.txt`, `.csv`).
- **Output**: Always append suffixes to output files (e.g., `_results.xlsx`).
- **Deletion**: Explicitly ask for confirmation before deleting any file.
- **Context**: Read `xrd_analyzer.py` fully before modifying `xrd_gui.py` to ensure API compatibility.

### Example Snippet
```python
import numpy as np
from typing import List, Optional

class DataProcessor:
    """数据处理类 (Data Processor)"""

    def smooth_data(self, data: np.ndarray, window: int = 5) -> np.ndarray:
        """
        对数据进行平滑处理 (Smooth the data)
        
        Args:
            data: 输入数组
            window: 窗口大小
            
        Returns:
            平滑后的数组
        """
        if window % 2 == 0:
            window += 1
        try:
            from scipy.signal import savgol_filter
            return savgol_filter(data, window, 3)
        except ImportError:
            # Fallback or error
            return data
```
