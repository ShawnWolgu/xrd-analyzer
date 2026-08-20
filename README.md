# XRD Analyzer

XRD Analyzer is an experimental Python desktop application for loading X-ray diffraction
scans, preprocessing intensity data, fitting Pseudo-Voigt peaks, and exporting fit results and
derived quantities for PZT thin-film analysis.

The repository is currently in a foundation refactor. The existing GUI remains usable, but
legacy numerical output must not yet be treated as independently validated scientific ground
truth. See [Scientific validation](docs/scientific-validation.md) before using derived values in
a publication or formal report.

The program reports the direct Bragg `d` spacing as a reflection-specific characteristic
length. Peak labels such as 002, 004, 111, 200, 222, or 400 do not trigger lattice-constant
multipliers or tetragonality inference.

## Current capabilities

- Load two-column text scans and combine multiple scan ranges.
- Apply the data-loading 2theta range as a permanent project-wide crop; out-of-range data and
  peaks are removed from the live plot, fitting input, workbook export, and figure export.
- Trim, smooth, and subtract polynomial or SNIP backgrounds.
- Add peaks manually or through automatic peak detection.
- Add a numerical peak either from its `2theta` position or by converting a theoretical
  interplanar spacing `d` with the first-order Bragg relation.
- Fit constant-background plus Pseudo-Voigt peak models with `lmfit`.
- Select Linear, Log, or Mixed objectives with a visible Log intensity floor `I0`.
- Manually include or exclude 2theta ranges without modifying the loaded scan.
- Keep each peak in optimize, frozen-complete-shape, or disabled state.
- Review solver convergence and boundary diagnostics before accepting a result as the next
  fitting guess.
- Display `R²_fit` only on the points that actually enter the current optimization, in linear
  intensity space; do not mix Linear, Log, and Mixed objectives into an ambiguously labeled
  RMSE.
- Run numerical fitting in a protected background thread with an explicit native stack and
  single-threaded BLAS section, preventing OpenBLAS covariance inversion from exhausting the
  macOS Qt worker-thread stack.
- Export workbooks that can be loaded again as project files, restoring data, peak guesses,
  fitted values, locks, peak states, fit controls, and the displayed candidate result.
- Export fitting figures.
- Display reflection-specific Bragg characteristic lengths and provisional Scherrer-related
  quantities in the GUI.

## Install

Use Python 3.10 or newer in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The historical `requirements.txt` remains available for runtime-only installation. Project
metadata and development dependencies are defined in `pyproject.toml`.

## Run

```bash
python run_analyzer.py
```

After editable installation, the `xrd-analyzer` command is also available.

The fitting workflow is intentionally user-controlled:

1. Choose the objective, optimizer, background, and optional fit ranges.
2. Set each peak to `优化`, `冻结`, or `禁用`.
3. Run the fit and inspect convergence, covariance, boundary hits, and warnings.
4. Use `接受当前结果作为下一轮初值` only after judging the candidate result.
5. Freeze an accepted complete peak shape when fitting other components manually.

The two kinds of ranges have different meanings:

- `数据加载与合并 → 应用范围` permanently crops the project data. Save or export before
  applying it if the discarded points may be needed later.
- The fit include/exclude fields only change which retained points enter the objective; they do
  not delete scan data.

An exported `.xlsx` report is also a project snapshot. Use `加载Excel项目` to restore it. New
reports preserve the complete session state; older reports that contain `Full_Data` and
`Peak_Parameters` can be opened on a best-effort basis when those sheets contain enough data.

The project wavelength is editable in the peak-management panel. Its default is
`Cu K-alpha-1 = 1.5406 angstrom`. The same explicit wavelength is used for inverse peak
placement, reported characteristic lengths, apparent Scherrer coherent-domain-size estimates,
and the Excel project snapshot. For a theoretical spacing, the program applies the first-order
relation `2theta = 2 asin(wavelength / (2d))`; it does not infer lattice-constant multipliers
from the peak name.

For Log and Mixed objectives the residual is
`log10(I_observed + I0) - log10(I_calculated + I0)`. The program does not choose a scientifically
correct `I0` automatically. Log fitting is rejected when the selected processed data contain
negative intensity.

To redraw an exported workbook:

```bash
xrd-plot-results path/to/results.xlsx --summary
```

## Verify

```bash
python -m pytest
python -m ruff check .
```

The suite includes strict expected-failure tests for confirmed legacy scientific issues. An
`XFAIL` records an unresolved problem; it is not a successful scientific validation.

## Repository map

- `xrd_analyzer.py`: legacy core loading, preprocessing, fitting, metrics, and reporting.
- `xrd_gui.py`: legacy PyQt5 interface and application state.
- `plot_from_excel.py`: plotting and summaries from exported workbooks.
- `tests/`: automated regression and scientific-contract tests.
- `docs/architecture.md`: current boundaries and incremental target structure.
- `docs/scientific-validation.md`: evidence status and scientific correction backlog.
- `.agents/skills/xrd-scientific-maintenance/`: repository-specific Codex workflow.

The planned package extraction is intentionally incremental. It will preserve a compatibility
surface while moving formulas and state out of GUI callbacks.

## Data and generated artifacts

The tracked scan, workbook, and figures are historical examples from the migrated project.
They may be used as regression artifacts, but they are not certified reference data. Never
overwrite an input scan during analysis.

## License

A distribution license has not yet been selected. Until one is added, no open-source license is
granted by this repository.
