# XRD Analyzer

Current release: **v1.0.0**

XRD Analyzer is a general-purpose Python desktop application for loading X-ray diffraction
scans, preprocessing intensity data, fitting Pseudo-Voigt peaks, restoring analysis projects,
and exporting fitted and derived quantities.

Version 1.0 establishes the maintained application baseline. Passing regression tests protect
the documented software behavior, but numerical output must not be treated as independently
validated scientific ground truth. See [Scientific validation](docs/scientific-validation.md)
before using derived values in a publication or formal report.

The program reports the direct Bragg `d` spacing as a reflection-specific characteristic
length. Peak labels such as 002, 004, 111, 200, 222, or 400 do not trigger lattice-constant
multipliers or tetragonality inference.

## Current capabilities

- Load two-column text scans and combine multiple scan ranges.
- Apply the data-loading 2theta range as a permanent project-wide crop; out-of-range data and
  peaks are removed from the live plot, fitting input, workbook export, and figure export.
- Trim, smooth, and subtract polynomial or SNIP backgrounds.
- Recompute preprocessing from an immutable raw scan and an explicit ordered operation list;
  applying the same controls twice no longer compounds filtering.
- Enforce a one-way frontend/backend dependency boundary: the PyQt frontend imports XRD
  behavior only through `xrd_backend.py`, while the backend dependency closure contains no PyQt.
- Add peaks manually by numerical input or by clicking the plotted scan.
- Add a numerical peak either from its `2theta` position or by converting a theoretical
  interplanar spacing `d` with the first-order Bragg relation.
- Import reusable TXT peak lists, or export active fitted peak positions to the same project
  `database/` directory in a round-trip-compatible format.
- Shift every configured peak and its search bounds by a common, explicit `2theta` offset using
  step buttons or a centered slider; shifting invalidates the previous fitted result.
- Fit constant-background plus Pseudo-Voigt peak models with `lmfit`.
- Select Linear, Log, or Mixed objectives with a visible Log intensity floor `I0`.
- Manually include or exclude 2theta ranges without modifying the loaded scan.
- Keep each peak in optimize, frozen-complete-shape, or disabled state.
- Undo and redo among the five most recent completed fitting results, restoring both table values
  and displayed total/component curves without recording intermediate configuration edits.
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
python main.py
```

After editable installation, the `xrd-analyzer` command is also available.
The historical `python run_analyzer.py` command remains as a compatibility wrapper.

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

Run the versioned core baseline gate separately with:

```bash
python -m pytest -m baseline
```

See [Core baseline testing](docs/baseline-testing.md) for evidence levels and baseline-update
rules.

The scientific fitting pipeline, Pseudo-Voigt equations, parameter definitions, objectives,
and result-interpretation boundaries are summarized in
[XRD fitting model](docs/fitting-model.md).

The suite contains regression tests for corrected legacy scientific issues. A passing suite
only verifies the documented contracts; it does not turn historical example data into certified
scientific ground truth.

## Repository map

- `xrd_analyzer.py`: compatibility facade plus the remaining fitting and reporting engine.
- `main.py`: the single application entry point and GUI startup orchestration.
- `xrd_gui.py`: legacy PyQt5 interface and application state.
- `xrd_backend.py`: the only public backend entrypoint used by the frontend, including
  `XRDApplicationService`.
- `xrd_session.py`: immutable scans, source scans, preprocessing provenance, and fit configuration.
- `xrd_io.py`: two-column scan loading, cropping, and multi-scan stitching.
- `xrd_preprocessing.py`: pure filtering and background transformations.
- `xrd_peaks.py`: peak guesses, locks, states, and fitted parameter semantics.
- `xrd_crystallography.py`: Bragg characteristic-length and apparent Scherrer calculations.
- `xrd_project.py`: versioned Excel project loading and restored-result compatibility.
- `plot_from_excel.py`: plotting and summaries from exported workbooks.
- `tests/`: automated regression and scientific-contract tests.
- `tests/baselines/`: versioned synthetic-reference and historical-regression specifications.
- `docs/architecture.md`: current boundaries and incremental target structure.
- `docs/fitting-model.md`: scientific fitting pipeline, equations, variables, and interpretation.
- `docs/scientific-validation.md`: evidence status and scientific correction backlog.
- `.agents/skills/xrd-scientific-maintenance/`: repository-specific Codex workflow.

The package extraction is incremental. Existing imports remain compatible while state and
scientific operations move out of GUI callbacks. Workbook schema version 3 records ordered
preprocessing steps, structured fit configuration, scan hashes, and retained-point counts.

## Data and generated artifacts

The tracked scan, workbook, and figures are historical examples from the migrated project.
They may be used as regression artifacts, but they are not certified reference data. Never
overwrite an input scan during analysis.

## License

A distribution license has not yet been selected. Until one is added, no open-source license is
granted by this repository.
