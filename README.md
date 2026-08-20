# XRD Analyzer

XRD Analyzer is an experimental Python desktop application for loading X-ray diffraction
scans, preprocessing intensity data, fitting Pseudo-Voigt peaks, and exporting fit results and
derived quantities for PZT thin-film analysis.

The repository is currently in a foundation refactor. The existing GUI remains usable, but
legacy numerical output must not yet be treated as independently validated scientific ground
truth. See [Scientific validation](docs/scientific-validation.md) before using derived values in
a publication or formal report.

## Current capabilities

- Load two-column text scans and combine multiple scan ranges.
- Trim, smooth, and subtract polynomial or SNIP backgrounds.
- Add peaks manually or through automatic peak detection.
- Fit constant-background plus Pseudo-Voigt peak models with `lmfit`.
- Export workbooks and fitting figures.
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
