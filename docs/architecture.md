# Architecture

## Current package layout

```text
main.py                         source-checkout launcher
src/xrd_analyzer/
  application.py               dependency checks and QApplication lifecycle
  gui.py                       PyQt widgets, interaction, rendering, fit thread
  backend.py                   public application-service boundary for the GUI
  engine.py                    fitting and reporting engine
  session.py                   immutable scan and analysis configuration
  io.py                        scan loading, cropping, and merging
  preprocessing.py             pure intensity transformations and provenance
  peaks.py                     peak state and fitted-parameter semantics
  crystallography.py           diffraction geometry and derived quantities
  project.py                   Excel project restoration
  i18n.py                      UI translations
  tools/                       optional conversion and workbook plotting tools
examples/                      API usage examples
tests/                         scientific, regression, GUI, and architecture tests
```

The installed `xrd-analyzer` command and `python -m xrd_analyzer` route to
`xrd_analyzer.application`. Root `main.py` remains the only application Python file at the
repository top level and delegates to the same startup function.

## Dependency boundary

```text
GUI -> backend/application service -> session and scientific core -> NumPy/SciPy/lmfit
                                \-> project and export adapters
```

- `gui.py` imports scientific behavior only from `.backend`.
- The backend dependency closure must not import PyQt.
- Scientific formulas stay in the core, not GUI callbacks.
- Export code consumes fitted result state rather than recomputing alternate physics.

These rules and the root-layout contract are enforced by
`tests/test_architecture_boundaries.py`.

## Data and state rules

- Keep loaded raw scans separate from processed scans.
- Preserve coordinate meaning, units, source identity, masks, and ordered preprocessing steps.
- Recompute preprocessing from raw data so repeated application does not compound filtering.
- Keep peak guesses separate from completed fit results.
- Record model, objective, constraints, optimizer status, uncertainty availability, and package
  versions with each result.

## Remaining extraction work

`engine.py` still combines model construction, optimization, metrics, and reporting. Future
structural work may separate fitting and reporting, but each extraction must preserve the
versioned synthetic baseline and leave the application runnable.
