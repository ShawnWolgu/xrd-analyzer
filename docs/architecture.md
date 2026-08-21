# Architecture

## Current state

The application now has an explicit frontend/backend boundary:

```text
Application entry point
  main.py (dependency check, QApplication lifecycle, main window startup)
      |
      v
Frontend
  xrd_gui.py (PyQt widgets, user interaction, rendering, background thread)
      |
      | only permitted XRD dependency
      v
Backend public API
  xrd_backend.py / XRDApplicationService
      |
      +-> xrd_session.py
      +-> xrd_io.py
      +-> xrd_preprocessing.py
      +-> xrd_peaks.py
      +-> xrd_crystallography.py
      +-> xrd_project.py
      +-> xrd_analyzer.py (remaining Fitter and Reporter compatibility module)
```

An automated architecture test parses imports and enforces both rules:

- `main.py` is the single application entry point and `xrd_gui.py` contains no `main()`;
- the frontend may import XRD behavior only from `xrd_backend`;
- the complete backend dependency closure may not import PyQt.

`XRDApplicationService` owns the active `AnalysisSession` and performs source loading, merging,
cropping, preprocessing transitions, fit-configuration transitions, and Fitter construction.
The frontend owns widgets, dialogs, tables, plots, and thread lifecycle.

Raw, processed, and source scans are immutable `ScanData` values. `AnalysisSession` owns their
relationship, the ordered preprocessing provenance, the permanent data range, and a validated
`FitConfiguration`. Transitional GUI array attributes remain synchronized views so old callers
continue to work. `xrd_analyzer.py` still combines fitting and reporting and is the next major
extraction boundary.

The versioned core baseline and its update policy are documented in
[`baseline-testing.md`](baseline-testing.md).

## Target boundaries

Keep the target deliberately small:

```text
src/xrd_analyzer/
  io.py               raw scan loading and validated scan objects
  preprocessing.py    pure transformations plus provenance
  peaks.py            peak assignments and parameter semantics
  fitting.py          model construction, constraints, objectives, fit results
  crystallography.py  Bragg characteristic-length and broadening calculations
  reporting.py        result schemas and export adapters
  session.py          immutable analysis configuration and application state
  gui/                 PyQt widgets and controllers
```

Dependency direction must remain one-way:

```text
GUI -> session/application -> scientific core -> NumPy/SciPy/lmfit
                         \-> export adapters
```

The scientific core must not import PyQt. Export adapters must not carry alternate versions of
scientific formulas. The GUI displays typed results and validation status from the core.

## Data and state rules

- Represent a loaded scan separately from a processed scan.
- Preserve the raw arrays, coordinate meaning, units, source identity, and masks.
- Represent each preprocessing operation and parameter explicitly and in order.
- Make repeated application idempotent from the same raw scan and configuration; do not filter
  the previous filtered display accidentally.
- Keep peak guesses and fitted results separate. Refinement may deliberately construct new
  guesses, but it must not mutate the record of the completed fit.
- Record model, objective, constraints, optimizer status, uncertainty availability, and package
  versions with every result.

## Incremental extraction sequence

1. Establish tests and document legacy scientific issues without changing formulas. **Done.**
2. Extract immutable scan/configuration/result data structures. **Scan and configuration done.**
3. Extract loading and preprocessing as pure functions. **Done with compatibility re-exports.**
4. Extract peak models and objectives while comparing legacy outputs. **Peak state done; fitting
   engine remains in the compatibility module.**
5. Extract Bragg characteristic-length calculations and correct each scientific issue in a
   separate, validated change.
6. Replace GUI-owned scientific state with a small controller/session object. **Data,
   preprocessing, sources, permanent range, and fit configuration are now Session-owned; peak
   editing and candidate results remain Fitter-owned.**
7. Move the compatibility modules into `src/` after callers and entry points are covered.

Do not perform a wholesale rewrite. Each extraction must leave the application runnable and the
evidence status of its results explicit.
