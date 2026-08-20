# Architecture

## Current state

The application currently has two large runtime modules:

```text
xrd_gui.py
  -> xrd_analyzer.py
       DataLoader -> Preprocessor -> Fitter -> Reporter
```

`xrd_analyzer.py` combines several scientific responsibilities, while `xrd_gui.py` combines
widgets, mutable analysis state, orchestration, physical interpretation, and export commands.
This layout makes it difficult to distinguish a structural change from a numerical one.

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

1. Establish tests and document legacy scientific issues without changing formulas.
2. Extract immutable scan/configuration/result data structures.
3. Extract loading and preprocessing as pure functions.
4. Extract peak models and objectives while comparing legacy outputs.
5. Extract Bragg characteristic-length calculations and correct each scientific issue in a
   separate, validated change.
6. Replace GUI-owned scientific state with a small controller/session object.
7. Move the compatibility modules into `src/` after callers and entry points are covered.

Do not perform a wholesale rewrite. Each extraction must leave the application runnable and the
evidence status of its results explicit.
