# Repository instructions

## Project purpose

This repository is a scientific Python desktop application for loading XRD scans,
preprocessing intensity data, fitting Pseudo-Voigt peaks, and exporting fit and
derived quantities for PZT thin-film analysis.

Treat numerical meaning and provenance as part of the public API. A refactor is not
successful if it preserves the GUI but changes scientific results without an explicit,
validated decision.

## Mandatory project skill

Use `.agents/skills/xrd-scientific-maintenance/SKILL.md` whenever work touches:

- `xrd_analyzer.py`, fitting, preprocessing, peak parameters, or derived quantities;
- `xrd_gui.py` state that changes data or fit configuration;
- exported numeric results, plots used for interpretation, or scientific terminology;
- tests or documentation that claim numerical or physical correctness.

Read the skill's model reference for model or physics changes and its validation
policy for every numerical change.

## Development workflow

1. Classify the change as structural, scientific-behavioral, GUI-only, or export-only.
2. Add or identify a test that can detect the relevant failure before implementation.
3. Keep structural refactors separate from scientific corrections.
4. Preserve raw input data and record every transformation needed to reproduce output.
5. Run the required checks and report any expected failures or unverified assumptions.

Do not call an existing output "ground truth" merely because a new implementation
matches it. Use `legacy baseline` until an analytical result, independent implementation,
certified reference, or experimental standard validates it.

## Architecture boundaries

- Core scientific code must not depend on PyQt.
- GUI code may orchestrate core services but must not implement scientific formulas.
- File loading must not silently overwrite raw data.
- Preprocessing must return new data and an explicit configuration/provenance record.
- Long-running fitting must stay outside the GUI thread.
- Export code must consume a result object; it must not recompute physics differently
  from the core layer.

The current two-file layout is legacy. Follow `docs/architecture.md` when extracting
modules, and preserve compatibility until callers and tests are migrated.

## Scientific guardrails

- State whether an angular value is `2theta` or `theta`, and state its unit.
- State whether a width is `sigma`, HWHM, or FWHM; never use an approximate conversion
  without naming the exact peak model.
- Report the first-order Bragg `d` spacing as a reflection-specific characteristic length.
- Do not infer lattice constants, `a/c`, tetragonality, or Miller-index multipliers from peak
  order or peak names.
- Do not label an uncorrected Scherrer estimate as grain size.
- Do not report generic chi-square, reduced chi-square, AIC, or BIC for a custom residual
  unless their definitions and statistical assumptions remain valid.
- Preserve fit status, parameter uncertainty, configuration, package versions, and input
  identity in exported scientific results.
- Do not globally suppress numerical warnings.

Known legacy concerns are documented in `docs/scientific-validation.md`. Do not fix one
incidentally inside a structural change.

## Commands

Use Python 3.10 or newer in an isolated environment.

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m ruff check .
```

For GUI smoke testing on a headless system:

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -c "import xrd_gui"
```

## Code conventions

- Add type hints to new public functions.
- Use concise Chinese docstrings for user-domain behavior; English is acceptable for
  infrastructure whose public API is already English.
- Prefer small pure functions and immutable configuration/result objects.
- Keep comments focused on model assumptions, units, and reasons.
- Raise contextual exceptions in core code and convert them to `QMessageBox` messages at
  the GUI boundary.
- Add tests under `tests/`; do not create executable `test.py` scripts in the repository
  root.

## Git scope

- Work on a feature branch for refactors.
- Do not mix generated outputs or sample-data changes into source refactors.
- Inspect the diff before staging. Commit and push only after explicit user authorization.
