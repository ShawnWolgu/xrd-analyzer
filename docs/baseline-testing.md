# Core baseline testing

The baseline suite is the release gate for changes to loading, preprocessing, fitting,
derived quantities, or project-state flow.

## Baseline layers

### Synthetic reference v1 — `VERIFIED`

`tests/baselines/core_synthetic_v1.json` defines a noiseless three-peak scan with known:

- constant background;
- normalized Pseudo-Voigt area, center, sigma, and Lorentzian fraction;
- peak bounds and initial guesses;
- wavelength and expected Bragg characteristic lengths;
- explicit absolute or relative tolerances.

`tests/test_core_baseline.py` generates the intensities with an independent analytical
implementation rather than using `lmfit` to generate its own test input. The production
`Fitter` must recover the injected parameters and the fit-mask linear-intensity `R²_fit`.
This validates synthetic parameter recovery, not performance on every experimental pattern.

### Historical scan v1 — `REGRESSION-ONLY`

`tests/baselines/historical_scan_v1.json` records the tracked example scan's:

- source-file SHA-256 and numeric-content SHA-256;
- point count, 2θ range and median step;
- intensity extrema and sum;
- deterministic Savitzky–Golay output statistics and peak position.

This layer detects unintended input or preprocessing drift. It is not a certified material
standard and does not prove that a fitted physical interpretation is correct.

## Required commands

Run the baseline gate alone:

```bash
python -m pytest -m baseline
```

Run the complete release gate:

```bash
python -m pytest
python -m ruff check .
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -c "import xrd_gui"
```

## Change policy

Do not update expected values merely because a refactor changes output.

1. Classify the change as structural or scientific behavior.
2. For structural work, the existing baseline must continue to pass.
3. For an intentional scientific change, add independent evidence and explain why the old
   expectation is wrong before changing the baseline.
4. Version the JSON file instead of silently overwriting an established baseline when the
   scientific contract changes.
5. Keep tolerances tied to analytical recovery, numerical conditioning, scan resolution, or a
   documented reference; do not loosen them only to obtain a pass.
