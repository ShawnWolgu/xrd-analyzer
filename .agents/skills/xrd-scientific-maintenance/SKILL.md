---
name: xrd-scientific-maintenance
description: Preserve numerical meaning, physical validity, provenance, and reproducibility while modifying, refactoring, reviewing, testing, or documenting this repository's XRD loading, preprocessing, lmfit peak models, fit metrics, Bragg characteristic-length or Scherrer calculations, exports, and GUI analysis state. Use for every change that can alter XRD inputs, fitted parameters, derived quantities, or scientific claims.
---

# XRD scientific maintenance

Protect scientific behavior while making the code easier to maintain. Separate a cleaner
implementation from a scientifically different implementation, and make both auditable.

## Classify the change

Choose exactly one primary class before editing:

- **Structural**: move, rename, split, type, or simplify code without intended numerical change.
- **Scientific behavior**: change a formula, model, residual, constraint, preprocessing step,
  default, tolerance, or interpretation.
- **GUI state**: change how configuration or data moves through the interface.
- **Export/presentation**: change serialization, labels, plots, or reported terminology.

Split the work when one diff would combine structural and scientific-behavior changes.

## Establish evidence first

1. Read `references/validation-policy.md` for every numerical or structural change.
2. Read `references/model-definitions.md` when touching peak parameters, diffraction geometry,
   characteristic lengths, fit metrics, or Scherrer estimates.
3. Identify the smallest test that would detect the unwanted change.
4. Run that test before editing. For a new correctness claim, demonstrate that the new test
   fails for the known reason before implementing the fix.
5. Label comparison with current output as `legacy baseline`, not correctness evidence.

Use an analytical limit, independent implementation, certified reference, or experimental
standard for scientific validation whenever available. If none exists, record the result as
`UNVERIFIED` and stop short of a correctness claim.

## Implement within boundaries

- Keep raw arrays immutable after loading.
- Represent preprocessing and fitting choices as explicit configuration.
- Put equations and parameter conversions in the core layer, never in GUI event handlers.
- Carry units and parameter semantics through result objects and exports.
- Preserve masks or provenance for missing and interpolated points; never fabricate an
  unmarked baseline across uncovered scan ranges.
- Keep fit success, messages, uncertainties, covariance availability, package versions, and
  input identity with the result.
- Report Bragg `d` as a reflection-specific characteristic length; do not infer lattice
  constants or Miller-index multipliers from peak order or peak names.
- Surface numerical warnings with context; do not suppress them globally.

## Verify the result

Run from the repository root:

```bash
python -m pytest
python -m ruff check .
```

For a structural refactor, compare representative tabular output:

```bash
python .agents/skills/xrd-scientific-maintenance/scripts/compare_tables.py \
  baseline.xlsx candidate.xlsx --sheet Fit_Results
```

Choose tolerances before running a scientific comparison and justify them from resolution,
noise, numerical conditioning, or an accepted reference. Do not loosen a tolerance merely to
make a test pass.

## Report evidence honestly

State which of these levels supports each affected result:

- `VERIFIED`: independent scientific or analytical evidence passes a justified tolerance.
- `REGRESSION-ONLY`: matches the current implementation but lacks independent validation.
- `UNVERIFIED`: no adequate validation exists or assumptions remain unresolved.

List expected failures separately from passing tests. Never describe an xfail or a regression
comparison as proof of scientific correctness.
