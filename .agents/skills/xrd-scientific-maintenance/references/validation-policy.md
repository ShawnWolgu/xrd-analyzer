# Validation policy

## Evidence levels

Use three explicit evidence labels:

1. `VERIFIED`: passes an analytical result, independent implementation, certified standard, or
   justified experimental reference.
2. `REGRESSION-ONLY`: agrees with the legacy implementation within a declared tolerance.
3. `UNVERIFIED`: lacks adequate independent evidence or still depends on an unresolved
   assumption.

A saved workbook or image from the current program is a regression artifact, not ground truth.

## Required checks by change class

| Change class | Minimum evidence |
|---|---|
| Structural | Passing unit tests plus representative legacy-output comparison |
| Scientific behavior | A test seen failing first, independent correctness evidence, and justified tolerance |
| GUI state | Core test for the state transition plus a headless import or focused UI smoke test |
| Export/presentation | Schema/round-trip test and a check that labels match core semantics |

## Numerical tolerances

Choose and document tolerances before implementation.

- Derive scientific tolerances from scan step, instrument resolution, noise, expected solver
  conditioning, or a reference uncertainty.
- Use tight machine-level tolerances only for deterministic algebraic equivalence.
- Report absolute and relative differences for regression comparisons.
- Do not round values before testing unless the export format itself defines the rounding.
- Never widen a tolerance without documenting the physical or numerical reason.

## Tests for scientific calculations

Prefer, in order:

1. exact analytical cases;
2. limiting cases and invariants;
3. an independent implementation using a different path;
4. certified reference data;
5. synthetic recovery with known injected parameters;
6. regression comparison, clearly labeled as regression-only.

Make a new test fail for the intended reason before a scientific fix. For known legacy defects,
use strict `xfail` tests with a precise reason and a link to the validation record. When fixed,
the resulting XPASS must force removal of the marker and review of the new evidence.

## Provenance requirements

For a reproducible result, retain:

- source path or stable input identifier and content hash;
- raw point count, range, units, masks, and interpolation/gap policy;
- ordered preprocessing operations and parameters;
- model names, peak assignments, bounds, constraints, initial values, and objective weights;
- optimizer, termination state, warnings, uncertainties, and package versions;
- formulas, wavelength, corrections, and constants used for derived quantities.

If an item is unavailable in the legacy implementation, mark it as missing rather than
inventing a value.
