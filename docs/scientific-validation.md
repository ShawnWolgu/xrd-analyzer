# Scientific validation status

This record separates confirmed implementation behavior from validated scientific behavior.
The migrated workbook and images are `REGRESSION-ONLY`; they are not ground truth.

## Evidence labels

- `VERIFIED`: checked against an analytical, independent, certified, or otherwise justified
  scientific reference.
- `REGRESSION-ONLY`: agrees with the legacy implementation within a declared tolerance.
- `UNVERIFIED`: assumptions or correctness evidence remain incomplete.

## Confirmed legacy concerns

| Topic | Evidence in current code | Potential impact | Foundation action |
|---|---|---|---|
| Partial numeric input row | The angle is appended before intensity conversion succeeds | `x` and `y` can have different lengths after a malformed row | Strict xfail; parse each row atomically in a separate input-integrity change |
| Pseudo-Voigt FWHM | `Peak.set_result` uses `2.3548 * sigma`; lmfit defines `fwhm = 2 * sigma` | Peak width and every downstream broadening estimate | Strict xfail; correct in a separate scientific change |
| Manual FWHM edit | GUI converts entered FWHM using an intermediate factor `2.2` | Refit starts from a parameter with different semantics | Document and test during GUI-state extraction |
| Pseudo-Voigt fallback height | Fallback formula uses a Gaussian coefficient different from lmfit's normalized model | Height can differ when direct model evaluation is unavailable | Add an independent formula test before correction |
| Legacy lattice terminology | The code labels Bragg `d` values as `a/c`, infers two reflections by angular order, and reports tetragonality | The same logic cannot safely cover 002/004/111/200/222/400 or unknown reflections | Replace with per-peak `characteristic length d`; do not infer lattice constants or multipliers |
| Scan gaps | Stitching fills uncovered grid points with `1e-5` without a mask | Artificial observations can enter preprocessing and fitting | Strict xfail; preserve gaps or validity masks |
| Minimum separation | The second peak's lower bound is based on the first peak's initial guess, not its fitted center | Fitted peaks can violate the requested pairwise separation | Strict xfail; implement a relational constraint |
| Mixed objective metrics | A log residual and normalized linear residual are concatenated, then generic chi-square/AIC/BIC values are reported | Statistical labels may not have their conventional meaning | Define objective-specific diagnostics before changing output |
| Preprocessing state | GUI preprocessing starts from the current displayed array | Repeated clicks compound transformations and are difficult to reproduce | Address during immutable session extraction |
| Warning handling | The core globally ignores warnings | Fit and numerical failures can be hidden | Remove only after focused warning tests exist |

Primary lmfit reference:
https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.PseudoVoigtModel

## What the foundation tests establish

The current tests establish only that:

- valid two-column text parsing and inclusive range trimming behave consistently;
- overlapping scan points are averaged in a deterministic simple case;
- polynomial background subtraction recovers an exact synthetic polynomial;
- the constructed lmfit model itself exposes `fwhm = 2 * sigma`;
- the tracked sample scan remains readable and structurally unchanged;
- known partial-row, FWHM, characteristic-length terminology, scan-gap, and separation problems
  remain visible as strict xfails.

They do not validate the complete fitting workflow against instrument standards, certified PZT
reference data, or an independent peak-fitting implementation.

## Required evidence for scientific corrections

1. State the equation, units, assignment, and assumptions before changing code.
2. Add a test that fails for the confirmed reason and show the failure.
3. Choose a tolerance from scan resolution, instrument resolution, uncertainty, or numerical
   analysis before implementation.
4. Correct the core calculation without mixing in module moves or UI redesign.
5. Propagate the typed result and terminology to exports and GUI.
6. Record whether the result becomes `VERIFIED`, stays `REGRESSION-ONLY`, or remains
   `UNVERIFIED`.
