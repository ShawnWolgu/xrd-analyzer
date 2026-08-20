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
| Pseudo-Voigt FWHM | lmfit defines `fwhm = 2 * sigma`; result display now reads lmfit's derived `fwhm` parameter (or uses the exact same relation as fallback) | Peak width and every downstream broadening estimate | Resolved with an analytical model-contract test |
| Manual FWHM edit | GUI converts entered FWHM exactly with `sigma = FWHM / 2`, checks type-specific bounds, and fixes sigma when selected | Entered width has the same semantics in display and refitting | Resolved with GUI edit, bounds, and fixed-parameter tests |
| Pseudo-Voigt fallback height | Fallback formula uses a Gaussian coefficient different from lmfit's normalized model | Height can differ when direct model evaluation is unavailable | Add an independent formula test before correction |
| Legacy lattice terminology | The legacy code labeled Bragg `d` values as `a/c`, inferred two reflections by angular order, and reported tetragonality | The same logic cannot safely cover 002/004/111/200/222/400 or unknown reflections | Resolved and analytically verified for direct Bragg `d`: report per-peak `characteristic length d`; do not infer lattice constants or multipliers |
| Scan gaps | Stitching fills uncovered grid points with `1e-5` without a mask | Artificial observations can enter preprocessing and fitting | Strict xfail; preserve gaps or validity masks |
| Minimum separation | The second peak's lower bound is based on the first peak's initial guess, not its fitted center | Fitted peaks can violate the requested pairwise separation | Strict xfail; implement a relational constraint |
| Mixed objective metrics | Linear, Log, and Mixed objectives are explicit; Log uses `log10(I + I0)` with a visible positive `I0`; the displayed `R²_fit` is computed only on the fit mask in linear intensity space; generic RMSE/reduced chi-square/AIC/BIC labels are not reported | Prevents excluded regions and incompatible residual scales from being folded into one headline score | Algebraically verified residual and fit-mask tests; the best scientific choice of objective and `I0` remains `UNVERIFIED` and user-controlled |
| Preprocessing state | GUI preprocessing starts from the current displayed array | Repeated clicks compound transformations and are difficult to reproduce | Address during immutable session extraction |
| Warning handling | Numerical warnings, solver success/message, covariance availability, evaluation count, and boundary hits are retained with the candidate result | Fit failures or degeneracy can otherwise be mistaken for a trustworthy solution | Global suppression removed; diagnostics are shown in the GUI and exported |
| macOS background-fit crash | Crash report identified `Thread stack size exceeded` in `FittingThread` during OpenBLAS `dgetrf_parallel` covariance inversion | Native `SIGBUS` terminates the whole GUI before Python can report an error | Reserve a 16 MiB worker stack and limit BLAS to one thread inside the fit; retain covariance calculation |

Primary lmfit reference:
https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.PseudoVoigtModel

## What the foundation tests establish

The current tests establish only that:

- valid two-column text parsing and inclusive range trimming behave consistently;
- overlapping scan points are averaged in a deterministic simple case;
- polynomial background subtraction recovers an exact synthetic polynomial;
- the constructed lmfit model itself exposes `fwhm = 2 * sigma`;
- the tracked sample scan remains readable and structurally unchanged;
- the Bragg calculation returns the injected per-reflection characteristic length without
  applying a multiplier from labels such as 004 or 111;
- the first-order inverse Bragg conversion returns exactly `2theta = 60 degrees` when
  `d = wavelength`, round-trips representative angles to machine precision, and rejects
  non-positive values or `d < wavelength / 2`;
- one explicit project wavelength is propagated to theoretical-`d` peak placement,
  characteristic lengths, uncorrected apparent Scherrer coherent-domain-size estimates, and
  the restorable workbook instead of using separate hidden constants;
- new workbooks use the characteristic-length column while the plotting reader still recognizes
  historical `d_spacing_Å` columns and per-peak values in legacy `Lattice_Parameters` sheets;
- the GUI displays the reflection label and characteristic length without `a/c` or tetragonality
  claims;
- known partial-row, scan-gap, and separation problems remain visible as strict xfails;
- displayed, manually entered, and fixed Pseudo-Voigt FWHM values follow the same exact
  `FWHM = 2 * sigma` definition, with film bounds of 0.02–3.00° and substrate bounds of
  0.02–2.00° in 2θ;
- a frozen peak fixes the four independent Pseudo-Voigt shape parameters: center, amplitude
  (area), sigma (FWHM), and fraction; disabled peaks remain explicit zero components;
- candidate fitted values remain separate from accepted next-run guesses until the user accepts
  them;
- the background worker reserves at least 8 MiB of native stack, applies a one-thread BLAS
  context, and repeatedly completes the `least_squares` covariance step for a three-peak case
  with one frozen peak;
- manual include/exclude ranges create a fit mask without modifying the stored scan arrays;
- `R²_fit` uses only that fit mask and is explicitly a linear-intensity diagnostic; plots and
  reports omit RMSE because Linear, Log, and Mixed objectives do not share one RMSE scale;
- the data-loading 2theta range performs an inclusive, irreversible in-session crop across raw,
  processed, source, fitting, plot, workbook, and figure data, and removes peaks whose centers
  fall outside the retained interval;
- a newly exported project workbook round-trips processed/raw data, fitted curves, peak names
  (including leading-zero hkl labels), bounds, guesses, fitted values, locks, peak states, and
  explicit GUI controls without rerunning the optimizer;
- zero-intensity Log residual behavior is analytically verified for a visible `I0`, while the
  selection of `I0` for an experiment remains a user scientific decision.

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
