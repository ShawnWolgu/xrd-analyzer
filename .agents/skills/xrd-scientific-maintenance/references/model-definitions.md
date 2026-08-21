# Model definitions and scientific semantics

Use these definitions unless the project records and validates an explicit replacement.

## Diffraction coordinate

- Treat the scan coordinate as `2theta` in degrees.
- Compute the Bragg angle as `theta = 2theta / 2` and convert it to radians for trigonometry.
- With first-order diffraction, use `d = wavelength / (2 sin(theta))`.
- Record the wavelength and radiation convention; `1.5406 angstrom` represents a common Cu
  K-alpha value but is not a universal default for every instrument or treatment of K-alpha
  splitting.

## lmfit PseudoVoigtModel

For `lmfit.models.PseudoVoigtModel`:

- `amplitude` is the integrated area of the normalized peak.
- `center` has the same unit as the scan coordinate.
- `sigma` is the shared half width at half maximum used by the Gaussian and Lorentzian terms.
- `fwhm = 2 * sigma`.
- `fraction` is the Lorentzian fraction and is bounded from 0 to 1.
- Peak height is
  `amplitude / sigma * ((1 - fraction) * sqrt(log(2) / pi) + fraction / pi)`.

Primary reference:
https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.PseudoVoigtModel

Do not substitute the Gaussian conversion `2.35482 * sigma` for this model.

## Project characteristic-length contract

Report the direct Bragg result `d = wavelength / (2 sin(theta))` as the
reflection-specific `characteristic length d`.

- Do not call this value a lattice constant.
- Do not infer `a`, `c`, `c/a`, tetragonality, or a Miller-index multiplier.
- Treat names such as 002, 004, 111, 200, 222, or 400 as annotations only; they do not alter
  the calculation.
- Do not infer a reflection assignment from angular order.

Any future lattice-parameter calculation requires a separate, explicitly requested
crystallographic model with validated reflection assignments. It must not be hidden inside the
generic peak reporter.

## Scherrer estimate

Use `D = K * wavelength / (beta * cos(theta))`, where `beta` is an FWHM in radians after any
instrumental-broadening correction required by the stated method.

Without instrumental correction and a justified shape factor, label the result as an
`apparent Scherrer coherent-domain-size estimate`. Do not call it grain size. State `K`, the
wavelength, width definition, correction method, and whether broadening from strain or defects
was separated.

## Fit objective and metrics

A concatenated log residual plus normalized linear residual is a custom objective. Report its
component definitions and weight. Do not automatically interpret its summed squares as a
conventional chi-square or use standard reduced chi-square, AIC, or BIC formulas unless the
likelihood and effective observation count justify them.

Keep raw-scale diagnostics such as RMSE and residual plots distinct from the optimization
objective. Record the number of observations, free parameters, masks, and weighting scheme.
