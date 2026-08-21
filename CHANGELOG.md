# Changelog

## v1.0.0 — 2026-08-21

- Established a general-purpose XRD Analyzer identity and standard `main.py` entry point.
- Separated GUI orchestration from backend session, preprocessing, I/O, peak, and project state.
- Added versioned Excel project restoration and project-wide `2theta` range cropping.
- Added manual, theoretical-`d`, and TXT peak-position workflows, including fitted-position export.
- Added complete-peak freeze/disable controls, fitted-result undo/redo, and common peak-position shift.
- Clarified Pseudo-Voigt parameter semantics, `R²_fit`, Bragg characteristic length, and scientific limits.
- Added scientific regression baselines, GUI-state tests, Ruff checks, and protected background fitting.
