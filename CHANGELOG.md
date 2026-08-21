# Changelog

## Unreleased

- Adopted the GNU General Public License v3.0 only (`GPL-3.0-only`).
- Added source-install and launch instructions for Windows, macOS, and Linux.
- Aligned the Python package metadata version with the v1.1.0 application release.
- Moved application modules into a standard `src/xrd_analyzer/` package while keeping
  `main.py` as the repository launcher.
- Removed migrated `230610` scan/export artifacts and ignored common local fitting outputs.

## v1.1.0 — 2026-08-21

- Added live Chinese, Japanese, and English UI switching from the upper-right language selector.
- Persisted the selected UI language in restorable Excel project state.
- Standardized built-in plot titles, axes, callouts, and legend labels in English.

## v1.0.0 — 2026-08-21

- Established a general-purpose XRD Analyzer identity and standard `main.py` entry point.
- Separated GUI orchestration from backend session, preprocessing, I/O, peak, and project state.
- Added versioned Excel project restoration and project-wide `2theta` range cropping.
- Added manual, theoretical-`d`, and TXT peak-position workflows, including fitted-position export.
- Added complete-peak freeze/disable controls, fitted-result undo/redo, and common peak-position shift.
- Clarified Pseudo-Voigt parameter semantics, `R²_fit`, Bragg characteristic length, and scientific limits.
- Added scientific regression baselines, GUI-state tests, Ruff checks, and protected background fitting.
