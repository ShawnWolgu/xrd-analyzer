"""Shared Matplotlib presentation defaults for XRD figures."""

from __future__ import annotations

import matplotlib


PLOT_FONT_FAMILY = "Arial"
PLOT_FONT_FALLBACKS = ("Arial Unicode MS", "Hiragino Sans GB", "DejaVu Sans")


def apply_plot_style() -> None:
    """Use Arial consistently in interactive, exported, and workbook-redrawn figures."""
    matplotlib.rcParams["font.family"] = [
        PLOT_FONT_FAMILY,
        *PLOT_FONT_FALLBACKS,
    ]
