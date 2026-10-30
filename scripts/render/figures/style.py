"""Shared style for the paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Springer LNCS body font (Computer Modern via LaTeX).
FIGURE_WIDTH = 6.0
# Dark green from ColorBrewer PRGn: colorblind-safe, 5.6:1 contrast on white.
DOT_COLOR = "#1b7837"
MEAN_COLOR = "#000000"
LNCS_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
}


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
