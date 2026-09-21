"""Shared plotting style, so every figure renders with the same typography.

A figure is scaled by LaTeX to the width of its slot.  If the figure is drawn wider than that slot, its text
shrinks by the same factor and the fonts stop matching between figures.  The two widths below are the slots AASTeX
gives, one column and the full text block; drawing at exactly those widths places every figure at unit scale, so a
9 pt label is 9 pt everywhere."""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl

COLUMN_WIDTH = 3.41      # inches, one AASTeX column (the paper draws each figure at the width of its slot)
FULL_WIDTH = 7.10        # inches, the full text block

INK = "#1A1A1A"
BLUE = "#3B75AF"
ORANGE = "#C4552B"
GREY = "#7A7F87"
FAINT = "#4A4F55"

RC = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5, "axes.linewidth": 0.8, "axes.grid": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.0, "lines.linewidth": 1.7,
    # not bbox="tight": cropping to content makes the saved width depend on the labels, and \includegraphics[width=...]
    # then rescales each figure by a different factor; each figure is drawn at exactly the width of its slot instead
    "figure.constrained_layout.use": False,
}


def pt(size: float) -> float:
    """A point size passed directly as fontsize=, kept here so that every figure sizes its annotations the same way."""
    return size


def use() -> None:
    mpl.rcParams.update(RC)


def save(fig, stem: Path) -> Path:
    """Write the PDF LaTeX includes."""
    pdf = stem.with_suffix(".pdf"); fig.savefig(pdf); return pdf
