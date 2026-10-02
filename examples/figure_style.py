"""The plot style of the tutorial figures: one font family, a few colours and one figure width, so that every
figure of the tutorial looks the same.  Call use() once before plotting."""
import matplotlib as mpl

FULL_WIDTH = 7.10        # inches, the width every tutorial figure is drawn at

# the colours: near-black for lines and text, a blue and an orange for data, a grey for the grid lines
INK = "#1A1A1A"
BLUE = "#3B75AF"
ORANGE = "#C4552B"
GREY = "#7A7F87"

# matplotlib settings: serif fonts, inward ticks on all four sides with minor ticks, no legend frame
RC = {
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8.5, "axes.labelsize": 9.5, "axes.titlesize": 9.5, "axes.linewidth": 0.8, "axes.grid": False,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
    "legend.frameon": False, "legend.fontsize": 8.0, "lines.linewidth": 1.7,
    "figure.constrained_layout.use": False,
}


def use():
    """Apply the style to every figure drawn after this call (it updates matplotlib's global rcParams)."""
    mpl.rcParams.update(RC)
