"""The figures of the tutorial, drawn with the typography of the paper (figure_style.py).

lattice(...)     the synthetic grid at three metallicities in the recovered coordinates, with the grid lines drawn through the
                 stars: Figure 3 of Ting & Saad (2026).
one_to_one(...)  transferred labels against the true ones, one panel per label.
"""
import numpy as np
import matplotlib.pyplot as plt
import figure_style as fs
from armillary import evaluate as ev

fs.use()
LABELS = [r"$T_{\rm eff}$ [K]", r"$\log g$", "[Fe/H]"]
SHORT = [r"$T_{\rm eff}$", r"$\log g$", "[Fe/H]"]


def _oriented(C, labels):
    """The sign of a coordinate is arbitrary: flip so that c1 rises with Teff and c2 with log g, as the label plane is drawn."""
    Q = C[:, :2].copy()
    if np.corrcoef(Q[:, 0], labels[:, 0])[0, 1] < 0: Q[:, 0] *= -1
    if np.corrcoef(Q[:, 1], labels[:, 1])[0, 1] < 0: Q[:, 1] *= -1
    return Q


def _grid_lines(ax, X, labels, color=fs.GREY, lw=0.45, alpha=0.8):
    """Lines of constant Teff and of constant log g through the points X of a (Teff, log g) grid."""
    uT, uG = np.unique(labels[:, 0]), np.unique(labels[:, 1]); lk = -np.ones((len(uT), len(uG)), int)
    lk[np.searchsorted(uT, labels[:, 0]), np.searchsorted(uG, labels[:, 1])] = np.arange(len(labels))
    for i in range(lk.shape[0]):
        row = lk[i][lk[i] >= 0]; ax.plot(X[row, 0], X[row, 1], "-", color=color, lw=lw, alpha=alpha, zorder=2)
    for k in range(lk.shape[1]):
        col = lk[:, k][lk[:, k] >= 0]; ax.plot(X[col, 0], X[col, 1], "-", color=color, lw=lw, alpha=alpha, zorder=2)


def lattice(C, labels, feh_values, width=fs.FULL_WIDTH):
    """The grid at three metallicities: the label plane above, the recovered coordinates below, grid lines through both (Figure 3)."""
    fig, axes = plt.subplots(2, 3, figsize=(width, 4.6)); fig.subplots_adjust(left=0.10, right=0.905, bottom=0.10, top=0.94, wspace=0.28, hspace=0.32)
    for j, v in enumerate(feh_values):
        m = np.isclose(labels[:, 2], v); L = labels[m]; Q = _oriented(C[m], L)
        for ax, X in ((axes[0, j], L[:, :2]), (axes[1, j], Q)):
            _grid_lines(ax, X, L); sc = ax.scatter(X[:, 0], X[:, 1], c=L[:, 1], cmap="viridis", s=6, lw=0, zorder=3, vmin=1, vmax=5)
        axes[0, j].set_xlim(7150, 3850); axes[0, j].set_ylim(5.25, 0.75); axes[0, j].set_title(f"[Fe/H] = {v:+.2f}", fontsize=9); axes[0, j].set_xlabel(LABELS[0])
        axes[1, j].invert_xaxis(); axes[1, j].invert_yaxis(); axes[1, j].set_xlabel("$c_1$"); axes[1, j].set_ylabel("$c_2$"); axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    axes[0, 0].set_ylabel(LABELS[1])
    axes[0, 0].text(-0.30, 0.5, "the label grid", transform=axes[0, 0].transAxes, fontsize=8.5, ha="center", va="center", rotation=90)
    axes[1, 0].text(-0.30, 0.5, "recovered coordinates", transform=axes[1, 0].transAxes, fontsize=8.5, ha="center", va="center", rotation=90)
    cax = fig.add_axes([0.92, 0.12, 0.014, 0.80]); fig.colorbar(sc, cax=cax, label=LABELS[1])
    return fig


def one_to_one(truth, pred, title=None, width=fs.FULL_WIDTH):
    """Transferred labels against the true ones for the stars given, one panel per label, the 1-sigma error in the corner."""
    lim = [(3900, 7100), (0.8, 5.2), (-2.1, 0.6)]; ticks = [[4000, 5000, 6000, 7000], [1, 2, 3, 4, 5], [-2, -1, 0]]
    fig, axes = plt.subplots(1, 3, figsize=(width, 2.6)); fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.86, wspace=0.3)
    for r, ax in enumerate(axes):
        lo, hi = lim[r]; x, y = truth[:, r], pred[:, r]
        ax.scatter(x, y, s=4, c=fs.BLUE, alpha=0.5, linewidths=0, rasterized=True); ax.plot([lo, hi], [lo, hi], color=fs.INK, lw=0.6, alpha=0.7)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xticks(ticks[r]); ax.set_yticks(ticks[r])
        e = ev.error(y - x); ax.text(0.04, 0.95, f"{e:.0f} K" if r == 0 else f"{e:.2f}" if r == 1 else f"{e:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_xlabel(SHORT[r] + " true"); ax.set_ylabel(SHORT[r] + " transferred")
    if title: fig.suptitle(title, fontsize=9)
    return fig
