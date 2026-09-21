"""The figures of the tutorial, drawn with the typography of the paper (figure_style.py).

lattice(...)     the synthetic grid at three metallicities in the recovered coordinates, with the grid lines drawn through the
                 stars: Figure 3 of Ting & Saad (2026).
schematic(...)   the schematic of the method (the paper's Figure 1) redrawn on real synthetic spectra: the seven steps on the
                 (Teff, log g) sheet of the grid at one metallicity, every panel computed with the package's own functions.
one_to_one(...)  transferred labels against the true ones, one panel per label.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path as MplPath
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.sparse.csgraph import dijkstra
from scipy.interpolate import RBFInterpolator
import figure_style as fs
from armillary import distance as dm, evaluate as ev

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


def _corners(labels):
    """The coolest giant and the hottest dwarf of a (Teff, log g) sheet: the two ends of the geodesic."""
    t = (labels[:, 0] - labels[:, 0].min()) / np.ptp(labels[:, 0]); g = (labels[:, 1] - labels[:, 1].min()) / np.ptp(labels[:, 1])
    return int(np.argmin(t + g)), int(np.argmax(t + g))


def _boundary(Q, labels):
    """The outline of the sheet in the coordinates Q: the outer grid lines, as a closed path."""
    uT, uG = np.unique(labels[:, 0]), np.unique(labels[:, 1]); lk = -np.ones((len(uT), len(uG)), int)
    lk[np.searchsorted(uT, labels[:, 0]), np.searchsorted(uG, labels[:, 1])] = np.arange(len(labels))
    ring = list(lk[0]) + list(lk[1:, -1]) + list(lk[-1, -2::-1]) + list(lk[-2:0:-1, 0]); ring = [i for i in ring if i >= 0]
    return MplPath(Q[ring])


def schematic(F, flux, wl, labels, labelled, Y, chunk=(0, None), width=fs.FULL_WIDTH):
    """Figure 1 on the fitted sheet.  F: the Fit of the sheet's spectra; flux, wl: its spectra; labels [N, 2] (Teff, log g);
    labelled: the indices whose Teff was given to propagate; Y [N]: the transferred Teff of every star."""
    N = len(flux); fig = plt.figure(figsize=(width, 4.2)); gs = fig.add_gridspec(2, 4, left=0.02, right=0.99, top=0.92, bottom=0.05, wspace=0.16, hspace=0.5, width_ratios=(1, 1, 1.15, 1.15))
    def panel(r, c, title, three_d=False):
        ax = fig.add_subplot(gs[r, c], projection="3d" if three_d else None); ax.set_title(title, fontsize=9, pad=3)
        if not three_d: ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
        return ax
    i0, i1 = _corners(labels)
    # (1) distance: the two corner stars and their cumulative absorption curves over one chunk
    a, b = chunk; b = b or len(wl); x = np.linspace(0, 1, b - a)
    fa, fb = F.fn[i0, a:b].astype(float), F.fn[i1, a:b].astype(float)
    Fa = dm.cumulative_curve(fa[None], np.ones((1, b - a), bool))[0]; Fb = dm.cumulative_curve(fb[None], np.ones((1, b - a), bool))[0]
    ax = panel(0, 0, "distance")
    ax.plot(x, fa + 1.15, color=fs.BLUE, lw=0.6); ax.plot(x, fb + 1.15, color=fs.ORANGE, lw=0.6)
    ax.fill_between(x, Fa, Fb, color=fs.GREY, alpha=0.35, lw=0); ax.plot(x, Fa, color=fs.BLUE, lw=0.8); ax.plot(x, Fb, color=fs.ORANGE, lw=0.8)
    ax.text(0.02, 1.1, f"{labels[i0, 0]:.0f} K, log g {labels[i0, 1]:.1f}", color=fs.BLUE, fontsize=6, va="top"); ax.text(0.98, 0.98, f"{labels[i1, 0]:.0f} K, log g {labels[i1, 1]:.1f}", color=fs.ORANGE, fontsize=6, va="top", ha="right")
    ax.set_xlim(0, 1); ax.set_ylim(-0.02, 2.3)
    # (2) all pairs: the table of distances, the stars ordered by their geodesic distance from the coolest giant
    D = dm.pairwise(F.features, block=256); dg, pred = dijkstra(F.lattice, directed=False, indices=i0, return_predecessors=True); order = np.argsort(dg)
    ax = panel(0, 1, "all pairs"); ax.imshow(D[np.ix_(order, order)], cmap="Greys", vmin=0, vmax=np.percentile(D, 99), interpolation="nearest"); ax.set_frame_on(True)
    for s in ax.spines.values(): s.set_linewidth(0.5)
    # (3) the graph and (4) a geodesic, on the spectra themselves: their projection on the first three principal components
    X = F.X_pca[:, :3]; X = (X - X.mean(0)) / X.std(0).max(); edges = np.array(F.lattice.nonzero()).T; edges = edges[edges[:, 0] < edges[:, 1]]
    path = [i1]
    while path[-1] != i0: path.append(pred[path[-1]])
    for c, title, with_path in ((2, "graph", False), (3, "geodesic", True)):
        ax = panel(0, c, title, three_d=True); ax.add_collection3d(Line3DCollection(X[edges], colors=fs.GREY, linewidths=0.3))
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=2.2, c=fs.INK, depthshade=False, linewidths=0)
        if with_path:
            Pp = X[path]; ax.plot(Pp[:, 0], Pp[:, 1], Pp[:, 2], color=fs.ORANGE, lw=1.4, zorder=5); ax.scatter(*X[[i0, i1]].T, s=14, c=fs.ORANGE, depthshade=False, zorder=6)
        ax.view_init(elev=28, azim=-52); ax.set_axis_off(); ax.set_box_aspect((1, 1, 0.8), zoom=1.3)
        ax.text2D(0.5, 0.0, "spectra in 3 principal components", transform=ax.transAxes, ha="center", va="top", fontsize=6, color=fs.FAINT)
    # (5) the coordinates, (6) refined, (7) the transferred label: the grid lines of constant Teff and log g through the recovered coordinates
    vmin, vmax = labels[:, 0].min(), labels[:, 0].max(); cmap = dict(cmap="Oranges", vmin=vmin - 0.3 * (vmax - vmin), vmax=vmax + 0.1 * (vmax - vmin))
    for c, C, title in ((0, F.C_geo, "coordinates"), (1, F.C, "refinement"), (2, F.C, "labels")):
        Q = _oriented(C, labels); ax = panel(1, c, title)
        if title != "labels":
            _grid_lines(ax, Q, labels); ax.scatter(Q[:, 0], Q[:, 1], s=3.5, c=fs.INK, zorder=3, linewidths=0)
        else:
            _grid_lines(ax, Q, labels, color="white", alpha=0.7)
            xs, ys = np.meshgrid(np.linspace(Q[:, 0].min(), Q[:, 0].max(), 100), np.linspace(Q[:, 1].min(), Q[:, 1].max(), 100), indexing="ij")
            field = RBFInterpolator(Q, Y, kernel="thin_plate_spline", smoothing=1.0)(np.column_stack([xs.ravel(), ys.ravel()])).reshape(xs.shape)
            mesh = ax.pcolormesh(xs, ys, field, shading="gouraud", zorder=1, rasterized=True, **cmap)
            mesh.set_clip_path(_boundary(Q, labels), transform=ax.transData)
            ax.scatter(Q[:, 0], Q[:, 1], s=4, c=Y, edgecolors=fs.INK, linewidths=0.2, zorder=3, **cmap)
            ax.scatter(Q[labelled, 0], Q[labelled, 1], s=40, marker="*", c=fs.ORANGE, edgecolors=fs.INK, linewidths=0.4, zorder=4)
        ax.set_xlim(Q[:, 0].min() - 0.05 * np.ptp(Q[:, 0]), Q[:, 0].max() + 0.05 * np.ptp(Q[:, 0])); ax.set_ylim(Q[:, 1].min() - 0.05 * np.ptp(Q[:, 1]), Q[:, 1].max() + 0.05 * np.ptp(Q[:, 1]))
    fig.add_subplot(gs[1, 3]).set_axis_off()
    for (x0, y0, x1, y1) in ((0.245, 0.72, 0.265, 0.72), (0.49, 0.72, 0.51, 0.72), (0.735, 0.72, 0.755, 0.72), (0.245, 0.25, 0.265, 0.25), (0.49, 0.25, 0.51, 0.25)):
        fig.patches.append(plt.matplotlib.patches.FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, color=fs.FAINT, lw=0.6))
    fig.patches.append(plt.matplotlib.patches.FancyArrowPatch((0.86, 0.53), (0.12, 0.48), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7, color=fs.FAINT, lw=0.6, connectionstyle="angle,angleA=-90,angleB=180,rad=4"))
    return fig


def pick_labelled(labels, where=((0.2, 0.25), (0.8, 0.3), (0.35, 0.8), (0.75, 0.85))):
    """A few labelled stars spread over a (Teff, log g) sheet: the ones nearest the given fractions of the label ranges."""
    t = (labels[:, 0] - labels[:, 0].min()) / np.ptp(labels[:, 0]); g = (labels[:, 1] - labels[:, 1].min()) / np.ptp(labels[:, 1])
    return np.array([int(np.argmin((t - a) ** 2 + (g - b) ** 2)) for a, b in where])


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
