"""Scores for coordinates and transferred labels, so that every analysis scores them the same way.

    error, error_r2     the error statistic (half the central 68 per cent range of the differences), and it with R^2 per label
    linear_probe_r2     how linearly the coordinates follow each label: R^2 of the best affine map coordinates -> labels
    quadratic_map, design   the same with a polynomial map, which can undo a smooth curvature but not a fold or a break
    pair_jitter         how far apart the coordinates of two spectra of the same star fall, in units of the local spacing
"""
import numpy as np
from scipy.spatial import cKDTree


def error(e, axis=0):
    """The error statistic: half the width of the central 68 per cent of the differences e (the 16th to 84th
    percentile range over two).  For Gaussian differences this is very nearly the standard deviation (0.994 sigma); for
    real ones it is the width of a Gaussian of the same core, insensitive to the few stars a graph places wrongly, which
    would inflate an rms.  It measures the spread only: a constant offset gives zero.  NaNs are ignored.  Every error the tutorial quotes uses this function."""
    e = np.asarray(e, float); lo, hi = np.nanpercentile(e, [16, 84], axis=axis); return (hi - lo) / 2


def error_r2(Y, P):
    """(the error per label, R^2 per label) of predicted labels P against reference labels Y, both [N, L].
    R^2 = 1 - sum of squared residuals / total sum of squares about the mean of Y."""
    Y = np.asarray(Y, float); P = np.asarray(P, float); e = P - Y
    return error(e), 1 - (e ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def linear_probe_r2(C, Y):
    """The linear probe: R^2 of the affine map c -> A c + b from the coordinates C [N, d] to the labels Y [N, L],
    fitted by least squares and scored on the same stars (one value per label).  Coordinates built from distances are
    defined only up to a rotation, so no single axis is privileged; the probe asks how closely the best affine combination
    of them follows each label.  1 for an exact linear relation, 0 for none."""
    A = np.column_stack([np.ones(len(C)), C]); coef, *_ = np.linalg.lstsq(A, Y, rcond=None); P = A @ coef
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def design(Z, degree):
    """The design matrix of a polynomial in the standardised coordinates Z [N, d]: a constant, every coordinate, and
    for degree >= 2 every product of two (squares included), for degree >= 3 every product of three."""
    n = Z.shape[1]; cols = [np.ones(len(Z))] + [Z[:, i] for i in range(n)]
    if degree >= 2: cols += [Z[:, i] * Z[:, j] for i in range(n) for j in range(i, n)]
    if degree >= 3: cols += [Z[:, i] * Z[:, j] * Z[:, k] for i in range(n) for j in range(i, n) for k in range(j, n)]
    return np.column_stack(cols)


def quadratic_map(C, Y, degree=2):
    """The polynomial map from the coordinates to the labels, fitted by least squares and scored on all rows.  It can
    correct a smooth curvature of the coordinates but cannot undo a fold or a break, so comparing it with the linear
    probe separates coordinates that are only curved from ones in which parts are displaced.
    Returns (P the mapped labels, the R^2 of each label as a list, the number of terms)."""
    Zc = (C - C.mean(0)) / (C.std(0) + 1e-12); A = design(Zc, degree); P = A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    r2 = 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0); return P, [float(v) for v in r2], A.shape[1]


def pair_jitter(C, is_main, is_pair):
    """The repeatability of the coordinates, from stars observed twice: the median over the pairs of |c_a - c_b| / sqrt 2
    (the scatter of one spectrum's coordinates), divided by the median distance of a main star to its 10th nearest
    main star (the local spacing of the sample), so that the number does not depend on the units of the coordinates.

    is_main: a mask of the stars that form the sample; is_pair: a mask of the repeat spectra, in which consecutive rows
    are the two spectra of one star."""
    Cp = C[is_pair]; jitter = np.linalg.norm(Cp[0::2] - Cp[1::2], axis=1) / np.sqrt(2)
    Cm = C[is_main]; r10 = cKDTree(Cm).query(Cm, k=11)[0][:, -1]          # k = 11: the star itself and 10 others
    return float(np.median(jitter) / np.median(r10))
