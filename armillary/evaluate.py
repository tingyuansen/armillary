"""The scores and probes the experiments share, so that every script scores the coordinates the same way.
    error, error_r2     the paper's error statistic (half the central 68 per cent range of the differences), and it with R^2 per label
    linear_probe_r2     R^2 of the affine map coordinates -> labels fitted on all stars (the linear map of Section 3.1)
    quadratic_map, design   polynomial maps of the standardised coordinates (the quadratic map of Section 3.1)
    pair_jitter         the separation of the two coordinates of a star observed twice (Section 3.2, survey_precision.py)
    labelled_pool       the random draw of the training set (survey_ladder.py --draw random, survey_competitors.py)
    window_giants_split, score_held, alpha_trend   the protocol and score of the sweeps and the degradation ladder
    two_gaussians, separation   a two-Gaussian fit and its separation D (the two alpha sequences, Section 4 and Figure 11)
    ridge_probe_r2      the split-half ridge probe of survey_coordinates.py
    grid_step_lengths, line_monotonicity, orientation_consistency, factorial_index, grid_neighbour_pairs, euclidean_metric   the synthetic-grid tests of Section 2.7
    local_linear, pick_kmedoids, fewshot, affine_r2_between, standardise   the side scores of the ten-thousand-star sweeps (competitor_sweep.py, autoencoder.py), kept as their record
    tolist              JSON-ready copies
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
LAB = ["Teff", "logg", "[Fe/H]", "[a/M]"]


def error(e, axis=0):
    """The paper's error statistic: half the width of the central 68 per cent of the differences e (the 16th to 84th
    percentile range over two), the width of a Gaussian of the same core, insensitive to the few stars a graph places
    wrongly.  Section 3.2 defines it once; every quoted error, figure annotation and score uses this function."""
    e = np.asarray(e, float); lo, hi = np.nanpercentile(e, [16, 84], axis=axis); return (hi - lo) / 2


def error_r2(Y, P):
    """(the error per label, R^2 per label) of predicted labels P against reference labels Y."""
    Y = np.asarray(Y, float); P = np.asarray(P, float); e = P - Y
    return error(e), 1 - (e ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def standardise(Cr, Cq):
    mu, sd = Cr.mean(0), Cr.std(0) + 1e-12; return (Cr - mu) / sd, (Cq - mu) / sd


def linear_probe_r2(C, Y):
    """R^2 of the affine image of the coordinates, fitted and scored on the same stars (one value per label)."""
    A = np.column_stack([np.ones(len(C)), C]); coef, *_ = np.linalg.lstsq(A, Y, rcond=None); P = A @ coef
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def design(Z, degree):
    """Polynomial features of standardised coordinates up to `degree` (1: affine, 2: quadratic, 3: cubic)."""
    n = Z.shape[1]; cols = [np.ones(len(Z))] + [Z[:, i] for i in range(n)]
    if degree >= 2: cols += [Z[:, i] * Z[:, j] for i in range(n) for j in range(i, n)]
    if degree >= 3: cols += [Z[:, i] * Z[:, j] * Z[:, k] for i in range(n) for j in range(i, n) for k in range(j, n)]
    return np.column_stack(cols)


def quadratic_map(C, Y, degree=2):
    """The polynomial map from the coordinates to the labels, fitted and scored on all rows; returns (P, r2 list, n_terms)."""
    Zc = (C - C.mean(0)) / (C.std(0) + 1e-12); A = design(Zc, degree); P = A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    r2 = 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0); return P, [float(v) for v in r2], A.shape[1]


def local_linear(Cr, Yr, Cq, k=25, loo=False, return_scatter=False):
    """Weighted local linear regression on the k nearest training stars in standardised coordinates."""
    Zr, Zq = standardise(Cr, Cq); kk = min(k, len(Cr) - (1 if loo else 0)); dist, idx = cKDTree(Zr).query(Zq, k=kk + 1 if loo else kk)
    if kk == 1: dist, idx = dist[:, None], idx[:, None]
    if loo: dist, idx = dist[:, 1:], idx[:, 1:]
    P = np.zeros((len(Cq), Yr.shape[1])); S = np.zeros_like(P)
    for i in range(len(Cq)):
        nb_ = idx[i]; dz = Zr[nb_] - Zq[i]; w = np.exp(-0.5 * (dist[i] / (dist[i].max() + 1e-12) * 2.0) ** 2)
        A = np.column_stack([np.ones(len(nb_)), dz]) * w[:, None]
        coef, *_ = np.linalg.lstsq(A, Yr[nb_] * w[:, None], rcond=None); P[i] = coef[0]
        if return_scatter:
            resid = Yr[nb_] - (np.column_stack([np.ones(len(nb_)), dz]) @ coef); S[i] = np.sqrt((w[:, None] * resid ** 2).sum(0) / w.sum())
    return (P, S, dist.mean(1)) if return_scatter else P


def pick_kmedoids(C, n, seed=0, n_init=3):
    """n training stars that cover the cloud without labels: k-means on the standardised coordinates, each centre
    replaced by the nearest star (duplicates possible)."""
    Z = (C - C.mean(0)) / (C.std(0) + 1e-12); km = KMeans(n_clusters=n, n_init=n_init, random_state=seed).fit(Z)
    return np.array([int(np.argmin(((Z - c) ** 2).sum(1))) for c in km.cluster_centers_])


def pair_jitter(C, is_main, is_pair):
    """Repeat-pair jitter: median over pairs of |c_a - c_b| / sqrt 2 (consecutive rows of is_pair are one star
    observed twice), divided by the median distance of a main star to its 10-th nearest main star."""
    Cp = C[is_pair]; jitter = np.linalg.norm(Cp[0::2] - Cp[1::2], axis=1) / np.sqrt(2)
    Cm = C[is_main]; r10 = cKDTree(Cm).query(Cm, k=11)[0][:, -1]
    return float(np.median(jitter) / np.median(r10))


def fewshot(C, Y, n, seeds=(0, 1, 2), propagate=None, k_ll=None):
    """One rung of the label ladder: n k-medoid training stars (one draw per seed), the rest predicted by the
    local-linear map or, with `propagate(cal, Y_cal) -> P for every star`, through the graph.  Mean RMSE."""
    errs = []
    for s in seeds:
        cal = pick_kmedoids(C, n, seed=s); rest = np.setdiff1d(np.arange(len(C)), cal)
        P = propagate(cal, Y[cal])[rest] if propagate is not None else local_linear(C[cal], Y[cal], C[rest], k=k_ll or min(25, max(6, n // 2)))
        errs.append(error_r2(Y[rest], P)[0])
    return np.mean(errs, 0)


# ------------------------------------------------------------------ the alpha-residual test
def alpha_trend(Yg):
    """The running median of [alpha/M] in [Fe/H] over the giants Yg (25 quantile bins), as a function."""
    e = np.quantile(Yg[:, 2], np.linspace(0, 1, 26)); c = 0.5 * (e[1:] + e[:-1])
    m = np.array([np.median(Yg[(Yg[:, 2] >= e[i]) & (Yg[:, 2] < e[i + 1]), 3]) if ((Yg[:, 2] >= e[i]) & (Yg[:, 2] < e[i + 1])).any() else np.nan for i in range(25)])
    ok = np.isfinite(m); return lambda f: np.interp(np.clip(f, c[ok][0], c[ok][-1]), c[ok], m[ok])




def labelled_pool(has_label, seed=0):
    """The label ladder's draw (survey_ladder.py, survey_competitors.py): every labelled star in one random order, so that
    the n training stars are pool[:n] and every other labelled star, pool[n:], is scored."""
    return np.random.default_rng(seed).permutation(np.where(np.asarray(has_label, bool))[0])


def window_giants_split(Y, has_label, test_frac=0.25, seed=0, window=(-0.9, -0.2)):
    """The scoring protocol of the sweeps, the scaling run and the degradation ladder (survey_sweep.py, survey_scaling.py, survey_degrade.py):
    giants = labelled, log g < 3.5, finite [alpha/M]; window giants = giants with window[0] < [Fe/H] < window[1];
    held = a random test_frac of the window giants, reserved before any training star is drawn; pool = the other
    labelled stars in random order, so that the training set of n is pool[:n] (nested draws).
    Returns (held, pool, giants mask, trend function)."""
    Y = np.asarray(Y, float); has = np.asarray(has_label, bool)
    g = has & (Y[:, 1] < 3.5) & np.isfinite(Y[:, 3]); trend = alpha_trend(Y[g])
    gi = np.where(g & (Y[:, 2] > window[0]) & (Y[:, 2] < window[1]))[0]
    rng = np.random.default_rng(seed); held = rng.permutation(gi)[: max(1, int(round(test_frac * len(gi))))]
    pool = rng.permutation(np.setdiff1d(np.where(has)[0], held))
    return held, pool, g, trend


def two_gaussians(x, seed=0):
    """A two-Gaussian fit to the sample x, fitted on the standardised sample and returned in its units: (means, widths, weights),
    the lower component first."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; m, s = x.mean(), x.std() + 1e-12
    g = GaussianMixture(2, random_state=seed, n_init=5).fit(((x - m) / s)[:, None]); o = np.argsort(g.means_.ravel())
    return g.means_.ravel()[o] * s + m, np.sqrt(g.covariances_.ravel()[o]) * s, g.weights_[o]


def separation(x, seed=0, min_weight=0.05):
    """The separation of the two components of two_gaussians(x), D = (mu_2 - mu_1) / sqrt((sigma_1^2 + sigma_2^2) / 2) (Ashman's D,
    two peaks showing from D of about 2); nan if one component is lighter than min_weight."""
    mu, sd, w = two_gaussians(x, seed)
    return float("nan") if w.min() < min_weight else float((mu[1] - mu[0]) / np.sqrt(0.5 * (sd[0] ** 2 + sd[1] ** 2)))


def score_held(Y, P, held, trend, slice_=(-0.70, -0.35), alpha_on=None, min_slice=11):
    """The score of predicted labels on the held-out stars `held`: the error (evaluate.error) and median offset (bias) per label, then on
    the stars of `held[alpha_on]` (all of them by default; the giants of a mixed held-out set) the R^2 and the
    correlation of the [alpha/M] residual about the trend, and the two-Gaussian separation on the [Fe/H] slice
    when it holds at least `min_slice` stars."""
    held = np.asarray(held); e = P[held] - Y[held]; rms = error(e); bias = np.median(e, 0)
    h = held if alpha_on is None else held[np.asarray(alpha_on)]
    rt = Y[h, 3] - trend(Y[h, 2]); rp = P[h, 3] - trend(P[h, 2])
    r2 = 1 - ((rp - rt) ** 2).sum() / ((rt - rt.mean()) ** 2).sum(); corr = float(np.corrcoef(rt, rp)[0, 1])
    sl = (Y[h, 2] > slice_[0]) & (Y[h, 2] < slice_[1])
    return dict(rmse=[float(v) for v in rms], bias=[float(v) for v in bias], alpha_r2=float(r2), corr=corr,
                sigma=separation(P[h][sl, 3]) if sl.sum() >= min_slice else float("nan"), n_slice=int(sl.sum()), n=int(len(held)))


def euclidean_metric(X):
    """Pixel (Euclidean) distance between spectra for all pairs, in double precision on centred data."""
    Xc = np.asarray(X, np.float64); Xc = Xc - Xc.mean(0); sq = np.einsum("ij,ij->i", Xc, Xc)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Xc @ Xc.T), 0.0)); np.fill_diagonal(D, 0.0); return 0.5 * (D + D.T)


def tolist(x):
    """JSON-ready copy of nested dicts/arrays."""
    if isinstance(x, dict): return {str(k): tolist(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [tolist(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, (np.floating, float)): return float(x)
    if isinstance(x, (np.integer, int)): return int(x)
    return x




def ridge_probe_r2(C, Y, train, test, alpha=1.0):
    """R^2 per label of a ridge regression from the coordinates, fitted on `train` and scored on `test`
    (the split-half probe the survey-scale runs report)."""
    from sklearn.linear_model import Ridge
    out = []
    for j in range(Y.shape[1]):
        p = Ridge(alpha).fit(C[train], Y[train, j]).predict(C[test]); out.append(float(1 - ((p - Y[test, j]) ** 2).sum() / ((Y[test, j] - Y[test, j].mean()) ** 2).sum()))
    return out


def affine_r2_between(A, B):
    """Mean R^2 of the columns of B predicted by an affine map of A: 1 when B is, up to an affine map, a
    function of A (is a learned latent more than the leading principal components?)."""
    M = np.column_stack([np.ones(len(A)), A]); P = M @ np.linalg.lstsq(M, B, rcond=None)[0]
    return float(np.mean(1 - ((B - P) ** 2).sum(0) / ((B - B.mean(0)) ** 2).sum(0)))


def factorial_index(lab):
    """The grid's own coordinates: the unique values of each of the three labels and a lookup table from
    (i, j, k) on the grid to the row of the star (-1 where the grid has no star)."""
    uniq = [np.unique(lab[:, j]) for j in range(3)]; gi = np.stack([np.searchsorted(uniq[j], lab[:, j]) for j in range(3)], 1)
    lookup = -np.ones(tuple(len(u) for u in uniq), int); lookup[tuple(gi.T)] = np.arange(len(lab)); return uniq, lookup


def grid_neighbour_pairs(lookup, axis):
    """Rows (i, j) of every pair of grid stars adjacent along one label axis."""
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in lookup.shape], indexing="ij"), -1).reshape(-1, 3)
    nb_ = ijk.copy(); nb_[:, axis] += 1; ok = nb_[:, axis] < lookup.shape[axis]
    return lookup[tuple(ijk[ok].T)], lookup[tuple(nb_[ok].T)]


def grid_step_lengths(D, grid_index, lookup, regions):
    """The ruler test: the median distance of one grid step along each label axis, in each region of the grid
    (a dict name -> boolean mask over the grid stars).  Returns [3, n_regions]."""
    tab = np.zeros((3, len(regions)))
    for ax in range(3):
        i, j = grid_neighbour_pairs(lookup, ax); d = D[grid_index[i], grid_index[j]]; tab[ax] = [np.median(d[m[i]]) for m in regions.values()]
    return tab


def line_monotonicity(C, lab):
    """The lattice test of Section 3: along every grid line (one label varying, the other two fixed) the
    absolute Spearman correlation between position along the line's principal axis in the coordinates and the
    grid order; the mean, the fraction of perfectly monotonic lines, and the mean per axis."""
    uniq_, lookup_ = factorial_index(lab); vals = {0: [], 1: [], 2: []}
    for a in range(3):
        others = [b_ for b_ in range(3) if b_ != a]
        for idx in np.ndindex(*[lookup_.shape[b_] for b_ in others]):
            sl = [slice(None)] * 3
            for b_, i in zip(others, idx): sl[b_] = i
            line = lookup_[tuple(sl)]; line = line[line >= 0]; Pl = C[line] - C[line].mean(0); u = np.linalg.svd(Pl, full_matrices=False)[2][0]
            vals[a].append(abs(spearmanr(Pl @ u, np.arange(len(line))).correlation))
    allv = sum(vals.values(), []); return dict(mean=float(np.mean(allv)), perfect=float(np.mean(np.isclose(allv, 1.0))), per_axis=[float(np.mean(vals[a])) for a in range(3)])


def orientation_consistency(C, lab):
    """The second lattice test: the sign of the determinant of the three grid steps out of every cell corner,
    against the majority sign (no fold reverses the orientation); the fraction consistent overall and in the
    interior of the grid."""
    uniq_, lookup_ = factorial_index(lab); dets, interior = [], []
    for i, j, k in np.ndindex(lookup_.shape[0] - 1, lookup_.shape[1] - 1, lookup_.shape[2] - 1):
        p, a, b_, c = lookup_[i, j, k], lookup_[i + 1, j, k], lookup_[i, j + 1, k], lookup_[i, j, k + 1]
        dets.append(np.linalg.det(np.stack([C[a] - C[p], C[b_] - C[p], C[c] - C[p]])))
        interior.append(0 < i < lookup_.shape[0] - 2 and 0 < j < lookup_.shape[1] - 2 and 0 < k < lookup_.shape[2] - 2)
    dets, interior = np.array(dets), np.array(interior); bad = dets * np.sign(np.median(dets)) < 0
    return dict(consistent=float(1 - bad.mean()), interior=float(1 - bad[interior].mean()), n=len(dets))

