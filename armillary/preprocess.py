"""The normalised flux: each spectrum divided by a locally fitted continuum.

Two continuum steps exist:

  running_continuum / prepare   for spectra that arrive with their instrumental response (DESI type): a
                                running upper-quantile continuum per detector segment, then a cut to the
                                pixels usable in most stars.
  local_renormalise             for every input: within each chunk of a chunking, a straight line through
                                the upper-quantile pixels, so that the depth 1 - f is measured from a local
                                continuum.  APOGEE aspcapStar spectra are pseudo-continuum-normalised by the
                                pipeline already and take only this step.

A good-pixel mask accompanies the flux throughout: bad pixels take no part in any fit, carry no absorption
in distance.py, and are filled with the mean of that pixel over the sample before the principal components of
refine.py (fill_bad).  segment_chunks() gives the chunk bounds of a chunking, equal-width chunks per detector segment.
"""
import numpy as np
import numba as nb


def running_continuum(flux, good, seg_id, window_px=300, q=0.90, step=None):
    """Per-segment running upper-quantile continuum through the good pixels (knots every window/4 pixels,
    linear interpolation between knots).  Returns flux / continuum and the continuum."""
    f = np.asarray(flux, np.float64); good = np.asarray(good, bool); N, P = f.shape; cont = np.ones((N, P))
    step = step or max(window_px // 4, 8); h = window_px // 2
    for s in np.unique(seg_id):
        idx = np.where(seg_id == s)[0]; a, b = idx[0], idx[-1] + 1
        knots = np.arange(a + h // 2, b - h // 2, step)
        if len(knots) < 2: knots = np.array([a, b - 1])
        cont[:, a:b] = _running_quantile_segment(f, good, knots.astype(np.int64), int(a), int(b), int(h), float(q))
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, np.nan)
    return f / cont, cont


@nb.njit(parallel=True, cache=True)
def _running_quantile_segment(f, good, knots, a, b, h, q):
    """The q-quantile of the good pixels in a window of half-width h around every knot, interpolated linearly
    between knots (numpy's default quantile and interp, compiled and run one star per thread)."""
    N = f.shape[0]; K = len(knots); out = np.empty((N, b - a))
    for n in nb.prange(N):
        vals = np.empty(K); ok = np.zeros(K, np.bool_)
        for i in range(K):
            lo = max(a, knots[i] - h); hi = min(b, knots[i] + h); cnt = 0
            for j in range(lo, hi):
                if good[n, j]: cnt += 1
            if cnt >= 10:
                w = np.empty(cnt); c = 0
                for j in range(lo, hi):
                    if good[n, j]: w[c] = f[n, j]; c += 1
                w.sort(); pos = q * (cnt - 1); k0 = int(np.floor(pos)); frac = pos - k0
                v = w[k0] + (w[min(k0 + 1, cnt - 1)] - w[k0]) * frac
                if np.isfinite(v) and v > 0: vals[i] = v; ok[i] = True
        m = 0
        for i in range(K):
            if ok[i]: m += 1
        if m >= 2:
            xs = np.empty(m); ys = np.empty(m); c = 0
            for i in range(K):
                if ok[i]: xs[c] = knots[i]; ys[c] = vals[i]; c += 1
            for x in range(a, b): out[n, x - a] = np.interp(np.float64(x), xs, ys)
        elif m == 1:
            for i in range(K):
                if ok[i]: v1 = vals[i]
            for x in range(a, b): out[n, x - a] = v1
        else:
            for x in range(a, b): out[n, x - a] = np.nan
    return out


def prepare(flux, err, good, seg_id, window_px=100, q=0.90, min_cover=0.9, lo=0.0, hi=1.5):
    """Continuum-normalise by the running continuum, drop pixels usable in fewer than min_cover of the stars,
    and clean the rest (a pixel is good only where the normalised flux lies in (lo, hi)).

    Returns (fn, err_n, ok, seg, common): the normalised flux with bad pixels set to 1.0, the error divided
    by the continuum, the good-pixel mask, the segment id of the kept pixels, and the pixel mask applied."""
    fnorm, cont = running_continuum(flux, good, seg_id, window_px, q)
    ok = good & np.isfinite(fnorm) & (fnorm > lo) & (fnorm < hi)
    common = ok.mean(0) > min_cover
    fnorm, ok, seg = fnorm[:, common], ok[:, common], np.asarray(seg_id)[common]
    e = (np.asarray(err, np.float64) / np.where(np.isfinite(cont), cont, 1))[:, common]
    return (np.where(ok, fnorm, 1.0).astype(np.float32), np.where(ok, e, 1.0).astype(np.float32),
            ok, seg, common)


def segment_chunks(seg_id, per_segment):
    """Chunk bounds [(a, b), ...] of one chunking: per_segment[s] equal-width chunks in detector segment s.
    Several chunkings, from coarse to fine, give the hierarchy of scales the distance is built on."""
    seg_id = np.asarray(seg_id); bounds = []
    for s, n in enumerate(per_segment):
        idx = np.where(seg_id == s)[0]
        if len(idx) == 0: raise ValueError(f"segment {s} has no pixels")
        e = np.linspace(idx[0], idx[-1] + 1, n + 1).astype(int)
        bounds += [(int(e[i]), int(e[i + 1])) for i in range(n)]
    return bounds


def local_renormalise(flux, bounds, good, q=0.85, floor=0.2, err=None):
    """The normalised flux: within each chunk (a, b) of `bounds`, divide the flux by a straight line fitted
    through the good pixels at or above the q-quantile of the chunk (the local continuum).  Bad pixels are
    divided too but never enter a fit.  Returns float32; with `err` given, (flux, err) both divided by the line."""
    f = np.asarray(flux, np.float64); out = f.copy(); good = np.asarray(good, bool)
    e_out = None if err is None else np.asarray(err, np.float64).copy()
    for a, b in bounds:
        seg_f, g = f[:, a:b], good[:, a:b].astype(np.float64)
        x = np.linspace(-1, 1, b - a); A = np.column_stack([np.ones(b - a), x])
        thr = np.array([np.quantile(r[m > 0], q) if m.sum() > 10 else -np.inf for r, m in zip(seg_f, g)])[:, None]
        w = g * (seg_f >= thr)
        # Multiplying a masked NaN or infinity by zero still gives NaN.
        fit_flux = np.where(g > 0, seg_f, 0.0)
        AtA = np.einsum("np,pi,pj->nij", w, A, A); Aty = np.einsum("np,pi,np->ni", w, A, fit_flux)
        coef = np.linalg.solve(AtA + 1e-6 * np.eye(2), Aty[..., None])[..., 0]
        line = np.clip(coef[:, :1] + coef[:, 1:] * x, floor, None); out[:, a:b] = seg_f / line
        if e_out is not None: e_out[:, a:b] = e_out[:, a:b] / line
    return out.astype(np.float32) if e_out is None else (out.astype(np.float32), e_out.astype(np.float32))


def fill_bad(fn, good):
    """The normalised flux with every bad pixel set to the mean of that pixel over the stars where it is good, so
    that it adds nothing to the covariance: the matrix refine.pca_flux decomposes and the input of the Euclidean
    competitors."""
    fn = np.asarray(fn, np.float32); good = np.asarray(good, bool)
    n = good.sum(0); mean = np.where(n > 0, np.where(good, fn, 0.0).sum(0) / np.maximum(n, 1), 1.0).astype(np.float32)
    return np.where(good, fn, mean[None, :])
