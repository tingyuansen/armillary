"""The normalised flux f: each spectrum divided by a locally fitted continuum.

The distance (distance.py) measures the absorption depth 1 - f, so the continuum it is measured from matters.  A slow
tilt of the continuum across a chunk would shift the absorption profile just as a real spectral feature does.  Two
continuum steps exist:

  running_continuum / prepare   for spectra that arrive with their instrumental response (flux-calibrated, not yet
                                normalised): a running upper-quantile continuum per detector segment, then a cut to
                                the pixels usable in most stars.  Config(continuum="running") applies it first.
  local_renormalise             for every input: within each chunk of a chunking, a straight line through the
                                upper-quantile pixels, so that the depth 1 - f is measured from a local continuum.
                                Spectra that a pipeline has already continuum-normalised take only this step, which
                                then removes the residual tilts of that normalisation.

A good-pixel mask accompanies the flux throughout: bad pixels take no part in any fit, carry no absorption in
distance.py, and are filled with the mean of that pixel over the sample before the principal components of refine.py
(fill_bad).  segment_chunks() gives the chunk bounds of a chunking, equal-width chunks per detector segment, so that no
chunk straddles a gap between detectors.
"""
import numpy as np
import numba as nb


def running_continuum(flux, good, seg_id, window_px=300, q=0.90, step=None):
    """A running upper-quantile continuum through the good pixels, computed separately in every detector segment.

    At knots every window/4 pixels (at least 8), the continuum is the q-quantile of the good pixels within
    window_px / 2 on either side; between the knots it is interpolated linearly.  An upper quantile follows the top of
    the flux, so it rides over the absorption lines instead of through them.  Returns (flux / continuum, continuum);
    the continuum is NaN wherever it could not be estimated (too few good pixels near every knot)."""
    f = np.asarray(flux, np.float64); good = np.asarray(good, bool); N, P = f.shape; cont = np.ones((N, P))
    step = step or max(window_px // 4, 8); h = window_px // 2
    for s in np.unique(seg_id):
        # the pixel range of this segment, and knots that keep half a window clear of its ends
        idx = np.where(seg_id == s)[0]; a, b = idx[0], idx[-1] + 1
        knots = np.arange(a + h // 2, b - h // 2, step)
        if len(knots) < 2: knots = np.array([a, b - 1])                      # a short segment: its two end pixels
        cont[:, a:b] = _running_quantile_segment(f, good, knots.astype(np.int64), int(a), int(b), int(h), float(q))
    cont = np.where(np.isfinite(cont) & (cont > 0), cont, np.nan)            # an unusable continuum becomes NaN
    return f / cont, cont


@nb.njit(parallel=True, cache=True)
def _running_quantile_segment(f, good, knots, a, b, h, q):
    """The q-quantile of the good pixels in a window of half-width h around every knot of one segment [a, b),
    interpolated linearly between knots.  It reproduces numpy's default (linear) quantile and np.interp, compiled and
    run one star per thread.  A knot with fewer than 10 good pixels, or a non-positive quantile, is skipped; a star
    with one usable knot gets a constant continuum, and one with none gets NaN."""
    N = f.shape[0]; K = len(knots); out = np.empty((N, b - a))
    for n in nb.prange(N):
        # ---- the quantile at every knot
        vals = np.empty(K); ok = np.zeros(K, np.bool_)
        for i in range(K):
            lo = max(a, knots[i] - h); hi = min(b, knots[i] + h); cnt = 0
            for j in range(lo, hi):
                if good[n, j]: cnt += 1
            if cnt >= 10:
                # gather the good pixels of the window, sort them, and interpolate the q-quantile between ranks
                w = np.empty(cnt); c = 0
                for j in range(lo, hi):
                    if good[n, j]: w[c] = f[n, j]; c += 1
                w.sort(); pos = q * (cnt - 1); k0 = int(np.floor(pos)); frac = pos - k0
                v = w[k0] + (w[min(k0 + 1, cnt - 1)] - w[k0]) * frac
                if np.isfinite(v) and v > 0: vals[i] = v; ok[i] = True
        # ---- interpolate between the usable knots
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
    """The first step for spectra with their instrumental response.  Divide by the running continuum, drop the
    pixels usable in fewer than a fraction min_cover of the stars, and clean the rest: a pixel is good only where the
    normalised flux is finite and lies in (lo, hi).  The pixel cut is common to all stars, so that every star keeps the
    same wavelength grid and the chunks mean the same thing for all of them.

    Returns (fn, err_n, ok, seg, common): the normalised flux with bad pixels set to 1.0 (no absorption), the error
    divided by the same continuum, the good-pixel mask, the segment id of the kept pixels, and the pixel mask applied."""
    fnorm, cont = running_continuum(flux, good, seg_id, window_px, q)
    ok = good & np.isfinite(fnorm) & (fnorm > lo) & (fnorm < hi)
    common = ok.mean(0) > min_cover                                          # pixels good in most stars
    fnorm, ok, seg = fnorm[:, common], ok[:, common], np.asarray(seg_id)[common]
    # the errors scale with the flux: divide them by the same continuum (1 where the continuum is undefined)
    e = (np.asarray(err, np.float64) / np.where(np.isfinite(cont), cont, 1))[:, common]
    return (np.where(ok, fnorm, 1.0).astype(np.float32), np.where(ok, e, 1.0).astype(np.float32),
            ok, seg, common)


def segment_chunks(seg_id, per_segment):
    """The chunk bounds [(a, b), ...] of one chunking: per_segment[s] equal-width chunks in detector segment s
    (segments numbered from 0, every segment in per_segment must have pixels).  Several chunkings, from coarse to
    fine, give the hierarchy of scales the distance is built on: a wide chunk compares the broad features of two
    spectra, a narrow one their individual lines."""
    seg_id = np.asarray(seg_id); bounds = []
    for s, n in enumerate(per_segment):
        idx = np.where(seg_id == s)[0]
        if len(idx) == 0: raise ValueError(f"segment {s} has no pixels")
        # n + 1 evenly spaced edges from the first to one past the last pixel of the segment
        e = np.linspace(idx[0], idx[-1] + 1, n + 1).astype(int)
        bounds += [(int(e[i]), int(e[i + 1])) for i in range(n)]
    return bounds


def local_renormalise(flux, bounds, good, q=0.85, floor=0.2, err=None):
    """The normalised flux f: within each chunk (a, b) of `bounds`, divide the flux by a straight line fitted through
    the good pixels at or above the q-quantile of the chunk (the local continuum).  The upper-quantile pixels are the
    ones nearest the continuum, so the line follows the top of the spectrum rather than the lines.  The line is
    floored at `floor` so that a chunk of very low flux cannot divide by a value near zero.

    Bad pixels are divided too, but never enter a fit.  A star with ten or fewer good pixels in a chunk gets a line
    through all its good pixels there.  Returns float32 [N, P]; with `err` given, (flux, err), both divided by the line."""
    f = np.asarray(flux, np.float64); out = f.copy(); good = np.asarray(good, bool)
    e_out = None if err is None else np.asarray(err, np.float64).copy()
    for a, b in bounds:
        seg_f, g = f[:, a:b], good[:, a:b].astype(np.float64)
        # the design matrix of a straight line in x, the pixel position scaled to [-1, 1] across the chunk
        x = np.linspace(-1, 1, b - a); A = np.column_stack([np.ones(b - a), x])
        # per star: the q-quantile of its good pixels; the fit weights are 1 on the good pixels at or above it
        thr = np.array([np.quantile(r[m > 0], q) if m.sum() > 10 else -np.inf for r, m in zip(seg_f, g)])[:, None]
        w = g * (seg_f >= thr)
        # Multiplying a masked NaN or infinity by zero still gives NaN, so the bad pixels are zeroed before the fit.
        fit_flux = np.where(g > 0, seg_f, 0.0)
        # weighted least squares for every star at once: (A^T w A) coef = A^T w f, with a tiny ridge for stability
        AtA = np.einsum("np,pi,pj->nij", w, A, A); Aty = np.einsum("np,pi,np->ni", w, A, fit_flux)
        coef = np.linalg.solve(AtA + 1e-6 * np.eye(2), Aty[..., None])[..., 0]
        line = np.clip(coef[:, :1] + coef[:, 1:] * x, floor, None); out[:, a:b] = seg_f / line
        if e_out is not None: e_out[:, a:b] = e_out[:, a:b] / line
    return out.astype(np.float32) if e_out is None else (out.astype(np.float32), e_out.astype(np.float32))


def fill_bad(fn, good):
    """The normalised flux with every bad pixel set to the mean of that pixel over the stars where it is good, so that
    it adds nothing to the covariance refine.pca_flux decomposes.  A pixel bad in every star is set to 1.0."""
    fn = np.asarray(fn, np.float32); good = np.asarray(good, bool)
    n = good.sum(0); mean = np.where(n > 0, np.where(good, fn, 0.0).sum(0) / np.maximum(n, 1), 1.0).astype(np.float32)
    return np.where(good, fn, mean[None, :])
