"""The distance between two spectra.

Every star becomes one feature vector, the concatenation over chunkings and chunks of its cumulative
absorption curve, scaled so that the L1 distance between two feature vectors IS the distance D between the spectra.  The neighbour search of lattice.py consumes these vectors and nothing else.

    cumulative_curve   rho and F of one chunk (the depth 1 - f, unclipped, zero on bad pixels, divided by its
                       integral over the chunk, then accumulated; with pixel errors given, the depth is weighted by the
                       pixel's inverse variance, the DESI setting of Config.error_weights)
    build_features     the feature vectors: F at every pixel / median W1 of the chunk, concatenated
    pairwise           blockwise exact L1 distances: the full matrix, or the k nearest of every star
    w1_pixel           the distance of one pair summed pixel by pixel, the reference the tests compare against
"""
import time
import numpy as np
import numba as nb


# ------------------------------------------------------------------ numba L1 kernels
@nb.njit(parallel=True, fastmath=True, cache=True)
def _l1_block(X, lo, hi, out):
    """L1 distance from rows lo:hi of X to every row of X, written into out[hi-lo, N] (float32)."""
    N, F = X.shape
    for i in nb.prange(hi - lo):
        xi = X[lo + i]
        for j in range(N):
            xj = X[j]
            s = 0.0
            for p in range(F):
                s += abs(xi[p] - xj[p])
            out[i, j] = np.float32(s)
    return out


@nb.njit(parallel=True, fastmath=True, cache=True)
def _l1_rows_to(X, rows, targets, out):
    """L1 distance from X[rows] to X[targets], written into out[len(rows), len(targets)]."""
    F = X.shape[1]
    for i in nb.prange(len(rows)):
        xi = X[rows[i]]
        for j in range(len(targets)):
            xj = X[targets[j]]
            s = 0.0
            for p in range(F):
                s += abs(xi[p] - xj[p])
            out[i, j] = np.float32(s)
    return out


@nb.njit(parallel=True, fastmath=True, cache=True)
def _l1_cand(X, rows, cand, out):
    """L1 distance from each of X[rows] to its own candidate list cand[i], written into out[len(rows), n_cand]."""
    F = X.shape[1]
    for i in nb.prange(len(rows)):
        xi = X[rows[i]]
        for j in range(cand.shape[1]):
            xj = X[cand[i, j]]
            s = 0.0
            for p in range(F):
                s += abs(xi[p] - xj[p])
            out[i, j] = np.float32(s)
    return out


# ------------------------------------------------------------------ the cumulative absorption curve
def cumulative_curve(fn_chunk, good_chunk, err_chunk=None):
    """The cumulative absorption curve F of every star in one chunk.

    depth = 1 - f on the good pixels and zero on bad pixels.  With `err_chunk` given (Config.error_weights, the
    DESI setting), the depth is weighted by w, the inverse variance 1 / sigma^2 of the pixel normalised to a mean
    of one over the good pixels of the chunk, so that a noisy pixel carries less of the chunk's absorption and the
    total keeps its scale; use this where the errors carry structure of their own (sky and detector, as in DESI) and not where
    they follow the photon noise (as in APOGEE, where it moves absorption from the weak lines into the strong ones, more for a
    faint star than a bright one, and worsens every result).  The depth is NOT clipped at zero: where noise puts the flux above the continuum it is negative and left so, because that noise cancels
    in the cumulative sum, whereas setting it to zero would add a pedestal that grows with the noise level.  These are signed cumulative profiles, not probability CDFs: their L1 distance agrees with one-dimensional Wasserstein distance only for nonnegative unit-normalised profiles. rho = depth / total, F = cumsum(rho), so F = 1 at the last pixel except in the guarded case below.

    Guard: a chunk whose net absorption |total| is below 1e-3 is divided by 1e-3 instead of by its total, so
    that a chunk with no absorption at all (or one whose positive and negative depths cancel) does not blow up;
    its curve is then the raw cumulative sum times 1e3."""
    f = np.asarray(fn_chunk, np.float64); g = np.asarray(good_chunk, bool)
    if err_chunk is None: depth = np.where(g, 1.0 - f, 0.0)
    else:
        w = np.where(g, 1.0 / np.maximum(np.asarray(err_chunk, np.float64), 1e-6) ** 2, 0.0)
        w /= np.maximum(w.sum(1, keepdims=True) / np.maximum(g.sum(1, keepdims=True), 1), 1e-300)
        depth = np.where(g, (1.0 - f) * w, 0.0)
    cdf = np.cumsum(depth, axis=-1); total = cdf[:, -1]; small = np.abs(total) < 1e-3
    F = cdf / np.where(small, 1e-3, total)[:, None]
    return F.astype(np.float32)


class FeatureInfo:
    """What build_features did: chunk bounds, medians and the feature layout."""
    def __init__(self):
        self.chunkings = []        # per chunking: list of (a, b)
        self.medians = []          # per chunking: array of the median W1 per chunk over the reference pairs
        self.offsets = []          # per chunking: (start, stop) columns in the feature vector
        self.n_ref = 0; self.n_pairs_median = 0; self.ref_index = None; self.seconds = 0.0

    @property
    def length(self): return self.offsets[-1][1] if self.offsets else 0

    def summary(self):
        return dict(n_chunkings=len(self.chunkings), chunks=[len(c) for c in self.chunkings], length=int(self.length),
                    n_ref=int(self.n_ref), n_pairs_median=int(self.n_pairs_median), seconds=float(self.seconds))


def _median_pairs(X, block=256):
    """Median of the L1 distance over all pairs among the rows of X."""
    D = pairwise(X, block=block); iu = np.triu_indices(len(X), 1)
    return float(np.median(D[iu]))


def build_features(fn, good, err, chunkings, n_ref=500, seed=0, block=256, log=None, ref_index=None):
    """The feature vectors whose L1 distance is the distance D between two spectra.

    For chunk c of a chunking, the block of the feature vector is  F_i^{(c)} / median_{pairs} W1^{(c)},  F at every
    pixel of the chunk (cumulative_curve; `err` None for the plain depth, the pixel errors for the weighted one), so that |Phi_i - Phi_j|_1 =
    sum over every chunk of every chunking of W1^{(c)}(i, j) / median W1^{(c)}, with W1 the sum over the pixels of
    |F_i - F_j| (the one-dimensional Wasserstein distance between the two absorption profiles).  The median is over every pair among `n_ref` stars drawn at random (seed `seed`),
    n_ref (n_ref - 1) / 2 pairs, the same draw for every chunk; `ref_index` names those stars explicitly instead.

    Returns (features float32 [N, length], FeatureInfo)."""
    t0 = time.time(); fn = np.asarray(fn, np.float32); good = np.asarray(good, bool); N = len(fn)
    err = None if err is None else np.asarray(err, np.float32); assert err is None or err.shape == fn.shape, (err.shape, fn.shape)
    rng = np.random.default_rng(seed); ref = np.asarray(ref_index) if ref_index is not None else rng.choice(N, min(n_ref, N), replace=False)
    info = FeatureInfo(); info.n_ref = len(ref); info.n_pairs_median = len(ref) * (len(ref) - 1) // 2
    info.ref_index = ref
    blocks = []; start = 0
    for s, bounds in enumerate(chunkings):
        cols, meds = [], []
        for a, b in bounds:
            F = cumulative_curve(fn[:, a:b], good[:, a:b], None if err is None else err[:, a:b])
            med = max(_median_pairs(F[ref], block), 1e-12)
            cols.append(F / np.float32(med)); meds.append(med)
        Phi_s = np.concatenate(cols, 1).astype(np.float32); blocks.append(Phi_s)
        info.chunkings.append(list(bounds)); info.medians.append(np.array(meds)); info.offsets.append((start, start + Phi_s.shape[1]))
        start += Phi_s.shape[1]
        if log: log(f"features: chunking {s + 1}/{len(chunkings)} ({len(bounds)} chunks) done, {start} columns so far")
    Phi = np.concatenate(blocks, 1).astype(np.float32); info.seconds = time.time() - t0
    return np.ascontiguousarray(Phi), info


# ------------------------------------------------------------------ exact blockwise distances
def pairwise(features, block=256, k=None, verbose=False):
    """Exact L1 distances between feature vectors, a block of rows at a time (memory O(block x N)).

    k=None returns the full N x N float32 matrix (only for N small enough to hold it).  Otherwise returns
    (nbr, dist): each star's k nearest by D, sorted, self excluded, int32 and float32 arrays [N, k]."""
    X = np.ascontiguousarray(features, np.float32); N = len(X); t0 = time.time()
    if k is None:
        D = np.empty((N, N), np.float32)
        for lo in range(0, N, block):
            hi = min(lo + block, N); _l1_block(X, lo, hi, D[lo:hi])
        return D
    if not isinstance(k, (int, np.integer)) or not 0 < k < N:
        raise ValueError(f"k = {k} must be a positive integer smaller than N = {N}")
    nbr = np.empty((N, k), np.int32); dst = np.empty((N, k), np.float32); buf = np.empty((block, N), np.float32)
    for lo in range(0, N, block):
        hi = min(lo + block, N); b = _l1_block(X, lo, hi, buf[: hi - lo])
        for i in range(hi - lo): b[i, lo + i] = np.inf                      # a star is not its own neighbour
        part = np.argpartition(b, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(b, part, axis=1), axis=1, kind="stable")
        idx = np.take_along_axis(part, order, axis=1)
        nbr[lo:hi] = idx; dst[lo:hi] = np.take_along_axis(b, idx, axis=1)
        if verbose and (lo // block) % 20 == 0:
            el = time.time() - t0; done = hi / N
            print(f"   knn {hi}/{N}  {el:6.0f}s elapsed, {el / max(done, 1e-9) - el:6.0f}s left", flush=True)
    return nbr, dst


# ------------------------------------------------------------------ the pixel-level reference
def w1_pixel(fn, good, err, bounds, i, j):
    # err None for the plain depth
    """The distance W1(i, j) for every chunk of one chunking, as the sum over every pixel of
    |F_i - F_j| (pixel units, no median normalisation): what the feature vectors' L1 distance reproduces."""
    out = []
    for a, b in bounds:
        F = cumulative_curve(np.asarray(fn)[[i, j], a:b], np.asarray(good)[[i, j], a:b], None if err is None else np.asarray(err)[[i, j], a:b]).astype(np.float64)
        out.append(float(np.abs(F[0] - F[1]).sum()))
    return np.array(out)
