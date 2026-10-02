"""The distance between two spectra.

Armillary compares two spectra by WHERE their absorption sits along the wavelength axis, not by how much absorption
they have.  Within each wavelength chunk, the absorption depth 1 - f of a star is divided by its total over the chunk
(so every star has one unit of absorption per chunk) and accumulated into a cumulative curve F.  The distance between
two stars in that chunk is the area between their two curves, sum_pixels |F_i - F_j|; for nonnegative profiles this is
the one-dimensional Wasserstein (earth mover's) distance, and the paper keeps that name for the signed profiles used here.
Each chunk's distance is divided by its median over pairs of stars, so that every chunk contributes in the same units,
and the result is summed over the chunks of several chunkings (a few wide chunks for the broad shape of the spectrum,
many narrow ones for individual lines).

The key implementation trick: because each chunk's distance is a sum of absolute differences over pixels, the whole
distance D is an L1 distance between two long vectors.  Each star becomes one feature vector, its cumulative curves for
every chunk of every chunking, each chunk scaled by 1 / (its median distance).  Then

    |Phi_i - Phi_j|_1 = sum over chunks c of W1^(c)(i, j) / median W1^(c) = D_ij,

so the N x N matrix of D never has to be formed: a nearest-neighbour search under the L1 metric on the feature vectors
(lattice.py) is a nearest-neighbour search under D.

    cumulative_curve   the cumulative absorption curve F of every star in one chunk
    build_features     the feature vectors Phi, for every chunk of every chunking, with the median scaling
    pairwise           exact L1 distances between feature vectors, a block of rows at a time
    w1_pixel           the chunk distances of one pair computed pixel by pixel: the reference the tests compare against
"""
import time
import numpy as np
import numba as nb


# ------------------------------------------------------------------ numba L1 kernels
# Three compiled kernels for the L1 distance between rows of the feature matrix X.  They differ only in which pairs
# they evaluate: one block of rows against every row (the exact search), a set of rows against a set of targets
# (bridging the graph, filling short rows), and each row against its own candidate list (re-scoring the candidates
# of nearest-neighbour descent).  All three accumulate in float64 and store float32, so that the exact search and the
# re-scored approximate search give bit-identical distances for the same pair.
@nb.njit(parallel=True, fastmath=True, cache=True)
def _l1_block(X, lo, hi, out):
    """L1 distance from rows lo:hi of X to every row of X, written into out[hi-lo, N] (float32)."""
    N, F = X.shape
    for i in nb.prange(hi - lo):                     # rows of the block in parallel
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
    """L1 distance from each X[rows[i]] to the stars of its own candidate list cand[i], written into out[len(rows), n_cand]."""
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

    fn_chunk, good_chunk [N, n_pix]: the normalised flux of the chunk and its good-pixel mask; err_chunk: the pixel
    errors, or None.  Returns F, float32 [N, n_pix].

    The depth is 1 - f on the good pixels and zero on the bad ones, so a bad pixel carries no absorption and the curve
    is flat across it.  The depth is NOT clipped at zero: where noise puts the flux above the continuum, the depth is
    negative and left so.  Noise scatters the flux above and below the continuum by equal amounts, so it cancels in
    the cumulative sum; clipping at zero would keep only the downward scatter and add a spurious absorption that grows
    with the noise level.  F is therefore a signed cumulative profile, not a probability distribution, and its L1
    distance equals the one-dimensional Wasserstein distance only for nonnegative profiles.

    With `err_chunk` (Config.error_weights), each pixel's depth is multiplied by a reliability weight w, the inverse
    variance 1 / sigma^2 of the pixel normalised to a mean of one over the good pixels of the chunk.  A noisy pixel then
    carries less of the chunk's absorption, while the total keeps its scale (only the relative errors matter).  Use it
    where the errors carry structure of their own (sky lines, detector features).  It is off by default because, where
    the errors follow the photon noise, the inverse variance is largest in the line cores, so the weights move
    absorption from the weak lines into the strong ones.

    rho = depth / total and F = cumsum(rho), so F rises from ~0 to 1 at the last pixel.  Guard: a chunk whose net
    absorption |total| is below 1e-3 is divided by 1e-3 instead, so that a chunk with no absorption (or with positive
    and negative depths that cancel) does not blow up; its curve is then the raw cumulative sum times 1e3."""
    f = np.asarray(fn_chunk, np.float64); g = np.asarray(good_chunk, bool)
    if err_chunk is None: depth = np.where(g, 1.0 - f, 0.0)
    else:
        # inverse variance on the good pixels (errors floored at 1e-6), zero on the bad ones ...
        w = np.where(g, 1.0 / np.maximum(np.asarray(err_chunk, np.float64), 1e-6) ** 2, 0.0)
        # ... divided by its mean over the good pixels of the row, so that only the RELATIVE errors matter
        w /= np.maximum(w.sum(1, keepdims=True) / np.maximum(g.sum(1, keepdims=True), 1), 1e-300)
        depth = np.where(g, (1.0 - f) * w, 0.0)
    # the running sum of the depth; its last value is the net absorption of the chunk
    cdf = np.cumsum(depth, axis=-1); total = cdf[:, -1]; small = np.abs(total) < 1e-3
    F = cdf / np.where(small, 1e-3, total)[:, None]          # normalise to one unit of absorption (the guard above)
    return F.astype(np.float32)


class FeatureInfo:
    """What build_features did: the chunk bounds, the median of every chunk, and where each chunking sits in the
    feature vector.  fit() keeps it on the Fit as Fit.info."""
    def __init__(self):
        self.chunkings = []        # per chunking: the list of chunk bounds (a, b), pixel indices
        self.medians = []          # per chunking: array of the median W1 of each chunk over the reference pairs
        self.offsets = []          # per chunking: the (start, stop) columns it occupies in the feature vector
        # the reference stars of the medians, how many pairs they give, and the time build_features took
        self.n_ref = 0; self.n_pairs_median = 0; self.ref_index = None; self.seconds = 0.0

    @property
    def length(self): return self.offsets[-1][1] if self.offsets else 0       # total columns of the feature vector

    def summary(self):
        """A JSON-ready summary: the number of chunkings and of chunks, the vector length, the median pairs, the time."""
        return dict(n_chunkings=len(self.chunkings), chunks=[len(c) for c in self.chunkings], length=int(self.length),
                    n_ref=int(self.n_ref), n_pairs_median=int(self.n_pairs_median), seconds=float(self.seconds))


def _median_pairs(X, block=256):
    """Median of the L1 distance over all pairs among the rows of X (each pair once: the upper triangle)."""
    D = pairwise(X, block=block); iu = np.triu_indices(len(X), 1)
    return float(np.median(D[iu]))


def build_features(fn, good, err, chunkings, n_ref=500, seed=0, block=256, log=None, ref_index=None):
    """The feature vectors whose L1 distance is the distance D between two spectra.

    fn, good [N, P]: the normalised flux and its good-pixel mask.  err [N, P]: the pixel errors for the error-weighted
    depth, or None for the plain depth.  chunkings: a list of chunkings, each a list of chunk bounds (a, b) from
    preprocess.segment_chunks.  Returns (features float32 [N, length], FeatureInfo).

    For chunk c, the block of the feature vector is F_i^(c) / median_pairs W1^(c): the cumulative curve at every pixel
    of the chunk, divided by the chunk's typical distance.  Summed directly, the chunk distances would be dominated by
    the widest or most line-rich chunks; after the division every chunk counts in the same units, the number of typical
    distances by which two stars differ in it.  Then |Phi_i - Phi_j|_1 = sum over every chunk of every chunking of
    W1^(c)(i, j) / median W1^(c), with W1 the sum over the pixels of |F_i - F_j|.

    The median is taken over every pair among min(n_ref, N) stars drawn at random (seed `seed`), n_ref (n_ref - 1) / 2
    pairs, the same stars for every chunk; `ref_index` names those stars explicitly instead.  A few hundred stars give
    a stable median at any sample size, so the medians cost the same for a thousand stars as for a million (the curves
    themselves cost O(N P))."""
    t0 = time.time(); fn = np.asarray(fn, np.float32); good = np.asarray(good, bool); N = len(fn)
    err = None if err is None else np.asarray(err, np.float32); assert err is None or err.shape == fn.shape, (err.shape, fn.shape)
    # the reference stars of the medians
    rng = np.random.default_rng(seed); ref = np.asarray(ref_index) if ref_index is not None else rng.choice(N, min(n_ref, N), replace=False)
    info = FeatureInfo(); info.n_ref = len(ref); info.n_pairs_median = len(ref) * (len(ref) - 1) // 2
    info.ref_index = ref
    blocks = []; start = 0
    for s, bounds in enumerate(chunkings):
        cols, meds = [], []
        for a, b in bounds:
            # the curves of every star in this chunk, then the chunk's median distance over the reference pairs
            # (floored, so that a chunk with no variation cannot divide by zero)
            F = cumulative_curve(fn[:, a:b], good[:, a:b], None if err is None else err[:, a:b])
            med = max(_median_pairs(F[ref], block), 1e-12)
            cols.append(F / np.float32(med)); meds.append(med)
        # one chunking's columns side by side; its place in the full vector is recorded in info.offsets
        Phi_s = np.concatenate(cols, 1).astype(np.float32); blocks.append(Phi_s)
        info.chunkings.append(list(bounds)); info.medians.append(np.array(meds)); info.offsets.append((start, start + Phi_s.shape[1]))
        start += Phi_s.shape[1]
        if log: log(f"features: chunking {s + 1}/{len(chunkings)} ({len(bounds)} chunks) done, {start} columns so far")
    # every chunking side by side: one row per star, one column per pixel per chunking
    Phi = np.concatenate(blocks, 1).astype(np.float32); info.seconds = time.time() - t0
    return np.ascontiguousarray(Phi), info


# ------------------------------------------------------------------ exact blockwise distances
def pairwise(features, block=256, k=None, verbose=False):
    """Exact L1 distances between feature vectors, a block of rows at a time (memory O(block x N)).

    k=None returns the full N x N float32 matrix (only for N small enough to hold it: the medians, the tests).
    Otherwise returns (nbr, dist): each star's k nearest neighbours by D, sorted by distance, the star itself excluded,
    as int32 and float32 arrays [N, k].  The cost is O(N^2 F) for F feature columns; nearest-neighbour descent
    (lattice.neighbours, search="nndescent") replaces it for large samples."""
    X = np.ascontiguousarray(features, np.float32); N = len(X); t0 = time.time()
    if k is None:
        # the full matrix, filled one block of rows at a time
        D = np.empty((N, N), np.float32)
        for lo in range(0, N, block):
            hi = min(lo + block, N); _l1_block(X, lo, hi, D[lo:hi])
        return D
    if not isinstance(k, (int, np.integer)) or not 0 < k < N:
        raise ValueError(f"k = {k} must be a positive integer smaller than N = {N}")
    nbr = np.empty((N, k), np.int32); dst = np.empty((N, k), np.float32); buf = np.empty((block, N), np.float32)
    for lo in range(0, N, block):
        # the distances from this block of rows to every star; only the k smallest of each row are kept
        hi = min(lo + block, N); b = _l1_block(X, lo, hi, buf[: hi - lo])
        for i in range(hi - lo): b[i, lo + i] = np.inf                      # a star is not its own neighbour
        # argpartition finds the k smallest without a full sort; a stable sort then orders those k by distance
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
    """The distance W1(i, j) for every chunk of one chunking, computed directly as the sum over every pixel of
    |F_i - F_j| (pixel units, no median normalisation).  This is what the L1 distance between the feature vectors must
    reproduce, chunk by chunk, after multiplying back by the medians; the tests compare the two.  `err` None for the
    plain depth, the pixel errors for the weighted one."""
    out = []
    for a, b in bounds:
        # the curves of the two stars alone in this chunk, then the area between them
        F = cumulative_curve(np.asarray(fn)[[i, j], a:b], np.asarray(good)[[i, j], a:b], None if err is None else np.asarray(err)[[i, j], a:b]).astype(np.float64)
        out.append(float(np.abs(F[0] - F[1]).sum()))
    return np.array(out)
