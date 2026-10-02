"""The whole method as one function.

fit(flux, good, segments, config) runs every step from the spectra to the refined coordinates and returns a Fit, which
holds every intermediate product, the timings of every step, and the label transfer as Fit.propagate().  Config holds
every parameter.  The same implementation is used at every sample size; only the neighbour search and the number of
landmarks change for large samples.

The steps of fit(), and the module of each:

    1. the normalised flux           preprocess.prepare (if continuum="running"), then preprocess.local_renormalise
    2. the feature vectors           distance.build_features: their L1 distance is the spectral distance D
    3. the neighbours and the graph  lattice.neighbours, lattice.lattice
    4. the geodesic distances        lattice.geodesics, from the landmarks
    5. the coordinates               coordinates.landmark_mds
    6. the refinement                refine.pca_flux, refine.lle_weights, refine.refine
and then, on the Fit,
    7. the label transfer            Fit.propagate -> labels.propagate, through the refinement's weights
"""
import time, json, dataclasses
from dataclasses import dataclass, field
import numpy as np
import numba as nb
from . import preprocess as pp, distance as dm, lattice as lm, coordinates as cm, refine as rm, labels as lb


@dataclass
class Config:
    """Every parameter of the method.  The defaults are the values of Ting & Saad (2026).

    The defaults suit continuum-normalised spectra on one detector segment, with every star a landmark and an exact
    neighbour search, which is fine up to a few tens of thousands of stars.  For larger samples set n_landmarks=800 and
    search="nndescent".  For several detector segments give `chunkings` one count per segment.  The fields that most
    often need changing for new data are chunkings, continuum, error_weights, d, n_landmarks and search; the rest are
    the method's settings and rarely need to change."""
    # --- step 1, the normalised flux
    continuum: str = "none"            # "none": the input is already continuum-normalised | "running": the input keeps its instrumental response, and preprocess.prepare divides by a running continuum first
    window_px: int = 100               # the window of the running continuum, in pixels (continuum="running")
    continuum_q: float = 0.90          # the upper quantile the running continuum follows
    min_cover: float = 0.9             # continuum="running" keeps only the pixels good in more than this fraction of the stars
    renorm_chunking: tuple = None      # the chunking whose chunks the local straight-line continuum is fitted in; None: the middle one of `chunkings`
    renorm_q: float = 0.85             # the fit uses the pixels at or above this quantile of each chunk
    # --- step 2, the distance
    chunkings: tuple = ((10,), (20,), (40,))   # one tuple per chunking, coarse to fine; each gives the number of chunks in every detector segment, e.g. ((4, 4, 3), (8, 7, 5)) for three segments
    error_weights: bool = False        # weight each pixel's depth by its inverse variance in the curves: True where the errors carry structure of their own (sky lines, detector features), False where they follow the photon noise
    n_ref: int = 500                   # stars whose pairs give the median W1 of every chunk (n_ref (n_ref-1)/2 pairs)
    # --- step 3, the neighbour graph (the lattice in the code)
    k_lattice: int = 30                # neighbours per star in the graph the geodesics run on
    search: str = "exact"              # "exact" (every pair; up to a few tens of thousands of stars) | "nndescent" (approximate; large samples)
    n_landmarks: int = None            # landmarks for the geodesics and the scaling; None: every star (memory and time grow as N^2)
    # --- step 5, the coordinates
    d: int = 4                         # the number of coordinates; read it from Fit.eigenvalues
    n_eig: int = 12                    # how many eigenvalues fit returns, to choose d from
    # --- step 6, the refinement
    n_pca: int = 30                    # the weights are fitted on this many principal components of the normalised flux (bad pixels at the pixel's sample mean)
    pca_fit_max: int = 30000           # the components are fitted on at most this many stars drawn at random, and applied to all
    k_refine: int = 100                # neighbours under D of the one weight matrix W
    reg: float = 1e-3                  # the ridge on the local Gram matrix G, times trace(G)
    rho: float = 0.003                 # anchor strength of the refinement, relative to tr[(I-W)^T(I-W)]/N; smaller pulls harder toward the neighbour relations; the paper chose it without labels, from repeat spectra of the same stars
    # --- step 7, the transfer: the same W as the refinement
    k_prop: int = None                 # None: k_refine, and the refinement's W is reused; a value builds a second W over that many neighbours
    mu: float = 3.0                    # anchor strength of the transfer, relative to tr[(I-W)^T(I-W)]/N as for rho; larger holds the training stars closer to their labels
    # --- implementation
    seed: int = 0                      # every random draw (median stars, landmarks, PCA subset, nn-descent)
    block: int = 256                   # rows per block of the exact search
    threads: int = None                # numba threads (None: numba's default)
    cg_tol: float = 1e-8               # relative residual at which the conjugate-gradient solves stop
    cg_maxiter: int = 20000            # their iteration limit

    def to_dict(self): return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        """A Config from a dict (as from to_dict or JSON).  Unknown keys raise ValueError, so a misspelt field is not
        silently ignored; the lists JSON returns are turned back into the tuples the fields expect."""
        d = dict(d); fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - fields
        if unknown: raise ValueError(f"unknown config keys {sorted(unknown)}")
        for k in ("chunkings",):
            if k in d: d[k] = tuple(tuple(c) for c in d[k])
        if "renorm_chunking" in d and d["renorm_chunking"] is not None: d["renorm_chunking"] = tuple(d["renorm_chunking"])
        return cls(**d)

    @classmethod
    def load(cls, path): return cls.from_dict(json.load(open(path)))

    def save(self, path): json.dump(self.to_dict(), open(path, "w"), indent=1)

    @property
    def k_prop_eff(self): return self.k_refine if self.k_prop is None else self.k_prop     # the neighbours the transfer uses


class Fit:
    """What fit() returns.  Attributes (N stars, P pixels, F feature columns, m landmarks):
        config, timings (dict, seconds per step), log_lines (list of lines)
        pixel_mask    the pixels kept (continuum="running" drops some); segments: the segments of the kept pixels
        fn, good, err the normalised flux f_i (float32 [N, P]), its good-pixel mask, and the errors after the same normalisation
        features      the feature vectors (float32 [N, F]); info: distance.FeatureInfo (chunk bounds, medians)
        nbr, dist     each star's k_max nearest neighbours under D and their distances (k_max = the largest of the k's)
        lattice       the graph G (csr); lattice_info (components, piece_sizes, bridges)
        landmarks     the landmark indices; eigenvalues: the n_eig largest of the scaling, to choose d from
        C_geo, C      the coordinates before and after the refinement (N x d); refine_residual per coordinate
        X_pca, W      the projections of the spectra and the refinement weights (csr)
    Methods: propagate(labels, labelled_index, ...) to transfer labels; save(path) to write the coordinates."""
    def __init__(self, config): self.config = config; self.timings = {}; self.log_lines = []; self._Wprop = {}

    # ---- the transfer
    def propagation_weights(self, k_prop=None):
        """W for the transfer.  With k_prop None or equal to k_refine, the refinement's W itself (the one weight
        matrix, as in the paper); otherwise a second matrix over k_prop neighbours under D by the same rule, built once
        and cached.  k_prop cannot exceed the neighbours fit searched (k_max)."""
        c = self.config; k_prop = k_prop or c.k_prop_eff
        if k_prop == c.k_refine and getattr(self, "W", None) is not None: return self.W
        if k_prop in self._Wprop: return self._Wprop[k_prop]
        if k_prop > self.nbr.shape[1]: raise ValueError(f"k_prop = {k_prop} exceeds the {self.nbr.shape[1]} neighbours searched")
        W = rm.lle_weights(self.X_pca, self.nbr[:, :k_prop], reg=c.reg)
        if k_prop == c.k_refine: self.W = W      # a fit stopped after the neighbours: this is the one weight matrix
        self._Wprop[k_prop] = W; return W

    def propagate(self, labels, labelled_index, mu=None, k_prop=None, log=None):
        """The labels of every star from those of the labelled stars.

        labels [N, L]: one row per star of the fit (any values on the unlabelled rows; [L, N] is accepted too);
        labelled_index: the rows whose labels are known.  mu and k_prop default to the config's.  Returns Y [N, L].
        The relative residuals of the solve are kept on Fit.propagate_residual."""
        c = self.config; W = self.propagation_weights(k_prop)
        labels = np.atleast_2d(np.asarray(labels, np.float64)); labels = labels if labels.shape[0] == W.shape[0] else labels.T
        Y, res = lb.propagate(W, labelled_index, labels[labelled_index], mu=c.mu if mu is None else mu, tol=c.cg_tol, maxiter=c.cg_maxiter, log=log)
        self.propagate_residual = res; return Y

    def save(self, path, **extra):
        """Write the coordinates and everything needed to draw or re-score them to a compressed .npz: C, C_geo, the
        eigenvalues, the landmarks, the neighbour lists, the configuration, the timings and the graph summary, plus any
        `extra` arrays.  The features and the flux are not written, so the file does not restore a Fit."""
        np.savez_compressed(path, C=self.C.astype(np.float32), C_geo=self.C_geo.astype(np.float32), eigenvalues=self.eigenvalues,
                            landmarks=self.landmarks, nbr=self.nbr, dist=self.dist, config=json.dumps(self.config.to_dict()),
                            timings=json.dumps(self.timings), lattice_info=json.dumps({k: v for k, v in self.lattice_info.items() if k != "bridge_edges"}), **extra)


def fit(flux, good, segments, config=None, err=None, log=print, stop=None):
    """The construction of the coordinates on one set of spectra.

    flux, good  [N, P] float32 and bool: the spectra on a common wavelength grid, and False on the bad pixels.
    segments    [P] int: the detector segment of every pixel, numbered from 0, so that no chunk straddles a gap.
    config      a Config (default Config()).
    err         [N, P]: the pixel standard deviations, used by continuum="running" and error_weights=True.  If omitted,
                an array of ones is used.  It is divided by the continuum and kept on the Fit.
    log         a function that receives one progress line per step (default print; None for silence).
    stop        "features" returns after the feature vectors (to check a neighbour search); "neighbours" or "lattice"
                after the neighbour search and the graph, with the projections the weights need (the label transfer
                needs only the graph, not the coordinates); None runs everything.
    Returns a Fit."""
    c = config or Config(); t_all = time.time(); F = Fit(c); lines = []
    # every progress line is time-stamped from the start of the fit, kept on the Fit, and passed to `log`
    def say(s):
        lines.append(f"[{time.time() - t_all:6.0f}s] {s}")
        if log: log(lines[-1])
    if c.threads: nb.set_num_threads(int(c.threads))
    flux = np.asarray(flux, np.float32); good = np.asarray(good, bool); segments = np.asarray(segments).astype(int); N = len(flux)
    say(f"fit: {N} spectra x {flux.shape[1]} pixels; search {c.search}; threads {nb.get_num_threads()}")

    # ---- 1. the normalised flux
    t = time.time()
    if err is None: err = np.ones_like(flux)
    if c.continuum == "running":
        # spectra with their instrumental response: divide by a running continuum and keep the pixels good in most stars
        fn0, err0, good, segments, common = pp.prepare(flux, err, good, segments, c.window_px, c.continuum_q, c.min_cover); F.pixel_mask = common
    elif c.continuum == "none":
        fn0, err0 = flux, err; F.pixel_mask = np.ones(flux.shape[1], bool)
    else: raise ValueError(f"unknown continuum {c.continuum!r}")
    # then, for every input, a straight-line local continuum in the chunks of one chunking (by default the middle one)
    renorm = c.renorm_chunking if c.renorm_chunking is not None else c.chunkings[len(c.chunkings) // 2]
    F.fn, F.err = pp.local_renormalise(fn0, pp.segment_chunks(segments, renorm), good, q=c.renorm_q, err=err0); F.good = good; F.segments = segments
    F.timings["normalise"] = time.time() - t; say(f"normalised flux: {F.fn.shape[1]} pixels, {F.timings['normalise']:.1f} s")

    # ---- 2. the feature vectors whose L1 distance is D
    t = time.time(); chunkings = [pp.segment_chunks(segments, per) for per in c.chunkings]
    F.features, F.info = dm.build_features(F.fn, good, F.err if c.error_weights else None, chunkings, n_ref=c.n_ref, seed=c.seed, block=c.block)
    F.timings["features"] = time.time() - t
    say(f"features{' (error-weighted depth)' if c.error_weights else ''}: {F.features.shape[1]} columns from {[len(ch) for ch in chunkings]} chunks, medians over {F.info.n_pairs_median} pairs of {F.info.n_ref} stars, {F.timings['features']:.1f} s")
    if stop == "features":
        F.timings["total"] = time.time() - t_all; F.log_lines = lines; say(f"stopped after the features ({F.timings['total']:.0f} s)"); return F

    # ---- 3. neighbours under D and the lattice
    # one search serves the graph, the refinement and the transfer: search the largest k any of them uses, and slice
    t = time.time(); k_max = max(c.k_lattice, c.k_refine, c.k_prop_eff)
    F.nbr, F.dist = lm.neighbours(F.features, k_max, search=c.search, block=c.block, seed=c.seed)
    F.timings["neighbours"] = time.time() - t; say(f"neighbours: k = {k_max} by {c.search} search, {F.timings['neighbours']:.1f} s")
    t = time.time(); F.lattice, F.lattice_info = lm.lattice(F.nbr[:, :c.k_lattice], F.dist[:, :c.k_lattice], F.features, log=say)
    F.timings["lattice"] = time.time() - t
    say(f"lattice: {F.lattice.nnz // 2} edges, {F.lattice.nnz / N:.1f} per star, {F.lattice_info['components']} component(s), {F.lattice_info['bridges']} bridge(s)")
    if stop in ("neighbours", "lattice"):
        # the projections are computed here too, so that Fit.propagate can build its weights without the coordinates
        t = time.time(); F.X_pca = rm.pca_flux(F.fn, good, c.n_pca, fit_max=c.pca_fit_max, seed=c.seed); F.timings["pca"] = time.time() - t
        F.timings["total"] = time.time() - t_all; F.log_lines = lines; say(f"stopped after the {stop} ({F.timings['total']:.0f} s)"); return F

    # ---- 4-5. geodesics from the landmarks and the coordinates
    # every star is a landmark unless n_landmarks is set and smaller than N; then n_landmarks stars drawn at random
    t = time.time()
    F.landmarks = np.arange(N) if (c.n_landmarks is None or c.n_landmarks >= N) else np.random.default_rng(c.seed).permutation(N)[:c.n_landmarks]
    Dl = lm.geodesics(F.lattice, F.landmarks); F.timings["geodesics"] = time.time() - t
    say(f"geodesics: Dijkstra from {len(F.landmarks)} landmark(s), {F.timings['geodesics']:.1f} s")
    t = time.time(); F.C_geo, F.eigenvalues = cm.landmark_mds(Dl, F.landmarks, c.d, c.n_eig); F.timings["mds"] = time.time() - t
    say(f"coordinates: d = {c.d}; eigenvalue ratios {np.round(F.eigenvalues[:8] / max(F.eigenvalues[0], 1e-300), 3).tolist()}, {F.timings['mds']:.1f} s")

    # ---- 6. the refinement: the projections, the weights over k_refine neighbours, then the sparse solve
    t = time.time(); F.X_pca = rm.pca_flux(F.fn, good, c.n_pca, fit_max=c.pca_fit_max, seed=c.seed)
    F.timings["pca"] = time.time() - t
    t = time.time(); F.W = rm.lle_weights(F.X_pca, F.nbr[:, :c.k_refine], reg=c.reg); F.timings["lle"] = time.time() - t
    say(f"refinement weights on the {c.n_pca}-component projection ({F.timings['pca']:.1f} s), k = {c.k_refine} neighbours under D ({F.timings['lle']:.1f} s)")
    t = time.time(); F.C, F.refine_residual = rm.refine(F.C_geo, F.W, rho=c.rho, tol=c.cg_tol, maxiter=c.cg_maxiter)
    F.timings["refine"] = time.time() - t
    say(f"refined: rho = {c.rho}, max relative residual {F.refine_residual.max():.1e}, {F.timings['refine']:.1f} s")
    F.timings["total"] = time.time() - t_all; F.log_lines = lines; say(f"done in {F.timings['total']:.0f} s")
    return F
