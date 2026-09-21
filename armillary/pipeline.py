"""The whole chain as one function, fit(flux, good, segments, config), with a Config dataclass and a Fit object holding
every intermediate product, the timings of every step, and the label transfer as Fit.propagate().  The same implementation
is used at every sample size."""
import time, json, dataclasses
from dataclasses import dataclass, field
import numpy as np
import numba as nb
from . import preprocess as pp, distance as dm, lattice as lm, coordinates as cm, refine as rm, labels as lb


@dataclass
class Config:
    """Every parameter of the method, with the values of the APOGEE runs of Ting & Saad (2026) as defaults."""
    # --- preprocessing: the flux divided by a locally fitted continuum
    continuum: str = "none"            # "none" (pipeline-normalised input, APOGEE) | "running" (preprocess.prepare, DESI type)
    window_px: int = 100               # running continuum window (continuum="running")
    continuum_q: float = 0.90          # its upper quantile
    min_cover: float = 0.9             # common pixel cut of prepare (fraction of stars a pixel must be good in)
    renorm_chunking: tuple = (8, 7, 5) # the chunking whose chunks the local straight-line continuum is fitted in
    renorm_q: float = 0.85             # its upper quantile
    # --- the distance
    chunkings: tuple = ((4, 4, 3), (8, 7, 5), (16, 14, 10))   # chunks per detector segment, coarse to fine
    error_weights: bool = False        # weight each pixel's depth by its inverse variance in the curves: True where the errors carry structure of their own (DESI, sky and detector), False where they follow the photon noise (APOGEE)
    n_ref: int = 500                   # stars whose pairs give the median W1 of every chunk (n_ref (n_ref-1)/2 pairs)
    # --- the neighbour graph (the code's lattice)
    k_lattice: int = 30
    search: str = "exact"              # "exact" | "nndescent"
    n_landmarks: int = None            # landmarks for the geodesics and the scaling; None: every star
    # --- the coordinates
    d: int = 4
    n_eig: int = 12                    # eigenvalues returned, to read the number of coordinates from
    # --- the refinement
    n_pca: int = 30                    # the projection the weights are fitted on: this many principal components of the normalised flux, bad pixels at the pixel's sample mean
    pca_fit_max: int = 30000           # the components are fitted on this many stars drawn at random and applied to all
    k_refine: int = 100                # neighbours under D of the one weight matrix W (100 on the 4,000-star test cube and at survey scale)
    reg: float = 1e-3                  # ridge on the local Gram matrix, x trace(G)
    rho: float = 0.003                 # anchor strength of the refinement, relative to tr[(I-W)^T(I-W)]/N; chosen without labels, at the minimum of the separation of repeat observations of the same star
    # --- the transfer: the same W as the refinement
    k_prop: int = None                 # None: k_refine, and the refinement's W is reused; a value builds a second W over that many neighbours
    mu: float = 1.0
    # --- implementation
    seed: int = 0
    block: int = 256                   # rows per block of the exact search
    threads: int = None                # numba threads (None: numba's default)
    cg_tol: float = 1e-8
    cg_maxiter: int = 20000

    def to_dict(self): return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
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

    # ---- the configurations of the published runs
    @property
    def k_prop_eff(self): return self.k_refine if self.k_prop is None else self.k_prop

    @classmethod
    def paper_apogee(cls, **kw):
        """The 4,000-star APOGEE test cube: one weight matrix over k_refine = k_prop = 100 neighbours under D, mu 1."""
        return cls(**kw)

    @classmethod
    def paper_apogee_survey(cls, **kw):
        """The survey-scale APOGEE chain: 800 landmarks, one weight matrix over k_refine = k_prop = 100 neighbours under D, mu 3."""
        d = dict(n_landmarks=800, k_refine=100, k_prop=None, mu=3.0, search="nndescent"); d.update(kw); return cls(**d)

    @classmethod
    def paper_desi(cls, **kw):
        """The DESI chain: running continuum (window 100 px, q 0.90), local renormalisation in (10, 6, 4),
        the depth weighted by the pixel's inverse variance (the errors carry sky and detector structure), four chunkings,
        800 landmarks, one weight matrix over k_refine = k_prop = 100 (APOGEE's setting, which DESI's sweep cannot separate from 200 or 400), mu 3."""
        d = dict(continuum="running", window_px=100, continuum_q=0.90, min_cover=0.9, renorm_chunking=(10, 6, 4), error_weights=True,
                 chunkings=((5, 3, 2), (10, 6, 4), (20, 12, 8), (40, 24, 16)), n_landmarks=800, k_refine=100, k_prop=None, mu=3.0, search="nndescent")
        d.update(kw); return cls(**d)

    @classmethod
    def paper_synthetic(cls, **kw):
        """The synthetic grid: one segment, one chunking of 32 chunks, d = 3, k_refine 50, the local continuum fitted in
        the same 32 chunks (the synthetic spectra are already normalised; the fit changes little)."""
        d = dict(chunkings=((32,),), renorm_chunking=(32,), d=3, k_refine=50); d.update(kw); return cls(**d)


class Fit:
    """What fit() returns.  Attributes (N stars, F features, m landmarks):
        config, timings (dict, seconds per step), log (list of lines)
        pixel_mask   the pixels kept (continuum="running" drops some); segments of the kept pixels
        fn, good, err the normalised flux f_i (float32 [N, P]), its good-pixel mask and its error after the same normalisation
        features     the feature vectors (float32 [N, F]); info: distance.FeatureInfo
        nbr, dist    each star's k_max nearest neighbours under D and the distances (k_max = max of the k's)
        lattice      G (csr), lattice_info (components, piece_sizes, bridges)
        landmarks, eigenvalues, C_geo (N x d), C (N x d), refine_residual
        X_pca, W     the projections of the spectra and the refinement weights (csr)
    Methods: propagate(labels, labelled_index, ...), save(path)."""
    def __init__(self, config): self.config = config; self.timings = {}; self.log_lines = []; self._Wprop = {}

    # ---- the transfer
    def propagation_weights(self, k_prop=None):
        """W for the transfer: the refinement's W (k_prop None or equal to k_refine: the one weight matrix), else a
        second matrix over k_prop neighbours under D by the same rule."""
        c = self.config; k_prop = k_prop or c.k_prop_eff
        if k_prop == c.k_refine and getattr(self, "W", None) is not None: return self.W
        if k_prop in self._Wprop: return self._Wprop[k_prop]
        if k_prop > self.nbr.shape[1]: raise ValueError(f"k_prop = {k_prop} exceeds the {self.nbr.shape[1]} neighbours searched")
        W = rm.lle_weights(self.X_pca, self.nbr[:, :k_prop], reg=c.reg)
        if k_prop == c.k_refine: self.W = W      # a fit stopped after the neighbours: this is the one weight matrix
        self._Wprop[k_prop] = W; return W

    def propagate(self, labels, labelled_index, mu=None, k_prop=None, log=None):
        """The labels of every star from those of the labelled stars.  labelled_index indexes
        the rows of `labels`, which are the stars of the graph."""
        c = self.config; W = self.propagation_weights(k_prop)
        labels = np.atleast_2d(np.asarray(labels, np.float64)); labels = labels if labels.shape[0] == W.shape[0] else labels.T
        Y, res = lb.propagate(W, labelled_index, labels[labelled_index], mu=c.mu if mu is None else mu, tol=c.cg_tol, maxiter=c.cg_maxiter, log=log)
        self.propagate_residual = res; return Y

    def save(self, path, **extra):
        """The coordinates and everything needed to draw or re-score them (not the features or the flux)."""
        np.savez_compressed(path, C=self.C.astype(np.float32), C_geo=self.C_geo.astype(np.float32), eigenvalues=self.eigenvalues,
                            landmarks=self.landmarks, nbr=self.nbr, dist=self.dist, config=json.dumps(self.config.to_dict()),
                            timings=json.dumps(self.timings), lattice_info=json.dumps({k: v for k, v in self.lattice_info.items() if k != "bridge_edges"}), **extra)


def fit(flux, good, segments, config=None, err=None, log=print, stop=None):
    """The construction of the coordinates on one set of spectra.

    flux, good  [N, P] float32 and bool;  segments [P] detector segment id of every pixel;  config: Config.
    err is needed only for continuum="running" (it is divided by the continuum and returned on the Fit).
    stop="features" returns after the feature vectors (to check a neighbour search), stop="neighbours" after the
    neighbour search and the lattice (the label transfer needs only the graph), stop="lattice" likewise; None runs
    everything."""
    c = config or Config(); t_all = time.time(); F = Fit(c); lines = []
    def say(s):
        lines.append(f"[{time.time() - t_all:6.0f}s] {s}")
        if log: log(lines[-1])
    if c.threads: nb.set_num_threads(int(c.threads))
    flux = np.asarray(flux, np.float32); good = np.asarray(good, bool); segments = np.asarray(segments).astype(int); N = len(flux)
    say(f"fit: {N} spectra x {flux.shape[1]} pixels; search {c.search}; threads {nb.get_num_threads()}")

    # ---- the normalised flux
    t = time.time()
    if err is None: err = np.ones_like(flux)
    if c.continuum == "running":
        fn0, err0, good, segments, common = pp.prepare(flux, err, good, segments, c.window_px, c.continuum_q, c.min_cover); F.pixel_mask = common
    elif c.continuum == "none":
        fn0, err0 = flux, err; F.pixel_mask = np.ones(flux.shape[1], bool)
    else: raise ValueError(f"unknown continuum {c.continuum!r}")
    renorm = c.renorm_chunking if c.renorm_chunking is not None else c.chunkings[len(c.chunkings) // 2]
    F.fn, F.err = pp.local_renormalise(fn0, pp.segment_chunks(segments, renorm), good, q=c.renorm_q, err=err0); F.good = good; F.segments = segments
    F.timings["normalise"] = time.time() - t; say(f"normalised flux: {F.fn.shape[1]} pixels, {F.timings['normalise']:.1f} s")

    # ---- the feature vectors whose L1 distance is D
    t = time.time(); chunkings = [pp.segment_chunks(segments, per) for per in c.chunkings]
    F.features, F.info = dm.build_features(F.fn, good, F.err if c.error_weights else None, chunkings, n_ref=c.n_ref, seed=c.seed, block=c.block)
    F.timings["features"] = time.time() - t
    say(f"features{' (error-weighted depth)' if c.error_weights else ''}: {F.features.shape[1]} columns from {[len(ch) for ch in chunkings]} chunks, medians over {F.info.n_pairs_median} pairs of {F.info.n_ref} stars, {F.timings['features']:.1f} s")
    if stop == "features":
        F.timings["total"] = time.time() - t_all; F.log_lines = lines; say(f"stopped after the features ({F.timings['total']:.0f} s)"); return F

    # ---- neighbours under D and the lattice
    t = time.time(); k_max = max(c.k_lattice, c.k_refine, c.k_prop_eff)
    F.nbr, F.dist = lm.neighbours(F.features, k_max, search=c.search, block=c.block, seed=c.seed)
    F.timings["neighbours"] = time.time() - t; say(f"neighbours: k = {k_max} by {c.search} search, {F.timings['neighbours']:.1f} s")
    t = time.time(); F.lattice, F.lattice_info = lm.lattice(F.nbr[:, :c.k_lattice], F.dist[:, :c.k_lattice], F.features, log=say)
    F.timings["lattice"] = time.time() - t
    say(f"lattice: {F.lattice.nnz // 2} edges, {F.lattice.nnz / N:.1f} per star, {F.lattice_info['components']} component(s), {F.lattice_info['bridges']} bridge(s)")
    if stop in ("neighbours", "lattice"):
        t = time.time(); F.X_pca = rm.pca_flux(F.fn, good, c.n_pca, fit_max=c.pca_fit_max, seed=c.seed); F.timings["pca"] = time.time() - t
        F.timings["total"] = time.time() - t_all; F.log_lines = lines; say(f"stopped after the {stop} ({F.timings['total']:.0f} s)"); return F

    # ---- geodesics from the landmarks and the coordinates
    t = time.time()
    F.landmarks = np.arange(N) if (c.n_landmarks is None or c.n_landmarks >= N) else np.random.default_rng(c.seed).permutation(N)[:c.n_landmarks]
    Dl = lm.geodesics(F.lattice, F.landmarks); F.timings["geodesics"] = time.time() - t
    say(f"geodesics: Dijkstra from {len(F.landmarks)} landmark(s), {F.timings['geodesics']:.1f} s")
    t = time.time(); F.C_geo, F.eigenvalues = cm.landmark_mds(Dl, F.landmarks, c.d, c.n_eig); F.timings["mds"] = time.time() - t
    say(f"coordinates: d = {c.d}; eigenvalue ratios {np.round(F.eigenvalues[:8] / max(F.eigenvalues[0], 1e-300), 3).tolist()}, {F.timings['mds']:.1f} s")

    # ---- the refinement
    t = time.time(); F.X_pca = rm.pca_flux(F.fn, good, c.n_pca, fit_max=c.pca_fit_max, seed=c.seed)
    F.timings["pca"] = time.time() - t
    t = time.time(); F.W = rm.lle_weights(F.X_pca, F.nbr[:, :c.k_refine], reg=c.reg); F.timings["lle"] = time.time() - t
    say(f"refinement weights on the {c.n_pca}-component projection ({F.timings['pca']:.1f} s), k = {c.k_refine} neighbours under D ({F.timings['lle']:.1f} s)")
    t = time.time(); F.C, F.refine_residual = rm.refine(F.C_geo, F.W, rho=c.rho, tol=c.cg_tol, maxiter=c.cg_maxiter)
    F.timings["refine"] = time.time() - t
    say(f"refined: rho = {c.rho}, max relative residual {F.refine_residual.max():.1e}, {F.timings['refine']:.1f} s")
    F.timings["total"] = time.time() - t_all; F.log_lines = lines; say(f"done in {F.timings['total']:.0f} s")
    return F


