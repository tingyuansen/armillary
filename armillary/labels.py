"""Transferring labels through the graph, and choosing which stars to label.

The coordinates, the graph and the weights W are built without labels.  The labels of a few stars, the training set,
are then carried to every other star through the same weights.  The problem has the form of the refinement with two
differences: the labels take the place of the coordinates, and the anchor acts only on the training set, pulling each
training star toward its known labels:

    Y = argmin |(I - W) Y|^2 + mu sum_{i in training set} |Y_i - y_i|^2,
    i.e.   [(I - W)^T (I - W) + mu P] Y = mu P Yhat,

with P the diagonal indicator of the training stars and Yhat their labels (zero elsewhere).  The labels of all stars
are determined together, so a star far from the training set receives its labels through the stars between it and
the nearest training stars.

    propagate      the solve above, mu scaled as in the refinement, mu = mu_0 x tr[(I-W)^T(I-W)] / N, by
                   preconditioned conjugate gradients, one label at a time
    density_draw   which stars to label: a random draw weighted toward the sparse regions of the coordinates

W is built by refine.lle_weights over each star's k_prop neighbours under D; when k_prop equals k_refine the
refinement's W is reused (pipeline.Fit.propagate does that)."""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .refine import _system, _check_cg


def propagate(W, labelled, Y_lab, mu=3.0, tol=1e-9, maxiter=20000, log=None):
    """The transfer.  W [N, N]: the weights; labelled: the indices of the training stars; Y_lab [n_lab, L]: their
    labels (L labels per star; the transpose is accepted).  Returns (Y [N, L], residuals [L]): the labels of every star
    (the training stars included, pulled toward but not fixed at their values) and the relative residual of every label.

    mu is dimensionless, scaled by the mean diagonal of (I-W)^T(I-W) as rho is in the refinement.  Larger mu holds the
    training stars closer to their labels.

    Each connected component of W must contain a training star: on a component without one, nothing anchors the labels,
    the system is singular and the labels there are undetermined.  Invalid training sets raise ValueError; a solve
    that fails to converge raises RuntimeError."""
    if not np.isfinite(mu) or mu <= 0: raise ValueError("mu must be finite and positive")
    N = W.shape[0]; M, Mt, tr, diag = _system(W); m = mu * tr / N             # m: the anchor strength in absolute units
    # ---- validate the training set before any solve
    labelled = np.asarray(labelled); Y_lab = np.atleast_2d(np.asarray(Y_lab, np.float64))
    if labelled.ndim != 1 or not len(labelled) or not np.issubdtype(labelled.dtype, np.integer):
        raise ValueError("labelled must be a nonempty one-dimensional array of integer indices")
    if np.any((labelled < 0) | (labelled >= N)) or len(np.unique(labelled)) != len(labelled):
        raise ValueError("labelled indices must be unique and in range")
    if Y_lab.shape[0] != len(labelled): Y_lab = Y_lab.T                       # accept [L, n_lab] as well
    if Y_lab.ndim != 2 or Y_lab.shape[0] != len(labelled) or not np.isfinite(Y_lab).all():
        raise ValueError("Y_lab must contain finite labels for each labelled star")
    # every connected piece of W (ignoring zero weights) must hold at least one training star
    support = W.tocsr(copy=True); support.eliminate_zeros()
    ncomp, component = connected_components(support, directed=False)
    del support
    if len(np.unique(component[labelled])) != ncomp:
        raise ValueError("every connected component of W must contain a labelled star; "
                         "labels on an unanchored component are undetermined")
    # ---- A = M^T M + mu P as a matrix-free operator, with its diagonal as the (Jacobi) preconditioner
    s = np.zeros(N); s[labelled] = m                                         # mu P
    dg = diag + s; dg = np.where(dg > 1e-12, dg, 1.0)
    A = LinearOperator((N, N), matvec=lambda x: Mt @ (M @ x) + s * x, dtype=np.float64)
    Pre = LinearOperator((N, N), matvec=lambda x: x / dg, dtype=np.float64)
    out = np.zeros((N, Y_lab.shape[1])); res = []
    for j in range(Y_lab.shape[1]):
        # one label at a time: the right-hand side mu P Yhat is nonzero only on the training stars
        b = np.zeros(N); b[labelled] = m * Y_lab[:, j]; t0 = time.time()
        x, info = cg(A, b, rtol=tol, maxiter=maxiter, M=Pre)
        r = float(np.linalg.norm(A.matvec(x) - b) / max(np.linalg.norm(b), 1e-30)); res.append(r); out[:, j] = x
        if log: log(f"propagate: column {j} cg info {info}, relative residual {r:.2e}, {time.time() - t0:.1f} s")
        _check_cg(info, r, tol, "propagate", j)
    return out, np.array(res)


def density_draw(C, n, candidates=None, p=0.25, seed=0):
    """n training stars drawn without replacement from `candidates` (default every star), with probability
    proportional to the local density of the sample over the coordinates C to the power -p.  Returns the indices.

    A training set drawn at random follows the sample's own density, so the rare regions (metal-poor stars, a minor
    sequence) are the last it reaches; until it does, the transfer has no training star there.  The coordinates exist
    before any label, so they can decide which stars to label: weighting by density^-p moves the draw toward the sparse
    regions.  This matters most for the smallest training sets.

    The density is taken from the distance r to the 20th nearest star, in the coordinates standardised over the
    candidates.  In d coordinates density ~ r^-d, so density^-p ~ r^(p d).  The density is measured against EVERY star
    of C, not only the candidates, so it is the density of the sample.  p = 1/4 is the paper's value; larger p places
    the draw at the edges of the sample and costs precision."""
    C = np.asarray(C, float); cand = np.arange(len(C)) if candidates is None else np.asarray(candidates)
    Z = (C - C[cand].mean(0)) / C[cand].std(0)                                # standardise over the candidates
    r = cKDTree(Z).query(Z[cand], k=21)[0][:, -1]; w = r ** (p * C.shape[1]); w = w / w.sum()   # k = 21: the star itself and 20 others
    return cand[np.random.default_rng(seed).choice(len(cand), n, replace=False, p=w)]
