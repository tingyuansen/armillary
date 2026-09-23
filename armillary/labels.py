"""Transferring labels through the graph.

    propagate   [(I - W)^T (I - W) + mu P] Y = mu P Yhat, P the diagonal indicator of the labelled stars, Yhat
                their labels; mu scaled as in the refinement, mu = mu_0 x tr[(I-W)^T(I-W)] / N; solved by
                preconditioned conjugate gradients, one label at a time.

W is built by refine.lle_weights over each star's k_prop neighbours under D; when k_prop equals k_refine the
refinement's W is reused (pipeline.Fit.propagate does that)."""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse.csgraph import connected_components
from .refine import _system, _check_cg


def propagate(W, labelled, Y_lab, mu=1.0, tol=1e-9, maxiter=20000, log=None):
    """The transfer.  labelled: the indices of the labelled stars; Y_lab [n_lab, L] their labels.
    Returns (Y [N, L], residuals): the labels of every star and the relative residual of every column.
    Each connected component of W must contain a labelled star. Invalid anchors raise ValueError;
    a solve that fails to converge raises RuntimeError."""
    if not np.isfinite(mu) or mu <= 0: raise ValueError("mu must be finite and positive")
    N = W.shape[0]; M, Mt, tr, diag = _system(W); m = mu * tr / N
    labelled = np.asarray(labelled); Y_lab = np.atleast_2d(np.asarray(Y_lab, np.float64))
    if labelled.ndim != 1 or not len(labelled) or not np.issubdtype(labelled.dtype, np.integer):
        raise ValueError("labelled must be a nonempty one-dimensional array of integer indices")
    if np.any((labelled < 0) | (labelled >= N)) or len(np.unique(labelled)) != len(labelled):
        raise ValueError("labelled indices must be unique and in range")
    if Y_lab.shape[0] != len(labelled): Y_lab = Y_lab.T
    if Y_lab.ndim != 2 or Y_lab.shape[0] != len(labelled) or not np.isfinite(Y_lab).all():
        raise ValueError("Y_lab must contain finite labels for each labelled star")
    support = W.tocsr(copy=True); support.eliminate_zeros()
    ncomp, component = connected_components(support, directed=False)
    del support
    if len(np.unique(component[labelled])) != ncomp:
        raise ValueError("every connected component of W must contain a labelled star; "
                         "labels on an unanchored component are undetermined")
    s = np.zeros(N); s[labelled] = m                                         # mu P
    dg = diag + s; dg = np.where(dg > 1e-12, dg, 1.0)
    A = LinearOperator((N, N), matvec=lambda x: Mt @ (M @ x) + s * x, dtype=np.float64)
    Pre = LinearOperator((N, N), matvec=lambda x: x / dg, dtype=np.float64)
    out = np.zeros((N, Y_lab.shape[1])); res = []
    for j in range(Y_lab.shape[1]):
        b = np.zeros(N); b[labelled] = m * Y_lab[:, j]; t0 = time.time()
        x, info = cg(A, b, rtol=tol, maxiter=maxiter, M=Pre)
        r = float(np.linalg.norm(A.matvec(x) - b) / max(np.linalg.norm(b), 1e-30)); res.append(r); out[:, j] = x
        if log: log(f"propagate: column {j} cg info {info}, relative residual {r:.2e}, {time.time() - t0:.1f} s")
        _check_cg(info, r, tol, "propagate", j)
    return out, np.array(res)
