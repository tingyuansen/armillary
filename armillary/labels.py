"""Transferring labels through the graph.

    propagate   [(I - W)^T (I - W) + mu P] Y = mu P Yhat, P the diagonal indicator of the labelled stars, Yhat
                their labels; mu scaled as in the refinement, mu = mu_0 x tr[(I-W)^T(I-W)] / N; solved by
                preconditioned conjugate gradients, one label at a time.

W is built by refine.lle_weights over each star's k_prop neighbours under D; when k_prop equals k_refine the
refinement's W is reused (pipeline.Fit.propagate does that)."""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from .refine import _system


def propagate(W, labelled, Y_lab, mu=1.0, tol=1e-9, maxiter=20000, log=None):
    """The transfer.  labelled: the indices of the labelled stars; Y_lab [n_lab, L] their labels.
    Returns (Y [N, L], residuals): the labels of every star and the relative residual of every column."""
    N = W.shape[0]; M, Mt, tr, diag = _system(W); m = mu * tr / N
    labelled = np.asarray(labelled); Y_lab = np.atleast_2d(np.asarray(Y_lab, np.float64))
    if Y_lab.shape[0] != len(labelled): Y_lab = Y_lab.T
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
        if r > 1e-5 and log: log(f"propagate: WARNING column {j} did not converge")
    return out, np.array(res)
