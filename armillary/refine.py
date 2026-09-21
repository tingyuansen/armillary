"""Section 2.4, the locally linear refinement: equations (lle), (refine) and (solve).

    pca_flux      the projection of each spectrum on the n_pca leading principal components
                  of the normalised flux (bad pixels at the pixel's sample mean); also the PCA competitor of 3.1
    lle_weights   the weights w_ij of equation (lle) over each star's neighbours UNDER D, summing to one, with
                  a ridge reg x trace(G) on the local Gram matrix G = Z Z^T; a sparse matrix W
    refine        equation (solve), [(I - W)^T (I - W) + mu I] C = mu C_geo, mu = rho tr[(I-W)^T(I-W)] / N,
                  by preconditioned conjugate gradients, one column at a time; no dense N x N anywhere
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.decomposition import PCA
from .preprocess import fill_bad


def pca_flux(fn, good, n_pca=100, fit_max=30000, seed=0, block=20000):
    """x_i of equation (lle): the projection of the normalised flux on its n_pca leading principal components.

    Bad pixels are set to the pixel's mean over the sample first (preprocess.fill_bad).  The components are fitted on every star when N <= fit_max, else on a random subset of fit_max stars
    (seed `seed`) and every star projected in blocks."""
    X = fill_bad(fn, good); N = len(X); n = min(n_pca, N - 1)
    if N <= fit_max: return PCA(n_components=n, random_state=seed).fit_transform(X)
    fit = np.random.default_rng(seed).choice(N, fit_max, replace=False)
    pca = PCA(n_components=n, random_state=seed).fit(X[fit])
    return np.concatenate([pca.transform(X[i:i + block]) for i in range(0, N, block)])


def lle_weights(X, nbr, reg=1e-3, block=None):
    """The weights of equation (lle): row i of W reconstructs x_i from x_j, j in nbr[i] (its neighbours under
    D), with sum_j w_ij = 1.  Roweis & Saul (2000): with Z = x_j - x_i stacked over the neighbours and
    G = Z Z^T the local Gram matrix, w solves (G + reg tr(G) I) w = 1 and is then normalised to sum one.
    The ridge reg x tr(G) makes the solve unique when the neighbours outnumber the projection's components (Section 2.4).
    Returns a csr matrix [N, N] with k entries per row."""
    X = np.asarray(X, np.float64); N, k = nbr.shape; vals = np.empty((N, k))
    block = block or max(16, int(2e8 / (k * k * 8)))
    I = np.arange(k)
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        Z = X[nbr[lo:hi]] - X[lo:hi, None, :]                 # (B, k, P)
        G = Z @ Z.transpose(0, 2, 1)                           # (B, k, k)
        tr = np.trace(G, axis1=1, axis2=2)
        G[:, I, I] += reg * np.where(tr > 0, tr, 1.0)[:, None]
        w = np.linalg.solve(G, np.ones((hi - lo, k, 1)))[..., 0]
        vals[lo:hi] = w / w.sum(1, keepdims=True)
    rows = np.repeat(np.arange(N), k)
    return sp.csr_matrix((vals.ravel(), (rows, nbr.ravel())), shape=(N, N))


def _system(W):
    """M = I - W, M^T M as an operator, tr(M^T M) and diag(M^T M), without forming the product."""
    N = W.shape[0]; M = (sp.eye(N, format="csr") - W).tocsr(); Mt = M.T.tocsr()
    MM = M.multiply(M); tr = float(MM.sum()); diag = np.asarray(MM.sum(0)).ravel()
    return M, Mt, tr, diag


def refine(C_geo, W, rho=0.003, tol=1e-8, maxiter=20000, log=None):
    """Equation (solve): C = argmin |(I - W) C|^2 + mu |C - C_geo|^2, i.e. [(I-W)^T(I-W) + mu I] C = mu C_geo,
    with the anchor strength mu = rho x tr[(I-W)^T(I-W)] / N (dimensionless rho).  Solved by conjugate
    gradients with a Jacobi preconditioner, one column of C at a time.
    Returns (C, residuals): the relative residual |A c - b| / |b| of every column, checked rather than trusted."""
    N = W.shape[0]; M, Mt, tr, diag = _system(W); mu = rho * tr / N
    Minv = 1.0 / np.maximum(diag + mu, 1e-12)
    A = LinearOperator((N, N), matvec=lambda v: Mt @ (M @ v) + mu * v, dtype=np.float64)
    P = LinearOperator((N, N), matvec=lambda v: Minv * v, dtype=np.float64)
    C_geo = np.asarray(C_geo, np.float64); out = np.empty_like(C_geo); res = []
    for j in range(C_geo.shape[1]):
        b = mu * C_geo[:, j]; t0 = time.time()
        x, info = cg(A, b, rtol=tol, maxiter=maxiter, M=P)
        r = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)); res.append(r); out[:, j] = x
        if log: log(f"refine: column {j} cg info {info}, relative residual {r:.2e}, {time.time() - t0:.1f} s")
    return out, np.array(res)
