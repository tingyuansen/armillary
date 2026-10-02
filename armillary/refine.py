"""The locally linear refinement of the coordinates.

The geodesic coordinates C_geo capture the large-scale arrangement of the stars but scatter on small scales: a
shortest path depends on which stars happen to sample the route, and even a small relative error in a long geodesic
distance can disturb a star's place among its immediate neighbours.  The refinement uses the local structure of the
manifold to reduce that scatter (following Roweis & Saul 2000).  On a smooth manifold every point is nearly a weighted
average of its nearest neighbours.  The weights w_ij that best reconstruct each star's spectrum from its neighbours'
spectra come from the spectra and D alone, so the noise of the geodesic coordinates does not enter them.  The refined
coordinates C balance two requirements, every star the same weighted average of its neighbours, and the large-scale
shape of C_geo:

    C = argmin |(I - W) C|^2 + mu |C - C_geo|^2,   i.e.   [(I - W)^T (I - W) + mu I] C = mu C_geo.

    pca_flux      the projection of each spectrum on the n_pca leading principal components of the normalised flux,
                  the vectors x_i the weights are fitted on (bad pixels at the pixel's sample mean)
    lle_weights   the weights w_ij over each star's neighbours UNDER D, summing to one, with a ridge
                  reg x trace(G) on the local Gram matrix G = Z Z^T; a sparse matrix W
    refine        the solve above, with mu = rho tr[(I-W)^T(I-W)] / N, by preconditioned conjugate gradients, one
                  column at a time; no dense N x N matrix is ever formed
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.decomposition import PCA
from .preprocess import fill_bad


def pca_flux(fn, good, n_pca=100, fit_max=30000, seed=0, block=20000):
    """The projection of the normalised flux on its n_pca leading principal components: the vectors x_i the
    refinement weights are fitted on.  The projection keeps the dominant spectral variation while reducing the pixel
    noise that enters the local fits.  Returns float [N, n] with n = min(n_pca, pixels, fit stars - 1).

    Bad pixels are first set to the pixel's mean over the sample (preprocess.fill_bad), so that they add nothing to the
    covariance.  The components are fitted on every star when N <= fit_max; otherwise on a random subset of fit_max
    stars (seed `seed`), and every star is then projected, `block` stars at a time to bound the memory."""
    X = fill_bad(fn, good); N = len(X); n = min(n_pca, X.shape[1], min(N, fit_max) - 1)
    if n < 1: raise ValueError("PCA needs at least two fit spectra, one pixel and one component")
    if N <= fit_max: return PCA(n_components=n, random_state=seed).fit_transform(X)
    fit = np.random.default_rng(seed).choice(N, fit_max, replace=False)
    pca = PCA(n_components=n, random_state=seed).fit(X[fit])
    return np.concatenate([pca.transform(X[i:i + block]) for i in range(0, N, block)])


def lle_weights(X, nbr, reg=1e-3, block=None):
    """The locally linear weights.  X [N, n]: the projections x_i (pca_flux); nbr [N, k]: each star's neighbours under
    D.  Returns W, a csr matrix [N, N] with k entries per row, each row summing to one.

    Row i of W reconstructs x_i from the x_j of its neighbours, x_i ~ sum_j w_ij x_j with sum_j w_ij = 1 (Roweis & Saul
    2000).  The weights have a closed form, one small linear solve per star: with Z the k x n matrix of the
    differences x_j - x_i and G = Z Z^T their k x k Gram matrix, w solves (G + reg tr(G) I) w = 1 and is then
    normalised to sum to one.  When the neighbours outnumber the projection's components, G is singular; the ridge
    reg x tr(G), scaled to G itself, makes the solve unique.

    The stars are processed in blocks (`block` stars, by default sized to about 200 MB of Gram matrices), with all the
    solves of a block done at once by numpy's batched linear algebra."""
    X = np.asarray(X, np.float64); N, k = nbr.shape; vals = np.empty((N, k))
    block = block or max(16, int(2e8 / (k * k * 8)))
    I = np.arange(k)
    for lo in range(0, N, block):
        hi = min(lo + block, N)
        Z = X[nbr[lo:hi]] - X[lo:hi, None, :]                 # (B, k, P): neighbour minus star
        G = Z @ Z.transpose(0, 2, 1)                           # (B, k, k): the local Gram matrices
        tr = np.trace(G, axis1=1, axis2=2)
        G[:, I, I] += reg * np.where(tr > 0, tr, 1.0)[:, None]   # the ridge on the diagonal (1 if the neighbours coincide)
        w = np.linalg.solve(G, np.ones((hi - lo, k, 1)))[..., 0]
        vals[lo:hi] = w / w.sum(1, keepdims=True)              # normalise to sum one
    # row i holds the weights at the columns of its neighbours
    rows = np.repeat(np.arange(N), k)
    return sp.csr_matrix((vals.ravel(), (rows, nbr.ravel())), shape=(N, N))


def _system(W):
    """The pieces of the normal equations: M = I - W, its transpose, tr(M^T M) and diag(M^T M).  M^T M is never
    formed: the solvers apply it to a vector as M^T (M v), two sparse products.  Its diagonal is the column sums of
    the element-wise square of M, and its trace their total."""
    N = W.shape[0]; M = (sp.eye(N, format="csr") - W).tocsr(); Mt = M.T.tocsr()
    MM = M.multiply(M); tr = float(MM.sum()); diag = np.asarray(MM.sum(0)).ravel()
    return M, Mt, tr, diag


def _check_cg(info, residual, tol, operation, column):
    """Raise RuntimeError for an incomplete or non-finite solve, even when no logger was supplied: a solve that did not
    converge is never returned as a result.  The residual may exceed the requested tolerance by a factor of ten,
    to allow for the difference between the solver's internal residual and the recomputed one."""
    if info != 0 or not np.isfinite(residual) or residual > max(10 * tol, 1e-12):
        raise RuntimeError(f"{operation}: column {column} did not converge "
                           f"(cg info {info}, relative residual {residual:.2e}, tolerance {tol:.2e})")


def refine(C_geo, W, rho=0.003, tol=1e-8, maxiter=20000, log=None):
    """The refinement.  C_geo [N, d]: the geodesic coordinates; W: the weights (lle_weights).  Returns (C [N, d],
    residuals [d]).

    C = argmin |(I - W) C|^2 + mu |C - C_geo|^2: row i of (I - W) C is star i's coordinate minus the weighted average of
    its neighbours', the amount by which the neighbour relation fails for that star.  The first term enforces the
    neighbour relations, the second holds the coordinates near C_geo.  Setting the gradient to zero gives one sparse
    linear system, [(I-W)^T(I-W) + mu I] C = mu C_geo.

    The anchor strength is mu = rho x tr[(I-W)^T(I-W)] / N, the dimensionless rho times the mean diagonal of
    (I-W)^T(I-W), so that rho means the same at any sample size and neighbour count.  As rho -> infinity C -> C_geo;
    smaller rho suppresses more strongly the variations that poorly satisfy the neighbour relations, which can include
    physical structure as well as noise.

    The system is solved by conjugate gradients with a Jacobi (diagonal) preconditioner, one column of C at a time.
    The relative residual |A c - b| / |b| of every column is recomputed and checked rather than trusted; any solve that
    fails to converge raises RuntimeError."""
    if not np.isfinite(rho) or rho <= 0: raise ValueError("rho must be finite and positive")
    N = W.shape[0]; M, Mt, tr, diag = _system(W); mu = rho * tr / N
    # A = M^T M + mu I as a matrix-free operator, and its inverse diagonal as the preconditioner
    Minv = 1.0 / np.maximum(diag + mu, 1e-12)
    A = LinearOperator((N, N), matvec=lambda v: Mt @ (M @ v) + mu * v, dtype=np.float64)
    P = LinearOperator((N, N), matvec=lambda v: Minv * v, dtype=np.float64)
    C_geo = np.asarray(C_geo, np.float64); out = np.empty_like(C_geo); res = []
    for j in range(C_geo.shape[1]):
        # one coordinate at a time: the right-hand side is mu times that column of C_geo
        b = mu * C_geo[:, j]; t0 = time.time()
        x, info = cg(A, b, rtol=tol, maxiter=maxiter, M=P)
        r = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300)); res.append(r); out[:, j] = x
        if log: log(f"refine: column {j} cg info {info}, relative residual {r:.2e}, {time.time() - t0:.1f} s")
        _check_cg(info, r, tol, "refine", j)
    return out, np.array(res)
