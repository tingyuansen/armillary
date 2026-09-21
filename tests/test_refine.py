"""The conjugate-gradient refinement equals the dense solve of equation (solve) on 500 stars."""
import numpy as np
from armillary import refine as rm, distance as dm


def _case(n=500, seed=0):
    r = np.random.default_rng(seed); X = r.normal(size=(n, 20)); X[:, 3:] *= 0.1
    Phi = X.astype(np.float32); nbr, _ = dm.pairwise(Phi, k=15)
    W = rm.lle_weights(X, nbr, reg=1e-3); C_geo = X[:, :3] + 0.3 * r.normal(size=(n, 3)); return W, C_geo


def test_weights_sum_to_one():
    W, _ = _case(); assert np.allclose(np.asarray(W.sum(1)).ravel(), 1.0)
    assert W.nnz == 500 * 15


def test_cg_equals_dense():
    W, C_geo = _case(); N = W.shape[0]; rho = 0.01
    M = np.eye(N) - W.toarray(); MtM = M.T @ M; mu = rho * np.trace(MtM) / N
    C_dense = np.linalg.solve(MtM + mu * np.eye(N), mu * C_geo)
    C_cg, res = rm.refine(C_geo, W, rho=rho, tol=1e-12)
    assert res.max() < 1e-10
    assert np.allclose(C_cg, C_dense, rtol=1e-7, atol=1e-8 * np.abs(C_dense).max())
