"""The refinement: the weights, conjugate gradients against the dense solve, and the principal-component projection."""
import numpy as np
import pytest
from armillary import refine as rm, distance as dm


def _case(n=500, seed=0):
    r = np.random.default_rng(seed); X = r.normal(size=(n, 20)); X[:, 3:] *= 0.1
    Phi = X.astype(np.float32); nbr, _ = dm.pairwise(Phi, k=15)
    W = rm.lle_weights(X, nbr, reg=1e-3); C_geo = X[:, :3] + 0.3 * r.normal(size=(n, 3)); return W, C_geo


def test_weights_sum_to_one():
    """Each row of W must sum to one and hold exactly k entries."""
    W, _ = _case(); assert np.allclose(np.asarray(W.sum(1)).ravel(), 1.0)
    assert W.nnz == 500 * 15


def test_cg_equals_dense():
    """The conjugate-gradient refinement must equal the dense solve of [(I-W)^T(I-W) + mu I] C = mu C_geo on 500 stars."""
    W, C_geo = _case(); N = W.shape[0]; rho = 0.01
    M = np.eye(N) - W.toarray(); MtM = M.T @ M; mu = rho * np.trace(MtM) / N
    C_dense = np.linalg.solve(MtM + mu * np.eye(N), mu * C_geo)
    C_cg, res = rm.refine(C_geo, W, rho=rho, tol=1e-12)
    assert res.max() < 1e-10
    assert np.allclose(C_cg, C_dense, rtol=1e-7, atol=1e-8 * np.abs(C_dense).max())


def test_nonconverged_refinement_is_not_returned_as_a_result():
    """A solve stopped before convergence must raise, never return unconverged coordinates."""
    W, C_geo = _case(n=80)
    with pytest.raises(RuntimeError, match="did not converge"):
        rm.refine(C_geo, W, maxiter=1)


@pytest.mark.parametrize("pixels,fit_max,expected", [(3, 30, 3), (8, 4, 3)])
def test_pca_caps_components_by_pixels_and_fit_sample(pixels, fit_max, expected):
    """The number of components is capped by the pixels and by the stars the components are fitted on."""
    flux = np.random.default_rng(0).normal(size=(20, pixels)).astype(np.float32)
    result = rm.pca_flux(flux, np.ones_like(flux, dtype=bool), n_pca=100, fit_max=fit_max)
    assert result.shape == (20, expected) and np.isfinite(result).all()
