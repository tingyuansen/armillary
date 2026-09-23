"""Propagation by conjugate gradients equals the dense solve of equation (propsolve)."""
import numpy as np
import pytest
import scipy.sparse as sp
from armillary import refine as rm, labels as lb, distance as dm


def _case(n=400, seed=1):
    r = np.random.default_rng(seed); X = r.normal(size=(n, 10)); Phi = X.astype(np.float32); nbr, _ = dm.pairwise(Phi, k=12)
    W = rm.lle_weights(X, nbr, reg=1e-3); Y = X[:, :2] @ np.array([[1.0, 0.5], [-0.3, 2.0]]) + 0.05 * r.normal(size=(n, 2))
    cal = r.choice(n, 40, replace=False); return W, Y, cal


def test_cg_equals_dense():
    W, Y, cal = _case(); N = W.shape[0]; mu = 1.0
    M = np.eye(N) - W.toarray(); MtM = M.T @ M; m = mu * np.trace(MtM) / N; s = np.zeros(N); s[cal] = 1.0
    B = np.zeros((N, 2)); B[cal] = m * Y[cal]; ref = np.linalg.solve(MtM + m * np.diag(s), B)
    P, res = lb.propagate(W, cal, Y[cal], mu=mu, tol=1e-12)
    assert res.max() < 1e-9
    assert np.allclose(P, ref, rtol=1e-6, atol=1e-7 * np.abs(ref).max())


def test_nonconverged_propagation_is_not_returned_as_a_result():
    W, Y, cal = _case(n=80)
    with pytest.raises(RuntimeError, match="did not converge"):
        lb.propagate(W, cal, Y[cal], maxiter=1)


def test_each_disconnected_component_needs_an_anchor():
    W = sp.block_diag([np.array([[0., 1.], [1., 0.]])] * 2, format="csr")
    with pytest.raises(ValueError, match="unanchored component"):
        lb.propagate(W, [0], [[5.]])
    result, _ = lb.propagate(W, [0, 2], [[5.], [9.]])
    np.testing.assert_allclose(result[:, 0], [5, 5, 9, 9], atol=1e-8)


@pytest.mark.parametrize("cal,labels", [([0, 0], [[1.], [2.]]),
                                       ([-1], [[1.]]), ([2], [[1.]]),
                                       ([0.5], [[1.]]), ([], []), ([0], [[np.nan]])])
def test_invalid_anchors_fail_clearly(cal, labels):
    W = sp.csr_matrix([[0., 1.], [1., 0.]])
    with pytest.raises(ValueError): lb.propagate(W, cal, labels)
