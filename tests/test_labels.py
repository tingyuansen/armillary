"""The label transfer: conjugate gradients against the dense solve, the checks on the training set, and the density draw."""
import numpy as np
import pytest
import scipy.sparse as sp
from armillary import refine as rm, labels as lb, distance as dm


def _case(n=400, seed=1):
    r = np.random.default_rng(seed); X = r.normal(size=(n, 10)); Phi = X.astype(np.float32); nbr, _ = dm.pairwise(Phi, k=12)
    W = rm.lle_weights(X, nbr, reg=1e-3); Y = X[:, :2] @ np.array([[1.0, 0.5], [-0.3, 2.0]]) + 0.05 * r.normal(size=(n, 2))
    cal = r.choice(n, 40, replace=False); return W, Y, cal


def test_cg_equals_dense():
    """The conjugate-gradient transfer must equal the dense solve of [(I-W)^T(I-W) + mu P] Y = mu P Yhat on 400 stars."""
    W, Y, cal = _case(); N = W.shape[0]; mu = 1.0
    M = np.eye(N) - W.toarray(); MtM = M.T @ M; m = mu * np.trace(MtM) / N; s = np.zeros(N); s[cal] = 1.0
    B = np.zeros((N, 2)); B[cal] = m * Y[cal]; ref = np.linalg.solve(MtM + m * np.diag(s), B)
    P, res = lb.propagate(W, cal, Y[cal], mu=mu, tol=1e-12)
    assert res.max() < 1e-9
    assert np.allclose(P, ref, rtol=1e-6, atol=1e-7 * np.abs(ref).max())


def test_nonconverged_propagation_is_not_returned_as_a_result():
    """A solve stopped before convergence must raise, never return unconverged labels."""
    W, Y, cal = _case(n=80)
    with pytest.raises(RuntimeError, match="did not converge"):
        lb.propagate(W, cal, Y[cal], maxiter=1)


def test_each_disconnected_component_needs_an_anchor():
    """A component of W without a training star leaves its labels undetermined and must raise; with one training star per component, each component takes its own label."""
    W = sp.block_diag([np.array([[0., 1.], [1., 0.]])] * 2, format="csr")
    with pytest.raises(ValueError, match="unanchored component"):
        lb.propagate(W, [0], [[5.]])
    result, _ = lb.propagate(W, [0, 2], [[5.], [9.]])
    np.testing.assert_allclose(result[:, 0], [5, 5, 9, 9], atol=1e-8)


@pytest.mark.parametrize("cal,labels", [([0, 0], [[1.], [2.]]),
                                       ([-1], [[1.]]), ([2], [[1.]]),
                                       ([0.5], [[1.]]), ([], []), ([0], [[np.nan]])])
def test_invalid_anchors_fail_clearly(cal, labels):
    """Duplicate, out-of-range, non-integer or empty indices, and non-finite labels, must raise ValueError."""
    W = sp.csr_matrix([[0., 1.], [1., 0.]])
    with pytest.raises(ValueError): lb.propagate(W, cal, labels)


def test_density_draw_favours_sparse_regions():
    """The draw must favour a sparse halo (10 per cent of the stars) over a dense core, return unique indices, repeat with the same seed, and stay within the candidates."""
    r = np.random.default_rng(0); C = np.vstack([r.normal(0, 0.1, (900, 2)), r.normal(0, 3.0, (100, 2))])   # a dense core and a sparse halo
    pick = lb.density_draw(C, 100, seed=1)
    assert len(np.unique(pick)) == 100 and (pick >= 900).mean() > 0.2                 # the halo has 10 percent of the stars
    assert np.array_equal(pick, lb.density_draw(C, 100, seed=1))
    cand = np.arange(0, 1000, 2); assert np.isin(lb.density_draw(C, 50, candidates=cand), cand).all()
