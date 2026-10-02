"""Landmark multidimensional scaling against classical multidimensional scaling on the full geodesic table."""
import numpy as np
from scipy.sparse.csgraph import dijkstra
from armillary import coordinates as cm, lattice as lm, distance as dm


def _graph(n=400, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, 3)) @ np.diag([3, 2, 1]); X = np.column_stack([X, np.zeros((n, 5))]).astype(np.float32)
    nbr, dist = dm.pairwise(X, k=12); G, _ = lm.lattice(nbr, dist, X); return G


def test_all_landmarks_equals_classical():
    """With every star a landmark, the landmark formula must reproduce classical MDS exactly: the same eigenvalues, the same coordinates up to the sign of each axis, and the same inner products."""
    G = _graph(); N = G.shape[0]; Dg = dijkstra(G, directed=False)
    C_ref, ev_ref = cm.classical_mds(Dg, d=3, n_eig=10)
    C_lm, ev_lm = cm.landmark_mds(lm.geodesics(G, np.arange(N)), np.arange(N), d=3, n_eig=10)
    assert np.allclose(ev_ref, ev_lm, rtol=1e-10, atol=1e-8 * ev_ref[0])
    sign = np.sign((C_ref * C_lm).sum(0)); assert np.allclose(C_ref, C_lm * sign, rtol=1e-9, atol=1e-9 * np.abs(C_ref).max())
    assert np.allclose(C_ref @ C_ref.T, C_lm @ C_lm.T, rtol=1e-9, atol=1e-9 * (C_ref ** 2).sum())


def test_few_landmarks_are_close():
    """With 80 landmarks out of 400 stars, the pairwise distances of the coordinates must still correlate with the classical ones above 0.99."""
    G = _graph(); N = G.shape[0]; Dg = dijkstra(G, directed=False); C_ref, _ = cm.classical_mds(Dg, d=3)
    lmk = np.random.default_rng(0).permutation(N)[:80]
    C_lm, _ = cm.landmark_mds(lm.geodesics(G, lmk), lmk, d=3)
    d_ref = np.linalg.norm(C_ref[:, None] - C_ref[None], axis=-1); d_lm = np.linalg.norm(C_lm[:, None] - C_lm[None], axis=-1)
    iu = np.triu_indices(N, 1); assert np.corrcoef(d_ref[iu], d_lm[iu])[0, 1] > 0.99
