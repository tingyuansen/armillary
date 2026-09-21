"""Bridging makes the lattice connected; the neighbour lists are what the graph is built from."""
import numpy as np
from scipy.sparse.csgraph import connected_components
from armillary import lattice as lm, distance as dm


def test_bridging_connects_pieces():
    r = np.random.default_rng(0); A = r.normal(size=(200, 6)); B = r.normal(size=(150, 6)) + 40; C = r.normal(size=(60, 6)) - 40
    X = np.vstack([A, B, C]).astype(np.float32); nbr, dist = dm.pairwise(X, k=8)
    G, info = lm.lattice(nbr, dist, X)
    assert info["components"] == 3 and info["bridges"] == 2
    assert connected_components(G, directed=False)[0] == 1
    assert (G != G.T).nnz == 0                                  # symmetric
    for i, j, w in info["bridge_edges"]:                       # each bridge is the shortest edge to the largest piece
        d = np.abs(X[i] - X[j]).sum(); assert np.isclose(w, d, rtol=1e-5)
        assert i >= 200 and j < 200
    Dl = lm.geodesics(G, [0, 250, 380]); assert np.isfinite(Dl).all()


def test_edge_weights_are_D():
    X = np.random.default_rng(2).normal(size=(120, 5)).astype(np.float32); nbr, dist = dm.pairwise(X, k=5)
    G, _ = lm.lattice(nbr, dist, X)
    for i in range(120):
        for j, d in zip(nbr[i], dist[i]): assert np.isclose(G[i, j], d)
