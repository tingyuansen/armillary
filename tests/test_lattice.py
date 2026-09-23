"""Bridging makes the lattice connected; the neighbour lists are what the graph is built from."""
import numpy as np
import pytest
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


def test_identical_spectra_remain_connected_at_zero_distance():
    X = np.zeros((4, 2), dtype=np.float32)
    nbr, dist = lm.neighbours(X, k=1)
    G, info = lm.lattice(nbr, dist, X)
    assert info["components"] == 1 and info["bridges"] == 0
    assert connected_components(G, directed=False)[0] == 1
    np.testing.assert_array_equal(lm.geodesics(G, np.arange(4)), np.zeros((4, 4)))


def test_zero_cost_bridge_joins_disconnected_identical_spectra():
    X = np.zeros((4, 2), dtype=np.float32)
    nbr = np.array([[1], [0], [3], [2]])
    G, info = lm.lattice(nbr, np.zeros((4, 1), dtype=np.float32), X)
    assert info["components"] == 2 and info["bridges"] == 1
    assert info["bridge_edges"][0][2] == 0
    assert G.nnz == 6 and np.all(G.data == 0)
    np.testing.assert_array_equal(lm.geodesics(G, [0]), np.zeros((1, 4)))


def test_zero_edges_preserve_mixed_distance_paths():
    X = np.array([[0], [0], [2]], dtype=np.float32)
    nbr, dist = lm.neighbours(X, k=1)
    G, _ = lm.lattice(nbr, dist, X)
    np.testing.assert_array_equal(lm.geodesics(G, [0]), [[0, 0, 2]])


@pytest.mark.parametrize("k", [0, -1, 3, 1.5])
@pytest.mark.parametrize("search", ["exact", "nndescent"])
def test_invalid_neighbour_count_fails_before_search(k, search):
    with pytest.raises(ValueError, match="positive integer"):
        lm.neighbours(np.zeros((3, 2), dtype=np.float32), k, search=search)
