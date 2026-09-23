"""The neighbour graph (the lattice) and the geodesic distances along it.

    neighbours    the k nearest neighbours of every star under D, by the exact blockwise search of
                  distance.pairwise or by nearest-neighbour descent (pynndescent, manhattan metric, seeded),
                  chosen by search="exact" | "nndescent".  Both return the same (nbr, dist) arrays, sorted by
                  the same float32 L1 distance, self excluded.
    lattice       the graph G: every star joined to its k nearest neighbours, edges symmetrised and weighted
                  by D; every separate piece joined to the largest by the single shortest edge between a
                  star inside it and a star of the largest piece.  No minimum spanning tree.
    geodesics     the geodesic distances from a set of landmarks, by Dijkstra's algorithm on G.
    overlap       how much two neighbour lists agree (to check an approximate search against the exact one).
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from .distance import pairwise, _l1_cand, _l1_rows_to, _l1_block


def neighbours(features, k, search="exact", block=256, seed=0, verbose=False):
    """Each star's k nearest neighbours under D (the L1 distance between feature vectors).

    Returns (nbr int32 [N, k], dist float32 [N, k]) sorted by distance, self excluded."""
    if not isinstance(k, (int, np.integer)) or not 0 < k < len(features):
        raise ValueError(f"k = {k} must be a positive integer smaller than N = {len(features)}")
    if search == "exact": return pairwise(features, block=block, k=k, verbose=verbose)
    if search == "nndescent": return _nndescent(features, k, seed=seed, verbose=verbose)
    raise ValueError(f"unknown search {search!r}: 'exact' or 'nndescent'")


def _nndescent(features, k, seed=0, verbose=False):
    """Nearest-neighbour descent (Dong et al. 2011) on the feature vectors under the manhattan (L1) metric.

    The search supplies candidate lists only.  Every distance is then recomputed with the exact kernel
    (_l1_cand, the kernel the exact search sums) and each row re-sorted on those values, so the result has the
    exact search's semantics: sorted by the same float32 L1 distance, self excluded, distances bit-identical
    where the neighbour is the same.  A row the descent left short (a heap slot never filled, marked -1) is
    replaced by its exact neighbours.

    overlap() measures how well it agrees with the exact search."""
    import warnings, pynndescent
    warnings.filterwarnings("ignore")
    X = np.ascontiguousarray(features, np.float32); N = len(X); t0 = time.time()
    index = pynndescent.NNDescent(X, n_neighbors=k + 1, metric="manhattan", random_state=seed, low_memory=True)
    raw, _ = index.neighbor_graph
    out = np.empty((N, k), np.int32); short = []
    for i in range(N):
        row = raw[i][(raw[i] != i) & (raw[i] >= 0)]
        if len(row) >= k: out[i] = row[:k]
        else: short.append(i); out[i] = 0
    if short:
        rows = np.asarray(short, np.int64); b = np.empty((len(rows), N), np.float32)
        _l1_rows_to(X, rows, np.arange(N, dtype=np.int64), b)
        for i, r in enumerate(rows): b[i, r] = np.inf
        out[rows] = np.argpartition(b, k, axis=1)[:, :k]
        if verbose: print(f"   nn-descent: {len(rows)} short row(s) filled exactly", flush=True)
    dst = _l1_cand(X, np.arange(N, dtype=np.int64), out.astype(np.int64), np.empty((N, k), np.float32))
    order = np.argsort(dst, axis=1, kind="stable")
    out = np.take_along_axis(out, order, axis=1); dst = np.take_along_axis(dst, order, axis=1)
    if verbose: print(f"   nn-descent: {N} rows in {time.time() - t0:6.0f}s", flush=True)
    return out, dst


def lattice(nbr, dist, features=None, verbose=False, log=None):
    """The lattice G from the neighbour lists.

    G has an edge (i, j) of weight D_ij whenever j is among the k nearest of i or i among the k nearest of j.
    If the graph falls into several pieces, each piece is joined to the largest by one edge, the shortest
    between a star inside it and a star of the largest piece, found from the feature vectors (needed only
    then).  Returns (G csr symmetric, info dict with the number of components, their sizes and the bridges)."""
    N, k = nbr.shape
    rows = np.repeat(np.arange(N), k); cols = nbr.ravel(); vals = np.asarray(dist, np.float64).ravel()
    # Sparse maximum drops explicit zero edges. Keep them during graph assembly
    # using a marker below any positive float32 distance, then restore exact zeros
    # before returning the graph (and before any geodesic calculation).
    zero_edge = np.nextafter(0.0, 1.0)
    vals = np.where(vals == 0, zero_edge, vals)
    m = rows != cols
    G = sp.coo_matrix((vals[m], (rows[m], cols[m])), shape=(N, N)).tocsr(); G = G.maximum(G.T)
    ncomp, lab = connected_components(G, directed=False)
    info = dict(components=int(ncomp), bridges=0, bridge_edges=[], piece_sizes=sorted(np.bincount(lab).tolist(), reverse=True))
    if log: log(f"lattice: k-NN graph has {ncomp} component(s)" + (f", sizes {info['piece_sizes']}" if ncomp > 1 else ""))
    if ncomp == 1:
        G.data[G.data == zero_edge] = 0.0
        return G, info
    if features is None: raise ValueError("the k-NN graph is disconnected; the feature vectors are needed to bridge it")
    X = np.ascontiguousarray(features, np.float32)
    order = np.argsort(np.bincount(lab))[::-1]; main = order[0]; tgt = np.where(lab == main)[0].astype(np.int64)
    er, ec, ev = [], [], []
    for c in order[1:]:
        idx = np.where(lab == c)[0].astype(np.int64); best = (np.inf, -1, -1)
        for lo in range(0, len(idx), 64):
            rows_ = idx[lo:lo + 64]; d = _l1_rows_to(X, rows_, tgt, np.empty((len(rows_), len(tgt)), np.float32))
            i_, j_ = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[i_, j_] < best[0]: best = (float(d[i_, j_]), int(rows_[i_]), int(tgt[j_]))
        er += [best[1], best[2]]; ec += [best[2], best[1]]; ev += [best[0], best[0]]
        info["bridge_edges"].append((best[1], best[2], best[0]))
    ev = np.asarray(ev); ev = np.where(ev == 0, zero_edge, ev)
    B = sp.coo_matrix((ev, (er, ec)), shape=(N, N)).tocsr(); G = G.maximum(B)
    G.data[G.data == zero_edge] = 0.0
    assert connected_components(G, directed=False)[0] == 1
    info["bridges"] = ncomp - 1
    if log: log(f"lattice: bridged {ncomp - 1} component(s)")
    return G, info


def geodesics(G, landmarks):
    """The geodesic distance from every landmark l to every star j: Dijkstra's algorithm on the
    lattice, one run per landmark.  Returns float64 [n_landmarks, N]."""
    Dl = dijkstra(G, directed=False, indices=np.asarray(landmarks))
    if not np.isfinite(Dl).all():
        raise ValueError("unreachable stars: the lattice is not connected")
    return Dl


def overlap(nbr_a, nbr_b, k):
    """Mean fraction of the first k neighbours shared between two neighbour lists, and the fraction of stars
    whose first neighbour agrees."""
    a, b = nbr_a[:, :k], nbr_b[:, :k]; N = len(a)
    ov = np.fromiter((len(np.intersect1d(a[i], b[i], assume_unique=False)) for i in range(N)), float, N) / k
    return float(ov.mean()), float(np.mean(a[:, 0] == b[:, 0]))
