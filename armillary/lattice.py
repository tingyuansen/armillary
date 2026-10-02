"""The neighbour graph (called the lattice in the code) and the geodesic distances along it.

The distance D (distance.py) is meant to measure LOCAL changes of a spectrum: it is trustworthy between neighbours,
but need not grow in proportion to the parameter separation between distant stars.  The distance along the manifold
between two distant stars is therefore assembled from short steps.  Every star is joined to its k nearest neighbours
under D, and the geodesic distance between two stars is the length of the shortest path through that graph, the sum of
D over the edges of the path.  The coordinates (coordinates.py) are then built from these geodesic distances.

    neighbours    the k nearest neighbours of every star under D, by the exact blockwise search of
                  distance.pairwise or by nearest-neighbour descent (pynndescent, manhattan metric, seeded),
                  chosen by search="exact" | "nndescent".  Both return the same (nbr, dist) arrays, sorted by
                  the same float32 L1 distance, the star itself excluded.
    lattice       the graph G: every star joined to its k nearest neighbours, the edges symmetrised and weighted
                  by D; every separate piece joined to the largest piece by the single shortest edge between them.
    geodesics     the geodesic distances from a set of landmarks, by Dijkstra's algorithm on G.
    overlap       how much two neighbour lists agree (to check an approximate search against the exact one).
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from .distance import pairwise, _l1_cand, _l1_rows_to, _l1_block


def neighbours(features, k, search="exact", block=256, seed=0, verbose=False):
    """Each star's k nearest neighbours under D, the L1 distance between feature vectors.

    search="exact" compares every pair (O(N^2), exact; fine up to a few tens of thousands of stars).
    search="nndescent" uses nearest-neighbour descent, which avoids evaluating every pair (for large samples).
    Returns (nbr int32 [N, k], dist float32 [N, k]), sorted by distance, the star itself excluded."""
    # validate k before either search starts, so that a bad value fails at once and the same way for both
    if not isinstance(k, (int, np.integer)) or not 0 < k < len(features):
        raise ValueError(f"k = {k} must be a positive integer smaller than N = {len(features)}")
    if search == "exact": return pairwise(features, block=block, k=k, verbose=verbose)
    if search == "nndescent": return _nndescent(features, k, seed=seed, verbose=verbose)
    raise ValueError(f"unknown search {search!r}: 'exact' or 'nndescent'")


def _nndescent(features, k, seed=0, verbose=False):
    """Nearest-neighbour descent (Dong et al. 2011) on the feature vectors under the manhattan (L1) metric.

    The descent starts from random neighbour lists and improves them by testing the neighbours of each star's current
    neighbours, so it never evaluates every pair.  Here it supplies the candidate lists only: every distance is then
    recomputed with the exact kernel (_l1_cand, the same float32 kernel the exact search uses) and each row re-sorted on
    those values.  The result therefore has the exact search's semantics: sorted by the same float32 L1 distance, the
    star itself excluded, and distances bit-identical wherever the neighbour is the same.  A row the descent left short
    (a heap slot never filled, marked -1) is replaced by that star's exact neighbours.

    overlap() measures how well the lists agree with the exact search."""
    import warnings, pynndescent
    warnings.filterwarnings("ignore")
    X = np.ascontiguousarray(features, np.float32); N = len(X); t0 = time.time()
    # ask for k + 1 neighbours: pynndescent's list normally includes the star itself, which is dropped below
    index = pynndescent.NNDescent(X, n_neighbors=k + 1, metric="manhattan", random_state=seed, low_memory=True)
    raw, _ = index.neighbor_graph
    out = np.empty((N, k), np.int32); short = []
    for i in range(N):
        # drop the star itself and the unfilled slots (-1); keep the first k of the rest
        row = raw[i][(raw[i] != i) & (raw[i] >= 0)]
        if len(row) >= k: out[i] = row[:k]
        else: short.append(i); out[i] = 0
    if short:
        # the rare rows with fewer than k valid candidates: an exact search of those stars against everyone
        rows = np.asarray(short, np.int64); b = np.empty((len(rows), N), np.float32)
        _l1_rows_to(X, rows, np.arange(N, dtype=np.int64), b)
        for i, r in enumerate(rows): b[i, r] = np.inf
        out[rows] = np.argpartition(b, k, axis=1)[:, :k]
        if verbose: print(f"   nn-descent: {len(rows)} short row(s) filled exactly", flush=True)
    # recompute every candidate's distance with the exact kernel, then sort each row by it (stable, as the exact search)
    dst = _l1_cand(X, np.arange(N, dtype=np.int64), out.astype(np.int64), np.empty((N, k), np.float32))
    order = np.argsort(dst, axis=1, kind="stable")
    out = np.take_along_axis(out, order, axis=1); dst = np.take_along_axis(dst, order, axis=1)
    if verbose: print(f"   nn-descent: {N} rows in {time.time() - t0:6.0f}s", flush=True)
    return out, dst


def lattice(nbr, dist, features=None, verbose=False, log=None):
    """The lattice G from the neighbour lists.

    nbr, dist [N, k]: each star's neighbours and distances (from neighbours()).  features: the feature vectors, needed
    only if the graph is disconnected.  Returns (G, a symmetric csr matrix [N, N] whose entries are the edge weights D;
    info, a dict with the number of components, their sizes, and the bridges).

    G has an edge (i, j) of weight D_ij whenever j is among the k nearest of i OR i among the k nearest of j (the
    union, so the graph is symmetric).  Such a graph can fall into separate pieces: a group of unusual stars may have
    all its nearest neighbours within the group, and a star in such a piece would have no path to the rest.  Each
    smaller piece is therefore joined to the LARGEST piece by one edge, the shortest of all the edges that could join
    the two, computed from the feature vectors.  The bridges make G connected, but they can span sparsely sampled
    regions, so they place the separate pieces only approximately."""
    N, k = nbr.shape
    # every (star, neighbour) pair as a sparse entry, weight D
    rows = np.repeat(np.arange(N), k); cols = nbr.ravel(); vals = np.asarray(dist, np.float64).ravel()
    # Sparse maximum drops explicit zero edges. Keep them during graph assembly
    # using a marker below any positive float32 distance, then restore exact zeros
    # before returning the graph (and before any geodesic calculation).
    zero_edge = np.nextafter(0.0, 1.0)
    vals = np.where(vals == 0, zero_edge, vals)
    m = rows != cols                                                         # never a self-loop
    # G.maximum(G.T) symmetrises: (i, j) is an edge if either star lists the other (both lists give the same weight)
    G = sp.coo_matrix((vals[m], (rows[m], cols[m])), shape=(N, N)).tocsr(); G = G.maximum(G.T)
    ncomp, lab = connected_components(G, directed=False)
    info = dict(components=int(ncomp), bridges=0, bridge_edges=[], piece_sizes=sorted(np.bincount(lab).tolist(), reverse=True))
    if log: log(f"lattice: k-NN graph has {ncomp} component(s)" + (f", sizes {info['piece_sizes']}" if ncomp > 1 else ""))
    if ncomp == 1:
        G.data[G.data == zero_edge] = 0.0
        return G, info
    # ---- several pieces: bridge each smaller one to the largest
    if features is None: raise ValueError("the k-NN graph is disconnected; the feature vectors are needed to bridge it")
    X = np.ascontiguousarray(features, np.float32)
    order = np.argsort(np.bincount(lab))[::-1]; main = order[0]; tgt = np.where(lab == main)[0].astype(np.int64)
    er, ec, ev = [], [], []                                                  # the bridge edges, both directions
    for c in order[1:]:
        # the shortest edge from any star of piece c to any star of the largest piece, 64 rows of c at a time
        idx = np.where(lab == c)[0].astype(np.int64); best = (np.inf, -1, -1)
        for lo in range(0, len(idx), 64):
            rows_ = idx[lo:lo + 64]; d = _l1_rows_to(X, rows_, tgt, np.empty((len(rows_), len(tgt)), np.float32))
            i_, j_ = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[i_, j_] < best[0]: best = (float(d[i_, j_]), int(rows_[i_]), int(tgt[j_]))
        er += [best[1], best[2]]; ec += [best[2], best[1]]; ev += [best[0], best[0]]
        info["bridge_edges"].append((best[1], best[2], best[0]))
    # add the bridges (zero-cost bridges protected by the same marker), restore the zeros, and confirm one component
    ev = np.asarray(ev); ev = np.where(ev == 0, zero_edge, ev)
    B = sp.coo_matrix((ev, (er, ec)), shape=(N, N)).tocsr(); G = G.maximum(B)
    G.data[G.data == zero_edge] = 0.0
    assert connected_components(G, directed=False)[0] == 1
    info["bridges"] = ncomp - 1
    if log: log(f"lattice: bridged {ncomp - 1} component(s)")
    return G, info


def geodesics(G, landmarks):
    """The geodesic distance from every landmark l to every star j: the length of the shortest path through G, by
    Dijkstra's algorithm, one run per landmark.  Apart from the bridges, every path is made of neighbour distances,
    where D is most informative, so the geodesic distance keeps growing with separation along the manifold where D
    itself no longer tracks it.  Running from M landmarks instead of all N stars costs O(M k N log N) instead of
    O(k N^2 log N).  Returns float64 [n_landmarks, N]."""
    Dl = dijkstra(G, directed=False, indices=np.asarray(landmarks))
    # an infinite distance means a star no path reaches: the graph was not connected (lattice() prevents this)
    if not np.isfinite(Dl).all():
        raise ValueError("unreachable stars: the lattice is not connected")
    return Dl


def overlap(nbr_a, nbr_b, k):
    """How much two neighbour lists agree: (the mean fraction of the first k neighbours shared between the two lists,
    the fraction of stars whose first neighbour is the same).  Used to check nearest-neighbour descent against the
    exact search."""
    a, b = nbr_a[:, :k], nbr_b[:, :k]; N = len(a)
    # per star: the size of the intersection of its two lists, as a fraction of k
    ov = np.fromiter((len(np.intersect1d(a[i], b[i], assume_unique=False)) for i in range(N)), float, N) / k
    return float(ov.mean()), float(np.mean(a[:, 0] == b[:, 0]))
