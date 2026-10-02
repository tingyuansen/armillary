"""Coordinates from geodesic distances, by multidimensional scaling.

Classical multidimensional scaling (Torgerson 1952) finds points c_i in d dimensions whose Euclidean separations
approximate a given table of distances.  The squared distance between two points is |c_i|^2 + |c_j|^2 - 2 c_i . c_j,
so the table of squared distances is -2 times the table of inner products plus the squared lengths.  Subtracting the
row and column means removes the squared lengths, and the factor -1/2 removes the -2:

    B = -1/2 J (D^g)^2 J,   J = I - (1/N) 1 1^T,    (D^g)^2 squared entry by entry.

For Euclidean distances B is exactly the table of inner products of centred coordinates, B = C C^T, so the eigenvectors
of B scaled by the square roots of their eigenvalues are the columns of C.  Keeping the d largest gives the d coordinates
that reproduce B as closely as any d coordinates can, and each eigenvalue measures how much of B one coordinate holds:
a few dominant eigenvalues suggest how many coordinates to keep.

    classical_mds   the above on the full N x N table (the reference for the tests)
    landmark_mds    the same on the m x m table among m landmarks; every other star is then placed from its geodesic
                    distances to the landmarks by the linear formula of de Silva & Tenenbaum (2004), as a position is
                    fixed from its distances to a few known points.  With every star a landmark it equals
                    classical_mds to machine precision (tested).

The eigenvalues are returned so that the number of coordinates to keep can be read from them; d is a parameter."""
import numpy as np
from scipy.linalg import eigh


def classical_mds(Dg, d, n_eig=12):
    """Coordinates from a full distance table Dg [N, N].  Returns C [N, d] and the n_eig largest eigenvalues of B."""
    N = len(Dg); D2 = np.asarray(Dg, np.float64) ** 2; rm = D2.mean(1, keepdims=True)
    # double centring written out: subtract the row and column means, add back the grand mean
    B = -0.5 * (D2 - rm - rm.T + D2.mean())                                  # = -1/2 J D2 J
    # only the largest n_eig eigenpairs are computed, then put in decreasing order
    n_eig = min(n_eig, N); vals, vecs = eigh(B, subset_by_index=[N - n_eig, N - 1]); vals, vecs = vals[::-1], vecs[:, ::-1]
    # coordinates = eigenvectors x sqrt(eigenvalue); a negative eigenvalue (non-Euclidean distances) contributes zero
    return vecs[:, :d] * np.sqrt(np.clip(vals[:d], 0, None)), vals


def landmark_mds(Dl, landmarks, d, n_eig=12):
    """Landmark MDS.  Dl [m, N] holds the geodesic distances from the m landmarks to every star; `landmarks` the
    indices of the landmarks among the N stars.  Returns C [N, d] and the n_eig largest eigenvalues.

    Classical MDS on the m x m landmark table, B = -1/2 J Dll^2 J, gives eigenvectors V and eigenvalues w, and the
    landmark coordinates V sqrt(w).  Every star i (landmark or not) is then placed by

        c_i = -1/2 L^+ (d_i^2 - mean_l d_l^2),   L^+ = (V / sqrt(w))^T,

    with d_i^2 the vector of its squared geodesic distances to the landmarks and mean_l d_l^2 the mean squared
    distance from each landmark to the others.  The bracket is the star's row of the centred table of squared
    distances, and L^+ turns it into coordinates (the inner products with the landmark coordinates, divided by the
    eigenvalues).  The eigendecomposition is of an m x m matrix, so the cost no longer grows as N^3.

    If fewer than d eigenvalues are positive, only those coordinates are computed and the rest are zero."""
    Dl = np.asarray(Dl, np.float64); landmarks = np.asarray(landmarks); m = len(landmarks)
    Dll = Dl[:, landmarks]                                                   # the landmark-to-landmark table
    J = np.eye(m) - np.ones((m, m)) / m                                      # the centring matrix
    B = -0.5 * J @ (Dll ** 2) @ J
    w, V = np.linalg.eigh((B + B.T) / 2)                                     # symmetrised against rounding
    o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]                            # decreasing eigenvalues
    n_eig = min(n_eig, m); dd = min(d, int((w[:d] > 0).sum()))              # coordinates with a positive eigenvalue
    Lp = (V[:, :dd] / np.sqrt(w[:dd])).T                                     # the pseudo-inverse transform
    mu = (Dll ** 2).mean(1)                                                  # mean squared distance per landmark
    C = -0.5 * (Lp @ (Dl ** 2 - mu[:, None])).T                              # every star at once, [N, dd]
    if dd < d: C = np.concatenate([C, np.zeros((C.shape[0], d - dd))], 1)
    return C, w[:n_eig]
