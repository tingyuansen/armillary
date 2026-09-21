"""Coordinates from geodesic distances.

    classical_mds   B = -1/2 J (D^g)^2 J, its top eigenvectors scaled by the square roots of the eigenvalues
                    (the reference for the tests, on the full N x N table).
    landmark_mds    the same on the m x m table among the landmarks, every other star placed by the linear
                    formula of de Silva & Tenenbaum (2004) from its geodesic distances to the landmarks.
                    With every star a landmark it equals classical_mds to machine precision (tested).

The eigenvalues are returned so that the number of coordinates to keep can be read from them; d is a parameter."""
import numpy as np
from scipy.linalg import eigh


def classical_mds(Dg, d, n_eig=12):
    """Coordinates from a full distance table.  Returns C [N, d] and the top n_eig eigenvalues of B."""
    N = len(Dg); D2 = np.asarray(Dg, np.float64) ** 2; rm = D2.mean(1, keepdims=True)
    B = -0.5 * (D2 - rm - rm.T + D2.mean())                                  # = -1/2 J D2 J
    n_eig = min(n_eig, N); vals, vecs = eigh(B, subset_by_index=[N - n_eig, N - 1]); vals, vecs = vals[::-1], vecs[:, ::-1]
    return vecs[:, :d] * np.sqrt(np.clip(vals[:d], 0, None)), vals


def landmark_mds(Dl, landmarks, d, n_eig=12):
    """Landmark MDS.  Dl [m, N] holds the geodesic distances from the m landmarks to every star.

    B = -1/2 J Dll^2 J on the m x m landmark table; its eigenvectors V and eigenvalues w give
    the landmark coordinates V sqrt(w).  Every star i is then placed by
        c_i = -1/2 L^+ (d_i^2 - mean_l d_l^2),   L^+ = (V / sqrt(w))^T,
    with d_i the vector of its geodesic distances to the landmarks and the mean over landmarks of the squared
    landmark-to-landmark distances subtracted.  Returns C [N, d] and the top n_eig eigenvalues."""
    Dl = np.asarray(Dl, np.float64); landmarks = np.asarray(landmarks); m = len(landmarks)
    Dll = Dl[:, landmarks]
    J = np.eye(m) - np.ones((m, m)) / m
    B = -0.5 * J @ (Dll ** 2) @ J
    w, V = np.linalg.eigh((B + B.T) / 2)
    o = np.argsort(w)[::-1]; w, V = w[o], V[:, o]
    n_eig = min(n_eig, m); dd = min(d, int((w[:d] > 0).sum()))
    Lp = (V[:, :dd] / np.sqrt(w[:dd])).T                                     # the pseudo-inverse transform
    mu = (Dll ** 2).mean(1)                                                  # mean squared distance per landmark
    C = -0.5 * (Lp @ (Dl ** 2 - mu[:, None])).T
    if dd < d: C = np.concatenate([C, np.zeros((C.shape[0], d - dd))], 1)
    return C, w[:n_eig]
