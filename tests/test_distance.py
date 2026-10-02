"""The distance: the L1 distance between feature vectors against the pixel-level integral, and the rules of the cumulative curve."""
import numpy as np
from armillary import distance as dm, preprocess as pp


def _pairs(n, m, seed=0):
    r = np.random.default_rng(seed); a = r.integers(0, n, m); b = (a + r.integers(1, n, m)) % n; return a, b


def test_features_reproduce_the_pixel_integral(synthetic):
    """The L1 distance between two feature vectors, chunk by chunk and multiplied back by the chunk medians, must equal the distance computed pixel by pixel from the two cumulative curves (one column per pixel)."""
    flux = synthetic["flux"]; N, P = flux.shape; good = np.ones((N, P), bool)
    bounds = pp.segment_chunks(np.zeros(P, int), (32,))
    fn = pp.local_renormalise(flux, bounds, good)
    err = np.ones_like(fn); Phi, info = dm.build_features(fn, good, err, [bounds], n_ref=100)
    assert Phi.shape[1] == P                                   # one column for each pixel
    med = info.medians[0]
    for i, j in zip(*_pairs(N, 30)):
        ref = dm.w1_pixel(fn, good, err, bounds, i, j)        # pixel units, one value for each chunk
        got = np.array([np.abs(Phi[i, a:b] - Phi[j, a:b]).sum() for a, b in bounds]) * med
        assert np.allclose(got, ref, rtol=2e-4, atol=1e-4), (got, ref)


def test_depth_is_unclipped_and_masked():
    """A negative depth (flux above the continuum) is kept, a bad pixel carries no absorption, and the guard divides a chunk of (almost) no net absorption by 1e-3 instead of its total."""
    f = np.array([[0.5, 1.2, 0.9, 1.0]]); g = np.array([[True, True, False, True]])
    F = dm.cumulative_curve(f, g, np.ones_like(f))[0]          # equal errors: the negative depth of pixel 1 stays
    assert np.allclose(F, np.cumsum([0.5, -0.2, 0.0, 0.0]) / 0.3)
    assert F[1] < F[0] and np.isclose(F[-1], 1.0)
    F0 = dm.cumulative_curve(np.ones((1, 4)), np.ones((1, 4), bool), np.ones((1, 4)))[0]
    assert np.allclose(F0, 0.0)                                # no absorption: the depth is divided by 1e-3
    Fn = dm.cumulative_curve(np.array([[1.0, 1.0, 1.0, 0.9995]]), np.ones((1, 4), bool), np.ones((1, 4)))[0]
    assert np.allclose(Fn[-1], 0.5)                            # |total| = 5e-4 < 1e-3: divided by 1e-3


def test_error_weights_scale_the_depth():
    """The error weights multiply each depth by the normalised inverse variance; only the relative errors matter, the curve still ends at one, and no errors means the plain depth."""
    f = np.array([[0.5, 0.5, 0.5, 1.0]]); g = np.ones((1, 4), bool)
    e = np.array([[1.0, 2.0, 1.0, 1.0]])                        # the variance of the second pixel is four times larger
    F = dm.cumulative_curve(f, g, e)[0]; w = 1 / e[0] ** 2; w = w / w.mean()
    assert np.allclose(F, np.cumsum(0.5 * w * np.array([1, 1, 1, 0])) / (0.5 * w[:3].sum()))
    assert np.allclose(dm.cumulative_curve(f, g, 3 * e)[0], F)  # only the relative errors are important
    assert np.isclose(F[-1], 1.0)
    assert np.allclose(dm.cumulative_curve(f, g, None)[0], dm.cumulative_curve(f, g, np.ones_like(f))[0])   # no errors: the plain depth


def test_pairwise_matches_numpy():
    """The blockwise exact search must match a direct numpy computation: the full matrix, and the k nearest neighbours with their distances."""
    X = np.random.default_rng(1).normal(size=(300, 50)).astype(np.float32)
    D = dm.pairwise(X, block=64); ref = np.abs(X[:, None, :] - X[None, :, :]).sum(-1)
    assert np.allclose(D, ref, rtol=1e-5, atol=1e-4)
    nbr, dist = dm.pairwise(X, block=64, k=7); np.fill_diagonal(ref, np.inf)
    assert (nbr == np.argsort(ref, 1)[:, :7]).mean() > 0.999
    assert np.allclose(dist, np.sort(ref, 1)[:, :7], rtol=1e-5, atol=1e-4)
