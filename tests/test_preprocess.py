"""The continuum steps: masked non-finite pixels must not affect the good pixels or the values that fill the bad ones."""
import numpy as np
from armillary import preprocess as pp


def test_local_continuum_ignores_masked_nonfinite_flux():
    """NaN and infinite flux on masked pixels must not change the local continuum, and so the normalised flux and errors on the good pixels."""
    x = np.linspace(-1, 1, 32)
    flux = np.vstack([1 + 0.1 * x, 0.9 - 0.05 * x])
    good = np.ones_like(flux, dtype=bool); good[:, [3, 11, 20]] = False
    masked = flux.copy(); masked[:, [3, 11, 20]] = [np.nan, np.inf, -np.inf]
    err = np.full_like(flux, 0.02)
    ref, ref_err = pp.local_renormalise(flux, [(0, 32)], good, err=err)
    actual, actual_err = pp.local_renormalise(masked, [(0, 32)], good, err=err)
    np.testing.assert_array_equal(actual[good], ref[good])
    np.testing.assert_array_equal(actual_err, ref_err)


def test_fill_bad_uses_only_good_values_and_handles_empty_columns():
    """A bad pixel takes the mean of the good values in its column; a column with no good value takes 1.0."""
    flux = np.array([[1, np.nan, np.inf], [3, 4, np.nan], [np.inf, 6, -np.inf]])
    good = np.array([[True, False, False], [True, True, False], [False, True, False]])
    expected = np.array([[1, 5, 1], [3, 4, 1], [2, 6, 1]], dtype=np.float32)
    np.testing.assert_array_equal(pp.fill_bad(flux, good), expected)
