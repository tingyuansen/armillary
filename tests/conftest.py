import os, numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH = os.path.join(ROOT, "examples", "synthetic_spectra.npz")   # the tutorial's data, shared by the tests


@pytest.fixture(scope="session")
def synthetic():
    """300 of the tutorial's synthetic spectra, enough for the fast tests: every ninth of the first 2,700, which span
    Teff 4000 to 5750 K and the full ranges of log g and [Fe/H]."""
    z = np.load(SYNTH); idx = np.arange(0, len(z["flux"]), 9)[:300]
    return dict(flux=z["flux"][idx].astype(np.float32), labels=z["labels"][idx].astype(float))


@pytest.fixture(scope="session")
def synthetic_all():
    """All 4,675 of the tutorial's synthetic spectra, for the slow nearest-neighbour check."""
    z = np.load(SYNTH); return dict(flux=z["flux"].astype(np.float32), labels=z["labels"].astype(float))
