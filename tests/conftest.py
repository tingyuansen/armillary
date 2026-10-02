import os, numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH = os.path.join(ROOT, "examples", "synthetic_spectra.npz")   # the tutorial's data, shared by the tests


@pytest.fixture(scope="session")
def synthetic():
    """300 of the tutorial's synthetic spectra (every ninth star), enough for the fast tests."""
    z = np.load(SYNTH); idx = np.arange(0, len(z["flux"]), 9)[:300]
    return dict(flux=z["flux"][idx].astype(np.float32), labels=z["labels"][idx].astype(float))


@pytest.fixture(scope="session")
def synthetic_all():
    """All 4,675 of the tutorial's synthetic spectra, for the slow nearest-neighbour check."""
    z = np.load(SYNTH); return dict(flux=z["flux"].astype(np.float32), labels=z["labels"].astype(float))
