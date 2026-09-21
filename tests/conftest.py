import os, numpy as np, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTH = os.path.join(ROOT, "examples", "synthetic_spectra.npz")
APOGEE = os.environ.get("ARMILLARY_APOGEE_CUBE", "")   # the paper's 4,000-star APOGEE test cube, for the slow nn-descent check


@pytest.fixture(scope="session")
def synthetic():
    """Three hundred of the tutorial's synthetic spectra."""
    z = np.load(SYNTH); idx = np.arange(0, len(z["flux"]), 9)[:300]
    return dict(flux=z["flux"][idx].astype(np.float32), labels=z["labels"][idx].astype(float))


@pytest.fixture(scope="session")
def apogee():
    if not APOGEE or not os.path.exists(APOGEE): pytest.skip("set ARMILLARY_APOGEE_CUBE to the paper's APOGEE test cube")
    z = np.load(APOGEE, allow_pickle=True); common = z["good"].mean(0) > 0.95
    return dict(flux=z["flux"][:, common], good=z["good"][:, common], err=z["err"][:, common], seg=z["segment_id"][common].astype(int), labels=z["labels"].astype(float), set_id=z["set_id"])
