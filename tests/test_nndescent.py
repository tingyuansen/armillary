"""Nearest-neighbour descent against the exact search, on all 4,675 tutorial spectra at k = 100."""
import time, numpy as np, pytest
from armillary import preprocess as pp, distance as dm, lattice as lm


@pytest.mark.slow
def test_nndescent_overlap_at_k100(synthetic_all):
    """The approximate lists must share more than 99 per cent of their neighbours with the exact ones, and give identical distances wherever the neighbour is the same."""
    flux = synthetic_all["flux"]; good = np.ones(flux.shape, bool); chunks = pp.segment_chunks(np.zeros(flux.shape[1], int), (32,))
    fn = pp.local_renormalise(flux, chunks, good)
    Phi, _ = dm.build_features(fn, good, None, [chunks])
    t = time.time(); nbr_e, dst_e = lm.neighbours(Phi, 100, "exact"); t_e = time.time() - t
    t = time.time(); nbr_a, dst_a = lm.neighbours(Phi, 100, "nndescent", seed=0); t_a = time.time() - t
    ov100, first = lm.overlap(nbr_e, nbr_a, 100); ov30, _ = lm.overlap(nbr_e, nbr_a, 30)
    print(f"\nexact {t_e:.0f}s, nn-descent {t_a:.0f}s; overlap k=30 {ov30:.4f}, k=100 {ov100:.4f}, same first {first:.4f}")
    assert ov100 > 0.99 and ov30 > 0.99
    same = nbr_e == nbr_a; assert np.allclose(dst_e[same], dst_a[same])     # where the lists agree, the distances are the same
