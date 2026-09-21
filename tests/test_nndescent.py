"""NN-descent against the exact search on the 4,800-spectrum APOGEE cube at k = 400.

At full scale the same comparison is experiments/survey_ann_check.py (survey/ann_check_<survey>.json, Section 2.6)."""
import time, numpy as np, pytest
from armillary import preprocess as pp, distance as dm, lattice as lm


@pytest.mark.slow
def test_nndescent_overlap_at_k400(apogee):
    flux, good, seg = apogee["flux"], apogee["good"], apogee["seg"]
    fn = pp.local_renormalise(flux, pp.segment_chunks(seg, (8, 7, 5)), good)
    chunkings = [pp.segment_chunks(seg, per) for per in ((4, 4, 3), (8, 7, 5), (16, 14, 10))]
    Phi, _ = dm.build_features(fn, good, apogee["err"], chunkings)
    t = time.time(); nbr_e, dst_e = lm.neighbours(Phi, 400, "exact"); t_e = time.time() - t
    t = time.time(); nbr_a, dst_a = lm.neighbours(Phi, 400, "nndescent", seed=0); t_a = time.time() - t
    ov400, first = lm.overlap(nbr_e, nbr_a, 400); ov30, _ = lm.overlap(nbr_e, nbr_a, 30)
    print(f"\nexact {t_e:.0f}s, nn-descent {t_a:.0f}s; overlap k=30 {ov30:.4f}, k=400 {ov400:.4f}, same first {first:.4f}")
    assert ov400 > 0.9 and ov30 > 0.9
    same = nbr_e == nbr_a; assert np.allclose(dst_e[same], dst_a[same])     # the same distances where the lists agree
