"""Armillary: coordinates for a spectroscopic survey from the spectra alone.

The package implements Section 2 of the paper (Construction of the Coordinates).  The steps and where each lives:

    preprocess.py    the normalised flux f_i (Section 2.1, "the flux divided by a locally fitted continuum")
    distance.py      the distance D between two spectra, equations (rho), (curve), (w1) and (metric)
    lattice.py       the neighbour graph (the code's lattice) and the geodesic distances, equation (geodesic)
    coordinates.py   coordinates from geodesic distances, equation (mds), by landmark MDS
    refine.py        the locally linear refinement, equations (lle), (refine) and (solve)
    labels.py        transferring labels through the graph, equations (propagate) and (propsolve)
    pipeline.py      Config and fit(): the whole chain with timings
    evaluate.py      the probes and scores the paper reports
"""
from . import preprocess, distance, lattice, coordinates, refine, labels, evaluate  # noqa: F401
from .pipeline import Config, fit, Fit  # noqa: F401
__version__ = "0.2.0"
