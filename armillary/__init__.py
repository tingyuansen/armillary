"""Armillary: coordinates for a spectroscopic survey from the spectra alone, and labels for every star from a few labelled ones.

The method, and where each step lives:

    preprocess.py    the normalised flux: each spectrum divided by a locally fitted continuum
    distance.py      the distance between two spectra, from their cumulative absorption in a hierarchy of wavelength chunks
    lattice.py       the neighbour graph (called the lattice in the code) and the geodesic distances along it
    coordinates.py   coordinates from the geodesic distances, by landmark multidimensional scaling
    refine.py        the locally linear refinement of the coordinates
    labels.py        the transfer of labels through the graph
    pipeline.py      Config and fit(): the whole chain with timings
    evaluate.py      the error statistic and the probes used to score coordinates and transferred labels

The method is described in Ting & Saad (2026), Armillary: a label-free coordinate system for stellar spectra.
"""
from . import preprocess, distance, lattice, coordinates, refine, labels, evaluate  # noqa: F401
from .pipeline import Config, fit, Fit  # noqa: F401
__version__ = "0.2.0"
