"""Armillary: coordinates for a set of stellar spectra from the spectra alone, and labels for every star from a few labelled ones.

A stellar spectrum is determined by a few parameters, so a set of spectra lies on a low-dimensional manifold with those
parameters as its coordinates.  Armillary recovers coordinates on that manifold without labels: a distance between
spectra, a graph of nearest neighbours under it, coordinates from the shortest paths through the graph, and a local
refinement.  The labels of a few stars are then carried to every other star through the same graph.

The method, and where each step lives:

    preprocess.py    the normalised flux: each spectrum divided by a locally fitted continuum
    distance.py      the distance between two spectra, from their cumulative absorption in a hierarchy of wavelength chunks
    lattice.py       the neighbour graph (called the lattice in the code) and the geodesic distances along it
    coordinates.py   coordinates from the geodesic distances, by landmark multidimensional scaling
    refine.py        the locally linear refinement of the coordinates
    labels.py        the transfer of labels through the graph, and the choice of which stars to label
    pipeline.py      Config and fit(): the whole chain with timings; Fit.propagate() for the transfer
    evaluate.py      the error statistic and the scores of coordinates and transferred labels

Most users need only Config, fit, Fit.propagate and labels.density_draw.  The method is described in
Ting & Saad (2026), Armillary: a label-free coordinate system for stellar spectra.
"""
from . import preprocess, distance, lattice, coordinates, refine, labels, evaluate  # noqa: F401
from .pipeline import Config, fit, Fit  # noqa: F401
__version__ = "0.3.1"
