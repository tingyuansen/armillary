# Armillary

Armillary makes a coordinate system for a set of stellar spectra from the spectra alone. It then transfers labels from a few labelled stars to all the other stars through the same neighbour graph. The labels can be effective temperature, surface gravity, abundances, or other measured quantities.

The method uses no model spectra and no training set to put the stars in the coordinates. Labels are necessary only for the last step. In the paper, a few dozen labelled stars give the main stellar-parameter structure of a sample of hundreds of thousands of stars.

Ting & Saad (2026), *Armillary: a label-free coordinate system for stellar spectra*, gives the method. This repository contains the package and a tutorial on synthetic spectra.

![The seven steps of Armillary](examples/armillary_schematic.png)

*The panels show the steps from left to right. Two spectra and their cumulative absorption curves: the area between the curves is the distance. The table of the distances between all the pairs. The neighbour graph on the manifold of the spectra. A geodesic along the graph. The coordinates from the geodesic distances. The same coordinates after the refinement. The labels of four labelled stars (the orange stars), transferred to all the other stars.*

## How it works

1. **The spectra.** Armillary divides each spectrum by a local continuum. It then makes one cumulative absorption curve for each wavelength chunk, with chunks of different widths.
2. **The distance.** The distance between two spectra is the sum of the areas between their curves. Each chunk is in units of its median over pairs of stars. This distance is the Wasserstein distance between the absorption profiles, extended to negative depths. Thus the noise above the continuum stays and is not clipped.
3. **The graph.** Each star is joined to its nearest neighbours under the distance, and the graph is made connected. The distances along the graph are the geodesic distances.
4. **The coordinates.** Landmark multidimensional scaling changes the geodesic distances into coordinates.
5. **The refinement.** Each spectrum is nearly a weighted sum of the spectra of its neighbours. The refinement makes the coordinates agree with these weights, and keeps them near the coordinates of step 4.
6. **The labels.** The labels go from the labelled stars to all the other stars, through the same weights. A penalty keeps each labelled star near its known labels.

| step | module | what it calculates |
| --- | --- | --- |
| 1, 2 | `armillary.preprocess`, `armillary.distance` | the normalised flux; the feature vectors, whose L1 distance is the distance between the spectra |
| 3 | `armillary.lattice` | the connected k-nearest-neighbour graph, and the geodesic distances from a set of landmarks |
| 4 | `armillary.coordinates` | the coordinates, by landmark multidimensional scaling |
| 5 | `armillary.refine` | the locally linear refinement of the coordinates |
| 6 | `armillary.labels` | the label transfer through the same weights, and the selection of the stars to label |
| | `armillary.pipeline` | `Config`, `fit` and `Fit`: all the steps in one function, with the time of each step |
| | `armillary.evaluate` | the error statistic and the scores of coordinates and transferred labels |

## Installation

Use Python 3.10 or later.

```bash
git clone https://github.com/tingyuansen/armillary.git
cd armillary
python -m pip install -e .
```

This command installs the package and the packages that it uses: NumPy, SciPy, scikit-learn, numba and pynndescent. numba compiles the distance kernels when you use them for the first time. To set the number of cores, use `Config.threads` or the environment variable `NUMBA_NUM_THREADS`.

To run the tests and the tutorial, install the optional packages:

```bash
python -m pip install -e ".[test,tutorial]"
python -m pytest tests -m "not slow"   # the fast tests
jupyter lab armillary_tutorial.ipynb
```

## Quick start

```python
import numpy as np, armillary as ar

z = np.load("examples/synthetic_spectra.npz")
flux = z["flux"].astype(np.float32)          # [N, P] continuum-normalised flux
good = np.ones(flux.shape, bool)             # [N, P] good pixels
segments = np.zeros(flux.shape[1], int)      # [P] the detector segment of each pixel

config = ar.Config(chunkings=((32,),), d=3, k_refine=50)   # one chunking of 32 chunks, three coordinates
F = ar.fit(flux, good, segments, config)
C = F.C                                      # the coordinates, [N, d]

labelled = ar.labels.density_draw(F.C, 100)  # the stars to label
Y = F.propagate(z["labels"], labelled_index=labelled)   # the labels of all the stars, [N, 3]
```

`fit` returns a `Fit`, which holds the intermediate products. These are the normalised flux, the feature vectors, the neighbours of each star with their distances, and the graph. They also include the eigenvalues, the coordinates before and after the refinement, and the time of each step. The `Fit` does not keep the geodesic distances.

`Fit.propagate` transfers a table of labels. `Fit.save(path)` writes the coordinates and the data that is necessary to plot or score them again. This file does not contain the full `Fit`. To transfer more labels later, keep the `Fit` in memory.

## The tutorial

[`armillary_tutorial.ipynb`](armillary_tutorial.ipynb) applies the full method to the 4,675 synthetic spectra in `examples/synthetic_spectra.npz`. [Payne-Zero](https://github.com/tingyuansen/payne-zero) calculated these spectra from 480 to 680 nm, at a resolving power of 10,000. The grid goes from 4000 to 7000 K in effective temperature, from 1 to 5 in surface gravity, and from −2 to +0.5 in metallicity. Each spectrum has its true labels.

The tutorial shows these items:

1. The showcase: the coordinates recover the label grid without labels, at three metallicities. This is Figure 3 of the paper, which uses the same grid at a resolving power of 20,000.
2. The seven panels of the schematic above, and the attribute of the `Fit` that holds each product.
3. The labels from 50 labelled stars, transferred to the other 4,625 stars. Then the same transfer at a signal-to-noise ratio of 30.
4. Noise that changes along the spectrum, with sky-line pixels: the error-weighted distance against the plain distance.
5. How to use the same calls on your own spectra.

The tutorial runs in two to three minutes on a laptop. `examples/tutorial_figures.py` makes the figures, and `examples/figure_style.py` sets their style.

## Configuration

`Config` holds all the parameters of the method, and `armillary/pipeline.py` describes each field. The defaults of the method settings are the values of the paper. The default `chunkings`, `n_landmarks` and `search` are a starting point: change them for your data. For a fixed configuration, Armillary uses no labels to make the coordinates.

Change these fields for your data:

| field | default | when to change it |
| --- | --- | --- |
| `chunkings` | `((10,), (20,), (40,))` | Give the number of chunks in each detector segment, for each chunking, from wide chunks to narrow chunks. For three segments, an example is `((4, 4, 3), (8, 7, 5), (16, 14, 10))`. |
| `continuum` | `"none"` | Set `"running"` if the spectra keep their instrumental response. |
| `error_weights` | `False` | Set `True` if the errors have a structure of their own, for example from sky lines or detector features. Then also give `err`. Keep `False` if the errors follow the photon noise. In that case, the weights move absorption into the strong lines. |
| `d` | `4` | Set the number of coordinates. Use `Fit.eigenvalues` to select it. |
| `n_landmarks`, `search` | `None`, `"exact"` | For more than a few tens of thousands of stars, set `n_landmarks=800` and `search="nndescent"`. |

`Config.save` and `Config.load` write and read a configuration as JSON.

## Input format

`fit(flux, good, segments, config=None, err=None, log=print, stop=None)`:

- `flux`: `[N, P]` float. The spectra on one wavelength grid. If `continuum="none"`, the spectra must be continuum-normalised.
- `good`: `[N, P]` bool. False on a bad pixel. A bad pixel is not used in a fit and has no absorption.
- `segments`: `[P]` int. The detector segment of each pixel, from 0. A chunk does not cross the gap between two segments.
- `err`: `[N, P]` float. The standard deviation of each pixel, in the units of the input flux. `fit` uses `err` only if `error_weights=True`. If you do not give `err`, `fit` uses an array of ones.
- `log`: a function that gets one progress line for each step. Use `log=None` for no output.
- `stop`: stop after an earlier step. The docstring of `fit` gives the values.

`Fit.propagate` uses an `[N, L]` array of labels, with one row for each star of the fit. `labelled_index` gives the rows with known labels. The values in the other rows are not used.

To select the stars to label, use `armillary.labels.density_draw(F.C, n)`. This function selects `n` stars as the paper does, with more stars in the sparse regions of the coordinates. Use `candidates` to select only from the stars that you can get labels for.

The indices of the labelled stars must be unique and in the range of the rows, and their labels must be finite. Each connected component of the weights must contain a labelled star. If the labelled stars are not valid, the functions raise `ValueError`. If a chunking does not give a number of chunks for each segment, `fit` raises `ValueError`. If the refinement or the transfer does not converge, the functions raise `RuntimeError`.

## Tests

```bash
python -m pytest tests -m "not slow"         # the fast tests, a few seconds
python -m pytest tests -m slow                # nearest-neighbour descent against the exact search, about 30 seconds
```

The tests compare the feature vectors with the distance added pixel by pixel. They compare landmark scaling with classical multidimensional scaling, and the conjugate-gradient refinement and transfer with dense solves. They also make sure that the bridges connect the graph with the shortest edges, that invalid inputs raise clear errors, and that the density draw selects more stars in sparse regions. The slow test compares nearest-neighbour descent with the exact search on all the tutorial spectra.

## Citation

If you use Armillary, cite Ting & Saad (2026), *Armillary: a label-free coordinate system for stellar spectra*.

## Licence

MIT. See `LICENSE`.
