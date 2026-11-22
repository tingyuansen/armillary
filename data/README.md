# The data of the paper

These tables are the data products of Ting & Saad (2026), *Armillary: a label-free coordinate system for stellar spectra*. They hold Armillary's coordinates of every star, the training sets, and the labels transferred from each training set. They also hold the other representations of Figure 5, the errors of Figure 8 and the synthetic grid of Figure 3. The label-transfer errors of Figures 6 to 11 and the R² values of Figure 5 can be computed again from these tables.

The tables come from the saved results of the paper's runs, with no new fit. The spectra are not included. They are public in APOGEE DR17 and DESI DR1, and the identifiers below select them.

## Files

| file | rows | content | figures |
| --- | --- | --- | --- |
| `armillary_apogee.fits` | 202,932 | APOGEE stars: ASPCAP labels, coordinates, training sets of 32 to 8192 stars, transferred labels | 4, 6, 7, 8 |
| `armillary_apogee_baselines.fits` | 202,932 | the same stars in PCA, t-SNE, UMAP, the autoencoder and the Sequencer ordering | 5 |
| `armillary_apogee_ladder.csv` | 30 | the 1σ errors of every representation for each training set | 8 |
| `armillary_desi.fits` | 104,799 | DESI stars: identifiers, ASPCAP labels of the cross-matched stars, signal-to-noise, coordinates, training sets of 8 to 4095 stars | 9, 10, 11 |
| `armillary_desi_sn100.fits` to `armillary_desi_sn20.fits` | 104,799 each | the transferred labels at each nominal signal-to-noise level: 100, 80, 60, 40 and 20 | 9, 10, 11 |
| `armillary_synthetic_grid.fits` | 4,675 | the synthetic label grid and its recovered coordinates | 3 |
| `SHA256SUMS` | | the checksum of each file | |

The rows of the baselines file are in the same order as `armillary_apogee.fits`. The rows of the five DESI level files are in the same order as `armillary_desi.fits`.

To read a table:

```python
from astropy.table import Table
apogee = Table.read("data/armillary_apogee.fits")
```

## Conventions

- Teff is in K. log g, [M/H], [Fe/H] and [α/M] are in dex.
- `C1` to `C4` are the refined coordinates. Their units are arbitrary, and a rotation of them is equally valid.
- `TRAIN_N<n>` is true for the stars of the training set of `n` stars.
- `<LABEL>_N<n>` is the label transferred from that training set, for every star. A training star is pulled toward its ASPCAP label and does not keep it exactly. The paper scores only the stars outside the training set.
- The paper's error is half the width of the central 68 per cent of the differences from ASPCAP, the 16th to 84th percentile range divided by two.
- The training sets were drawn with probability proportional to the local density over the coordinates to the power −1/4 (Section 3.2 of the paper). Each table gives the members of each set, so the sets need not be drawn again.

## `armillary_apogee.fits`

The APOGEE DR17 main-survey stars with no ASPCAP bad bit, no star-level flag, finite labels and a signal-to-noise above 70. Each star has the combined spectrum of highest signal-to-noise.

| column | description |
| --- | --- |
| `APOGEE_ID` | APOGEE identifier |
| `ASPCAP_TEFF`, `ASPCAP_LOGG`, `ASPCAP_M_H`, `ASPCAP_ALPHA_M` | the calibrated ASPCAP `TEFF`, `LOGG`, `M_H` and `ALPHA_M` of allStar, Teff to 0.1 K and the others to 0.0001 dex |
| `SNR` | the allStar signal-to-noise of the spectrum used |
| `C1` to `C4` | Armillary's four coordinates (Figure 4) |
| `TRAIN_N<n>` | true for the training stars, for n = 32, 128, 512, 2048 and 8192 |
| `TEFF_N<n>`, `LOGG_N<n>`, `M_H_N<n>`, `ALPHA_M_N<n>` | the labels transferred from that training set (Figures 6 to 8) |

## `armillary_apogee_baselines.fits`

The representations that Figure 5 compares with Armillary, for the same stars. Section 3.1 of the paper describes how each was made.

| column | description |
| --- | --- |
| `APOGEE_ID` | APOGEE identifier |
| `PCA_1` to `PCA_4` | the first four principal components |
| `TSNE_1` to `TSNE_3` | the three t-SNE coordinates |
| `UMAP_1` to `UMAP_4` | the four UMAP coordinates |
| `AE_1` to `AE_4` | the four latents of the autoencoder |
| `SEQUENCER_ORDER` | the one-dimensional ordering of the Sequencer |

## `armillary_apogee_ladder.csv`

The 1σ errors of Figure 8, over every star outside each training set. Armillary transfers the labels through its graph. Each other representation is read by a multilayer perceptron, at the weight decay of lowest error for each label.

| column | description |
| --- | --- |
| `REPRESENTATION` | `Armillary (graph)`, `autoencoder`, `PCA`, `UMAP`, `t-SNE` or `Sequencer` |
| `N_TRAIN` | the number of training stars: 32, 128, 512, 2048 or 8192 |
| `ERR_TEFF`, `ERR_LOGG`, `ERR_M_H`, `ERR_ALPHA_M` | the 1σ error of each label |

## `armillary_desi.fits`

The 4799 stars of the APOGEE sample that DESI DR1 observed, with a successful and unflagged stellar fit and a red-camera signal-to-noise above 100, and 100,000 further DESI stars with no ASPCAP label. The graph of the paper has 104,801 spectra: two cross-matched stars have two spectra each. This table has one row per star.

| column | description |
| --- | --- |
| `TARGETID`, `SURVEY`, `PROGRAM` | the DESI identifiers of the spectrum used |
| `APOGEE_ID` | the APOGEE identifier of a cross-matched star, empty otherwise |
| `ASPCAP_TEFF`, `ASPCAP_LOGG`, `ASPCAP_FE_H`, `ASPCAP_ALPHA_M` | the calibrated ASPCAP `TEFF`, `LOGG`, `FE_H` and `ALPHA_M` of a cross-matched star, NaN otherwise |
| `SNR` | the median red-camera signal-to-noise of the observed spectrum |
| `SNR_L100` to `SNR_L20` | the same after the noise of each nominal level is added |
| `C1` to `C4` | Armillary's four coordinates of the observed spectra |
| `TRAIN_N<n>` | true for the training stars, for n = 8, 32, 128, 512, 1024, 2048 and 4095 |

The DESI run uses ASPCAP's `FE_H` as its metallicity label. For these 4799 stars it differs from `M_H` by 0.007 dex (1σ). The largest training set has 4096 spectra of 4095 stars, and the paper calls it 4096.

## `armillary_desi_sn<L>.fits`

One file for each nominal signal-to-noise level `L` of Section 4: 100, 80, 60, 40 and 20. At each level the graph was built again from the degraded spectra, with the same training sets.

| column | description |
| --- | --- |
| `TARGETID` | DESI identifier, row by row as `armillary_desi.fits` |
| `TEFF_N<n>`, `LOGG_N<n>`, `FE_H_N<n>`, `ALPHA_M_N<n>` | the labels transferred from each training set |

## `armillary_synthetic_grid.fits`

The 25 × 17 × 11 grid of Section 2.7, with spectra from Payne-Zero.

| column | description |
| --- | --- |
| `TEFF`, `LOGG`, `M_H` | the labels of the grid star |
| `C1` to `C3` | its recovered coordinates (Figure 3) |
