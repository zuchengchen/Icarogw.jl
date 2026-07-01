# Release Notes Draft

## Full-Science Migration Milestone

This milestone completes the Julia-native migration of the local Python
`../icarogw` scientific feature set that is in scope for this repository.
Package code remains pure Julia and does not depend on a Python runtime bridge.

## Added

- Catalog and skymap workflows:
  `IcarogwCatalog`, `GwcosmoCatalog`, `GalaxyCatalog`, HEALPix/NUNIQ helpers,
  `LigoSkyMap`, posterior/injection pixelization, magnitude-threshold maps,
  effective galaxy interpolants, catalog diagnostics, and catalog plotting
  helpers.
- Catalog preprocessing:
  Python-compatible `pixel_*.hdf5` helpers, redshift-grid builders,
  magnitude-threshold builders, per-pixel interpolants, and aggregation into
  runtime `IcarogwCatalog` files.
- Bright-siren and dark-siren rates:
  catalog-aware CBC rates, vanilla EM counterpart rates, low-latency skymap EM
  rates, and PE-only pSEOB weighting.
- Stochastic/OmegaGW workflows:
  `dedf`, `precompute_omega_weights`, `spectral_siren_omega_gw`,
  `StochasticData`, CSV/HDF5 stochastic readers, stochastic-only likelihoods,
  and Poisson CBC plus stochastic `joint_loglikelihood` for the Python vanilla
  stochastic-spectrum path.
- Migration evidence:
  expanded API mapping, migration notes, gap audit, benchmarks, offline science
  workflow example, tutorial, and module review evidence.

## Changed

- Python mutable wrapper classes are represented by Julia structs,
  `ParameterSchema`, and composable model pieces.
- Python file-format compatibility is covered for the migrated HDF5/FITS/CSV
  workflows, while pickle remains intentionally unsupported.
- Default tests remain offline. Public-data integration tests are opt-in and
  skip by default.

## Excluded Or Future Extension Points

- Condor/HTCondor helper generation.
- Pickle support.
- CuPy/GPU backend switching and CUDA/GPU execution.
- Python runtime bridges in package code.
- Distributed catalog job orchestration.
- Collaboration-specific stochastic covariance/data-product APIs.
- Catalog/EM stochastic joint likelihoods beyond the Python vanilla
  stochastic-spectrum path.
