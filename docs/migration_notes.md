# Migration Notes

`Icarogw.jl` is not a line-by-line translation of Python `icarogw`. It keeps
the scientific contracts of the first-version scope while using Julia-native
data layouts and dispatch.

## Native Julia Design

The Python project relies on mutable wrapper classes whose `update(**kwargs)`
methods rebuild internal state before likelihood evaluation. The Julia rewrite
separates user-friendly parameter handling from the likelihood hot path:

```julia
schema = parameter_schema(SimplePowerLawPopulation)
theta = prior_transform(schema, u)
model = materialize(SimplePowerLawPopulation, theta)
loglikelihood(model, data)
```

`NamedTuple` input is supported for readability, but sampler-facing evaluation
uses `AbstractVector{<:Real}` and an ordered `ParameterSchema`.

## Data Containers

`PosteriorSamples`, `PosteriorSampleSet`, and `InjectionSet` store values in
dense `Matrix{Float64}` plus a small vector of column names. They can be
constructed from `NamedTuple`, `Dict`, `DataFrame`, or any Tables.jl-compatible
object. CSV and HDF5 IO are supported for first-version workflows.

Pickle is intentionally unsupported.

## Likelihood Interface

The core likelihood is a function, not a `bilby.Likelihood` subclass:

```julia
loglikelihood(SimplePowerLawPopulation, data, theta)
```

It implements posterior-sample reweighting, injection selection correction,
Poisson-rate likelihood, shape-only likelihood, and no-event likelihood.
Diagnostics are returned through `LikelihoodDiagnostics` rather than printed.

Scalar likelihood evaluation is single-core and deterministic. Batch evaluation
uses `theta_matrix` with shape `nparameters x npoints`; `parallel=true` is an
explicit opt-in.

## Dynesty Integration

The core package does not import `Dynesty`. `DynestyInterface.jl` provides
closures:

```julia
problem = dynesty_problem(SimplePowerLawPopulation, data)
sampler = Dynesty.NestedSampler(problem.loglikelihood, problem.prior_transform, problem.ndim)
```

The example in `examples/dynesty_population_inference.jl` loads local
`../Dynesty.jl` if available and otherwise prints a clear skip message.

## Cosmology

The Python implementation delegates baseline distances to Astropy and tabulates
interpolants. The Julia version computes flat `ΛCDM`, constant-`w`, and CPL
`w0-wa` distances with `QuadGK` and inverts luminosity distance with `Roots`.
This is simpler for the first native version and avoids a Python dependency.
Future performance work can add explicit cosmology workspaces or interpolation
caches.

Catalog luminosity formulas that do not require skymap or catalog file formats
are native Julia: `GalaxyLuminosityFunction` covers Python `galaxy_MF`, and
`LogPowerLawAbsMagnitudeRate` covers `log_powerlaw_absM_rate`. The Julia
normalization uses direct quadrature so common Schechter slopes remain finite
where the Python incomplete-gamma branch can return `NaN`.

Catalog formula helpers that are independent of pixelated catalog files are
also native Julia. `KCorrection` and `DeprecatedKCorrection` cover Python
`kcorr` and `kcorr_dep`; `LegacyGalaxyLuminosityFunction` covers Python
`galaxy_MF_dep`; `em_likelihood_prior_differential_volume` covers the EM
redshift likelihood-prior helper for `uniform`, `gaussian`, and
`gaussian_nocom` modes. The first FITS/HEALPix/NUNIQ skymap core and the
Python-compatible `IcarogwCatalog`/`GwcosmoCatalog` runtime HDF5 readers now
exist.

`GalaxyCatalog` provides a lightweight compatibility layer for Python
`galaxy_catalog` single-file HDF5 products. `create_hdf5` writes the `/catalog`
group with Python-style zero-based stored `sky_indices`; the Julia reader
converts those pixels to 1-based rows internally. `calculate_mthr!`,
`return_counts_map`, `calc_mthr`, and `effective_galaxy_number_interpolant`
cover the runtime threshold-map, empty-catalog, and effective-count paths.

Pixelated catalog preprocessing is available as file-level Julia helpers:
`create_pixelated_catalogs`, `clear_empty_pixelated_files`,
`remove_nans_pixelated_files`, `calculate_mthr_pixelated_files`,
`get_redshift_grid_for_files`, `initialize_icarogw_catalog`, and
`calculate_interpolant_files`. They write Python-style `pixel_*.hdf5` shards,
filled-pixel lists, NaN masks, magnitude thresholds, redshift grids, and
per-pixel effective-galaxy interpolants. The Julia-only
`build_icarogw_catalog_from_pixelated_files!` helper then aggregates those
shards into the single HDF5 layout consumed by `IcarogwCatalog`.
Large-scale job orchestration and EM counterpart rate/likelihood integration
remain open catalog workflow gaps.

## Model Composition

Python wrapper classes such as `PowerLaw_PowerLaw`,
`PowerLaw_PowerLaw_Gaussian`, and redshift-linear mass wrappers are represented
by composable Julia pieces: `PowerLawStationary`, `PowerLawLinear`,
`GaussianStationary`, `GaussianLinear`, `MixtureMassPrior`,
`RedshiftMixtureMassPrior`, and `RedshiftConditionalMassDistribution`. This
keeps the sampler-facing API compact while covering the same model families.

Additional CBC coordinate systems are expressed as separate rate model structs:
`CBCMass1Rate`, `CBCMchirpQRate`, `CBCSingleMassRate`,
`CBCTotalMassQRate`, `CBCRedshiftPrimaryQRate`,
`CBCCatalogVanillaRate`, and `CBCCatalogSkyMapRate`. The catalog-aware rates
consume runtime catalog interpolants and a pixelized `:sky_indices` column; PE
weights use sky-dependent catalog values, while injection weights use the
Python-compatible averaged or empty-catalog completeness paths.

Bright-siren counterpart rates are represented by
`CBCVanillaEMCounterpartRate` and `CBCLowLatencySkyMapEMCounterpartRate`.
The vanilla EM model expects posterior samples with `:mass_1`, `:mass_2`,
`:luminosity_distance`, and `:z_EM`; event weights follow Python's weighted
redshift-KDE construction, while injections use GW-only selection correction.
The low-latency skymap EM model expects `:z_EM`, `:right_ascension`, and
`:declination` posterior columns plus one `LigoSkyMap` per event.

Spin variants are represented by `SpinWeightedRate(base_model, spin_prior)`,
which composes `DefaultSpinPrior` or `GaussianSpinPrior` with the mass/redshift
rate model instead of duplicating every Python spin wrapper class.

## Posterior Workflows

Dependency-light posterior workflows are pure Julia functions. `build_parallel_posterior`
creates the matrix workspace used by Python `posterior_samples_catalog`, while
`add_counterpart` attaches EM redshift samples as a `z_EM` column. Sky-direction
filtering can compose with the `:sky_indices` column produced by
`pixelize`/`pixelize_with_catalog` for posterior samples and injections.

## Skymap Core

Julia uses `FITSIO.jl` and `Healpix.jl` for the first skymap runtime layer.
`radec2skymap`, `radec2indeces`, and `indices2radec` cover Python's HEALPix
coordinate helpers, with Julia 1-based indexing by default and an explicit
`zero_based=true` compatibility mode. `LigoSkyMap` reads multi-order FITS
tables with `UNIQ`, `PROBDENSITY`, `DISTMU`, and `DISTSIGMA`, implements the
minimal NUNIQ/MOC lookup needed by Python catalog workflows, and evaluates the
Python-compatible 3D posterior/likelihood. Posterior and injection containers
can now be pixelized against either HEALPix or catalog NUNIQ/MOC rows. Full
catalog and EM rate workflows remain a separate phase that will consume these
primitives.

## Stochastic Data

Python stochastic likelihoods consume dictionaries with `freqs`, `Cf`, and
`sigma2s`. Julia represents the same diagonal Gaussian data as `StochasticData`
and provides `read_stochastic_csv`, `write_stochastic_hdf5`, and
`read_stochastic_hdf5` for lightweight file workflows. Richer covariance or
collaboration-specific stochastic products remain future API design rather
than a Python formula gap.

## Performance Strategy

The first hot path avoids `Dict`, `DataFrame`, and dynamic wrapper updates
inside likelihood loops. Further performance work should focus on:

- precomputed cosmology interpolation workspaces
- fewer temporary vectors in diagnostics
- generated or cached column access plans
- benchmarked specialization for common rate models

## Known Differences

- Numerical distances may differ slightly from Astropy because integration and
  constants are native Julia.
- Schechter luminosity-function normalization uses direct quadrature instead
  of Python's incomplete-gamma branch for negative `alpha + 1` values.
- The simulation helpers are quick, seeded mock-data tools rather than full
  detector simulations. Python-style `injection_set_generator` support is
  exposed as an RNG-explicit helper and returns native `InjectionSet` data
  alongside the Python-style truth columns.
- Python wrapper classes are represented by composable Julia structs rather
  than one mutable class per model family.

## Planned And Excluded Features

Planned after first-version core:

- galaxy catalog / dark siren / bright siren workflows
- mixed stochastic/catalog/EM likelihoods
- optional additional file migration formats such as NPZ

Explicitly excluded:

- Condor / HTCondor scripts
- pickle
- `cupy_pal.py` backend switching
- Python bridge dependencies in package code
- CUDA/GPU backend for first version

## License And Attribution

The Python source project in `../icarogw` declares `EUPL-1.2`. This Julia
rewrite keeps the EUPL license text and attribution, and describes itself as a
native Julia migration rather than an independent reimplementation under a
different license.
