# Migration Gap Audit

This audit is the control surface for completing the local Python
`../icarogw` scientific feature set in native Julia. It complements
`docs/api_mapping.md`, which records the first-version mapping that already
exists. The machine-readable source of this audit is
`docs/migration_gap_audit.csv`.

## Status Meanings

- `implemented`: a native Julia implementation already exists and is covered by
  at least the first-version tests or reference fixtures.
- `partial`: Julia has a related abstraction, but the Python scientific
  behavior is not fully covered yet.
- `missing`: no Julia implementation exists for the scientific behavior.
- `excluded`: intentionally out of scope for this goal.

## Fixture Priority

- `existing`: static reference fixtures are already committed.
- `high`: needed before or during the next implementation phase.
- `medium`: needed before marking the corresponding module complete.
- `low`: useful for legacy compatibility or edge families, but not a phase
  blocker unless the implementation chooses to expose that workflow.
- `none`: excluded or not applicable.

## Migration Phases

The current recommended order is:

1. Gap audit and fixture-generation infrastructure.
2. Skymap and HEALPix-style coordinate helpers.
3. Catalog preprocessing, runtime catalog types, k-corrections, and EM helpers.
4. Stochastic background and `Omega_GW`, implemented once behind a unified API.
5. Catalog/EM/stochastic rate models and likelihood integration.
6. Remaining priors, spin families, simulation helpers, utility helpers, and
   migration readers.
7. Observability, workspace/cache diagnostics, tutorials, examples, benchmarks,
   and release notes.

Each phase should keep the repository testable. Stage commits are allowed after
the phase's relevant verification passes, but this audit should not be marked
complete until the final review checklist proves full scientific coverage.

## Current Largest Gaps

- `catalog.py`: complete pixelated catalog preparation, `galaxy_catalog`,
  EM counterpart rates, and GW/EM workflow integration. The
  `IcarogwCatalog` and `GwcosmoCatalog` runtime HDF5 readers now cover
  NUNIQ/HEALPix lookup, magnitude thresholds, sky-dependent and averaged
  effective galaxy interpolants, and empty-catalog mode; k-corrections,
  Schechter luminosity functions, absolute-magnitude rates, and EM redshift
  helpers are also implemented separately.
- `stochastic.py` and `omega_gw.py`: duplicated `dEdf`, omega-weight, and
  spectral-siren logic is unified in Julia through deterministic
  energy-spectrum, vanilla spectral-siren, Gaussian stochastic-only, simple
  stochastic CSV/HDF5 readers, and vanilla CBC+stochastic likelihood helpers.
  Richer covariance/data-product APIs and catalog/EM mixed stochastic
  likelihoods remain open.
- `rates.py` and `likelihood.py`: EM counterpart rates, full stochastic
  data-product support, and catalog/EM stochastic joint likelihoods.
  Catalog-aware CBC rates now consume runtime catalog interpolants and
  pixelized `:sky_indices`. Standard pSEOB weighting composes through
  `SpinWeightedRate`, while Python's pSEOB dummy injection asymmetry remains a
  specialized gap.
- `posterior_samples.py` and `injections.py`: higher-level catalog-aware
  workflows. Dependency-light posterior parallel workspaces, counterpart
  redshift-column attachment, HEALPix/catalog pixelization, non-catalog cuts,
  effective-sample-size, expected-detection, and reweighting helpers are
  implemented as pure Julia functions.
- `conversions.py`: HEALPix coordinate helpers and the first `LigoSkyMap`
  multi-order FITS/NUNIQ workspace are implemented, and the runtime catalog
  readers plus posterior/injection pixelization consume those skymap
  primitives. Higher-level catalog/EM workflows still need integration. Joint
  effective-spin KDE helpers are covered by RNG-explicit Julia implementations.
- `priors.py` and `wrappers.py`: standalone advanced priors, extended spin
  families, and dependency-free dip/Farah/bin/multi-peak paired mass wrapper
  compositions are implemented with fixture coverage. Redshift-linear mixture
  wrapper families are represented by explicit `RedshiftMixtureMassPrior`
  weight functions.
- `simulation.py`: detector/source-frame SNR helpers, measurement-noise
  helpers, flat-SNR scaling, deterministic quick likelihood factors,
  frequency/SNR cuts, mass/distance proposal generators, Python-style
  injection-set generation, quick data preparation, and quick PE resampling are
  implemented with fixture or unit-test coverage.
- `utils.py` and `cupy_pal.py`: non-Condor validation helper and CPU bounds
  helpers are implemented. Condor and GPU switching remain excluded.

## Reference Fixture Policy

Use local `../icarogw` as the Python source of truth. The formal Julia package
must not depend on Python, but fixture-generation scripts may import the local
Python package and write small static CSV/HDF5/NPZ-style reference artifacts
under `test/reference/`.

For stochastic and simulation workflows that use Monte Carlo draws, fixture
scripts must set deterministic NumPy seeds and use small sample sizes for unit
fixtures. Public real-data workflows belong in `test/integration/` and may
download/cache data outside committed large files.

If Python behavior appears numerically unstable or scientifically suspect,
document the difference and test the normal operating region instead of blindly
reproducing a bug.

## Stop Conditions

Stop and ask before:

- adding another major dependency with unclear Julia 1.10 support,
- relying on an external tool as part of the normal Julia package workflow,
- treating ambiguous Python behavior as normative,
- using data with unclear license or authentication requirements,
- dropping Python file-format compatibility for an important workflow,
- changing the Julia minimum version,
- introducing a formal Python bridge, or
- declaring completion with any required verification failed or unrun.
