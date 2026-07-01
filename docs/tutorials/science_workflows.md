# Science Workflow Tutorial

This tutorial shows how the migrated Julia APIs fit together without requiring
Python or public-data downloads. The runnable version is
[`examples/science_workflows.jl`](../../examples/science_workflows.jl).

## What It Covers

- Build a toy `IcarogwCatalog` HDF5 file and evaluate a catalog dark-siren
  likelihood with `CBCCatalogVanillaRate`.
- Evaluate vanilla EM counterpart rates with `CBCVanillaEMCounterpartRate`.
- Evaluate low-latency skymap EM counterpart rates with `LigoSkyMap` and
  `CBCLowLatencySkyMapEMCounterpartRate`.
- Compute a stochastic background spectrum with `precompute_omega_weights` and
  `spectral_siren_omega_gw`.
- Combine Poisson CBC and stochastic likelihoods with `joint_loglikelihood`.
- Create Dynesty-compatible closures through `dynesty_problem`.

## Run

From the repository root:

```sh
julia --project=. examples/science_workflows.jl
```

The example creates all HDF5 and skymap-like inputs in a temporary directory.
It should print finite log likelihoods for every workflow and exit without
writing committed artifacts.

## Data Policy

Default examples and tests are offline. Public real-data workflows belong in
`test/integration/` and are opt-in through `ICAROGW_RUN_PUBLIC_INTEGRATION=1`.
Large or downloaded files should live in the integration cache rather than in
the repository.
