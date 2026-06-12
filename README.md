# Icarogw.jl

`Icarogw.jl` is a native Julia rewrite of the adjacent Python
[`icarogw`](../icarogw) project for compact-binary population inference. The
Julia package is designed around typed models, array-backed data containers,
log-space likelihood evaluation, sampler-independent parameter schemas, and
examples that can run without a Python bridge.

This repository is in migration. The current milestone contains a working
first vertical slice:

- flat `ΛCDM` cosmology and modified-gravity luminosity-distance wrappers
- source/detector mass and distance conversions
- core mass, mass-ratio, spin, and redshift priors/rates
- posterior and injection containers with CSV/HDF5 IO
- selection-corrected hierarchical likelihoods with diagnostics
- no-event, Poisson-rate, and shape-only likelihood modes
- simulation helpers for toy posterior samples and injections
- `Plots.jl` plotting helpers
- local `../Dynesty.jl` compatible closures and example

The formal implementation is native Julia. Package code in `src/` must not use
`PyCall.jl`, `PythonCall.jl`, `CondaPkg.jl`, or another Python bridge.

## Installation

From this checkout:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The package targets Julia `1.10` compatibility. Development on newer Julia
versions is fine, but first-version code avoids Julia 1.11-only APIs.

## Quickstart

```julia
using Icarogw
using Random

rng = MersenneTwister(42)
truth = SimplePowerLawPopulation()
data = simulate_population_data(rng, truth; nevents=2, nsamples=128, ndetected=120)

schema = parameter_schema(SimplePowerLawPopulation)
theta = pack(schema, (
    alpha=2.0, beta=1.0, mmin=5.0, mmax=80.0,
    gamma=0.0, R0=25.0, H0=67.7, Om0=0.308,
))

logl = loglikelihood(SimplePowerLawPopulation, data, theta)
diag = likelihood_diagnostics(SimplePowerLawPopulation, data, theta)
println(logl)
println(diag.per_event_neff)
```

## Examples

Run the end-to-end example:

```sh
julia --project=. examples/basic_population_inference.jl
```

Run the Dynesty integration example:

```sh
julia --project=. examples/dynesty_population_inference.jl
```

The Dynesty example uses the local adjacent checkout `../Dynesty.jl` when it is
available. If that path is missing or incompatible, the example prints a clear
skip message instead of failing mysteriously.

## Relationship To Python `icarogw`

This package is a Julia-native migration of the Python project in `../icarogw`,
which declares `EUPL-1.2`. The Julia rewrite keeps attribution and uses the
same license text by default. It does not copy the Python implementation line by
line; APIs are renamed and restructured around Julia dispatch, typed structs,
and explicit parameter schemas.

See [docs/api_mapping.md](docs/api_mapping.md) for the Python-to-Julia mapping
and [docs/migration_notes.md](docs/migration_notes.md) for design differences.

## Not Supported In First Version

- Condor / HTCondor helper generation
- pickle
- Python GPU backend switching and `cupy_pal.py`
- CUDA/GPU backends
- full galaxy catalog / dark siren / bright siren workflow
- stochastic background / `Omega_GW`
- stochastic/catalog/EM-counterpart mixed likelihoods

Catalog, stochastic, and `Omega_GW` modules are explicit planned placeholders
that throw clear errors.

## Development

Run tests:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

Run benchmarks:

```sh
julia --project=. benchmark/benchmarks.jl
```

No automatic push is performed by the migration workflow.
