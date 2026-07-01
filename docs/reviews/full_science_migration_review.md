# Full Science Migration Review

## Module

- Julia module or workflow: full native Julia science workflow migration.
- Python reference files and APIs: local `../icarogw/icarogw` modules listed in
  `docs/migration_gap_audit.csv`.
- Migration phase: final full-science migration closure.
- Reviewer/date: Codex, 2026-07-01.

## Coverage

- Python scientific APIs covered: cosmology/conversions, priors/wrappers,
  rates, posterior/injection containers and workflows, likelihoods, simulation
  helpers, catalog/skymap runtimes and preprocessing, bright-siren and
  dark-siren workflows, stochastic/OmegaGW, non-Condor utilities, and CPU bounds
  helpers.
- Python APIs intentionally redesigned: mutable wrapper classes are replaced by
  typed Julia structs, `ParameterSchema`, pure likelihood functions, and
  composable rate/prior pieces; Python class inheritance is replaced by Julia
  dispatch.
- Python APIs intentionally excluded: Condor/HTCondor helpers, pickle, CuPy/GPU
  backend switching, Python bridge dependencies, CUDA/GPU backend execution,
  private/authenticated downloads, CLI/config orchestration.
- File formats covered: CSV, HDF5 catalog/posterior/injection/stochastic data,
  FITS LIGO skymaps, Python-style pixelated catalog HDF5 shards, and
  Python-compatible runtime catalog HDF5 layouts.
- Public examples or tutorials updated: `examples/basic_population_inference.jl`,
  `examples/dynesty_population_inference.jl`,
  `examples/science_workflows.jl`, `examples/README.md`,
  `docs/tutorials/science_workflows.md`, and `docs/release_notes.md`.

## Numerical Evidence

- Fixture files: committed reference fixtures under `test/reference/`; generated
  fixture tooling documented in `scripts/fixtures/README.md`.
- Fixture generation command: `python scripts/fixtures/generate_python_reference.py --suite all-small`.
- Julia verification command: `julia --project=. test/runtests.jl` and
  `julia --project=. -e 'using Pkg; Pkg.test()'`.
- Tolerances and rationale: unit tests use exact checks for structural/file
  compatibility and approximate checks for floating-point formulas; tolerances
  follow native Julia quadrature/root-finding and Python fixture stability.
- Known differences from Python: native Julia cosmology integration may differ
  slightly from Astropy; Schechter normalization uses direct quadrature where
  Python's incomplete-gamma branch can be unstable; wrappers are immutable and
  composable rather than mutable Python classes.

## Integration Evidence

- Offline tests: `julia --project=. test/runtests.jl`.
- Package tests: `julia --project=. -e 'using Pkg; Pkg.test()'`.
- Public-data integration tests: `julia --project=. test/integration/runtests.jl`
  passes and skips public-data suites by default; opt in with
  `ICAROGW_RUN_PUBLIC_INTEGRATION=1`.
- Benchmark coverage: `julia --project=. benchmark/benchmarks.jl` covers core
  likelihoods, stochastic/OmegaGW, catalog runtime and preprocessing, skymap
  core, EM counterpart rates, and PE-only pSEOB weighting.
- Manual example outputs: `examples/basic_population_inference.jl`,
  `examples/science_workflows.jl`, and `examples/dynesty_population_inference.jl`
  are runnable from the repository root; the Dynesty example skips cleanly when
  local `../Dynesty.jl` is absent.

## Risks

- Scientific assumptions: stochastic joint likelihood follows the Python vanilla
  stochastic-spectrum path; catalog/EM stochastic model design is future work.
- Dependency or ecosystem risks: FITSIO, HDF5, Healpix, Plots, and numerical
  dependencies remain Julia 1.10-compatible in `Project.toml`.
- Data licensing or provenance risks: default tests and examples use synthetic
  data only; public-data tests are opt-in and should cache downloads outside
  committed fixtures.
- Performance or scalability risks: distributed catalog orchestration and GPU
  execution are excluded; future work should add explicit workspaces/caches for
  large cosmology/catalog grids.
- Backward compatibility risks: Python API names are documented but not mirrored
  class-for-class; Julia APIs prefer dispatch and typed containers.

## Completion Decision

- Remaining limitations: excluded/future items are documented in
  `README.md`, `docs/migration_notes.md`, and `docs/release_notes.md`.
- Stop conditions reviewed: no Python bridge, no Julia minimum-version change,
  no private/auth data, no unsupported dependency trade-off, and no failed
  required verification in the final run.
- Ready to mark module phase complete: yes.
