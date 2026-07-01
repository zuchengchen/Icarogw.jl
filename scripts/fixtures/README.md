# Python Reference Fixture Generation

These scripts are development tools for producing small, static fixtures from
the local Python reference checkout at `../icarogw`.

The formal Julia package must not depend on Python. Python is used here only to
generate reference artifacts that can be committed under `test/reference/` when
they are small, deterministic, and license-safe.

## Environment

Use a Python environment capable of importing the local reference package. The
reference package declares Python 3.12 and dependencies such as `bilby`,
`mhealpy`, `ligo.skymap`, `astropy`, and `seaborn`.

From this repository, a typical invocation is:

```sh
python scripts/fixtures/generate_python_reference.py --suite core
```

The script prepends `../icarogw` to `sys.path`; it does not install or modify
the Python reference checkout.

## Suites

- `core`: tiny conversion fixture used to smoke-test the Python environment.
- `stochastic-smoke`: deterministic `omega_gw.dEdf` values for the Julia
  stochastic/OmegaGW implementation. This suite loads `omega_gw.py` directly
  instead of importing the Python package, so it does not require catalog
  dependencies such as `healpy`.
- `simulation-utils-smoke`: deterministic formulas from `simulation.py`,
  `utils.py`, and `cupy_pal.py`. It uses lightweight import stubs for optional
  Python dependencies and does not exercise random draws or catalog/skymap
  workflows.
- `priors-rates-smoke`: deterministic formulas from `priors.py` plus the
  `CBC_mixte_pop_rate` logaddexp mixture identity. It covers window helpers,
  `Bivariate2DGaussian`, and mixture-rate arithmetic without mutable Python
  wrapper state.
- `catalog-smoke`: dependency-light Schechter luminosity-function and
  absolute-magnitude rate formulas from `cosmology.py`; it does not load
  catalog/skymap files or optional HEALPix dependencies.
- `catalog-formulas-smoke`: dependency-light K-correction and EM redshift
  likelihood-prior formulas from `catalog.py`; it stubs optional HDF5/HEALPix
  imports and does not exercise catalog file workflows.
- `all-small`: all currently implemented small suites.

Generated files use the prefix `generated_reference_`. Review them before
renaming or committing them as stable fixtures.
