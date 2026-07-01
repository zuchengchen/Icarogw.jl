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
- `stochastic-smoke`: deterministic `omega_gw.dEdf` values for the future
  Julia stochastic/OmegaGW implementation.
- `catalog-smoke`: reserved for the catalog migration phase and intentionally
  not implemented yet.
- `all-small`: all currently implemented small suites.

Generated files use the prefix `generated_reference_`. Review them before
renaming or committing them as stable fixtures.
