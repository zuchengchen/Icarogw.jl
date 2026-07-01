# Reference Fixtures

These CSV files are frozen numerical outputs from the adjacent reference
implementation for first-version core behavior. Runtime tests load only these
static tables; they do not import or execute the adjacent implementation.

The comparisons intentionally allow small tolerances where the native Julia
implementation uses direct quadrature while the reference implementation used
interpolated tables or closed-form special-function expressions.

`reference_stochastic_dedf.csv` was generated from local
`../icarogw/icarogw/omega_gw.py` with
`python scripts/fixtures/generate_python_reference.py --suite stochastic-smoke`.
It loads that source file directly so the deterministic spectrum fixture does
not require optional catalog/skymap dependencies such as `healpy`.

New reference fixtures for the full scientific migration should be generated
with scripts under `scripts/fixtures/`, reviewed, and then renamed from the
temporary `generated_reference_*` prefix before committing. The formal Julia
package must not import Python at runtime.
