# Reference Fixtures

These CSV files are frozen numerical outputs from the adjacent reference
implementation for first-version core behavior. Runtime tests load only these
static tables; they do not import or execute the adjacent implementation.

The comparisons intentionally allow small tolerances where the native Julia
implementation uses direct quadrature while the reference implementation used
interpolated tables or closed-form special-function expressions.
