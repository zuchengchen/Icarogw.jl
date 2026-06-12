# Examples

Run examples from the repository root with `julia --project=. path/to/example.jl`.

## `basic_population_inference.jl`

Builds a toy population model, simulates posterior samples and injections,
evaluates the selection-corrected likelihood, prints diagnostics, and writes a
basic diagnostic plot to `examples/basic_population_diagnostics.png`.

Expected output includes a finite log likelihood, per-event effective sample
sizes, `xi`, and `N_expected`.

Optional dependencies: `Plots.jl` is part of the main project dependencies.

## `dynesty_population_inference.jl`

Uses `dynesty_problem` to create sampler-compatible closures and runs a small
local `../Dynesty.jl` nested-sampling smoke test when that checkout is
available. If the adjacent package is unavailable or incompatible, the script
prints a skip message.

Optional dependency: local `../Dynesty.jl`.
