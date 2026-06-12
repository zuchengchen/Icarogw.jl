# Notebook-Oriented Workflow

This repository does not require committed notebook outputs for the first
migration milestone. Use the examples as notebook seeds:

1. Start Julia with `julia --project=.`.
2. Load `Icarogw`, `Random`, and `Plots`.
3. Copy the workflow from `examples/basic_population_inference.jl` into a
   Pluto, IJulia, or VS Code notebook.
4. Keep large posterior and injection arrays in HDF5 via `write_hdf5` rather
   than pickle.

The sampler-facing pattern is:

```julia
schema = parameter_schema(SimplePowerLawPopulation)
theta = prior_transform(schema, rand(length(schema)))
loglikelihood(SimplePowerLawPopulation, data, theta)
```
