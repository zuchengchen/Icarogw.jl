using BenchmarkTools
using Icarogw
using Random

rng = MersenneTwister(7)
model = SimplePowerLawPopulation()
data = simulate_population_data(rng, model; nevents=2, nsamples=128, ndetected=128, ntotal=1000, zmax=0.8)
schema = parameter_schema(SimplePowerLawPopulation)
theta = pack(schema, (
    alpha=2.0, beta=1.0, mmin=5.0, mmax=80.0,
    gamma=0.0, R0=25.0, H0=67.7, Om0=0.308,
))

println("likelihood evaluation")
display(@benchmark loglikelihood(SimplePowerLawPopulation, $data, $theta))

println("diagnostics and weighting")
display(@benchmark likelihood_diagnostics(SimplePowerLawPopulation, $data, $theta))

println("simulation mock generation")
display(@benchmark simulate_population_data($rng, $model; nevents=1, nsamples=64, ndetected=64, ntotal=500, zmax=0.5))
