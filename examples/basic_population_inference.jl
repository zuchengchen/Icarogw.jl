using Icarogw
using Plots
using Random

rng = MersenneTwister(1234)

truth = SimplePowerLawPopulation()
data = simulate_population_data(rng, truth; nevents=2, nsamples=96, ndetected=96, ntotal=800, zmax=0.8)

schema = parameter_schema(SimplePowerLawPopulation)
theta = pack(schema, (
    alpha=2.0,
    beta=1.0,
    mmin=5.0,
    mmax=80.0,
    gamma=0.0,
    R0=25.0,
    H0=67.7,
    Om0=0.308,
))

logl = loglikelihood(SimplePowerLawPopulation, data, theta)
diag = likelihood_diagnostics(SimplePowerLawPopulation, data, theta)

println("loglikelihood = ", logl)
println("per-event Neff = ", diag.per_event_neff)
println("injection Neff = ", diag.injection_neff)
println("xi = ", diag.xi)
println("N_expected = ", diag.N_expected)

plt = plot_likelihood_diagnostics(diag)
savefig(plt, joinpath(@__DIR__, "basic_population_diagnostics.png"))
println("wrote ", joinpath(@__DIR__, "basic_population_diagnostics.png"))
