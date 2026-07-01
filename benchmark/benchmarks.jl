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

freqs = collect(range(20.0, 100.0; length=16))
omega_weights = precompute_omega_weights(MersenneTwister(9), freqs; tmp_min=5.0, tmp_max=80.0, n=256)

println("stochastic Omega_GW spectrum")
display(@benchmark spectral_siren_omega_gw($model, $omega_weights))

zgrid = collect(range(0.01, 0.3; length=64))
kc = KCorrection("bJ-glade+")
legacy_lf = LegacyGalaxyLuminosityFunction("K"; epsilon=0.8)
catalog_cosmology = FlatLambdaCDM(zmax=1.0)

println("catalog formula helpers")
display(@benchmark $kc($zgrid))
display(@benchmark background_effective_galaxy_density($legacy_lf, [-26.0, -22.0, -18.0]))
display(@benchmark em_likelihood_prior_differential_volume($zgrid, 0.1, 0.03, $catalog_cosmology; ptype="gaussian"))
