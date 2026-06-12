module Simulation

using DataFrames
using Distributions
using Random
using StatsBase
using ..Cosmology
using ..Conversions
using ..DataContainers
using ..Priors
using ..Rates

export simulate_sources,
    snr_samples,
    apply_snr_cut,
    generate_posterior_samples,
    generate_injections,
    simulate_population_data

"""
    simulate_sources(rng, n, model; zmax=2)

Draw source-frame component masses and redshifts from a population model. The
current implementation supports `SimplePowerLawPopulation` and returns a
`NamedTuple` with source and detector-frame columns.
"""
function simulate_sources(rng::AbstractRNG, n::Integer, model::SimplePowerLawPopulation; zmax=2.0)
    m1s, m2s = rand_prior(rng, model.mass, n)
    z = sample_comoving_volume(rng, model.cosmology, n; zmin=1e-4, zmax)
    m1d, m2d, dl = source_to_detector(m1s, m2s, z, model.cosmology)
    return (mass_1_source=m1s, mass_2_source=m2s, redshift=z, mass_1=m1d, mass_2=m2d,
        luminosity_distance=dl, chirp_mass=chirp_mass(m1d, m2d), mass_ratio=m2d ./ m1d)
end
simulate_sources(n::Integer, model::SimplePowerLawPopulation; rng=Random.default_rng(), kwargs...) =
    simulate_sources(rng, n, model; kwargs...)

"""
    snr_samples(rng, mass_1, mass_2, luminosity_distance; kwargs...)

Approximate observed SNR samples using the Python simulation scaling. Masses are
detector-frame solar masses and luminosity distance is in Mpc.
"""
function snr_samples(rng::AbstractRNG, mass_1, mass_2, luminosity_distance; numdet=3, rho_s=9.0, dL_s=1.5, Md_s=25.0, theta=nothing)
    n = length(mass_1)
    theta_vec = theta === nothing ? rand(rng, Uniform(0, 1.4), n) : Float64.(theta)
    md = chirp_mass(mass_1, mass_2)
    rho_true = rho_s .* theta_vec .* (md ./ Md_s).^(5 / 6) .* ((dL_s * 1000) ./ luminosity_distance)
    rho_obs = Vector{Float64}(undef, n)
    for i in eachindex(rho_obs)
        rho_obs[i] = sqrt(rand(rng, NoncentralChisq(2numdet, rho_true[i]^2)))
    end
    return (rho_true=rho_true, theta=theta_vec, rho_obs=rho_obs)
end
snr_samples(mass_1, mass_2, luminosity_distance; rng=Random.default_rng(), kwargs...) =
    snr_samples(rng, mass_1, mass_2, luminosity_distance; kwargs...)

"""
    apply_snr_cut(samples, snr; snr_threshold=12, fgw_cut=15)

Return a Boolean detection mask applying an observed SNR threshold and an ISCO
frequency cut.
"""
function apply_snr_cut(samples, snr; snr_threshold=12.0, fgw_cut=15.0)
    fisco = f_gw_isco.(samples.mass_1, samples.mass_2)
    return (snr .>= snr_threshold) .& (fisco .>= fgw_cut)
end

"""
    generate_posterior_samples(rng, truth; nsamples=256, frac_mass_sigma=0.08, frac_dl_sigma=0.15)

Generate quick toy posterior samples around one true detector-frame event. The
detector-frame prior is taken as uniform in the local proposal box, sufficient
for smoke tests and examples.
"""
function generate_posterior_samples(rng::AbstractRNG, truth; nsamples=256, frac_mass_sigma=0.08, frac_dl_sigma=0.15, event_name::Symbol=:event)
    m1 = abs.(rand(rng, Normal(truth.mass_1, max(0.1, frac_mass_sigma * truth.mass_1)), nsamples))
    m2 = abs.(rand(rng, Normal(truth.mass_2, max(0.1, frac_mass_sigma * truth.mass_2)), nsamples))
    for i in eachindex(m1)
        if m2[i] > m1[i]
            m1[i], m2[i] = m2[i], m1[i]
        end
    end
    dl = abs.(rand(rng, Normal(truth.luminosity_distance, max(1.0, frac_dl_sigma * truth.luminosity_distance)), nsamples))
    prior = fill(1.0, nsamples)
    return PosteriorSamples((mass_1=m1, mass_2=m2, luminosity_distance=dl, prior=prior); event_name)
end

"""
    generate_injections(rng, model; ndetected=200, ntotal=2000, zmax=2, snr_threshold=8)

Generate an injection set by repeatedly drawing sources and applying the SNR
and ISCO detection cuts.
"""
function generate_injections(rng::AbstractRNG, model::SimplePowerLawPopulation; ndetected=200, ntotal=2000, zmax=2.0, snr_threshold=8.0, Tobs=1.0)
    m1 = Float64[]
    m2 = Float64[]
    dl = Float64[]
    generated = 0
    while length(m1) < ndetected && generated < max(ntotal, ndetected)
        batch = min(4096, max(ntotal - generated, ndetected))
        s = simulate_sources(rng, batch, model; zmax)
        snr = snr_samples(rng, s.mass_1, s.mass_2, s.luminosity_distance).rho_obs
        mask = apply_snr_cut(s, snr; snr_threshold)
        append!(m1, s.mass_1[mask])
        append!(m2, s.mass_2[mask])
        append!(dl, s.luminosity_distance[mask])
        generated += batch
    end
    ntake = min(ndetected, length(m1))
    ntake > 0 || throw(ArgumentError("no injections passed the detection cut; lower snr_threshold or increase ntotal"))
    prior = fill(1.0, ntake)
    return InjectionSet((mass_1=m1[1:ntake], mass_2=m2[1:ntake], luminosity_distance=dl[1:ntake], prior=prior); ntotal=max(ntotal, generated), Tobs)
end
generate_injections(model::SimplePowerLawPopulation; rng=Random.default_rng(), kwargs...) =
    generate_injections(rng, model; kwargs...)

"""
    simulate_population_data(rng, model; nevents=3, nsamples=256, ndetected=300)

Build a small end-to-end dataset containing toy posterior samples and
injections. This is intended for examples, tests, and benchmarks.
"""
function simulate_population_data(rng::AbstractRNG, model::SimplePowerLawPopulation; nevents=3, nsamples=256, ndetected=300, ntotal=3000, zmax=1.5)
    sources = simulate_sources(rng, max(20, 5nevents), model; zmax)
    snr = snr_samples(rng, sources.mass_1, sources.mass_2, sources.luminosity_distance).rho_obs
    mask = apply_snr_cut(sources, snr; snr_threshold=6.0)
    idx = findall(mask)
    length(idx) >= nevents || (idx = collect(1:nevents))
    events = PosteriorSamples[]
    for (j, i) in enumerate(idx[1:nevents])
        truth = (mass_1=sources.mass_1[i], mass_2=sources.mass_2[i], luminosity_distance=sources.luminosity_distance[i])
        push!(events, generate_posterior_samples(rng, truth; nsamples, event_name=Symbol("event$j")))
    end
    inj = generate_injections(rng, model; ndetected, ntotal, zmax, snr_threshold=6.0)
    return PopulationData(PosteriorSampleSet(events), inj)
end
simulate_population_data(model::SimplePowerLawPopulation; rng=Random.default_rng(), kwargs...) =
    simulate_population_data(rng, model; kwargs...)

end
