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
    chirp_mass_detector,
    f_gw,
    z_to_dl,
    dl_to_z,
    dvc_dz_fullsky,
    snr_samples,
    snr_samples_source,
    snr_samples_detector,
    snr_samples_flat,
    chirp_mass_noise,
    mass_ratio_noise,
    theta_noise,
    noise,
    apply_snr_cut,
    snr_and_freq_cut,
    snr_cut_flat,
    likelihood_evaluation,
    generate_posterior_samples,
    generate_injections,
    simulate_population_data

const PLANCK15_FLATLCDM = FlatLambdaCDM(H0=67.7, Om0=0.308, zmax=14.0)

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
    chirp_mass_detector(m1_source, m2_source, z)

Detector-frame chirp mass `(1 + z) * chirp_mass(m1_source, m2_source)`.
"""
chirp_mass_detector(m1_source, m2_source, z) = chirp_mass(m1_source, m2_source) .* (1 .+ z)

"""
    f_gw(m1_source, m2_source, z)

Approximate detector-frame GW frequency in Hz from the Python `f_GW` helper.
"""
f_gw(m1_source, m2_source, z) = f_gw_isco.(m1_source .* (1 .+ z), m2_source .* (1 .+ z))

"""
    z_to_dl(z; cosmology=PLANCK15_FLATLCDM)

Convert redshift to luminosity distance in Mpc using the package's Planck-like
default cosmology unless another cosmology is supplied.
"""
z_to_dl(z; cosmology::AbstractCosmology=PLANCK15_FLATLCDM) = luminosity_distance(cosmology, z)

"""
    dl_to_z(dl; cosmology=PLANCK15_FLATLCDM)

Invert luminosity distance in Mpc using the package's Planck-like default
cosmology unless another cosmology is supplied.
"""
dl_to_z(dl; cosmology::AbstractCosmology=PLANCK15_FLATLCDM) = redshift_at_luminosity_distance(cosmology, dl)

"""
    dvc_dz_fullsky(z; cosmology=PLANCK15_FLATLCDM)

Full-sky differential comoving volume in Gpc^3 per unit redshift.
"""
dvc_dz_fullsky(z; cosmology::AbstractCosmology=PLANCK15_FLATLCDM) = dvc_dz(cosmology, z)

_as_vector(x::AbstractVector{<:Real}) = Float64.(x)
_as_vector(x::Real) = [Float64(x)]
_restore_shape(template::Real, values::Vector{Float64}) = only(values)
_restore_shape(template, values::Vector{Float64}) = values

function _snr_samples_detector(rng::AbstractRNG, mass_1, mass_2, luminosity_distance; numdet=3, rho_s=9.0, dL_s=1.5, Md_s=25.0, theta=nothing)
    m1 = _as_vector(mass_1)
    m2 = _as_vector(mass_2)
    dl = _as_vector(luminosity_distance)
    length(m1) == length(m2) == length(dl) || throw(ArgumentError("mass and distance inputs must have the same length"))
    n = length(m1)
    theta_vec = theta === nothing ? rand(rng, Uniform(0, 1.4), n) : _as_vector(theta)
    length(theta_vec) == n || throw(ArgumentError("theta must have length $n"))
    md = chirp_mass(m1, m2)
    rho_true = rho_s .* theta_vec .* (md ./ Md_s).^(5 / 6) .* ((dL_s * 1000) ./ dl)
    rho_obs = Vector{Float64}(undef, n)
    for i in eachindex(rho_obs)
        rho_obs[i] = sqrt(rand(rng, NoncentralChisq(2numdet, rho_true[i]^2)))
    end
    return (rho_true=rho_true, theta=theta_vec, rho_obs=rho_obs)
end

"""
    snr_samples(rng, mass_1, mass_2, luminosity_distance; kwargs...)

Approximate observed SNR samples using the Python simulation scaling. Masses are
detector-frame solar masses and luminosity distance is in Mpc.
"""
snr_samples(rng::AbstractRNG, mass_1, mass_2, luminosity_distance; kwargs...) =
    _snr_samples_detector(rng, mass_1, mass_2, luminosity_distance; kwargs...)
snr_samples(mass_1, mass_2, luminosity_distance; rng=Random.default_rng(), kwargs...) =
    snr_samples(rng, mass_1, mass_2, luminosity_distance; kwargs...)

"""
    snr_samples_detector(rng, m1_detector, m2_detector, luminosity_distance; kwargs...)

Detector-frame version of the SNR scaling. This is the Julia-native equivalent
of Python `snr_samples_det`.
"""
snr_samples_detector(rng::AbstractRNG, mass_1, mass_2, luminosity_distance; kwargs...) =
    _snr_samples_detector(rng, mass_1, mass_2, luminosity_distance; kwargs...)
snr_samples_detector(mass_1, mass_2, luminosity_distance; rng=Random.default_rng(), kwargs...) =
    snr_samples_detector(rng, mass_1, mass_2, luminosity_distance; kwargs...)

"""
    snr_samples_source(rng, m1_source, m2_source, z; cosmology=PLANCK15_FLATLCDM, kwargs...)

Source-frame version of the SNR scaling. This mirrors Python `snr_samples`:
source masses are converted to detector-frame chirp mass and redshift is
converted to luminosity distance before drawing observed SNRs.
"""
function snr_samples_source(rng::AbstractRNG, mass_1_source, mass_2_source, z; cosmology::AbstractCosmology=PLANCK15_FLATLCDM, kwargs...)
    m1s = _as_vector(mass_1_source)
    m2s = _as_vector(mass_2_source)
    zs = _as_vector(z)
    length(m1s) == length(m2s) == length(zs) || throw(ArgumentError("source masses and redshifts must have the same length"))
    m1d = m1s .* (1 .+ zs)
    m2d = m2s .* (1 .+ zs)
    dl = luminosity_distance(cosmology, zs)
    return _snr_samples_detector(rng, m1d, m2d, dl; kwargs...)
end
snr_samples_source(mass_1_source, mass_2_source, z; rng=Random.default_rng(), kwargs...) =
    snr_samples_source(rng, mass_1_source, mass_2_source, z; kwargs...)

"""
    snr_samples_flat(z; alpha=1)

Flat-PSD toy SNR scaling `alpha / z`.
"""
snr_samples_flat(z; alpha=1.0) = alpha ./ z

function _noise_vector(rng::AbstractRNG, values, rho_obs, scale)
    vals = _as_vector(values)
    rho = _as_vector(rho_obs)
    length(vals) == length(rho) || throw(ArgumentError("values and rho_obs must have the same length"))
    out = vals .+ randn(rng, length(vals)) .* (scale .* vals .* 10 ./ rho)
    return _restore_shape(values, out)
end

"""
    chirp_mass_noise(rng, Md, rho_obs)

Draw noisy detector-frame chirp masses with Python's quick-PE approximation.
"""
chirp_mass_noise(rng::AbstractRNG, Md, rho_obs) = _noise_vector(rng, Md, rho_obs, 1e-3)
chirp_mass_noise(Md, rho_obs; rng=Random.default_rng()) = chirp_mass_noise(rng, Md, rho_obs)

"""
    mass_ratio_noise(rng, q, rho_obs)

Draw noisy mass-ratio measurements with Python's quick-PE approximation.
"""
mass_ratio_noise(rng::AbstractRNG, q, rho_obs) = _noise_vector(rng, q, rho_obs, 0.25)
mass_ratio_noise(q, rho_obs; rng=Random.default_rng()) = mass_ratio_noise(rng, q, rho_obs)

"""
    theta_noise(rng, theta, rho_obs)

Draw noisy projection-factor measurements with Python's quick-PE approximation.
"""
function theta_noise(rng::AbstractRNG, theta, rho_obs)
    theta_vec = _as_vector(theta)
    rho = _as_vector(rho_obs)
    length(theta_vec) == length(rho) || throw(ArgumentError("theta and rho_obs must have the same length"))
    out = theta_vec .+ randn(rng, length(theta_vec)) .* (0.3 .* 10 ./ rho)
    return _restore_shape(theta, out)
end
theta_noise(theta, rho_obs; rng=Random.default_rng()) = theta_noise(rng, theta, rho_obs)

"""
    noise(rng, Md, q, theta, rho_obs)

Draw noisy `(Md, q, theta)` quick-PE measurements.
"""
function noise(rng::AbstractRNG, Md, q, theta, rho_obs)
    return chirp_mass_noise(rng, Md, rho_obs), mass_ratio_noise(rng, q, rho_obs), theta_noise(rng, theta, rho_obs)
end
noise(Md, q, theta, rho_obs; rng=Random.default_rng()) = noise(rng, Md, q, theta, rho_obs)

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
    snr_and_freq_cut(m1_source, m2_source, z, snr; snr_threshold=12, fgw_cut=15)

Return indices passing the observed-SNR and detector-frame frequency cuts.
"""
function snr_and_freq_cut(m1_source, m2_source, z, snr; snr_threshold=12.0, fgw_cut=15.0)
    mask = (snr .>= snr_threshold) .& (f_gw(m1_source, m2_source, z) .> fgw_cut)
    return findall(mask)
end

"""
    snr_cut_flat(snr; snr_threshold=1)

Return indices passing a flat toy SNR cut.
"""
snr_cut_flat(snr; snr_threshold=1.0) = findall(snr .>= snr_threshold)

"""
    likelihood_evaluation(rhos, qs, Mds, thetas, rho_obs, q_obs, Md_obs, theta_obs; numdet=3)

Evaluate the quick-PE likelihood used by Python `simulation.py`, combining a
noncentral-chi-square SNR term with Gaussian terms for mass ratio,
detector-frame chirp mass, and projection factor.
"""
function likelihood_evaluation(rhos, qs, Mds, thetas, rho_obs, q_obs, Md_obs, theta_obs; numdet=3)
    rho = _as_vector(rhos)
    q = _as_vector(qs)
    md = _as_vector(Mds)
    th = _as_vector(thetas)
    n = length(rho)
    length(q) == n && length(md) == n && length(th) == n || throw(ArgumentError("model arrays must have the same length"))
    rho_o = rho_obs isa AbstractVector ? Float64.(rho_obs) : fill(Float64(rho_obs), n)
    q_o = q_obs isa AbstractVector ? Float64.(q_obs) : fill(Float64(q_obs), n)
    md_o = Md_obs isa AbstractVector ? Float64.(Md_obs) : fill(Float64(Md_obs), n)
    th_o = theta_obs isa AbstractVector ? Float64.(theta_obs) : fill(Float64(theta_obs), n)
    length(rho_o) == n && length(q_o) == n && length(md_o) == n && length(th_o) == n ||
        throw(ArgumentError("observed values must be scalars or length $n"))

    out = Vector{Float64}(undef, n)
    for i in eachindex(out)
        out[i] = Distributions.pdf(NoncentralChisq(2numdet, rho[i]^2), rho_o[i]^2) *
            Distributions.pdf(Normal(q[i], 0.25 * q[i] * 10 / rho_o[i]), q_o[i]) *
            Distributions.pdf(Normal(md[i], 1e-3 * md[i] * 10 / rho_o[i]), md_o[i]) *
            Distributions.pdf(Normal(th[i], 0.3 * 10 / rho_o[i]), th_o[i])
    end
    return _restore_shape(rhos, out)
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
