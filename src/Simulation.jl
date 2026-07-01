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
    dvc_dz_reweight,
    apply_snr_cut,
    snr_and_freq_cut,
    snr_cut_flat,
    likelihood_evaluation,
    generate_mass_inj,
    generate_single_mass_inj,
    generate_dL_inj,
    generate_dL_inj_uniform,
    generate_dL_inj_z_uniform,
    injection_set_generator,
    quick_data_preparation,
    pe_quick_generation_samples,
    PE_quick_generation_samples,
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
    dvc_dz_reweight(rng, m1, m2, z; cosmology=PLANCK15_FLATLCDM, extra=())

Resample source-frame masses and redshifts with weights proportional to
`dVc/dz / (1+z)`, matching Python `dVc_dz_reweight`. Optional `extra`
vectors are resampled with the same indices and returned after `(m1, m2, z)`.
"""
function dvc_dz_reweight(rng::AbstractRNG, m1, m2, z; cosmology::AbstractCosmology=PLANCK15_FLATLCDM, extra=())
    m1v = _as_vector(m1)
    m2v = _as_vector(m2)
    zv = _as_vector(z)
    length(m1v) == length(m2v) == length(zv) || throw(ArgumentError("m1, m2, and z must have the same length"))
    extras = Tuple(_as_vector(e) for e in extra)
    all(length(e) == length(zv) for e in extras) || throw(ArgumentError("extra arrays must match z length"))
    weights = dvc_dz(cosmology, zv) ./ (1 .+ zv)
    total = sum(weights)
    total > 0 && isfinite(total) || throw(ArgumentError("invalid dVc/dz reweighting weights"))
    idx = sample(rng, 1:length(zv), Weights(weights ./ total), length(zv); replace=true)
    resampled = (m1v[idx], m2v[idx], zv[idx])
    isempty(extras) && return resampled
    return (resampled..., (e[idx] for e in extras)...)
end
dvc_dz_reweight(m1, m2, z; rng=Random.default_rng(), kwargs...) =
    dvc_dz_reweight(rng, m1, m2, z; kwargs...)

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

_require_key(params, key) = haskey(params, key) ? params[key] : throw(ArgumentError("missing mass-model parameter $key"))
_mass_param(params, key) = float(_require_key(params, key))

function _mass_distribution_from_python_name(mass_model::AbstractString, params)
    if mass_model == "PowerLaw"
        mmin = _mass_param(params, :mmin)
        mmax = _mass_param(params, :mmax)
        return ConditionalMassDistribution(PowerLaw(mmin, mmax, -_mass_param(params, :alpha)),
            PowerLaw(mmin, mmax, _mass_param(params, :beta)))
    elseif mass_model == "PowerLawPeak"
        mmin = _mass_param(params, :mmin)
        mmax = _mass_param(params, :mmax)
        mu_g = _mass_param(params, :mu_g)
        sigma_g = _mass_param(params, :sigma_g)
        primary = PowerLawGaussian(mmin, mmax, -_mass_param(params, :alpha),
            _mass_param(params, :lambda_peak), mu_g, sigma_g, mmin, mu_g + 5sigma_g)
        return ConditionalMassDistribution(primary, PowerLaw(mmin, mmax, _mass_param(params, :beta)))
    elseif mass_model == "MultiPeak"
        mmin = _mass_param(params, :mmin)
        mmax = _mass_param(params, :mmax)
        mu_low = _mass_param(params, :mu_g_low)
        sigma_low = _mass_param(params, :sigma_g_low)
        mu_high = _mass_param(params, :mu_g_high)
        sigma_high = _mass_param(params, :sigma_g_high)
        primary = PowerLawTwoGaussians(mmin, mmax, -_mass_param(params, :alpha),
            _mass_param(params, :lambda_g), _mass_param(params, :lambda_g_low),
            mu_low, sigma_low, mmin, mu_low + 5sigma_low,
            mu_high, sigma_high, mmin, mu_high + 5sigma_high)
        return ConditionalMassDistribution(primary, PowerLaw(mmin, mmax, _mass_param(params, :beta)))
    else
        throw(ArgumentError("unsupported mass_model $mass_model; choose PowerLaw, PowerLawPeak, or MultiPeak"))
    end
end

"""
    generate_mass_inj(rng, n, mass_model, params)

Generate source-frame `(m1, m2)` samples and proposal densities for Python
`simulation.generate_mass_inj` mass models: `"PowerLaw"`, `"PowerLawPeak"`,
and `"MultiPeak"`. Parameters may be a `NamedTuple` or dictionary with symbol
keys matching the Python names.
"""
function generate_mass_inj(rng::AbstractRNG, n::Integer, mass_model::AbstractString, params)
    prior = _mass_distribution_from_python_name(mass_model, params)
    m1, m2 = rand_prior(rng, prior, n)
    density = Priors.pdf(prior, m1, m2)
    return (mass_1_source=m1, mass_2_source=m2, prior=density, distribution=prior)
end
generate_mass_inj(n::Integer, mass_model::AbstractString, params; rng=Random.default_rng()) =
    generate_mass_inj(rng, n, mass_model, params)

"""
    generate_single_mass_inj(rng, n, "PowerLaw", params)

Generate one-dimensional source masses and proposal densities for Python
`generate_single_mass_inj`.
"""
function generate_single_mass_inj(rng::AbstractRNG, n::Integer, mass_model::AbstractString, params)
    mass_model == "PowerLaw" || throw(ArgumentError("generate_single_mass_inj currently supports only PowerLaw"))
    prior = PowerLaw(_mass_param(params, :mmin), _mass_param(params, :mmax), -_mass_param(params, :alpha))
    mass = rand_prior(rng, prior, n)
    return (mass_source=mass, prior=Priors.pdf(prior, mass), distribution=prior)
end
generate_single_mass_inj(n::Integer, mass_model::AbstractString, params; rng=Random.default_rng()) =
    generate_single_mass_inj(rng, n, mass_model, params)

"""
    generate_dL_inj(rng, n, zmax; cosmology=PLANCK15_FLATLCDM)

Draw luminosity distances with Python's `powerlaw(a=3, loc=0.1,
scale=dL(zmax)-10)` proposal and return distances plus proposal densities.
"""
function generate_dL_inj(rng::AbstractRNG, n::Integer, zmax::Real; cosmology::AbstractCosmology=PLANCK15_FLATLCDM)
    beta = z_to_dl(zmax; cosmology)
    scale = beta - 10.0
    scale > 0 || throw(ArgumentError("zmax gives an invalid Python dL proposal scale"))
    u = rand(rng, n)
    dL = 0.1 .+ scale .* u .^ (1 / 3)
    density = 3 .* ((dL .- 0.1) ./ scale) .^ 2 ./ scale
    return (luminosity_distance=dL, prior=density)
end
generate_dL_inj(n::Integer, zmax::Real; rng=Random.default_rng(), kwargs...) =
    generate_dL_inj(rng, n, zmax; kwargs...)

"""
    generate_dL_inj_uniform(rng, n, zmax; cosmology=PLANCK15_FLATLCDM)

Uniform luminosity-distance proposal matching Python
`generate_dL_inj_uniform`.
"""
function generate_dL_inj_uniform(rng::AbstractRNG, n::Integer, zmax::Real; cosmology::AbstractCosmology=PLANCK15_FLATLCDM)
    scale = z_to_dl(zmax; cosmology)
    scale > 0 || throw(ArgumentError("zmax gives an invalid uniform dL proposal scale"))
    dL = 0.1 .+ scale .* rand(rng, n)
    return (luminosity_distance=dL, prior=fill(1 / scale, n))
end
generate_dL_inj_uniform(n::Integer, zmax::Real; rng=Random.default_rng(), kwargs...) =
    generate_dL_inj_uniform(rng, n, zmax; kwargs...)

"""
    generate_dL_inj_z_uniform(rng, n, zmax; cosmology=PLANCK15_FLATLCDM)

Uniform-redshift proposal returned in luminosity distance, with density
converted by `ddL/dz`, matching Python `generate_dL_inj_z_uniform`.
"""
function generate_dL_inj_z_uniform(rng::AbstractRNG, n::Integer, zmax::Real; cosmology::AbstractCosmology=PLANCK15_FLATLCDM)
    z = 0.1 .+ zmax .* rand(rng, n)
    dL = z_to_dl(z; cosmology)
    prior = fill(1 / zmax, n) ./ ddl_dz(cosmology, z)
    return (luminosity_distance=dL, redshift=z, prior=prior)
end
generate_dL_inj_z_uniform(n::Integer, zmax::Real; rng=Random.default_rng(), kwargs...) =
    generate_dL_inj_z_uniform(rng, n, zmax; kwargs...)

"""
    injection_set_generator(rng, Ninj, Ntot, mass_model, params; ...)

Julia-native equivalent of Python `simulation.injection_set_generator`. It
keeps drawing batches of `Ntot` trial injections until at least `Ninj` pass the
SNR/frequency cuts, then returns source and detector-frame truth columns,
proposal priors, counts, and an `InjectionSet` ready for likelihood code.
"""
function injection_set_generator(rng::AbstractRNG, Ninj::Integer, Ntot::Integer, mass_model::AbstractString, params;
    zmax=5.0, snrthr=12.0, snr_threshold=snrthr, fgw_cut=15.0, numdet=3, rho_s=9.0,
    dL_s=1.5, Md_s=25.0, theta=nothing, Tobs=1.0, cosmology::AbstractCosmology=PLANCK15_FLATLCDM)
    Ninj > 0 && Ntot > 0 || throw(ArgumentError("Ninj and Ntot must be positive"))
    m1s_all = Float64[]
    m2s_all = Float64[]
    z_all = Float64[]
    snr_true_all = Float64[]
    m1d_all = Float64[]
    m2d_all = Float64[]
    dL_all = Float64[]
    prior_all = Float64[]
    generated = 0

    while length(m1s_all) < Ninj
        mass_draw = generate_mass_inj(rng, Ntot, mass_model, params)
        distance_draw = generate_dL_inj(rng, Ntot, zmax; cosmology)
        z_draw = dl_to_z(distance_draw.luminosity_distance; cosmology)
        prior = mass_draw.prior .* distance_draw.prior .* (1 .+ z_draw) .^ (-2)
        theta_batch = theta === nothing ? nothing : _as_vector(theta)
        if theta_batch !== nothing && length(theta_batch) != Ntot
            throw(ArgumentError("theta must have length Ntot when supplied to injection_set_generator"))
        end
        snr = snr_samples_source(rng, mass_draw.mass_1_source, mass_draw.mass_2_source, z_draw;
            cosmology, numdet, rho_s, dL_s, Md_s, theta=theta_batch)
        idx = snr_and_freq_cut(mass_draw.mass_1_source, mass_draw.mass_2_source, z_draw, snr.rho_obs;
            snr_threshold, fgw_cut)
        m1d = mass_draw.mass_1_source[idx] .* (1 .+ z_draw[idx])
        m2d = mass_draw.mass_2_source[idx] .* (1 .+ z_draw[idx])
        append!(m1s_all, mass_draw.mass_1_source[idx])
        append!(m2s_all, mass_draw.mass_2_source[idx])
        append!(z_all, z_draw[idx])
        append!(snr_true_all, snr.rho_true[idx])
        append!(m1d_all, m1d)
        append!(m2d_all, m2d)
        append!(dL_all, distance_draw.luminosity_distance[idx])
        append!(prior_all, prior[idx])
        generated += Ntot
    end

    keep = 1:Ninj
    injections = InjectionSet((mass_1=m1d_all[keep], mass_2=m2d_all[keep],
        luminosity_distance=dL_all[keep], prior=prior_all[keep]); ntotal=generated, Tobs)
    return (mass_1_source=m1s_all[keep], mass_2_source=m2s_all[keep], redshift=z_all[keep],
        snr=snr_true_all[keep], mass_1=m1d_all[keep], mass_2=m2d_all[keep],
        luminosity_distance=dL_all[keep], prior=prior_all[keep],
        ntotal_generated=generated, ndetected=length(m1s_all), injections=injections)
end
injection_set_generator(Ninj::Integer, Ntot::Integer, mass_model::AbstractString, params; rng=Random.default_rng(), kwargs...) =
    injection_set_generator(rng, Ninj, Ntot, mass_model, params; kwargs...)

"""
    quick_data_preparation(rng, m1, m2, z; reweight=true, ...)

Prepare the noisy quick-PE inputs used by `pe_quick_generation_samples`. The
return value is a named tuple containing source masses, redshifts, projection
factor `theta`, detected indices, observed SNR, observed mass ratio, observed
detector-frame chirp mass, and observed projection factor.
"""
function quick_data_preparation(rng::AbstractRNG, m1_astro, m2_astro, zmerg_astro; numdet=3, rho_s=9.0,
    dL_s=1.5, Md_s=25.0, snr_threshold=12.0, fgw_cut=15.0, theta=nothing,
    reweight::Bool=true, cosmology::AbstractCosmology=PLANCK15_FLATLCDM)
    if reweight
        if theta === nothing
            m1, m2, z = dvc_dz_reweight(rng, m1_astro, m2_astro, zmerg_astro; cosmology)
            theta_vec = nothing
        else
            m1, m2, z, theta_vec = dvc_dz_reweight(rng, m1_astro, m2_astro, zmerg_astro; cosmology, extra=(theta,))
        end
    else
        m1 = _as_vector(m1_astro)
        m2 = _as_vector(m2_astro)
        z = _as_vector(zmerg_astro)
        theta_vec = theta === nothing ? nothing : _as_vector(theta)
    end
    length(m1) == length(m2) == length(z) || throw(ArgumentError("m1, m2, and z must have the same length"))
    if theta_vec !== nothing && length(theta_vec) != length(z)
        throw(ArgumentError("theta must match z length"))
    end

    snr = snr_samples_source(rng, m1, m2, z; cosmology, numdet, rho_s, dL_s, Md_s, theta=theta_vec)
    md = chirp_mass_detector(m1, m2, z)
    q = mass_ratio.(m1, m2)
    md_obs, q_obs, theta_obs = noise(rng, md, q, snr.theta, snr.rho_obs)
    idx_cut = snr_and_freq_cut(m1, m2, z, snr.rho_obs; snr_threshold, fgw_cut)
    return (mass_1_source=m1, mass_2_source=m2, redshift=z, theta=snr.theta,
        detected_indices=idx_cut, rho_obs=snr.rho_obs, mass_ratio_obs=q_obs,
        chirp_mass_detector_obs=md_obs, theta_obs=theta_obs, rho_true=snr.rho_true)
end
quick_data_preparation(m1_astro, m2_astro, zmerg_astro; rng=Random.default_rng(), kwargs...) =
    quick_data_preparation(rng, m1_astro, m2_astro, zmerg_astro; kwargs...)

function _positive_uniform_draws(rng, lo, hi, n, label)
    hi > lo || throw(ArgumentError("invalid proposal interval for $label"))
    return rand(rng, Uniform(lo, hi), n)
end

"""
    pe_quick_generation_samples(rng, m1, m2, z, theta, idx, rho_obs, q_obs, Md_obs, theta_obs; ...)

Generate quick posterior samples for detected events using the same
importance-sampling structure as Python `PE_quick_generation_samples`.
`idx` uses Julia's 1-based indexing. Use the `Ngen` keyword to control the
proposal pool size; tests and examples should keep it small.
"""
function pe_quick_generation_samples(rng::AbstractRNG, m1, m2, z, theta, idx, rho_obs, q_obs, Md_obs, theta_obs;
    Ninj=5, Nsamp=1000, numdet=3, rho_s=9.0, dL_s=1.5, Md_s=25.0, Ngen=10_000,
    cosmology::AbstractCosmology=PLANCK15_FLATLCDM)
    m1v = _as_vector(m1)
    m2v = _as_vector(m2)
    zv = _as_vector(z)
    thetav = _as_vector(theta)
    rho_obsv = _as_vector(rho_obs)
    q_obsv = _as_vector(q_obs)
    md_obsv = _as_vector(Md_obs)
    theta_obsv = _as_vector(theta_obs)
    n = length(m1v)
    all(length(v) == n for v in (m2v, zv, thetav, rho_obsv, q_obsv, md_obsv, theta_obsv)) ||
        throw(ArgumentError("quick PE input arrays must have the same length"))
    isempty(idx) && throw(ArgumentError("idx must contain at least one detected event"))
    chosen = sample(rng, collect(idx), Ninj; replace=true)
    posterior = Dict{String,NamedTuple}()
    truth = Dict{String,NamedTuple}()

    dl_all = z_to_dl(zv; cosmology)
    for i in chosen
        1 <= i <= n || throw(ArgumentError("idx contains out-of-bounds event index $i"))
        uncmass = 1.1 * 0.7 / (rho_obsv[i] / 8)
        m1s = _positive_uniform_draws(rng, max((1 - uncmass) * m1v[i], 0.1), (1 + uncmass) * m1v[i], Ngen, "m1")
        m2s = _positive_uniform_draws(rng, max((1 - uncmass) * m2v[i], 0.1), (1 + uncmass) * m2v[i], Ngen, "m2")
        for j in eachindex(m1s)
            if m1s[j] < m2s[j]
                m1s[j], m2s[j] = m2s[j], m1s[j]
            end
        end

        uncdl = 3 * 1.2 / (rho_obsv[i] / 8)
        dls = _positive_uniform_draws(rng, max((1 - uncdl) * dl_all[i], 1.0), (1 + uncdl) * dl_all[i], Ngen, "dL")
        zs = dl_to_z(dls; cosmology)
        thetas = rand(rng, Uniform(0, 1.4), Ngen)
        qs = mass_ratio.(m1s, m2s)
        mds = chirp_mass_detector(m1s, m2s, zs)
        rhos = snr_samples_source(rng, m1s, m2s, zs; cosmology, numdet, rho_s, dL_s, Md_s, theta=thetas).rho_true
        prior = (dls .* (1 .+ zs)).^2
        likelihood_tot = likelihood_evaluation(rhos, qs, mds, thetas, rho_obsv[i], q_obsv[i], md_obsv[i], theta_obsv[i]; numdet) .* prior
        total = sum(likelihood_tot)
        total > 0 && isfinite(total) || throw(ArgumentError("quick PE proposal weights are zero or non-finite"))
        resampled = sample(rng, 1:Ngen, Weights(likelihood_tot ./ total), Nsamp; replace=true)
        key = string(i)
        posterior[key] = (mass_1_source=m1s[resampled], mass_2_source=m2s[resampled],
            redshift=zs[resampled], mass_1=m1s[resampled] .* (1 .+ zs[resampled]),
            mass_2=m2s[resampled] .* (1 .+ zs[resampled]), chirp_mass_detector=mds[resampled],
            mass_ratio=qs[resampled], rho=rhos[resampled], luminosity_distance=dls[resampled],
            theta=thetas[resampled])
        truth[key] = (mass_1_source=m1v[i], mass_2_source=m2v[i], redshift=zv[i],
            mass_1=m1v[i] * (1 + zv[i]), mass_2=m2v[i] * (1 + zv[i]),
            chirp_mass_detector=chirp_mass_detector(m1v[i], m2v[i], zv[i]),
            mass_ratio=mass_ratio(m1v[i], m2v[i]), luminosity_distance=dl_all[i], theta=thetav[i])
    end
    return (posterior_samples=posterior, truth=truth, indices=chosen)
end
pe_quick_generation_samples(m1, m2, z, theta, idx, rho_obs, q_obs, Md_obs, theta_obs; rng=Random.default_rng(), kwargs...) =
    pe_quick_generation_samples(rng, m1, m2, z, theta, idx, rho_obs, q_obs, Md_obs, theta_obs; kwargs...)
const PE_quick_generation_samples = pe_quick_generation_samples

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
