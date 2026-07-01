module Likelihood

using Base.Threads
using Random
using StatsBase
using ..Cosmology
using ..DataContainers
using ..Priors: logpdf
using ..Rates
using ..SkyMaps: evaluate_3d_posterior_likelihood
using ..Utils: logsumexp

export LikelihoodOptions,
    LikelihoodDiagnostics,
    event_logweights,
    injection_logweights,
    effective_sample_size,
    expected_number_detections,
    reweight_injections,
    reweight_posterior_samples,
    likelihood_diagnostics,
    loglikelihood,
    loglikelihood_batch,
    no_event_loglikelihood

"""
    LikelihoodOptions(; shape_only=false, poisson=true, neff_event_min=0,
                       neff_injection_min=0, likelihood_variance_threshold=Inf)

Numerical and statistical options for population likelihood evaluation. Scalar
`loglikelihood` is deterministic and single-threaded; parallelism is exposed
only through `loglikelihood_batch(...; parallel=true)`.
"""
Base.@kwdef struct LikelihoodOptions
    shape_only::Bool = false
    poisson::Bool = true
    neff_event_min::Float64 = 0.0
    neff_injection_min::Float64 = 0.0
    likelihood_variance_threshold::Float64 = Inf
end

"""
    LikelihoodDiagnostics

Structured diagnostics for a population likelihood evaluation.

Fields include:
`per_event_neff`, `injection_neff`, `xi`, `N_expected`,
`max_log_weight`, `min_log_weight`, `has_nan`, `has_inf`,
`underflow_risk`, and `likelihood_variance`.
"""
Base.@kwdef struct LikelihoodDiagnostics
    per_event_neff::Vector{Float64}
    injection_neff::Float64
    xi::Float64
    N_expected::Float64
    max_log_weight::Float64
    min_log_weight::Float64
    has_nan::Bool
    has_inf::Bool
    underflow_risk::Bool
    likelihood_variance::Float64
    accepted::Bool
    reason::String = ""
end

_required_columns(::SimplePowerLawPopulation) = (:mass_1, :mass_2, :luminosity_distance)
_required_columns(::CBCVanillaRate) = (:mass_1, :mass_2, :luminosity_distance)
_required_columns(::CBCMass1Rate) = (:mass_1, :mass_ratio, :luminosity_distance)
_required_columns(::CBCMchirpQRate) = (:chirp_mass, :mass_ratio, :luminosity_distance)
_required_columns(::CBCSingleMassRate) = (:mass_1, :luminosity_distance)
_required_columns(::CBCTotalMassQRate) = (:total_mass, :mass_ratio, :luminosity_distance)
_required_columns(::CBCRedshiftPrimaryQRate) = (:mass_1, :mass_ratio, :luminosity_distance)
_required_columns(::CBCCatalogVanillaRate) = (:mass_1, :mass_2, :luminosity_distance, :sky_indices)
_required_columns(::CBCCatalogSkyMapRate) = (:luminosity_distance, :sky_indices)
_required_columns(::CBCVanillaEMCounterpartRate) = (:mass_1, :mass_2, :luminosity_distance, :z_EM)
_required_columns(::CBCLowLatencySkyMapEMCounterpartRate) = (:z_EM, :right_ascension, :declination)
_required_columns(model::SpinWeightedRate) = (_required_columns(model.base)..., model.spin_columns...)
_required_columns(model::PEOnlySpinWeightedRate) = (_required_columns(model.base)..., model.spin_columns...)
function _required_columns(model::MixtureRate)
    cols1 = _required_columns(model.rate1)
    cols2 = _required_columns(model.rate2)
    cols1 == cols2 || throw(ArgumentError("MixtureRate components must use the same event columns for likelihood evaluation"))
    return cols1
end
_required_injection_columns(model) = _required_columns(model)
_required_injection_columns(::CBCVanillaEMCounterpartRate) = (:mass_1, :mass_2, :luminosity_distance)
_required_injection_columns(::CBCLowLatencySkyMapEMCounterpartRate) = (:luminosity_distance,)
_required_injection_columns(model::SpinWeightedRate) = (_required_injection_columns(model.base)..., model.spin_columns...)
_required_injection_columns(model::PEOnlySpinWeightedRate) = _required_injection_columns(model.base)

_is_scale_free(model) = getproperty(model, :scale_free)
_is_scale_free(model::SpinWeightedRate) = _is_scale_free(model.base)
_is_scale_free(model::PEOnlySpinWeightedRate) = _is_scale_free(model.base)

function _weighted_kde_logpdf(samples::AbstractVector{<:Real}, logweights::AbstractVector{<:Real}, points::AbstractVector{<:Real})
    length(samples) == length(logweights) || throw(ArgumentError("KDE samples and logweights must have the same length"))
    keep = isfinite.(samples) .& isfinite.(logweights)
    any(keep) || return fill(-Inf, length(points))
    x = Float64.(samples[keep])
    lw = Float64.(logweights[keep])
    lnorm = logsumexp(lw)
    isfinite(lnorm) || return fill(-Inf, length(points))
    w = exp.(lw .- lnorm)
    μ = sum(w .* x)
    denom = 1 - sum(abs2, w)
    variance = denom > eps(Float64) ? sum(w .* (x .- μ) .^ 2) / denom : sum(w .* (x .- μ) .^ 2)
    neff = inv(sum(abs2, w))
    bandwidth = sqrt(max(variance, 0.0)) * neff^(-1 / 5)
    if !(bandwidth > 0 && isfinite(bandwidth))
        bandwidth = max(1e-6, 1e-3 * (abs(μ) + 1))
    end
    lognorm = -log(bandwidth) - 0.5log(2pi)
    return [logsumexp(log.(w) .+ lognorm .- 0.5 .* ((p .- x) ./ bandwidth) .^ 2) for p in points]
end

function _em_vanilla_event_logweights(base::CBCVanillaEMCounterpartRate, ps::PosteriorSamples, extra_logweights=nothing)
    m1 = column(ps, :mass_1)
    m2 = column(ps, :mass_2)
    dl = column(ps, :luminosity_distance)
    z_em = column(ps, :z_EM)
    base_logw = Vector{Float64}(undef, length(ps.prior))
    z_gw = Vector{Float64}(undef, length(ps.prior))
    @inbounds for i in eachindex(base_logw)
        base_logw[i] = Rates._em_vanilla_base_logweight(base, m1[i], m2[i], dl[i], ps.prior[i])
        extra_logweights === nothing || (base_logw[i] += extra_logweights[i])
        z_gw[i] = redshift_at_luminosity_distance(base.cosmology, dl[i])
    end
    event_log_evidence = logsumexp(base_logw) - log(length(base_logw))
    return event_log_evidence .+ _weighted_kde_logpdf(z_gw, base_logw, z_em) .+ Rates._rate_scale(base)
end

function _event_logweights(model, ps::PosteriorSamples)
    cols = map(name -> column(ps, name), _required_columns(model))
    out = Vector{Float64}(undef, length(ps.prior))
    @inbounds for i in eachindex(out)
        out[i] = log_event_rate(model, (col[i] for col in cols)..., ps.prior[i])
    end
    return out
end
_event_logweights(model::CBCVanillaEMCounterpartRate, ps::PosteriorSamples) =
    _em_vanilla_event_logweights(model, ps)

function _event_logweights(model::SpinWeightedRate{<:CBCVanillaEMCounterpartRate}, ps::PosteriorSamples)
    spin_cols = map(name -> column(ps, name), model.spin_columns)
    extra = Vector{Float64}(undef, length(ps.prior))
    @inbounds for i in eachindex(extra)
        extra[i] = logpdf(model.spin_prior, (col[i] for col in spin_cols)...)
    end
    return _em_vanilla_event_logweights(model.base, ps, extra)
end

function _event_logweights(model::CBCLowLatencySkyMapEMCounterpartRate, ps::PosteriorSamples)
    z = column(ps, :z_EM)
    ra = column(ps, :right_ascension)
    dec = column(ps, :declination)
    dl = luminosity_distance(model.cosmology, z)
    _, sky_likelihood = evaluate_3d_posterior_likelihood(Rates._skymap_for_event(model, ps.event_name), dl, ra, dec)
    logw = Vector{Float64}(undef, length(ps.prior))
    @inbounds for i in eachindex(logw)
        if ps.prior[i] > 0 && sky_likelihood[i] > 0
            logw[i] = Rates.log_rate(model.redshift_rate, z[i]) + log(dvc_dz(model.cosmology, z[i])) -
                log(ps.prior[i]) - log1p(z[i]) + log(sky_likelihood[i])
        else
            logw[i] = -Inf
        end
    end
    event_log_evidence = logsumexp(logw) - log(length(logw)) + Rates._rate_scale(model)
    return fill(event_log_evidence, length(logw))
end

"""
    event_logweights(model, posterior_samples)

Return per-sample detector-frame log weights for one event posterior.
"""
event_logweights(model, ps::PosteriorSamples) = _event_logweights(model, ps)

function _injection_logweights(model, inj::InjectionSet)
    cols = map(name -> column(inj, name), _required_injection_columns(model))
    out = Vector{Float64}(undef, length(inj.prior))
    @inbounds for i in eachindex(out)
        out[i] = log_injection_rate(model, (col[i] for col in cols)..., inj.prior[i])
    end
    return out
end

"""
    injection_logweights(model, injections)

Return per-detected-injection detector-frame log weights.
"""
injection_logweights(model, inj::InjectionSet) = _injection_logweights(model, inj)

function _neff_from_logs(logw::AbstractVector{<:Real})
    isempty(logw) && return 0.0
    l1 = logsumexp(logw)
    l2 = logsumexp(2 .* logw)
    isfinite(l1) && isfinite(l2) || return 0.0
    return exp(2l1 - l2)
end

"""
    effective_sample_size(logweights)

Effective sample size `(sum w)^2 / sum(w^2)` computed stably from log weights.
"""
effective_sample_size(logweights::AbstractVector{<:Real}) = _neff_from_logs(logweights)
effective_sample_size(model, ps::PosteriorSamples) = effective_sample_size(event_logweights(model, ps))
effective_sample_size(model, inj::InjectionSet) = effective_sample_size(injection_logweights(model, inj))

"""
    expected_number_detections(model, injections)

Expected number of detections `Tobs * sum(weights) / ntotal`, matching the
Python injection helper's pseudo-rate convention.
"""
function expected_number_detections(model, injections::InjectionSet)
    logw = injection_logweights(model, injections)
    return injections.Tobs * exp(logsumexp(logw) - log(injections.ntotal))
end

function _normalized_probabilities(logw::AbstractVector{<:Real})
    isempty(logw) && throw(ArgumentError("cannot resample from empty log weights"))
    lnorm = logsumexp(logw)
    isfinite(lnorm) || throw(ArgumentError("log weights have zero or non-finite total probability"))
    probs = exp.(logw .- lnorm)
    total = sum(probs)
    total > 0 && isfinite(total) || throw(ArgumentError("invalid resampling probabilities"))
    return probs ./ total
end

"""
    reweight_injections(rng, model, injections, nsamples; replace=true)

Draw a weighted injection subset according to detector-frame rate weights.
"""
function reweight_injections(rng::AbstractRNG, model, injections::InjectionSet, nsamples::Integer; replace::Bool=true)
    probs = _normalized_probabilities(injection_logweights(model, injections))
    if !replace && nsamples > length(injections.prior)
        throw(ArgumentError("cannot draw more samples than available when replace=false"))
    end
    idx = sample(rng, 1:length(injections.prior), Weights(probs), nsamples; replace)
    return subset_injections(injections, idx)
end
reweight_injections(model, injections::InjectionSet, nsamples::Integer; rng=Random.default_rng(), kwargs...) =
    reweight_injections(rng, model, injections, nsamples; kwargs...)

"""
    reweight_posterior_samples(rng, model, posterior_samples, nsamples; replace=true)

Draw posterior samples according to detector-frame rate weights.
"""
function reweight_posterior_samples(rng::AbstractRNG, model, ps::PosteriorSamples, nsamples::Integer; replace::Bool=true)
    probs = _normalized_probabilities(event_logweights(model, ps))
    if !replace && nsamples > length(ps.prior)
        throw(ArgumentError("cannot draw more samples than available when replace=false"))
    end
    idx = sample(rng, 1:length(ps.prior), Weights(probs), nsamples; replace)
    return subset_posterior_samples(ps, idx)
end
reweight_posterior_samples(model, ps::PosteriorSamples, nsamples::Integer; rng=Random.default_rng(), kwargs...) =
    reweight_posterior_samples(rng, model, ps, nsamples; kwargs...)

function _diagnose(logw_events, logw_inj, data::PopulationData, log_xi, options::LikelihoodOptions)
    per_event_neff = [_neff_from_logs(w) for w in logw_events]
    inj_neff = _neff_from_logs(logw_inj)
    allw = reduce(vcat, (logw_events..., logw_inj))
    maxw = isempty(allw) ? -Inf : maximum(allw)
    minw = isempty(allw) ? Inf : minimum(allw)
    has_nan = any(isnan, allw)
    has_inf = any(isinf, allw)
    underflow_risk = isfinite(maxw) && any(w -> isfinite(w) && maxw - w > 700, allw)
    xi = exp(log_xi)
    Nexp = data.injections.Tobs * xi
    nev = length(data.posteriors)
    variance = inj_neff > 0 ? (nev^2 / inj_neff) * max(0.0, 1 - inj_neff / data.injections.ntotal) : Inf
    for (neff, ps) in zip(per_event_neff, data.posteriors.events)
        variance += neff > 0 ? (1 / neff) * max(0.0, 1 - neff / length(ps.prior)) : Inf
    end

    accepted = true
    reason = ""
    if inj_neff < options.neff_injection_min
        accepted = false
        reason = "injection effective sample size below threshold"
    elseif any(<(options.neff_event_min), per_event_neff)
        accepted = false
        reason = "per-event effective sample size below threshold"
    elseif variance > options.likelihood_variance_threshold
        accepted = false
        reason = "likelihood variance above threshold"
    elseif has_nan
        accepted = false
        reason = "NaN log weight detected"
    end

    return LikelihoodDiagnostics(
        per_event_neff=per_event_neff,
        injection_neff=inj_neff,
        xi=xi,
        N_expected=Nexp,
        max_log_weight=maxw,
        min_log_weight=minw,
        has_nan=has_nan,
        has_inf=has_inf,
        underflow_risk=underflow_risk,
        likelihood_variance=variance,
        accepted=accepted,
        reason=reason,
    )
end

function _evaluate(model, data::PopulationData, options::LikelihoodOptions)
    validate(data)
    logw_events = [_event_logweights(model, ps) for ps in data.posteriors.events]
    logw_inj = _injection_logweights(model, data.injections)
    log_event_means = [logsumexp(w) - log(length(w)) for w in logw_events]
    log_xi = logsumexp(logw_inj) - log(data.injections.ntotal)
    diagnostics = _diagnose(logw_events, logw_inj, data, log_xi, options)
    diagnostics.accepted || return -Inf, diagnostics

    if options.shape_only || _is_scale_free(model)
        value = sum(log_event_means) - length(data.posteriors) * log_xi
    elseif options.poisson
        value = -diagnostics.N_expected + length(data.posteriors) * log(data.injections.Tobs) + sum(log_event_means)
    else
        value = sum(log_event_means)
    end
    isfinite(value) || return -Inf, diagnostics
    return value, diagnostics
end

"""
    likelihood_diagnostics(model, data[, theta]; options=LikelihoodOptions())

Return `LikelihoodDiagnostics` for a model or model type. When `theta` is
provided for `SimplePowerLawPopulation`, it is materialized through that
model's parameter schema.
"""
function likelihood_diagnostics(model, data::PopulationData; options::LikelihoodOptions=LikelihoodOptions())
    _, diag = _evaluate(model, data, options)
    return diag
end
function likelihood_diagnostics(::Type{SimplePowerLawPopulation}, data::PopulationData, theta; options::LikelihoodOptions=LikelihoodOptions(), kwargs...)
    model = materialize(SimplePowerLawPopulation, theta; kwargs...)
    return likelihood_diagnostics(model, data; options)
end

"""
    loglikelihood(model, data[, theta]; options=LikelihoodOptions())

Selection-corrected hierarchical population likelihood. For matrix batch input,
use `loglikelihood_batch`; its dimension convention is `nparameters x npoints`.
"""
function loglikelihood(model, data::PopulationData; options::LikelihoodOptions=LikelihoodOptions())
    val, _ = _evaluate(model, data, options)
    return val
end
function loglikelihood(::Type{SimplePowerLawPopulation}, data::PopulationData, theta::AbstractVector; options::LikelihoodOptions=LikelihoodOptions(), kwargs...)
    model = materialize(SimplePowerLawPopulation, theta; kwargs...)
    return loglikelihood(model, data; options)
end
function loglikelihood(::Type{SimplePowerLawPopulation}, data::PopulationData, theta::NamedTuple; options::LikelihoodOptions=LikelihoodOptions(), kwargs...)
    model = materialize(SimplePowerLawPopulation, theta; kwargs...)
    return loglikelihood(model, data; options)
end

"""
    loglikelihood_batch(model_type, data, theta_matrix; parallel=false)

Evaluate a batch of parameter vectors. `theta_matrix` must have shape
`nparameters x npoints`; the returned vector has length `npoints`.
"""
function loglikelihood_batch(model_type::Type{SimplePowerLawPopulation}, data::PopulationData, theta_matrix::AbstractMatrix; parallel::Bool=false, options::LikelihoodOptions=LikelihoodOptions(), kwargs...)
    out = Vector{Float64}(undef, size(theta_matrix, 2))
    if parallel
        @threads for j in axes(theta_matrix, 2)
            out[j] = loglikelihood(model_type, data, view(theta_matrix, :, j); options, kwargs...)
        end
    else
        for j in axes(theta_matrix, 2)
            out[j] = loglikelihood(model_type, data, view(theta_matrix, :, j); options, kwargs...)
        end
    end
    return out
end

"""
    no_event_loglikelihood(model, injections)

No-event/upper-limit likelihood `-N_expected` using injections only.
"""
function no_event_loglikelihood(model, injections::InjectionSet)
    logw = _injection_logweights(model, injections)
    log_xi = logsumexp(logw) - log(injections.ntotal)
    return -injections.Tobs * exp(log_xi)
end

end
