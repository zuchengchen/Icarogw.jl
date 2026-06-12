module Likelihood

using Base.Threads
using ..DataContainers
using ..Rates
using ..Utils: logsumexp

export LikelihoodOptions,
    LikelihoodDiagnostics,
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
_required_columns(model::SpinWeightedRate) = (_required_columns(model.base)..., model.spin_columns...)

_is_scale_free(model) = getproperty(model, :scale_free)
_is_scale_free(model::SpinWeightedRate) = _is_scale_free(model.base)

function _event_logweights(model, ps::PosteriorSamples)
    cols = map(name -> column(ps, name), _required_columns(model))
    out = Vector{Float64}(undef, length(ps.prior))
    @inbounds for i in eachindex(out)
        out[i] = log_event_rate(model, (col[i] for col in cols)..., ps.prior[i])
    end
    return out
end

function _injection_logweights(model, inj::InjectionSet)
    cols = map(name -> column(inj, name), _required_columns(model))
    out = Vector{Float64}(undef, length(inj.prior))
    @inbounds for i in eachindex(out)
        out[i] = log_event_rate(model, (col[i] for col in cols)..., inj.prior[i])
    end
    return out
end

function _neff_from_logs(logw::AbstractVector{<:Real})
    isempty(logw) && return 0.0
    l1 = logsumexp(logw)
    l2 = logsumexp(2 .* logw)
    isfinite(l1) && isfinite(l2) || return 0.0
    return exp(2l1 - l2)
end

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
