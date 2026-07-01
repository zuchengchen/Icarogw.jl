module Rates

using ..Cosmology
using ..Conversions
using ..Catalog
using ..Priors
using ..Utils: logaddexp
using SpecialFunctions: beta, gamma

import ..Priors: ParameterSchema, ParameterSpec, parameter_schema, pack, unpack, logpdf

export AbstractRedshiftRate,
    PowerLawRate,
    MadauRate,
    MadauGammaRate,
    BetaRate,
    BetaLineRate,
    CBCVanillaRate,
    CBCMass1Rate,
    CBCMchirpQRate,
    CBCSingleMassRate,
    CBCTotalMassQRate,
    CBCRedshiftPrimaryQRate,
    CBCCatalogVanillaRate,
    CBCCatalogSkyMapRate,
    CBCVanillaEMCounterpartRate,
    CBCLowLatencySkyMapEMCounterpartRate,
    MixtureRate,
    SpinWeightedRate,
    SimplePowerLawPopulation,
    materialize,
    log_event_rate,
    log_injection_rate,
    parameter_schema

abstract type AbstractRedshiftRate end

"""
    PowerLawRate(gamma)

Redshift evolution `ψ(z) = (1+z)^gamma`.
"""
struct PowerLawRate <: AbstractRedshiftRate
    gamma::Float64
end
log_rate(r::PowerLawRate, z::Real) = r.gamma * log1p(z)

"""
    MadauRate(gamma, kappa, zp)

Madau-Dickinson-like redshift evolution used by Python `md_rate`.
"""
struct MadauRate <: AbstractRedshiftRate
    gamma::Float64
    kappa::Float64
    zp::Float64
end
function log_rate(r::MadauRate, z::Real)
    return log1p((1 + r.zp)^(-r.gamma - r.kappa)) +
        r.gamma * log1p(z) -
        log1p(((1 + z) / (1 + r.zp))^(r.gamma + r.kappa))
end

"""
    MadauGammaRate(gamma, kappa, zp, a, b, c)

Madau rate plus a gamma-shaped perturbation in log-rate.
"""
struct MadauGammaRate <: AbstractRedshiftRate
    gamma::Float64
    kappa::Float64
    zp::Float64
    a::Float64
    b::Float64
    c::Float64
end
function log_rate(r::MadauGammaRate, z::Real)
    md = log_rate(MadauRate(r.gamma, r.kappa, r.zp), z)
    gamma_pdf = z < 0 ? 0.0 : r.b^r.a * z^(r.a - 1) * exp(-r.b * z) / gamma(r.a)
    return md + r.c * gamma_pdf
end

"""
    BetaRate(a, b, c)

Rate whose log-evolution is `c * beta_pdf(z; a, b)`. It is mainly useful when
redshift has been scaled to `[0, 1]`, matching the Python behavior.
"""
struct BetaRate <: AbstractRedshiftRate
    a::Float64
    b::Float64
    c::Float64
end
function log_rate(r::BetaRate, z::Real)
    0 <= z <= 1 || return 0.0
    beta_pdf = z^(r.a - 1) * (1 - z)^(r.b - 1) / beta(r.a, r.b)
    return r.c * beta_pdf
end

"""
    BetaLineRate(a, b, c, d)

Beta log-rate for `z <= d` continued by a tangent line for `z > d`.
"""
struct BetaLineRate <: AbstractRedshiftRate
    a::Float64
    b::Float64
    c::Float64
    d::Float64
end
function log_rate(r::BetaLineRate, z::Real)
    base = BetaRate(r.a, r.b, r.c)
    z <= r.d && return log_rate(base, z)
    h = 1e-6
    slope = (log_rate(base, r.d + h) - log_rate(base, r.d - h)) / (2h)
    return slope * (z - r.d) + log_rate(base, r.d)
end

log_rate(r::AbstractRedshiftRate, z::AbstractArray) = map(v -> log_rate(r, v), z)

abstract type AbstractCBCRateModel end

"""
    SimplePowerLawPopulation(cosmology; kwargs...)

Concrete population model used for the vertical-slice workflow. It combines a
primary power-law mass distribution, conditional secondary-mass distribution,
power-law redshift evolution, and optional rate scale `R0`.
"""
Base.@kwdef struct SimplePowerLawPopulation <: AbstractCBCRateModel
    cosmology::FlatLambdaCDM = FlatLambdaCDM()
    mass::ConditionalMassDistribution = ConditionalMassDistribution(PowerLaw(5, 80, -2.0), PowerLaw(5, 80, 1.0))
    redshift_rate::PowerLawRate = PowerLawRate(0.0)
    R0::Float64 = 25.0
    scale_free::Bool = false
end

"""
    CBCVanillaRate(cosmology, mass_distribution, redshift_rate; R0=1, scale_free=false)

Rate model for events represented by detector-frame `(mass_1, mass_2,
luminosity_distance)`.
"""
struct CBCVanillaRate{C<:AbstractCosmology,M,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    cosmology::C
    mass_distribution::M
    redshift_rate::R
    R0::Float64
    scale_free::Bool
end
CBCVanillaRate(cosmology, mass_distribution, redshift_rate; R0=1.0, scale_free=false) =
    CBCVanillaRate(cosmology, mass_distribution, redshift_rate, float(R0), Bool(scale_free))

"""
    CBCMass1Rate(cosmology, mass_distribution, q_distribution, redshift_rate; R0=1, scale_free=false)

Rate model for detector-frame `(mass_1, mass_ratio, luminosity_distance)`.
"""
struct CBCMass1Rate{C<:AbstractCosmology,M,Q,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    cosmology::C
    mass_distribution::M
    q_distribution::Q
    redshift_rate::R
    R0::Float64
    scale_free::Bool
end
CBCMass1Rate(cosmology, mass_distribution, q_distribution, redshift_rate; R0=1.0, scale_free=false) =
    CBCMass1Rate(cosmology, mass_distribution, q_distribution, redshift_rate, float(R0), Bool(scale_free))

"""
    CBCMchirpQRate(cosmology, chirp_mass_distribution, q_distribution, redshift_rate; R0=1, scale_free=false)

Rate model for detector-frame `(chirp_mass, mass_ratio, luminosity_distance)`.
"""
struct CBCMchirpQRate{C<:AbstractCosmology,M,Q,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    cosmology::C
    chirp_mass_distribution::M
    q_distribution::Q
    redshift_rate::R
    R0::Float64
    scale_free::Bool
end
CBCMchirpQRate(cosmology, mchirp_distribution, q_distribution, redshift_rate; R0=1.0, scale_free=false) =
    CBCMchirpQRate(cosmology, mchirp_distribution, q_distribution, redshift_rate, float(R0), Bool(scale_free))

"""
    CBCSingleMassRate(cosmology, mass_distribution, redshift_rate; R0=1, scale_free=false)

Rate model for detector-frame `(mass_1, luminosity_distance)` workflows such
as single-object or one-dimensional mass analyses.
"""
struct CBCSingleMassRate{C<:AbstractCosmology,M,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    cosmology::C
    mass_distribution::M
    redshift_rate::R
    R0::Float64
    scale_free::Bool
end
CBCSingleMassRate(cosmology, mass_distribution, redshift_rate; R0=1.0, scale_free=false) =
    CBCSingleMassRate(cosmology, mass_distribution, redshift_rate, float(R0), Bool(scale_free))

"""
    CBCTotalMassQRate(cosmology, total_mass_distribution, q_distribution, redshift_rate; R0=1, scale_free=false)

Rate model for detector-frame `(total_mass, mass_ratio, luminosity_distance)`.
"""
struct CBCTotalMassQRate{C<:AbstractCosmology,M,Q,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    cosmology::C
    total_mass_distribution::M
    q_distribution::Q
    redshift_rate::R
    R0::Float64
    scale_free::Bool
end
CBCTotalMassQRate(cosmology, total_mass_distribution, q_distribution, redshift_rate; R0=1.0, scale_free=false) =
    CBCTotalMassQRate(cosmology, total_mass_distribution, q_distribution, redshift_rate, float(R0), Bool(scale_free))

"""
    CBCRedshiftPrimaryQRate(cosmology, mass_distribution, q_distribution, redshift_rate; R0=1, scale_free=false)

Rate model for detector-frame `(mass_1, mass_ratio, luminosity_distance)` where
the primary-mass distribution may depend on redshift through
`logpdf(mass_distribution, m1_source, z)`.
"""
struct CBCRedshiftPrimaryQRate{C<:AbstractCosmology,M,Q,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    cosmology::C
    mass_distribution::M
    q_distribution::Q
    redshift_rate::R
    R0::Float64
    scale_free::Bool
end
CBCRedshiftPrimaryQRate(cosmology, mass_distribution, q_distribution, redshift_rate; R0=1.0, scale_free=false) =
    CBCRedshiftPrimaryQRate(cosmology, mass_distribution, q_distribution, redshift_rate, float(R0), Bool(scale_free))

"""
    CBCCatalogVanillaRate(catalog, cosmology, mass_distribution, redshift_rate; Rgal=1, scale_free=false)

Catalog-aware CBC rate for detector-frame `(mass_1, mass_2,
luminosity_distance, sky_indices)` samples. Event weights use the
sky-dependent catalog interpolant; injection weights use the sky-averaged
catalog correction, matching Python `CBC_catalog_vanilla_rate`.
"""
struct CBCCatalogVanillaRate{Cat,C<:AbstractCosmology,B<:AbstractCosmology,M,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    catalog::Cat
    cosmology::C
    background_cosmology::B
    mass_distribution::M
    redshift_rate::R
    Rgal::Float64
    scale_free::Bool
end
function CBCCatalogVanillaRate(catalog, cosmology::AbstractCosmology, mass_distribution, redshift_rate::AbstractRedshiftRate;
    Rgal::Real=1.0, scale_free::Bool=false, background_cosmology=nothing)
    bg = background_cosmology === nothing ? _background_cosmology(cosmology) : background_cosmology
    return CBCCatalogVanillaRate(catalog, cosmology, bg, mass_distribution, redshift_rate, float(Rgal), scale_free)
end

"""
    CBCCatalogSkyMapRate(catalog, cosmology, redshift_rate; Rgal=1, scale_free=false)

Catalog-aware rate for skymap-only event coordinates
`(luminosity_distance, sky_indices)`. Event weights use the sky-dependent
catalog interpolant; injection weights use the Python empty-catalog
completeness correction.
"""
struct CBCCatalogSkyMapRate{Cat,C<:AbstractCosmology,B<:AbstractCosmology,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    catalog::Cat
    cosmology::C
    background_cosmology::B
    redshift_rate::R
    Rgal::Float64
    scale_free::Bool
end
function CBCCatalogSkyMapRate(catalog, cosmology::AbstractCosmology, redshift_rate::AbstractRedshiftRate;
    Rgal::Real=1.0, scale_free::Bool=false, background_cosmology=nothing)
    bg = background_cosmology === nothing ? _background_cosmology(cosmology) : background_cosmology
    return CBCCatalogSkyMapRate(catalog, cosmology, bg, redshift_rate, float(Rgal), scale_free)
end

"""
    CBCVanillaEMCounterpartRate(cosmology, mass_distribution, redshift_rate; R0=1, scale_free=false)

Bright-siren CBC rate for posterior samples carrying an EM redshift column
`:z_EM`. Event likelihood weights follow Python `CBC_vanilla_EM_counterpart`:
the GW posterior samples are first weighted by the vanilla CBC rate, then a
weighted redshift KDE is evaluated at the EM redshift samples. Injection
weights use the vanilla CBC selection correction without EM-observatory bias.
"""
struct CBCVanillaEMCounterpartRate{C<:AbstractCosmology,M,R<:AbstractRedshiftRate} <: AbstractCBCRateModel
    cosmology::C
    mass_distribution::M
    redshift_rate::R
    R0::Float64
    scale_free::Bool
end
CBCVanillaEMCounterpartRate(cosmology::AbstractCosmology, mass_distribution, redshift_rate::AbstractRedshiftRate;
    R0::Real=1.0, scale_free::Bool=false) =
    CBCVanillaEMCounterpartRate(cosmology, mass_distribution, redshift_rate, float(R0), Bool(scale_free))

"""
    CBCLowLatencySkyMapEMCounterpartRate(cosmology, redshift_rate, skymaps; R0=1, scale_free=false, event_names=nothing)

Low-latency bright-siren rate for EM counterpart candidates with columns
`:z_EM`, `:right_ascension`, and `:declination`. The event calculation combines
the full-sky redshift rate with the matching `LigoSkyMap` 3D localization
likelihood. When `event_names` is omitted, skymaps are matched to
`:event1`, `:event2`, ...; a single skymap also works for a single event.
"""
struct CBCLowLatencySkyMapEMCounterpartRate{C<:AbstractCosmology,R<:AbstractRedshiftRate,S} <: AbstractCBCRateModel
    cosmology::C
    redshift_rate::R
    skymaps::Vector{S}
    event_names::Vector{Symbol}
    R0::Float64
    scale_free::Bool
end
function CBCLowLatencySkyMapEMCounterpartRate(cosmology::AbstractCosmology, redshift_rate::AbstractRedshiftRate,
    skymaps; R0::Real=1.0, scale_free::Bool=false, event_names=nothing)
    maps = collect(skymaps)
    !isempty(maps) || throw(ArgumentError("at least one skymap is required"))
    names = event_names === nothing ? [Symbol("event$i") for i in 1:length(maps)] : Symbol.(collect(event_names))
    length(names) == length(maps) || throw(ArgumentError("event_names must have the same length as skymaps"))
    return CBCLowLatencySkyMapEMCounterpartRate(cosmology, redshift_rate, maps, names, float(R0), Bool(scale_free))
end
function CBCLowLatencySkyMapEMCounterpartRate(cosmology::AbstractCosmology, redshift_rate::AbstractRedshiftRate,
    skymaps::AbstractDict; R0::Real=1.0, scale_free::Bool=false)
    names = Symbol.(collect(keys(skymaps)))
    maps = collect(values(skymaps))
    return CBCLowLatencySkyMapEMCounterpartRate(cosmology, redshift_rate, maps; R0, scale_free, event_names=names)
end

"""
    SpinWeightedRate(base_model, spin_prior, spin_columns)

Compose a CBC rate model with an independent spin prior. `spin_columns` must be
either `(:chi_1, :chi_2, :cos_t_1, :cos_t_2)` for `DefaultSpinPrior`-style
component-spin priors or `(:chi_eff, :chi_p)` for `GaussianSpinPrior`-style
effective-spin priors.
"""
struct SpinWeightedRate{B<:AbstractCBCRateModel,S,C<:Tuple} <: AbstractCBCRateModel
    base::B
    spin_prior::S
    spin_columns::C
end
SpinWeightedRate(base::AbstractCBCRateModel, spin_prior) =
    SpinWeightedRate(base, spin_prior, _default_spin_columns(spin_prior))
_default_spin_columns(::DefaultSpinPrior) = (:chi_1, :chi_2, :cos_t_1, :cos_t_2)
_default_spin_columns(::GaussianComponentSpinPrior) = (:chi_1, :chi_2, :cos_t_1, :cos_t_2)
_default_spin_columns(::EvolvingGaussianSpinPrior) = (:chi_1, :chi_2, :cos_t_1, :cos_t_2, :mass_1_source, :mass_2_source)
_default_spin_columns(::BetaWindowGaussianSpinPrior) = (:chi_1, :chi_2, :cos_t_1, :cos_t_2, :mass_1_source, :mass_2_source)
_default_spin_columns(::BetaWindowBetaSpinPrior) = (:chi_1, :chi_2, :cos_t_1, :cos_t_2, :mass_1_source, :mass_2_source)
_default_spin_columns(::ECOTotallyReflectiveSpinPrior) = (:chi_1, :chi_2)
_default_spin_columns(::PSEOBGaussianPrior) = (:domega220, :dtau220)
_default_spin_columns(::GaussianSpinPrior) = (:chi_eff, :chi_p)

const _simple_schema = ParameterSchema(
    ParameterSpec(:alpha; lower=0.5, upper=5.0, default=2.0, description="positive primary-mass power-law slope; density uses -alpha"),
    ParameterSpec(:beta; lower=-2.0, upper=6.0, default=1.0, description="secondary conditional power-law exponent"),
    ParameterSpec(:mmin; lower=2.0, upper=20.0, default=5.0, unit="Msun"),
    ParameterSpec(:mmax; lower=30.0, upper=120.0, default=80.0, unit="Msun"),
    ParameterSpec(:gamma; lower=-4.0, upper=8.0, default=0.0, description="redshift evolution exponent"),
    ParameterSpec(:R0; lower=0.1, upper=200.0, prior=:loguniform, default=25.0, unit="Gpc^-3 yr^-1"),
    ParameterSpec(:H0; lower=50.0, upper=90.0, default=67.7, unit="km s^-1 Mpc^-1"),
    ParameterSpec(:Om0; lower=0.05, upper=0.6, default=0.308),
)

parameter_schema(::Type{SimplePowerLawPopulation}) = _simple_schema
parameter_schema(::SimplePowerLawPopulation) = _simple_schema

"""
    materialize(::Type{SimplePowerLawPopulation}, theta; schema=parameter_schema(...), zmax=10, scale_free=false)

Build an immutable population model from a sampler vector or named parameter
tuple.
"""
function materialize(::Type{SimplePowerLawPopulation}, theta; schema=parameter_schema(SimplePowerLawPopulation), zmax=10.0, scale_free=false)
    nt = theta isa NamedTuple ? theta : unpack(schema, theta)
    nt.mmin < nt.mmax || throw(ArgumentError("mmin must be smaller than mmax"))
    p1 = PowerLaw(nt.mmin, nt.mmax, -nt.alpha)
    p2 = PowerLaw(nt.mmin, nt.mmax, nt.beta)
    return SimplePowerLawPopulation(
        cosmology=FlatLambdaCDM(H0=nt.H0, Om0=nt.Om0, zmax=zmax),
        mass=ConditionalMassDistribution(p1, p2),
        redshift_rate=PowerLawRate(nt.gamma),
        R0=nt.R0,
        scale_free=scale_free,
    )
end
materialize(model::SimplePowerLawPopulation, theta=nothing; kwargs...) = theta === nothing ? model : materialize(SimplePowerLawPopulation, theta; kwargs...)

_rate_scale(model) = model.scale_free ? 0.0 : log(model.R0)
_rate_scale(model::SpinWeightedRate) = _rate_scale(model.base)
_rate_scale(model::Union{CBCCatalogVanillaRate,CBCCatalogSkyMapRate}) = model.scale_free ? 0.0 : log(model.Rgal)
_rate_model_scale_free(model) = hasproperty(model, :scale_free) ? getproperty(model, :scale_free) : false
_rate_model_scale_free(model::SpinWeightedRate) = _rate_model_scale_free(model.base)
_background_cosmology(c::AbstractCosmology) = hasproperty(c, :base) ? getproperty(c, :base) : c
_log_positive(x::Real) = x > 0 && isfinite(x) ? log(x) : -Inf

"""
    MixtureRate(rate1, rate2, lambda_pop)

Convex mixture of two CBC rate models, matching Python `CBC_mixte_pop_rate`:
`log(lambda_pop * rate1 + (1 - lambda_pop) * rate2)`. Component rate models
must accept the same event-coordinate arguments.
"""
struct MixtureRate{R1<:AbstractCBCRateModel,R2<:AbstractCBCRateModel} <: AbstractCBCRateModel
    rate1::R1
    rate2::R2
    lambda_pop::Float64
    scale_free::Bool
    function MixtureRate(rate1::AbstractCBCRateModel, rate2::AbstractCBCRateModel, lambda_pop::Real)
        0 <= lambda_pop <= 1 || throw(ArgumentError("lambda_pop must lie in [0, 1]"))
        scale_free = _rate_model_scale_free(rate1) && _rate_model_scale_free(rate2)
        return new{typeof(rate1),typeof(rate2)}(rate1, rate2, float(lambda_pop), scale_free)
    end
end

"""
    log_event_rate(model, event, prior)

Evaluate detector-frame event log weight contribution for a named event row.
The event must contain either `(mass_1, mass_2, luminosity_distance)`,
`(mass_1, mass_ratio, luminosity_distance)`, or `(chirp_mass, mass_ratio,
luminosity_distance)`, depending on model type. `prior` is the PE or injection
draw prior density in the same detector-frame variables.
"""
function log_event_rate(model::SimplePowerLawPopulation, mass_1, mass_2, luminosity_distance, prior)
    return log_event_rate(CBCVanillaRate(model.cosmology, model.mass, model.redshift_rate; R0=model.R0, scale_free=model.scale_free),
        mass_1, mass_2, luminosity_distance, prior)
end

function log_event_rate(model::CBCVanillaRate, mass_1, mass_2, luminosity_distance, prior)
    return _cbc_vanilla_logweight_no_scale(model.cosmology, model.mass_distribution, model.redshift_rate,
        mass_1, mass_2, luminosity_distance, prior) + _rate_scale(model)
end

function _cbc_vanilla_logweight_no_scale(cosmology::AbstractCosmology, mass_distribution, redshift_rate::AbstractRedshiftRate,
    mass_1, mass_2, luminosity_distance, prior)
    prior > 0 || return -Inf
    m1s, m2s, z = detector_to_source(mass_1, mass_2, luminosity_distance, cosmology)
    logjac = log(detector_to_source_jacobian(z, cosmology))
    mass_logpdf = applicable(logpdf, mass_distribution, m1s, m2s, z) ?
        logpdf(mass_distribution, m1s, m2s, z) : logpdf(mass_distribution, m1s, m2s)
    return mass_logpdf + log_rate(redshift_rate, z) +
        log(dvc_dz(cosmology, z)) - log(prior) - logjac - log1p(z)
end

function log_event_rate(model::CBCMass1Rate, mass_1, q, luminosity_distance, prior)
    prior > 0 || return -Inf
    z = redshift_at_luminosity_distance(model.cosmology, luminosity_distance)
    m1s = mass_1 / (1 + z)
    return logpdf(model.mass_distribution, m1s) + logpdf(model.q_distribution, q) +
        log_rate(model.redshift_rate, z) + log(dvc_dz(model.cosmology, z)) -
        log(prior) - log(detector_to_source_jacobian_q(z, model.cosmology)) - log1p(z) + _rate_scale(model)
end

function log_event_rate(model::CBCMchirpQRate, mchirp, q, luminosity_distance, prior)
    prior > 0 || return -Inf
    z = redshift_at_luminosity_distance(model.cosmology, luminosity_distance)
    mcs = mchirp / (1 + z)
    return logpdf(model.chirp_mass_distribution, mcs) + logpdf(model.q_distribution, q) +
        log_rate(model.redshift_rate, z) + log(dvc_dz(model.cosmology, z)) -
        log(prior) - log(detector_to_source_jacobian_q(z, model.cosmology)) - log1p(z) + _rate_scale(model)
end

function log_event_rate(model::CBCSingleMassRate, mass_1, luminosity_distance, prior)
    prior > 0 || return -Inf
    z = redshift_at_luminosity_distance(model.cosmology, luminosity_distance)
    ms = mass_1 / (1 + z)
    return logpdf(model.mass_distribution, ms) + log_rate(model.redshift_rate, z) +
        log(dvc_dz(model.cosmology, z)) - log(prior) -
        log(detector_to_source_jacobian_single_mass(z, model.cosmology)) - log1p(z) + _rate_scale(model)
end

function log_event_rate(model::CBCTotalMassQRate, total_mass, q, luminosity_distance, prior)
    prior > 0 || return -Inf
    z = redshift_at_luminosity_distance(model.cosmology, luminosity_distance)
    mts = total_mass / (1 + z)
    return logpdf(model.total_mass_distribution, mts) + logpdf(model.q_distribution, q) +
        log_rate(model.redshift_rate, z) + log(dvc_dz(model.cosmology, z)) -
        log(prior) - log(detector_to_source_jacobian_q(z, model.cosmology)) - log1p(z) + _rate_scale(model)
end

function log_event_rate(model::CBCRedshiftPrimaryQRate, mass_1, q, luminosity_distance, prior)
    prior > 0 || return -Inf
    z = redshift_at_luminosity_distance(model.cosmology, luminosity_distance)
    m1s = mass_1 / (1 + z)
    return logpdf(model.mass_distribution, m1s, z) + logpdf(model.q_distribution, q) +
        log_rate(model.redshift_rate, z) + log(dvc_dz(model.cosmology, z)) -
        log(prior) - log(detector_to_source_jacobian_q(z, model.cosmology)) - log1p(z) + _rate_scale(model)
end

function _catalog_effective_logdensity(model, z, sky_index; average::Bool)
    dcat, dbg = Catalog.effective_galaxy_number_interpolant(model.catalog, z, sky_index, model.background_cosmology; average)
    return _log_positive(dcat + dbg)
end

function _catalog_empty_logdensity(model::CBCCatalogSkyMapRate, z)
    dngal = background_effective_galaxy_density(model.catalog.luminosity_function, -Inf, z, model.catalog.abs_magnitude_rate) *
        dvc_dz_dOmega(model.background_cosmology, z)
    return _log_positive(dngal)
end

function log_event_rate(model::CBCCatalogVanillaRate, mass_1, mass_2, luminosity_distance, sky_index, prior)
    prior > 0 || return -Inf
    m1s, m2s, z = detector_to_source(mass_1, mass_2, luminosity_distance, model.cosmology)
    logdensity = _catalog_effective_logdensity(model, z, sky_index; average=false)
    isfinite(logdensity) || return -Inf
    mass_logpdf = applicable(logpdf, model.mass_distribution, m1s, m2s, z) ?
        logpdf(model.mass_distribution, m1s, m2s, z) : logpdf(model.mass_distribution, m1s, m2s)
    return mass_logpdf + log_rate(model.redshift_rate, z) + logdensity -
        log1p(z) - log(detector_to_source_jacobian(z, model.cosmology)) - log(prior) + _rate_scale(model)
end

function log_injection_rate(model::CBCCatalogVanillaRate, mass_1, mass_2, luminosity_distance, sky_index, prior)
    prior > 0 || return -Inf
    m1s, m2s, z = detector_to_source(mass_1, mass_2, luminosity_distance, model.cosmology)
    logdensity = _catalog_effective_logdensity(model, z, sky_index; average=true)
    isfinite(logdensity) || return -Inf
    mass_logpdf = applicable(logpdf, model.mass_distribution, m1s, m2s, z) ?
        logpdf(model.mass_distribution, m1s, m2s, z) : logpdf(model.mass_distribution, m1s, m2s)
    return mass_logpdf + log_rate(model.redshift_rate, z) + logdensity -
        log1p(z) - log(detector_to_source_jacobian(z, model.cosmology)) - log(prior) + _rate_scale(model)
end

function log_event_rate(model::CBCCatalogSkyMapRate, luminosity_distance, sky_index, prior)
    prior > 0 || return -Inf
    z = redshift_at_luminosity_distance(model.cosmology, luminosity_distance)
    logdensity = _catalog_effective_logdensity(model, z, sky_index; average=false)
    isfinite(logdensity) || return -Inf
    return log_rate(model.redshift_rate, z) + logdensity -
        log1p(z) - log(abs(ddl_dz(model.cosmology, z))) - log(prior) + _rate_scale(model)
end

function log_injection_rate(model::CBCCatalogSkyMapRate, luminosity_distance, sky_index, prior)
    prior > 0 || return -Inf
    z = redshift_at_luminosity_distance(model.cosmology, luminosity_distance)
    logdensity = _catalog_empty_logdensity(model, z)
    isfinite(logdensity) || return -Inf
    return log_rate(model.redshift_rate, z) + logdensity -
        log1p(z) - log(abs(ddl_dz(model.cosmology, z))) - log(prior) + _rate_scale(model)
end

function _em_vanilla_base_logweight(model::CBCVanillaEMCounterpartRate, mass_1, mass_2, luminosity_distance, prior)
    return _cbc_vanilla_logweight_no_scale(model.cosmology, model.mass_distribution, model.redshift_rate,
        mass_1, mass_2, luminosity_distance, prior)
end

function log_injection_rate(model::CBCVanillaEMCounterpartRate, mass_1, mass_2, luminosity_distance, prior)
    return _em_vanilla_base_logweight(model, mass_1, mass_2, luminosity_distance, prior) + _rate_scale(model)
end

function log_injection_rate(model::CBCLowLatencySkyMapEMCounterpartRate, luminosity_distance, prior)
    prior > 0 || return -Inf
    z = redshift_at_luminosity_distance(model.cosmology, luminosity_distance)
    return log_rate(model.redshift_rate, z) + log(dvc_dz(model.cosmology, z)) -
        log(prior) - log(abs(ddl_dz(model.cosmology, z))) - log1p(z) + _rate_scale(model)
end

function _skymap_for_event(model::CBCLowLatencySkyMapEMCounterpartRate, event_name::Symbol)
    idx = findfirst(==(event_name), model.event_names)
    idx !== nothing && return model.skymaps[idx]
    length(model.skymaps) == 1 && return only(model.skymaps)
    throw(ArgumentError("no skymap registered for event $event_name"))
end

function log_event_rate(model::SpinWeightedRate, args...)
    nbase = length(args) - length(model.spin_columns) - 1
    nbase >= 1 || throw(ArgumentError("SpinWeightedRate requires base event columns, spin columns, and prior"))
    base_args = args[1:nbase]
    spin_args = args[(nbase + 1):(end - 1)]
    prior = args[end]
    return log_event_rate(model.base, base_args..., prior) + logpdf(model.spin_prior, spin_args...)
end

function log_injection_rate(model::SpinWeightedRate, args...)
    nbase = length(args) - length(model.spin_columns) - 1
    nbase >= 1 || throw(ArgumentError("SpinWeightedRate requires base event columns, spin columns, and prior"))
    base_args = args[1:nbase]
    spin_args = args[(nbase + 1):(end - 1)]
    prior = args[end]
    return log_injection_rate(model.base, base_args..., prior) + logpdf(model.spin_prior, spin_args...)
end

function log_event_rate(model::MixtureRate, args...)
    l1 = model.lambda_pop == 0 ? -Inf : log(model.lambda_pop) + log_event_rate(model.rate1, args...)
    l2 = model.lambda_pop == 1 ? -Inf : log1p(-model.lambda_pop) + log_event_rate(model.rate2, args...)
    return logaddexp(l1, l2)
end

function log_injection_rate(model::MixtureRate, args...)
    l1 = model.lambda_pop == 0 ? -Inf : log(model.lambda_pop) + log_injection_rate(model.rate1, args...)
    l2 = model.lambda_pop == 1 ? -Inf : log1p(-model.lambda_pop) + log_injection_rate(model.rate2, args...)
    return logaddexp(l1, l2)
end

log_injection_rate(model, args...) = log_event_rate(model, args...)

end
