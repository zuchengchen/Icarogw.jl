module Catalog

using ..Cosmology
using QuadGK
using Random

import ..Cosmology: background_effective_galaxy_density,
    log_luminosity_function,
    log_luminosity_pdf,
    luminosity_function,
    luminosity_pdf,
    sample_luminosity_function

export LegacyGalaxyLuminosityFunction,
    galaxy_MF_dep,
    KCorrection,
    DeprecatedKCorrection,
    kcorr,
    kcorr_dep,
    user_normal,
    em_likelihood_prior_differential_volume,
    EM_likelihood_prior_differential_volume,
    catalog_planned

const _LEGACY_GALAXY_BANDS = Set(("W1", "K", "bJ"))
const _MODERN_KCORR_BANDS = Set(("W1-glade+", "K-glade+", "bJ-glade+", "W1-upglade", "g-upglade", "r-upglade"))
const _DEPRECATED_KCORR_BANDS = Set(("W1", "K", "bJ"))

function _legacy_galaxy_band_parameters(band::AbstractString)
    if band == "W1"
        return (-28.0, -16.6, -24.09, -1.12, 1.45e-2 * 1e9)
    elseif band == "bJ"
        return (-22.0, -16.5, -19.66, -1.21, 1.61e-2 * 1e9)
    elseif band == "K"
        return (-27.0, -19.0, -23.39, -1.09, 1.16e-2 * 1e9)
    else
        throw(ArgumentError("unknown legacy galaxy luminosity-function band $band; expected one of $(join(sort(collect(_LEGACY_GALAXY_BANDS)), ", "))"))
    end
end

_little_h(c::Cosmology.AbstractCosmology) =
    hasproperty(c, :H0) ? getproperty(c, :H0) / 100 : _little_h(getproperty(c, :base))

_optional_float(value, name) =
    value === nothing ? throw(ArgumentError("$name must be provided when band is not set")) : float(value)

"""
    LegacyGalaxyLuminosityFunction(; band, cosmology=FlatLambdaCDM())
    LegacyGalaxyLuminosityFunction(; Mmin, Mmax, Mstar, alpha, phistar, cosmology=FlatLambdaCDM())

Dependency-light compatibility helper for Python `catalog.galaxy_MF_dep`. It
uses the legacy `W1`, `K`, and `bJ` band names and stores the observed
little-`h` shifted Schechter parameters. Prefer `GalaxyLuminosityFunction` for
new code.
"""
struct LegacyGalaxyLuminosityFunction
    band::String
    Mmin::Float64
    Mmax::Float64
    Mstar::Float64
    alpha::Float64
    phistar::Float64
    little_h::Float64
    Mminobs::Float64
    Mmaxobs::Float64
    Mstarobs::Float64
    phistarobs::Float64
    norm::Float64
    effective_epsilon::Union{Nothing,Float64}
    effective_delta_grid::Vector{Float64}
    effective_integral_grid::Vector{Float64}
end

function LegacyGalaxyLuminosityFunction(; band=nothing, Mmin=nothing, Mmax=nothing, Mstar=nothing,
    alpha=nothing, phistar=nothing, cosmology::Cosmology.AbstractCosmology=Cosmology.FlatLambdaCDM(),
    epsilon=nothing)
    label = band === nothing ? "" : String(band)
    params = band === nothing ?
        (_optional_float(Mmin, "Mmin"), _optional_float(Mmax, "Mmax"),
            _optional_float(Mstar, "Mstar"), _optional_float(alpha, "alpha"),
            _optional_float(phistar, "phistar")) :
        _legacy_galaxy_band_parameters(label)
    Mminv, Mmaxv, Mstarv, alphav, phistarv = params
    Mminv < Mmaxv || throw(ArgumentError("Mmin must be smaller than Mmax"))
    phistarv > 0 || throw(ArgumentError("phistar must be positive"))
    h = _little_h(cosmology)
    h > 0 || throw(ArgumentError("cosmology H0 must be positive"))
    hshift = 5log10(h)
    Mminobs = Mminv + hshift
    Mmaxobs = Mmaxv + hshift
    Mstarobs = Mstarv + hshift
    phistarobs = phistarv * h^3
    norm = _legacy_luminosity_norm(Mminobs, Mmaxobs, Mstarobs, alphav, phistarobs)
    eps = epsilon === nothing ? nothing : float(epsilon)
    delta_grid, integral_grid = eps === nothing ?
        (Float64[], Float64[]) :
        _legacy_effective_density_grid(Mminv, Mmaxv, Mstarv, alphav, eps)
    return LegacyGalaxyLuminosityFunction(label, Mminv, Mmaxv, Mstarv, alphav, phistarv, h,
        Mminobs, Mmaxobs, Mstarobs, phistarobs, norm, eps, delta_grid, integral_grid)
end
LegacyGalaxyLuminosityFunction(band::AbstractString; kwargs...) =
    LegacyGalaxyLuminosityFunction(; band, kwargs...)

const galaxy_MF_dep = LegacyGalaxyLuminosityFunction

function _legacy_luminosity_norm(Mminobs::Real, Mmaxobs::Real, Mstarobs::Real, alpha::Real, phistarobs::Real)
    value, _ = quadgk(M -> exp(_legacy_log_luminosity_unbounded(M, Mstarobs, alpha, phistarobs)),
        Mminobs, Mmaxobs; rtol=1e-8)
    return value
end

function _legacy_log_luminosity_unbounded(M::Real, Mstarobs::Real, alpha::Real, phistarobs::Real)
    x = 0.4 * (Mstarobs - M)
    return log(0.4 * log(10) * phistarobs) + (alpha + 1) * x * log(10) - 10.0^x
end

function _legacy_effective_density_integral(Mmin::Real, Mmax::Real, Mstar::Real, alpha::Real, epsilon::Real, Mthr::Real)
    xmin = 10.0^(0.4 * (Mstar - Mmax))
    xmax = 10.0^(0.4 * (Mstar - Mthr))
    xmax <= xmin && return 0.0
    value, _ = quadgk(x -> x^(alpha + epsilon) * exp(-x), xmin, xmax; rtol=1e-8)
    return value
end

function _legacy_effective_density_grid(Mmin::Real, Mmax::Real, Mstar::Real, alpha::Real, epsilon::Real)
    mgrid = collect(range(Mmin, Mmax; length=100))
    deltas = reverse(Mstar .- mgrid)
    integrals = reverse(map(M -> _legacy_effective_density_integral(Mmin, Mmax, Mstar, alpha, epsilon, M), mgrid))
    return deltas, integrals
end

function _interp_clamped(x::Real, xs::AbstractVector, ys::AbstractVector)
    length(xs) == length(ys) || throw(ArgumentError("interpolation arrays must have the same length"))
    isempty(xs) && throw(ArgumentError("interpolation arrays must not be empty"))
    x <= first(xs) && return first(ys)
    x >= last(xs) && return last(ys)
    hi = searchsortedfirst(xs, x)
    lo = hi - 1
    t = (x - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + t * (ys[hi] - ys[lo])
end

function log_luminosity_function(g::LegacyGalaxyLuminosityFunction, M::Real)
    g.Mminobs <= M <= g.Mmaxobs || return -Inf
    return _legacy_log_luminosity_unbounded(M, g.Mstarobs, g.alpha, g.phistarobs)
end
log_luminosity_function(g::LegacyGalaxyLuminosityFunction, M::AbstractArray) =
    map(m -> log_luminosity_function(g, m), M)
luminosity_function(g::LegacyGalaxyLuminosityFunction, M) = exp.(log_luminosity_function(g, M))

function log_luminosity_pdf(g::LegacyGalaxyLuminosityFunction, M::Real)
    lval = log_luminosity_function(g, M)
    isfinite(lval) || return -Inf
    return lval - log(g.norm)
end
log_luminosity_pdf(g::LegacyGalaxyLuminosityFunction, M::AbstractArray) =
    map(m -> log_luminosity_pdf(g, m), M)
luminosity_pdf(g::LegacyGalaxyLuminosityFunction, M) = exp.(log_luminosity_pdf(g, M))

function sample_luminosity_function(rng::AbstractRNG, g::LegacyGalaxyLuminosityFunction, n::Integer)
    grid = collect(range(g.Mminobs, g.Mmaxobs; length=10_000))
    weights = luminosity_pdf(g, grid)
    cdf = cumsum(weights)
    total = last(cdf)
    total > 0 && isfinite(total) || throw(ArgumentError("invalid legacy luminosity-function CDF"))
    cdf ./= total
    out = Vector{Float64}(undef, n)
    @inbounds for i in eachindex(out)
        u = rand(rng)
        j = searchsortedfirst(cdf, u)
        if j <= 1
            out[i] = first(grid)
        else
            t = (u - cdf[j - 1]) / (cdf[j] - cdf[j - 1])
            out[i] = grid[j - 1] + t * (grid[j] - grid[j - 1])
        end
    end
    return out
end
sample_luminosity_function(g::LegacyGalaxyLuminosityFunction, n::Integer; rng=Random.default_rng()) =
    sample_luminosity_function(rng, g, n)

function background_effective_galaxy_density(g::LegacyGalaxyLuminosityFunction, Mthr::Real)
    g.effective_epsilon === nothing && throw(ArgumentError(
        "construct LegacyGalaxyLuminosityFunction with epsilon to enable effective galaxy density",
    ))
    return g.phistarobs * _interp_clamped(g.Mstarobs - Mthr, g.effective_delta_grid, g.effective_integral_grid)
end
background_effective_galaxy_density(g::LegacyGalaxyLuminosityFunction, Mthr::AbstractArray) =
    map(M -> background_effective_galaxy_density(g, M), Mthr)

"""
    KCorrection(band)

Dependency-light K-correction formulas from Python `catalog.kcorr`. Supported
bands are `W1-glade+`, `K-glade+`, `bJ-glade+`, `W1-upglade`, `g-upglade`, and
`r-upglade`. The upGLADE bands use the local linearization
`k0 + dkbydz * (z - z0)`.
"""
struct KCorrection
    band::String
    function KCorrection(band::AbstractString)
        band in _MODERN_KCORR_BANDS || throw(ArgumentError(
            "unknown K-correction band $band; expected one of $(join(sort(collect(_MODERN_KCORR_BANDS)), ", "))",
        ))
        return new(String(band))
    end
end

const kcorr = KCorrection

_required_kcorr(value, name) =
    value === nothing ? throw(ArgumentError("$name must be provided for upGLADE K-corrections")) : value

function (k::KCorrection)(z::Real; k0=nothing, dkbydz=nothing, z0=nothing)
    if k.band == "W1-glade+"
        return -(4.44e-2 + 2.67z + 1.33z^2 - 1.59z^3)
    elseif k.band == "K-glade+"
        return -6.0 * log10(1 + z)
    elseif k.band == "bJ-glade+"
        return (z + 6z^2) / (1 + 20z^3)
    else
        return _required_kcorr(k0, "k0") + _required_kcorr(dkbydz, "dkbydz") * (z - _required_kcorr(z0, "z0"))
    end
end
(k::KCorrection)(z::AbstractArray; k0=nothing, dkbydz=nothing, z0=nothing) =
    k.band in ("W1-upglade", "g-upglade", "r-upglade") ?
    _required_kcorr(k0, "k0") .+ _required_kcorr(dkbydz, "dkbydz") .* (z .- _required_kcorr(z0, "z0")) :
    map(zi -> k(zi), z)

"""
    DeprecatedKCorrection(band)

Legacy K-correction formulas from Python `catalog.kcorr_dep`. Supported bands
are `W1`, `K`, and `bJ`.
"""
struct DeprecatedKCorrection
    band::String
    function DeprecatedKCorrection(band::AbstractString)
        band in _DEPRECATED_KCORR_BANDS || throw(ArgumentError(
            "unknown deprecated K-correction band $band; expected one of $(join(sort(collect(_DEPRECATED_KCORR_BANDS)), ", "))",
        ))
        return new(String(band))
    end
end

const kcorr_dep = DeprecatedKCorrection

function (k::DeprecatedKCorrection)(z::Real)
    if k.band == "W1"
        return -(4.44e-2 + 2.67z + 1.33z^2 - 1.59z^3)
    elseif k.band == "K"
        return -6.0 * log10(1 + z)
    else
        return (z + 6z^2) / (1 + 15z^3)
    end
end
(k::DeprecatedKCorrection)(z::AbstractArray) = map(zi -> k(zi), z)

"""
    user_normal(x, mu, sigma)

Normalized Gaussian density used by the Python catalog EM helper.
"""
function user_normal(x::Real, mu::Real, sigma::Real)
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    return inv(sqrt(2pi) * sigma) * exp(-0.5 * ((x - mu) / sigma)^2)
end
user_normal(x::AbstractArray, mu::Real, sigma::Real) = map(xi -> user_normal(xi, mu, sigma), x)

_cosmology_zmax(c::Cosmology.AbstractCosmology) =
    hasproperty(c, :zmax) ? getproperty(c, :zmax) : _cosmology_zmax(getproperty(c, :base))

function _trapz(y::AbstractVector, x::AbstractVector)
    length(y) == length(x) || throw(ArgumentError("trapz inputs must have the same length"))
    length(y) >= 2 || return 0.0
    total = 0.0
    @inbounds for i in 2:length(y)
        total += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    end
    return total
end

function _em_likelihood_scalar(z::Real, zobs::Real, sigmaz::Real, cosmology::Cosmology.AbstractCosmology,
    numsigma::Real, ptype::AbstractString)
    sigmaz > 0 || throw(ArgumentError("sigmaz must be positive"))
    zvalmin = max(1e-6, zobs - numsigma * sigmaz)
    zvalmax = min(zobs + 5sigmaz, _cosmology_zmax(cosmology))
    zvalmax <= zvalmin && return 0.0

    if ptype == "uniform"
        in_window = zobs - numsigma * sigmaz <= z <= zobs + numsigma * sigmaz
        in_window || return 0.0
        denom = Cosmology.comoving_volume(cosmology, zvalmax) - Cosmology.comoving_volume(cosmology, zvalmin)
        denom > 0 && isfinite(denom) || throw(ArgumentError("uniform EM prior normalization failed"))
        return 4pi * Cosmology.dvc_dz_dOmega(cosmology, z) / denom
    elseif ptype == "gaussian"
        zproxy = collect(range(zvalmin, zvalmax; length=5000))
        y = Cosmology.dvc_dz_dOmega(cosmology, zproxy) .* user_normal(zproxy, zobs, sigmaz)
        normfact = _trapz(y, zproxy)
        normfact > 0 && isfinite(normfact) || throw(ArgumentError("gaussian EM prior normalization failed"))
        return Cosmology.dvc_dz_dOmega(cosmology, z) * user_normal(z, zobs, sigmaz) / normfact
    elseif ptype == "gaussian_nocom"
        zproxy = collect(range(zvalmin, zvalmax; length=5000))
        y = user_normal(zproxy, zobs, sigmaz)
        normfact = _trapz(y, zproxy)
        normfact > 0 && isfinite(normfact) || throw(ArgumentError("gaussian_nocom EM prior normalization failed"))
        return user_normal(z, zobs, sigmaz) / normfact
    else
        throw(ArgumentError("unknown EM likelihood prior type $ptype; expected uniform, gaussian, or gaussian_nocom"))
    end
end

"""
    em_likelihood_prior_differential_volume(z, zobs, sigmaz, cosmology; Numsigma=3, ptype="uniform")

Redshift likelihood times prior helper from Python
`EM_likelihood_prior_differential_volume`. `ptype="uniform"` multiplies a
top-hat EM redshift likelihood by a uniform-in-comoving-volume prior;
`"gaussian"` uses a Gaussian EM likelihood with the same comoving-volume
factor; `"gaussian_nocom"` normalizes the Gaussian without the comoving-volume
factor.
"""
function em_likelihood_prior_differential_volume(z::Real, zobs::Real, sigmaz::Real,
    cosmology::Cosmology.AbstractCosmology; Numsigma::Real=3.0, ptype="uniform")
    return _em_likelihood_scalar(z, zobs, sigmaz, cosmology, Numsigma, String(ptype))
end
function em_likelihood_prior_differential_volume(z::AbstractArray, zobs::Real, sigmaz::Real,
    cosmology::Cosmology.AbstractCosmology; Numsigma::Real=3.0, ptype="uniform")
    ptype_string = String(ptype)
    sigmaz > 0 || throw(ArgumentError("sigmaz must be positive"))
    zvalmin = max(1e-6, zobs - Numsigma * sigmaz)
    zvalmax = min(zobs + 5sigmaz, _cosmology_zmax(cosmology))
    zvalmax <= zvalmin && return zeros(Float64, size(z))

    if ptype_string == "uniform"
        denom = Cosmology.comoving_volume(cosmology, zvalmax) - Cosmology.comoving_volume(cosmology, zvalmin)
        denom > 0 && isfinite(denom) || throw(ArgumentError("uniform EM prior normalization failed"))
        lo = zobs - Numsigma * sigmaz
        hi = zobs + Numsigma * sigmaz
        return map(zi -> lo <= zi <= hi ? 4pi * Cosmology.dvc_dz_dOmega(cosmology, zi) / denom : 0.0, z)
    elseif ptype_string == "gaussian"
        zproxy = collect(range(zvalmin, zvalmax; length=5000))
        y = Cosmology.dvc_dz_dOmega(cosmology, zproxy) .* user_normal(zproxy, zobs, sigmaz)
        normfact = _trapz(y, zproxy)
        normfact > 0 && isfinite(normfact) || throw(ArgumentError("gaussian EM prior normalization failed"))
        return map(zi -> Cosmology.dvc_dz_dOmega(cosmology, zi) * user_normal(zi, zobs, sigmaz) / normfact, z)
    elseif ptype_string == "gaussian_nocom"
        zproxy = collect(range(zvalmin, zvalmax; length=5000))
        y = user_normal(zproxy, zobs, sigmaz)
        normfact = _trapz(y, zproxy)
        normfact > 0 && isfinite(normfact) || throw(ArgumentError("gaussian_nocom EM prior normalization failed"))
        return map(zi -> user_normal(zi, zobs, sigmaz) / normfact, z)
    else
        throw(ArgumentError("unknown EM likelihood prior type $ptype_string; expected uniform, gaussian, or gaussian_nocom"))
    end
end

const EM_likelihood_prior_differential_volume = em_likelihood_prior_differential_volume

"""
    catalog_planned()

Galaxy catalog, dark-siren, and bright-siren functionality is planned but not
implemented in the first native Julia version. This placeholder exists so users
get an explicit error instead of a silent partial implementation.
"""
function catalog_planned()
    throw(ErrorException("Catalog and EM-counterpart functionality is planned, not implemented in Icarogw.jl first-version scope."))
end

end
